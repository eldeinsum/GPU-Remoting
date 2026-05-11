#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_DRV(expr)                                                                                               \
    do {                                                                                                               \
        CUresult result = (expr);                                                                                      \
        if (result != CUDA_SUCCESS) {                                                                                   \
            const char *name = nullptr;                                                                                 \
            const char *text = nullptr;                                                                                 \
            cuGetErrorName(result, &name);                                                                              \
            cuGetErrorString(result, &text);                                                                            \
            std::cerr << #expr << " failed: " << (name ? name : "unknown") << " " << (text ? text : "")            \
                      << std::endl;                                                                                    \
            return 1;                                                                                                   \
        }                                                                                                               \
    } while (0)

#define CHECK_CUDA(expr)                                                                                              \
    do {                                                                                                               \
        cudaError_t result = (expr);                                                                                   \
        if (result != cudaSuccess) {                                                                                   \
            std::cerr << #expr << " failed: " << cudaGetErrorString(result) << " (" << static_cast<int>(result)       \
                      << ")" << std::endl;                                                                            \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

#define CHECK_NVRTC(expr)                                                                                              \
    do {                                                                                                               \
        nvrtcResult result = (expr);                                                                                    \
        if (result != NVRTC_SUCCESS) {                                                                                  \
            std::cerr << #expr << " failed: " << nvrtcGetErrorString(result) << std::endl;                            \
            return 1;                                                                                                   \
        }                                                                                                               \
    } while (0)

static int compile_cubin(std::vector<char> *cubin, int major, int minor)
{
    static const char source[] = R"(
extern "C" __device__ int device_value = 3;
extern "C" __managed__ int managed_value = 19;

extern "C" __global__ void library_kernel(int *out, int value)
{
    out[0] = value + device_value;
}
)";

    nvrtcProgram program;
    CHECK_NVRTC(nvrtcCreateProgram(&program, source, "test_library_api_extra.cu", 0, nullptr, nullptr));

    std::string arch = "--gpu-architecture=sm_" + std::to_string(major) + std::to_string(minor);
    const char *options[] = {arch.c_str(), "--std=c++11"};
    nvrtcResult compile_result = nvrtcCompileProgram(program, 2, options);
    size_t log_size = 0;
    nvrtcGetProgramLogSize(program, &log_size);
    if (log_size > 1) {
        std::string log(log_size, '\0');
        nvrtcGetProgramLog(program, &log[0]);
        std::cerr << log << std::endl;
    }
    if (compile_result != NVRTC_SUCCESS) {
        std::cerr << "nvrtcCompileProgram failed: " << nvrtcGetErrorString(compile_result) << std::endl;
        nvrtcDestroyProgram(&program);
        return 1;
    }

    size_t cubin_size = 0;
    CHECK_NVRTC(nvrtcGetCUBINSize(program, &cubin_size));
    cubin->resize(cubin_size);
    CHECK_NVRTC(nvrtcGetCUBIN(program, cubin->data()));
    CHECK_NVRTC(nvrtcDestroyProgram(&program));
    return 0;
}

static int launch_and_check(CUfunction function, CUdeviceptr d_output, int expected)
{
    CHECK_DRV(cuMemsetD32(d_output, 0, 1));
    int value = expected - 11;
    void *args[] = {
        (void *)&d_output,
        (void *)&value,
    };
    CHECK_DRV(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
    CHECK_DRV(cuCtxSynchronize());
    int output = 0;
    CHECK_DRV(cuMemcpyDtoH(&output, d_output, sizeof(output)));
    if (output != expected) {
        std::cerr << "library kernel produced " << output << " expected " << expected << std::endl;
        return 1;
    }
    return 0;
}

static int runtime_launch_and_check(cudaKernel_t kernel, void *d_output, int value, int expected)
{
    CHECK_CUDA(cudaMemset(d_output, 0, sizeof(int)));
    void *args[] = {
        (void *)&d_output,
        (void *)&value,
    };
    CHECK_CUDA(cudaLaunchKernel((const void *)kernel, dim3(1), dim3(1), args, 0, nullptr));
    CHECK_CUDA(cudaDeviceSynchronize());
    int output = 0;
    CHECK_CUDA(cudaMemcpy(&output, d_output, sizeof(output), cudaMemcpyDeviceToHost));
    if (output != expected) {
        std::cerr << "runtime library kernel produced " << output << " expected " << expected << std::endl;
        return 1;
    }
    return 0;
}

static int runtime_cooperative_launch_and_check(cudaKernel_t kernel, void *d_output, int value, int expected)
{
    CHECK_CUDA(cudaMemset(d_output, 0, sizeof(int)));
    void *args[] = {
        (void *)&d_output,
        (void *)&value,
    };
    CHECK_CUDA(cudaLaunchCooperativeKernel((const void *)kernel, dim3(1), dim3(1), args, 0, nullptr));
    CHECK_CUDA(cudaDeviceSynchronize());
    int output = 0;
    CHECK_CUDA(cudaMemcpy(&output, d_output, sizeof(output), cudaMemcpyDeviceToHost));
    if (output != expected) {
        std::cerr << "runtime cooperative library kernel produced " << output << " expected " << expected << std::endl;
        return 1;
    }
    return 0;
}

static int write_temp_cubin(const std::vector<char> &cubin, std::string *path)
{
    *path = "/tmp/gpu_remoting_library_api_" + std::to_string(getpid()) + ".cubin";
    std::ofstream file(*path, std::ios::binary);
    if (!file) {
        std::cerr << "failed to open temporary cubin path: " << *path << std::endl;
        return 1;
    }
    file.write(cubin.data(), cubin.size());
    return file.good() ? 0 : 1;
}

int main()
{
    CHECK_DRV(cuInit(0));
    CUdevice device;
    CHECK_DRV(cuDeviceGet(&device, 0));

    int major = 0;
    int minor = 0;
    CHECK_DRV(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CHECK_DRV(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

    CUcontext context;
    CHECK_DRV(cuDevicePrimaryCtxRetain(&context, device));
    CHECK_DRV(cuCtxSetCurrent(context));

    std::vector<char> cubin;
    if (compile_cubin(&cubin, major, minor) != 0) {
        return 1;
    }

    CUlibrary file_library;
    std::string cubin_path;
    if (write_temp_cubin(cubin, &cubin_path) != 0) {
        return 1;
    }
    CHECK_DRV(cuLibraryLoadFromFile(&file_library, cubin_path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
    CHECK_DRV(cuLibraryUnload(file_library));

    CUlibrary library;
    CHECK_DRV(cuLibraryLoadData(&library, cubin.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));

    CUkernel kernel;
    CHECK_DRV(cuLibraryGetKernel(&kernel, library, "library_kernel"));

    CUlibrary owner;
    CHECK_DRV(cuKernelGetLibrary(&owner, kernel));
    if (owner != library) {
        std::cerr << "cuKernelGetLibrary returned the wrong library" << std::endl;
        return 1;
    }

    const char *kernel_name = nullptr;
    CHECK_DRV(cuKernelGetName(&kernel_name, kernel));
    if (kernel_name == nullptr || std::strcmp(kernel_name, "library_kernel") != 0) {
        std::cerr << "cuKernelGetName returned " << (kernel_name ? kernel_name : "(null)") << std::endl;
        return 1;
    }

    unsigned int kernel_count = 0;
    CHECK_DRV(cuLibraryGetKernelCount(&kernel_count, library));
    if (kernel_count != 1) {
        std::cerr << "unexpected library kernel count: " << kernel_count << std::endl;
        return 1;
    }
    std::vector<CUkernel> kernels(kernel_count);
    CHECK_DRV(cuLibraryEnumerateKernels(kernels.data(), kernel_count, library));
    if (kernels[0] == nullptr) {
        std::cerr << "cuLibraryEnumerateKernels returned null" << std::endl;
        return 1;
    }
    const char *enumerated_kernel_name = nullptr;
    CHECK_DRV(cuKernelGetName(&enumerated_kernel_name, kernels[0]));
    if (enumerated_kernel_name == nullptr || std::strcmp(enumerated_kernel_name, "library_kernel") != 0) {
        std::cerr << "cuKernelGetName for enumerated kernel returned "
                  << (enumerated_kernel_name ? enumerated_kernel_name : "(null)") << std::endl;
        return 1;
    }
    CUlibrary enumerated_owner;
    CHECK_DRV(cuKernelGetLibrary(&enumerated_owner, kernels[0]));
    if (enumerated_owner != library) {
        std::cerr << "cuKernelGetLibrary returned the wrong library for enumerated kernel" << std::endl;
        return 1;
    }

    int max_threads = 0;
    CHECK_DRV(cuKernelGetAttribute(&max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel, device));
    if (max_threads < 1) {
        std::cerr << "unexpected max threads per block: " << max_threads << std::endl;
        return 1;
    }

    size_t param_count = 0;
    CUresult param_result = cuKernelGetParamCount(kernel, &param_count);
    if (param_result == CUDA_SUCCESS) {
        if (param_count != 2) {
            std::cerr << "unexpected kernel param count: " << param_count << std::endl;
            return 1;
        }
        size_t offset = 0;
        size_t size = 0;
        CHECK_DRV(cuKernelGetParamInfo(kernel, 0, &offset, &size));
        if (size != sizeof(CUdeviceptr)) {
            std::cerr << "unexpected first param size: " << size << std::endl;
            return 1;
        }
    } else if (param_result != CUDA_ERROR_NOT_SUPPORTED) {
        CHECK_DRV(param_result);
    }

    CUdeviceptr d_global = 0;
    size_t global_size = 0;
    CHECK_DRV(cuLibraryGetGlobal(&d_global, &global_size, library, "device_value"));
    if (global_size != sizeof(int)) {
        std::cerr << "unexpected global size: " << global_size << std::endl;
        return 1;
    }
    int global_value = 11;
    CHECK_DRV(cuMemcpyHtoD(d_global, &global_value, sizeof(global_value)));

    CUdeviceptr d_managed = 0;
    size_t managed_size = 0;
    CHECK_DRV(cuLibraryGetManaged(&d_managed, &managed_size, library, "managed_value"));
    if (managed_size != sizeof(int)) {
        std::cerr << "unexpected managed size: " << managed_size << std::endl;
        return 1;
    }
    int managed_value = 29;
    CHECK_DRV(cuMemcpyHtoD(d_managed, &managed_value, sizeof(managed_value)));
    int managed_output = 0;
    CHECK_DRV(cuMemcpyDtoH(&managed_output, d_managed, sizeof(managed_output)));
    if (managed_output != managed_value) {
        std::cerr << "managed value mismatch: " << managed_output << std::endl;
        return 1;
    }

    CUdeviceptr d_output = 0;
    CHECK_DRV(cuMemAlloc(&d_output, sizeof(int)));

    if (launch_and_check(reinterpret_cast<CUfunction>(kernels[0]), d_output, 33) != 0) {
        return 1;
    }

    CUfunction enumerated_function;
    CHECK_DRV(cuKernelGetFunction(&enumerated_function, kernels[0]));
    const char *enumerated_function_name = nullptr;
    CHECK_DRV(cuFuncGetName(&enumerated_function_name, enumerated_function));
    if (enumerated_function_name == nullptr || std::strcmp(enumerated_function_name, "library_kernel") != 0) {
        std::cerr << "cuFuncGetName for enumerated kernel returned "
                  << (enumerated_function_name ? enumerated_function_name : "(null)") << std::endl;
        return 1;
    }
    if (launch_and_check(enumerated_function, d_output, 55) != 0) {
        return 1;
    }

    if (launch_and_check(reinterpret_cast<CUfunction>(kernel), d_output, 42) != 0) {
        return 1;
    }

    CUfunction function;
    CHECK_DRV(cuKernelGetFunction(&function, kernel));
    const char *function_name = nullptr;
    CHECK_DRV(cuFuncGetName(&function_name, function));
    if (function_name == nullptr || std::strcmp(function_name, "library_kernel") != 0) {
        std::cerr << "cuFuncGetName returned " << (function_name ? function_name : "(null)") << std::endl;
        return 1;
    }
    size_t function_param_count = 0;
    CHECK_DRV(cuFuncGetParamCount(function, &function_param_count));
    if (function_param_count != 2) {
        std::cerr << "unexpected function param count: " << function_param_count << std::endl;
        return 1;
    }
    size_t function_param_offset = 0;
    size_t function_param_size = 0;
    CHECK_DRV(cuFuncGetParamInfo(function, 0, &function_param_offset, &function_param_size));
    if (function_param_size != sizeof(CUdeviceptr)) {
        std::cerr << "unexpected function first param size: " << function_param_size << std::endl;
        return 1;
    }
    CUmodule function_module = nullptr;
    CHECK_DRV(cuFuncGetModule(&function_module, function));
    if (function_module == nullptr) {
        std::cerr << "cuFuncGetModule returned null" << std::endl;
        return 1;
    }
    CUfunctionLoadingState loading_state;
    CHECK_DRV(cuFuncIsLoaded(&loading_state, function));
    CHECK_DRV(cuFuncLoad(function));
    if (launch_and_check(function, d_output, 77) != 0) {
        return 1;
    }

    CUmodule module;
    CHECK_DRV(cuLibraryGetModule(&module, library));
    CUfunction module_function;
    CHECK_DRV(cuModuleGetFunction(&module_function, module, "library_kernel"));
    CHECK_DRV(cuFuncGetName(&function_name, module_function));
    if (function_name == nullptr || std::strcmp(function_name, "library_kernel") != 0) {
        std::cerr << "cuFuncGetName for module function returned "
                  << (function_name ? function_name : "(null)") << std::endl;
        return 1;
    }
    CHECK_DRV(cuFuncGetModule(&function_module, module_function));
    if (function_module != module) {
        std::cerr << "cuFuncGetModule returned the wrong module" << std::endl;
        return 1;
    }
    if (launch_and_check(module_function, d_output, 91) != 0) {
        return 1;
    }

    cudaLibrary_t runtime_file_library;
    CHECK_CUDA(cudaLibraryLoadFromFile(&runtime_file_library, cubin_path.c_str(), nullptr, nullptr, 0, nullptr, nullptr,
                                       0));
    CHECK_CUDA(cudaLibraryUnload(runtime_file_library));
    unlink(cubin_path.c_str());

    cudaLibrary_t runtime_library;
    CHECK_CUDA(cudaLibraryLoadData(&runtime_library, cubin.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));

    cudaKernel_t runtime_kernel;
    CHECK_CUDA(cudaLibraryGetKernel(&runtime_kernel, runtime_library, "library_kernel"));

    unsigned int runtime_kernel_count = 0;
    CHECK_CUDA(cudaLibraryGetKernelCount(&runtime_kernel_count, runtime_library));
    if (runtime_kernel_count != 1) {
        std::cerr << "unexpected runtime library kernel count: " << runtime_kernel_count << std::endl;
        return 1;
    }
    std::vector<cudaKernel_t> runtime_kernels(runtime_kernel_count);
    CHECK_CUDA(cudaLibraryEnumerateKernels(runtime_kernels.data(), runtime_kernel_count, runtime_library));
    if (runtime_kernels[0] == nullptr) {
        std::cerr << "cudaLibraryEnumerateKernels returned null" << std::endl;
        return 1;
    }

    void *runtime_global = nullptr;
    size_t runtime_global_size = 0;
    CHECK_CUDA(cudaLibraryGetGlobal(&runtime_global, &runtime_global_size, runtime_library, "device_value"));
    if (runtime_global_size != sizeof(int)) {
        std::cerr << "unexpected runtime global size: " << runtime_global_size << std::endl;
        return 1;
    }
    int runtime_global_value = 17;
    CHECK_CUDA(cudaMemcpy(runtime_global, &runtime_global_value, sizeof(runtime_global_value), cudaMemcpyHostToDevice));

    void *runtime_managed = nullptr;
    size_t runtime_managed_size = 0;
    CHECK_CUDA(cudaLibraryGetManaged(&runtime_managed, &runtime_managed_size, runtime_library, "managed_value"));
    if (runtime_managed_size != sizeof(int)) {
        std::cerr << "unexpected runtime managed size: " << runtime_managed_size << std::endl;
        return 1;
    }
    int runtime_managed_value = 41;
    CHECK_CUDA(
        cudaMemcpy(runtime_managed, &runtime_managed_value, sizeof(runtime_managed_value), cudaMemcpyHostToDevice));
    int runtime_managed_output = 0;
    CHECK_CUDA(
        cudaMemcpy(&runtime_managed_output, runtime_managed, sizeof(runtime_managed_output), cudaMemcpyDeviceToHost));
    if (runtime_managed_output != runtime_managed_value) {
        std::cerr << "runtime managed value mismatch: " << runtime_managed_output << std::endl;
        return 1;
    }

    cudaFuncAttributes runtime_attributes;
    CHECK_CUDA(cudaFuncGetAttributes(&runtime_attributes, (const void *)runtime_kernel));
    if (runtime_attributes.maxThreadsPerBlock < 1) {
        std::cerr << "unexpected runtime max threads per block: " << runtime_attributes.maxThreadsPerBlock << std::endl;
        return 1;
    }
    CHECK_CUDA(cudaKernelSetAttributeForDevice(runtime_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 0, 0));

    void *runtime_output = nullptr;
    CHECK_CUDA(cudaMalloc(&runtime_output, sizeof(int)));
    if (runtime_launch_and_check(runtime_kernel, runtime_output, 23, 40) != 0) {
        return 1;
    }
    if (runtime_launch_and_check(runtime_kernels[0], runtime_output, 31, 48) != 0) {
        return 1;
    }
    int runtime_cooperative = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&runtime_cooperative, cudaDevAttrCooperativeLaunch, 0));
    if (runtime_cooperative) {
        if (runtime_cooperative_launch_and_check(runtime_kernel, runtime_output, 37, 54) != 0) {
            return 1;
        }
    } else {
        std::cout << "runtime cooperative launch unsupported; skipped" << std::endl;
    }
    CHECK_CUDA(cudaFree(runtime_output));
    CHECK_CUDA(cudaLibraryUnload(runtime_library));

    CHECK_DRV(cuMemFree(d_output));
    CHECK_DRV(cuLibraryUnload(library));
    CHECK_DRV(cuDevicePrimaryCtxRelease(device));

    std::cout << "library API test passed" << std::endl;
    return 0;
}
