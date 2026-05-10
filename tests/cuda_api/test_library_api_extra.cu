#include <cuda.h>
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
    unlink(cubin_path.c_str());

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

    CUdeviceptr d_output = 0;
    CHECK_DRV(cuMemAlloc(&d_output, sizeof(int)));

    if (launch_and_check(reinterpret_cast<CUfunction>(kernel), d_output, 42) != 0) {
        return 1;
    }

    CUfunction function;
    CHECK_DRV(cuKernelGetFunction(&function, kernel));
    if (launch_and_check(function, d_output, 77) != 0) {
        return 1;
    }

    CUmodule module;
    CHECK_DRV(cuLibraryGetModule(&module, library));
    CUfunction module_function;
    CHECK_DRV(cuModuleGetFunction(&module_function, module, "library_kernel"));
    if (launch_and_check(module_function, d_output, 91) != 0) {
        return 1;
    }

    CHECK_DRV(cuMemFree(d_output));
    CHECK_DRV(cuLibraryUnload(library));
    CHECK_DRV(cuDevicePrimaryCtxRelease(device));

    std::cout << "library API test passed" << std::endl;
    return 0;
}
