#include <cuda.h>
#include <nvrtc.h>

#include <iostream>
#include <string>
#include <vector>

#define CHECK_DRV(expr)                                                        \
    do {                                                                       \
        CUresult result = (expr);                                              \
        if (result != CUDA_SUCCESS) {                                          \
            const char *name = nullptr;                                        \
            const char *text = nullptr;                                        \
            cuGetErrorName(result, &name);                                     \
            cuGetErrorString(result, &text);                                   \
            std::cerr << #expr << " failed: "                                 \
                      << (name ? name : "unknown") << " "                    \
                      << (text ? text : "") << std::endl;                     \
            return 1;                                                          \
        }                                                                      \
    } while (0)

#define CHECK_NVRTC(expr)                                                      \
    do {                                                                       \
        nvrtcResult result = (expr);                                           \
        if (result != NVRTC_SUCCESS) {                                         \
            std::cerr << #expr << " failed: "                                 \
                      << nvrtcGetErrorString(result) << std::endl;             \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static int compile_cubin(std::vector<char> *cubin, int major, int minor)
{
    static const char source[] = R"(
extern "C" __global__ void cooperative_kernel(int *out, int value)
{
    out[0] = value + static_cast<int>(blockIdx.x);
}
)";

    nvrtcProgram program;
    CHECK_NVRTC(nvrtcCreateProgram(&program, source, "test_cooperative_launch_extra.cu", 0, nullptr, nullptr));

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
        std::cerr << "nvrtcCompileProgram failed: "
                  << nvrtcGetErrorString(compile_result) << std::endl;
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

int main()
{
    CHECK_DRV(cuInit(0));
    CUdevice device;
    CHECK_DRV(cuDeviceGet(&device, 0));

    int cooperative = 0;
    CHECK_DRV(cuDeviceGetAttribute(&cooperative, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, device));
    if (!cooperative) {
        std::cout << "cooperative launch unsupported; skipped" << std::endl;
        return 0;
    }

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

    CUmodule module;
    CHECK_DRV(cuModuleLoadData(&module, cubin.data()));
    CUfunction function;
    CHECK_DRV(cuModuleGetFunction(&function, module, "cooperative_kernel"));

    CUdeviceptr d_output = 0;
    CHECK_DRV(cuMemAlloc(&d_output, sizeof(int)));
    CHECK_DRV(cuMemsetD32(d_output, 0, 1));

    int value = 41;
    void *args[] = {&d_output, &value};
    CHECK_DRV(cuLaunchCooperativeKernel(function,
                                        1, 1, 1,
                                        1, 1, 1,
                                        0,
                                        nullptr,
                                        args));
    CHECK_DRV(cuCtxSynchronize());

    int output = 0;
    CHECK_DRV(cuMemcpyDtoH(&output, d_output, sizeof(output)));
    if (output != value) {
        std::cerr << "cooperative launch output mismatch" << std::endl;
        return 1;
    }

    CHECK_DRV(cuMemFree(d_output));
    CHECK_DRV(cuModuleUnload(module));
    CHECK_DRV(cuDevicePrimaryCtxRelease(device));

    std::cout << "cooperative launch API test passed" << std::endl;
    return 0;
}
