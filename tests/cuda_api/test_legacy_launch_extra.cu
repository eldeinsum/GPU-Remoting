#include <cuda.h>
#include <nvrtc.h>

#include <algorithm>
#include <cstring>
#include <iostream>
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
extern "C" __global__ void legacy_kernel(int *out, int value, float delta)
{
    out[blockIdx.x] = value + static_cast<int>(delta) + static_cast<int>(blockIdx.x);
}
)";

    nvrtcProgram program;
    CHECK_NVRTC(nvrtcCreateProgram(&program, source, "test_legacy_launch_extra.cu", 0, nullptr, nullptr));

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

static int configure_params(CUfunction function,
                            CUdeviceptr d_output,
                            int value,
                            float delta,
                            size_t param_size,
                            size_t output_offset,
                            size_t value_offset,
                            size_t delta_offset)
{
    CHECK_DRV(cuParamSetv(function, static_cast<int>(output_offset), &d_output, sizeof(d_output)));
    CHECK_DRV(cuParamSeti(function, static_cast<int>(value_offset), static_cast<unsigned int>(value)));
    CHECK_DRV(cuParamSetf(function, static_cast<int>(delta_offset), delta));
    CHECK_DRV(cuParamSetSize(function, static_cast<unsigned int>(param_size)));
    return 0;
}

static int expect_values(CUdeviceptr d_output, const std::vector<int> &expected)
{
    std::vector<int> output(expected.size(), 0);
    CHECK_DRV(cuMemcpyDtoH(output.data(), d_output, output.size() * sizeof(int)));
    if (output != expected) {
        std::cerr << "legacy launch output mismatch" << std::endl;
        return 1;
    }
    return 0;
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

    CUmodule module;
    CHECK_DRV(cuModuleLoadData(&module, cubin.data()));
    CUfunction function;
    CHECK_DRV(cuModuleGetFunction(&function, module, "legacy_kernel"));

    size_t output_offset = 0;
    size_t output_size = 0;
    size_t value_offset = 0;
    size_t value_size = 0;
    size_t delta_offset = 0;
    size_t delta_size = 0;
    CHECK_DRV(cuFuncGetParamInfo(function, 0, &output_offset, &output_size));
    CHECK_DRV(cuFuncGetParamInfo(function, 1, &value_offset, &value_size));
    CHECK_DRV(cuFuncGetParamInfo(function, 2, &delta_offset, &delta_size));

    size_t param_size = std::max(output_offset + output_size,
                        std::max(value_offset + value_size,
                                 delta_offset + delta_size));

    CUdeviceptr d_output = 0;
    CHECK_DRV(cuMemAlloc(&d_output, 2 * sizeof(int)));
    CHECK_DRV(cuFuncSetBlockShape(function, 1, 1, 1));
    CHECK_DRV(cuFuncSetSharedSize(function, 0));

    CHECK_DRV(cuMemsetD32(d_output, 0, 2));
    if (configure_params(function, d_output, 7, 5.0f, param_size,
                         output_offset, value_offset, delta_offset) != 0) {
        return 1;
    }
    CHECK_DRV(cuLaunch(function));
    CHECK_DRV(cuCtxSynchronize());
    if (expect_values(d_output, std::vector<int>{12, 0}) != 0) {
        return 1;
    }

    CHECK_DRV(cuMemsetD32(d_output, 0, 2));
    if (configure_params(function, d_output, 17, 5.0f, param_size,
                         output_offset, value_offset, delta_offset) != 0) {
        return 1;
    }
    CHECK_DRV(cuLaunchGrid(function, 2, 1));
    CHECK_DRV(cuCtxSynchronize());
    if (expect_values(d_output, std::vector<int>{22, 23}) != 0) {
        return 1;
    }

    CUstream stream;
    CHECK_DRV(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
    CHECK_DRV(cuMemsetD32Async(d_output, 0, 2, stream));
    if (configure_params(function, d_output, 23, 5.0f, param_size,
                         output_offset, value_offset, delta_offset) != 0) {
        return 1;
    }
    CHECK_DRV(cuLaunchGridAsync(function, 2, 1, stream));
    CHECK_DRV(cuStreamSynchronize(stream));
    if (expect_values(d_output, std::vector<int>{28, 29}) != 0) {
        return 1;
    }

    CHECK_DRV(cuStreamDestroy(stream));
    CHECK_DRV(cuMemFree(d_output));
    CHECK_DRV(cuModuleUnload(module));
    CHECK_DRV(cuDevicePrimaryCtxRelease(device));

    std::cout << "legacy launch API test passed" << std::endl;
    return 0;
}
