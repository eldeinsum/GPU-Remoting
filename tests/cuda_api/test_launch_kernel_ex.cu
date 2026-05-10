#include <cuda.h>
#include <nvrtc.h>

#include <cstdlib>
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
extern "C" __global__ void ex_kernel(
    int num_tokens,
    int max_tokens,
    const int *starts,
    const long long *positions,
    const int *block_table,
    int block_table_stride,
    int block_size,
    long long *out,
    const void *unused0,
    const void *unused1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_tokens) {
        long long value = positions[i] + starts[0] + block_table[i % block_table_stride] + block_size;
        value += max_tokens;
        value += unused0 == nullptr ? 0 : 1000000;
        value += unused1 == nullptr ? 0 : 2000000;
        out[i] = value;
    }
}
)";

    nvrtcProgram program;
    CHECK_NVRTC(nvrtcCreateProgram(&program, source, "test_launch_kernel_ex.cu", 0, nullptr, nullptr));

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

static int read_cubin_file(std::vector<char> *cubin, const char *path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "failed to open cubin: " << path << std::endl;
        return 1;
    }
    cubin->assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    if (cubin->empty()) {
        std::cerr << "empty cubin: " << path << std::endl;
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
    const char *cubin_path = std::getenv("GPU_REMOTING_LAUNCH_EX_CUBIN");
    if (cubin_path != nullptr && cubin_path[0] != '\0') {
        if (read_cubin_file(&cubin, cubin_path) != 0) {
            return 1;
        }
    } else if (compile_cubin(&cubin, major, minor) != 0) {
        return 1;
    }
    const char *dump_path = std::getenv("GPU_REMOTING_LAUNCH_EX_DUMP_CUBIN");
    if (dump_path != nullptr && dump_path[0] != '\0') {
        std::ofstream file(dump_path, std::ios::binary);
        if (!file) {
            std::cerr << "failed to open cubin dump path: " << dump_path << std::endl;
            return 1;
        }
        file.write(cubin.data(), cubin.size());
    }

    CUmodule module;
    CHECK_DRV(cuModuleLoadData(&module, cubin.data()));
    CUfunction function;
    CHECK_DRV(cuModuleGetFunction(&function, module, "ex_kernel"));

    const int count = 257;
    const int max_tokens = 1024;
    const int stride = 16;
    const int block_size = 64;

    std::vector<int> starts(1, 7);
    std::vector<long long> positions(count);
    std::vector<int> block_table(stride);
    std::vector<long long> output(count, 0);
    for (int i = 0; i < count; ++i) {
        positions[i] = i * 3;
    }
    for (int i = 0; i < stride; ++i) {
        block_table[i] = i * 5;
    }

    CUdeviceptr d_starts = 0;
    CUdeviceptr d_positions = 0;
    CUdeviceptr d_block_table = 0;
    CUdeviceptr d_output = 0;
    CHECK_DRV(cuMemAlloc(&d_starts, starts.size() * sizeof(starts[0])));
    CHECK_DRV(cuMemAlloc(&d_positions, positions.size() * sizeof(positions[0])));
    CHECK_DRV(cuMemAlloc(&d_block_table, block_table.size() * sizeof(block_table[0])));
    CHECK_DRV(cuMemAlloc(&d_output, output.size() * sizeof(output[0])));
    CHECK_DRV(cuMemcpyHtoD(d_starts, starts.data(), starts.size() * sizeof(starts[0])));
    CHECK_DRV(cuMemcpyHtoD(d_positions, positions.data(), positions.size() * sizeof(positions[0])));
    CHECK_DRV(cuMemcpyHtoD(d_block_table, block_table.data(), block_table.size() * sizeof(block_table[0])));

    const void *unused0 = nullptr;
    const void *unused1 = nullptr;
    void *args[] = {
        (void *)&count,
        (void *)&max_tokens,
        (void *)&d_starts,
        (void *)&d_positions,
        (void *)&d_block_table,
        (void *)&stride,
        (void *)&block_size,
        (void *)&d_output,
        (void *)&unused0,
        (void *)&unused1,
    };

    CUlaunchConfig config = {};
    config.gridDimX = (count + 127) / 128;
    config.gridDimY = 1;
    config.gridDimZ = 1;
    config.blockDimX = 128;
    config.blockDimY = 1;
    config.blockDimZ = 1;
    config.sharedMemBytes = 0;
    config.hStream = nullptr;
    config.attrs = nullptr;
    config.numAttrs = 0;

    CHECK_DRV(cuLaunchKernelEx(&config, function, args, nullptr));
    CHECK_DRV(cuCtxSynchronize());
    CHECK_DRV(cuMemcpyDtoH(output.data(), d_output, output.size() * sizeof(output[0])));

    for (int i = 0; i < count; ++i) {
        long long expected = positions[i] + starts[0] + block_table[i % stride] + block_size + max_tokens;
        if (output[i] != expected) {
            std::cerr << "mismatch at " << i << ": got " << output[i] << " expected " << expected << std::endl;
            return 1;
        }
    }

    CHECK_DRV(cuMemFree(d_output));
    CHECK_DRV(cuMemFree(d_block_table));
    CHECK_DRV(cuMemFree(d_positions));
    CHECK_DRV(cuMemFree(d_starts));
    CHECK_DRV(cuModuleUnload(module));
    CHECK_DRV(cuDevicePrimaryCtxRelease(device));

    std::cout << "cuLaunchKernelEx test passed for sm_" << major << minor << std::endl;
    return 0;
}
