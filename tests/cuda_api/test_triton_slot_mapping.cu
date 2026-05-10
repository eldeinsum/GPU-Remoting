#include <cuda.h>
#include <nvrtc.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
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
extern "C" __global__ void _compute_slot_mapping_kernel(
    int num_tokens,
    int max_tokens,
    const int *query_start_loc,
    const long long *positions,
    const int *block_table,
    int block_table_stride,
    int block_size,
    long long *slot_mapping,
    const void *unused0,
    const void *unused1)
{
    int num_reqs = gridDim.x - 1;
    int req = blockIdx.x;
    int tid = threadIdx.x;
    if (req < num_reqs) {
        int begin = query_start_loc[req];
        int end = query_start_loc[req + 1];
        for (int token = begin + tid; token < end; token += blockDim.x) {
            long long position = positions[token];
            int block_index = static_cast<int>(position / block_size);
            int block_number = block_table[req * block_table_stride + block_index];
            slot_mapping[token] = static_cast<long long>(block_number) * block_size + (position % block_size);
        }
    } else {
        for (int token = num_tokens + tid; token < max_tokens; token += blockDim.x) {
            slot_mapping[token] = -1;
        }
    }
    (void)unused0;
    (void)unused1;
}
)";

    nvrtcProgram program;
    CHECK_NVRTC(nvrtcCreateProgram(&program, source, "test_triton_slot_mapping.cu", 0, nullptr, nullptr));

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

static int read_cubin_file(std::vector<char> *data, const char *path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "failed to open cubin: " << path << std::endl;
        return 1;
    }
    data->assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    if (data->empty()) {
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
    const char *cubin_path = std::getenv("GPU_REMOTING_SLOT_MAPPING_CUBIN");
    if (cubin_path != nullptr && cubin_path[0] != '\0') {
        if (read_cubin_file(&cubin, cubin_path) != 0) {
            return 1;
        }
    } else if (compile_cubin(&cubin, major, minor) != 0) {
        return 1;
    }
    const char *dump_path = std::getenv("GPU_REMOTING_SLOT_MAPPING_DUMP_CUBIN");
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
    CHECK_DRV(cuModuleGetFunction(&function, module, "_compute_slot_mapping_kernel"));

    const int num_reqs = 4;
    const int num_tokens = 38;
    const int max_tokens = 8192;
    const int block_table_stride = 64;
    const int block_size = 16;

    std::vector<int> query_start_loc = {0, 9, 15, 26, 38};
    std::vector<long long> positions;
    positions.reserve(num_tokens);
    const int prompt_lens[] = {9, 6, 11, 12};
    for (int len : prompt_lens) {
        for (int i = 0; i < len; ++i) {
            positions.push_back(i);
        }
    }

    std::vector<int> block_table(num_reqs * block_table_stride, 0);
    for (int req = 0; req < num_reqs; ++req) {
        block_table[req * block_table_stride] = req + 1;
    }

    std::vector<long long> slot_mapping(max_tokens, 12345);

    CUdeviceptr d_query_start_loc = 0;
    CUdeviceptr d_positions = 0;
    CUdeviceptr d_block_table = 0;
    CUdeviceptr d_slot_mapping = 0;
    CHECK_DRV(cuMemAlloc(&d_query_start_loc, query_start_loc.size() * sizeof(query_start_loc[0])));
    CHECK_DRV(cuMemAlloc(&d_positions, positions.size() * sizeof(positions[0])));
    CHECK_DRV(cuMemAlloc(&d_block_table, block_table.size() * sizeof(block_table[0])));
    CHECK_DRV(cuMemAlloc(&d_slot_mapping, slot_mapping.size() * sizeof(slot_mapping[0])));
    CHECK_DRV(cuMemcpyHtoD(d_query_start_loc, query_start_loc.data(), query_start_loc.size() * sizeof(query_start_loc[0])));
    CHECK_DRV(cuMemcpyHtoD(d_positions, positions.data(), positions.size() * sizeof(positions[0])));
    CHECK_DRV(cuMemcpyHtoD(d_block_table, block_table.data(), block_table.size() * sizeof(block_table[0])));
    CHECK_DRV(cuMemcpyHtoD(d_slot_mapping, slot_mapping.data(), slot_mapping.size() * sizeof(slot_mapping[0])));

    const void *unused0 = nullptr;
    const void *unused1 = nullptr;
    void *args[] = {
        (void *)&num_tokens,
        (void *)&max_tokens,
        (void *)&d_query_start_loc,
        (void *)&d_positions,
        (void *)&d_block_table,
        (void *)&block_table_stride,
        (void *)&block_size,
        (void *)&d_slot_mapping,
        (void *)&unused0,
        (void *)&unused1,
    };

    CUlaunchConfig config = {};
    config.gridDimX = num_reqs + 1;
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
    CHECK_DRV(cuMemcpyDtoH(slot_mapping.data(), d_slot_mapping, slot_mapping.size() * sizeof(slot_mapping[0])));

    int token_offset = 0;
    for (int req = 0; req < num_reqs; ++req) {
        int block_number = block_table[req * block_table_stride];
        for (int pos = 0; pos < prompt_lens[req]; ++pos) {
            long long expected = static_cast<long long>(block_number) * block_size + pos;
            if (slot_mapping[token_offset] != expected) {
                std::cerr << "mismatch at token " << token_offset << ": got " << slot_mapping[token_offset]
                          << " expected " << expected << std::endl;
                return 1;
            }
            ++token_offset;
        }
    }
    for (int i = num_tokens; i < max_tokens; ++i) {
        if (slot_mapping[i] != -1) {
            std::cerr << "tail mismatch at " << i << ": got " << slot_mapping[i] << " expected -1" << std::endl;
            return 1;
        }
    }

    CHECK_DRV(cuMemFree(d_slot_mapping));
    CHECK_DRV(cuMemFree(d_block_table));
    CHECK_DRV(cuMemFree(d_positions));
    CHECK_DRV(cuMemFree(d_query_start_loc));
    CHECK_DRV(cuModuleUnload(module));
    CHECK_DRV(cuDevicePrimaryCtxRelease(device));

    std::cout << "triton slot mapping test passed" << std::endl;
    return 0;
}
