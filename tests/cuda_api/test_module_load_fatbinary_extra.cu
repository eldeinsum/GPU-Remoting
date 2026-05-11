#include <cuda.h>
#include <nvrtc.h>

#include <cstdint>
#include <cstring>
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

#pragma pack(push, 1)
struct FatBinaryHeader {
    std::uint32_t magic;
    std::uint16_t version;
    std::uint16_t header_size;
    std::uint64_t fat_size;
};

struct FatBinaryCodeHeader {
    std::uint16_t kind;
    std::uint16_t unknown02;
    std::uint32_t header_size;
    std::uint32_t code_size;
    std::uint32_t unknown0c;
    std::uint32_t compressed_size;
    std::uint32_t options_offset;
    std::uint16_t minor;
    std::uint16_t major;
    std::uint32_t arch;
    std::uint32_t name_offset;
    std::uint32_t name_size;
    std::uint32_t flags;
    std::uint32_t unknown2c;
    std::uint32_t unknown30;
    std::uint32_t unknown34;
    std::uint32_t decompressed_size;
    std::uint32_t unknown3c;
};
#pragma pack(pop)

static_assert(sizeof(FatBinaryHeader) == 16, "unexpected fat binary header size");
static_assert(sizeof(FatBinaryCodeHeader) == 64, "unexpected fat binary code header size");

static int compile_cubin(std::vector<char> *cubin, int major, int minor)
{
    static const char source[] = R"(
extern "C" __global__ void fatbinary_kernel(int *out, int value)
{
    out[0] = value + 7;
}
)";

    nvrtcProgram program;
    CHECK_NVRTC(nvrtcCreateProgram(&program, source, "test_module_load_fatbinary_extra.cu", 0, nullptr, nullptr));

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

static std::vector<unsigned char> make_fatbinary(const std::vector<char> &cubin,
                                                 int major,
                                                 int minor)
{
    FatBinaryHeader header = {};
    header.magic = 0xBA55ED50;
    header.version = 1;
    header.header_size = sizeof(FatBinaryHeader);
    header.fat_size = sizeof(FatBinaryCodeHeader) + cubin.size();

    FatBinaryCodeHeader code = {};
    code.kind = 2;
    code.unknown02 = 0x101;
    code.header_size = sizeof(FatBinaryCodeHeader);
    code.code_size = static_cast<std::uint32_t>(cubin.size());
    code.options_offset = 0;
    code.minor = static_cast<std::uint16_t>(minor);
    code.major = static_cast<std::uint16_t>(major);
    code.arch = static_cast<std::uint32_t>(major * 10 + minor);
    code.flags = 0x11;

    std::vector<unsigned char> fatbin(sizeof(header) + sizeof(code) + cubin.size());
    unsigned char *cursor = fatbin.data();
    std::memcpy(cursor, &header, sizeof(header));
    cursor += sizeof(header);
    std::memcpy(cursor, &code, sizeof(code));
    cursor += sizeof(code);
    std::memcpy(cursor, cubin.data(), cubin.size());
    return fatbin;
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
    std::vector<unsigned char> fatbin = make_fatbinary(cubin, major, minor);

    CUmodule module;
    CHECK_DRV(cuModuleLoadFatBinary(&module, fatbin.data()));

    CUfunction function;
    CHECK_DRV(cuModuleGetFunction(&function, module, "fatbinary_kernel"));

    const char *name = nullptr;
    CHECK_DRV(cuFuncGetName(&name, function));
    if (name == nullptr || std::strcmp(name, "fatbinary_kernel") != 0) {
        std::cerr << "unexpected function name: " << (name ? name : "(null)") << std::endl;
        return 1;
    }

    size_t param_count = 0;
    CHECK_DRV(cuFuncGetParamCount(function, &param_count));
    if (param_count != 2) {
        std::cerr << "unexpected parameter count: " << param_count << std::endl;
        return 1;
    }

    CUdeviceptr d_output = 0;
    CHECK_DRV(cuMemAlloc(&d_output, sizeof(int)));
    int value = 35;
    void *args[] = {&d_output, &value};
    CHECK_DRV(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
    CHECK_DRV(cuCtxSynchronize());

    int output = 0;
    CHECK_DRV(cuMemcpyDtoH(&output, d_output, sizeof(output)));
    if (output != 42) {
        std::cerr << "fatbinary kernel output mismatch: " << output << std::endl;
        return 1;
    }

    CHECK_DRV(cuMemFree(d_output));
    CHECK_DRV(cuModuleUnload(module));
    CHECK_DRV(cuDevicePrimaryCtxRelease(device));

    std::cout << "module load fatbinary API test passed" << std::endl;
    return 0;
}
