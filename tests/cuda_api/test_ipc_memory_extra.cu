#include <cuda.h>
#include <cuda_runtime.h>

#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t result = (call);                                           \
        if (result != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s failed: %s (%d)\n", #call,               \
                         cudaGetErrorString(result), static_cast<int>(result)); \
            return 1;                                                          \
        }                                                                      \
    } while (0)

#define CHECK_DRV(call)                                                        \
    do {                                                                       \
        CUresult result = (call);                                              \
        if (result != CUDA_SUCCESS) {                                          \
            const char *name = nullptr;                                        \
            cuGetErrorName(result, &name);                                     \
            std::fprintf(stderr, "%s failed: %s (%d)\n", #call,               \
                         name == nullptr ? "unknown" : name,                  \
                         static_cast<int>(result));                            \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static int from_hex(char c)
{
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    return -1;
}

template <typename Handle>
static std::string encode_handle(const Handle &handle)
{
    static const char kHex[] = "0123456789abcdef";
    const unsigned char *bytes =
        reinterpret_cast<const unsigned char *>(handle.reserved);
    std::string output;
    output.reserve(sizeof(handle.reserved) * 2);
    for (size_t i = 0; i < sizeof(handle.reserved); ++i) {
        output.push_back(kHex[bytes[i] >> 4]);
        output.push_back(kHex[bytes[i] & 0xf]);
    }
    return output;
}

template <typename Handle>
static int decode_handle(const char *hex, Handle *handle)
{
    if (hex == nullptr || handle == nullptr ||
        std::strlen(hex) != sizeof(handle->reserved) * 2) {
        return 1;
    }
    unsigned char *bytes = reinterpret_cast<unsigned char *>(handle->reserved);
    for (size_t i = 0; i < sizeof(handle->reserved); ++i) {
        int high = from_hex(hex[i * 2]);
        int low = from_hex(hex[i * 2 + 1]);
        if (high < 0 || low < 0) {
            return 1;
        }
        bytes[i] = static_cast<unsigned char>((high << 4) | low);
    }
    return 0;
}

static int check_values(const std::vector<unsigned int> &values)
{
    for (size_t i = 0; i < values.size(); ++i) {
        unsigned int expected = 0x7000u + static_cast<unsigned int>(i);
        if (values[i] != expected) {
            std::fprintf(stderr, "value mismatch at %zu: got %u expected %u\n",
                         i, values[i], expected);
            return 1;
        }
    }
    return 0;
}

static int run_driver_child(const char *handle_hex)
{
    constexpr size_t kCount = 64;
    constexpr size_t kBytes = kCount * sizeof(unsigned int);

    CUipcMemHandle handle = {};
    if (decode_handle(handle_hex, &handle) != 0) {
        std::fprintf(stderr, "invalid driver IPC handle encoding\n");
        return 1;
    }

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaFree(nullptr));
    CHECK_DRV(cuInit(0));

    CUdeviceptr imported = 0;
    CHECK_DRV(cuIpcOpenMemHandle(
        &imported, handle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS));

    std::vector<unsigned int> output(kCount, 0);
    CHECK_DRV(cuMemcpyDtoH(output.data(), imported, kBytes));
    if (check_values(output) != 0) {
        return 1;
    }

    CHECK_DRV(cuIpcCloseMemHandle(imported));
    return 0;
}

static int run_runtime_child(const char *handle_hex)
{
    constexpr size_t kCount = 64;
    constexpr size_t kBytes = kCount * sizeof(unsigned int);

    cudaIpcMemHandle_t handle = {};
    if (decode_handle(handle_hex, &handle) != 0) {
        std::fprintf(stderr, "invalid runtime IPC handle encoding\n");
        return 1;
    }

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaFree(nullptr));

    void *imported = nullptr;
    CHECK_CUDA(cudaIpcOpenMemHandle(&imported, handle,
                                    cudaIpcMemLazyEnablePeerAccess));

    std::vector<unsigned int> output(kCount, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), imported, kBytes,
                          cudaMemcpyDeviceToHost));
    if (check_values(output) != 0) {
        return 1;
    }

    CHECK_CUDA(cudaIpcCloseMemHandle(imported));
    return 0;
}

static int run_child_process(const char *program, const char *mode,
                             const std::string &handle_hex)
{
    pid_t pid = fork();
    if (pid < 0) {
        std::perror("fork");
        return 1;
    }
    if (pid == 0) {
        execl(program, program, mode, handle_hex.c_str(),
              static_cast<char *>(nullptr));
        std::perror("execl");
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        std::perror("waitpid");
        return 1;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        std::fprintf(stderr, "%s child failed: status=%d\n", mode, status);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 3 && std::strcmp(argv[1], "--child-driver") == 0) {
        return run_driver_child(argv[2]);
    }
    if (argc == 3 && std::strcmp(argv[1], "--child-runtime") == 0) {
        return run_runtime_child(argv[2]);
    }

    constexpr size_t kCount = 64;
    constexpr size_t kBytes = kCount * sizeof(unsigned int);

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaFree(nullptr));
    CHECK_DRV(cuInit(0));

    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));

    int unified_addressing = 0;
    CHECK_DRV(cuDeviceGetAttribute(
        &unified_addressing, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device));
    if (!unified_addressing) {
        std::puts("memory IPC API unsupported on this device");
        return 0;
    }

    CUdeviceptr device_ptr = 0;
    CHECK_DRV(cuMemAlloc(&device_ptr, kBytes));

    std::vector<unsigned int> input(kCount, 0);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = 0x7000u + static_cast<unsigned int>(i);
    }
    CHECK_DRV(cuMemcpyHtoD(device_ptr, input.data(), kBytes));
    CHECK_DRV(cuCtxSynchronize());

    CUipcMemHandle handle = {};
    CHECK_DRV(cuIpcGetMemHandle(&handle, device_ptr));
    std::string driver_handle_hex = encode_handle(handle);

    if (run_child_process(argv[0], "--child-driver", driver_handle_hex) != 0) {
        return 1;
    }

    CHECK_DRV(cuMemFree(device_ptr));

    void *runtime_ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&runtime_ptr, kBytes));
    CHECK_CUDA(cudaMemcpy(runtime_ptr, input.data(), kBytes,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaIpcMemHandle_t runtime_handle = {};
    CHECK_CUDA(cudaIpcGetMemHandle(&runtime_handle, runtime_ptr));
    std::string runtime_handle_hex = encode_handle(runtime_handle);

    if (run_child_process(argv[0], "--child-runtime", runtime_handle_hex) != 0) {
        return 1;
    }

    CHECK_CUDA(cudaFree(runtime_ptr));

    std::puts("memory IPC API test passed");
    return 0;
}
