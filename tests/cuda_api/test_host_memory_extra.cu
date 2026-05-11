#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static int check_values(const std::vector<int> &values, int base)
{
    for (size_t i = 0; i < values.size(); ++i) {
        int expected = base + static_cast<int>(i);
        if (values[i] != expected) {
            std::fprintf(stderr, "value mismatch at %zu: got %d expected %d\n",
                         i, values[i], expected);
            return 1;
        }
    }
    return 0;
}

static int check_runtime_host_device_pointer(void *host)
{
    void *device = nullptr;
    cudaError_t result = cudaHostGetDevicePointer(&device, host, 0);
    if (result == cudaSuccess) {
        if (device == nullptr) {
            std::fprintf(stderr, "runtime host device pointer is null\n");
            return 1;
        }
        return 0;
    }
    if (result == cudaErrorNotSupported && device == nullptr) {
        return 0;
    }
    std::fprintf(stderr, "cudaHostGetDevicePointer failed: %s (%d)\n",
                 cudaGetErrorString(result), static_cast<int>(result));
    return 1;
}

static int check_driver_host_device_pointer(void *host)
{
    CUdeviceptr device = 0;
    CUresult result = cuMemHostGetDevicePointer(&device, host, 0);
    if (result == CUDA_SUCCESS) {
        if (device == 0) {
            std::fprintf(stderr, "driver host device pointer is null\n");
            return 1;
        }
        return 0;
    }
    if (result == CUDA_ERROR_NOT_SUPPORTED && device == 0) {
        return 0;
    }
    const char *name = nullptr;
    cuGetErrorName(result, &name);
    std::fprintf(stderr, "cuMemHostGetDevicePointer failed: %s (%d)\n",
                 name == nullptr ? "unknown" : name, static_cast<int>(result));
    return 1;
}

static int run_runtime_host_memory()
{
    constexpr int kCount = 64;
    constexpr size_t kBytes = kCount * sizeof(int);

    CHECK_CUDA(cudaSetDevice(0));

    int *host = nullptr;
    CHECK_CUDA(cudaHostAlloc(
        reinterpret_cast<void **>(&host), kBytes, cudaHostAllocPortable));
    for (int i = 0; i < kCount; ++i) {
        host[i] = 100 + i;
    }

    unsigned int flags = 0;
    CHECK_CUDA(cudaHostGetFlags(&flags, host + 3));
    if ((flags & cudaHostAllocPortable) == 0) {
        std::fprintf(stderr, "runtime host allocation flags mismatch: %u\n", flags);
        return 1;
    }
    if (check_runtime_host_device_pointer(host) != 0) {
        return 1;
    }

    int *device = nullptr;
    std::vector<int> output(kCount, 0);
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&device), kBytes));
    CHECK_CUDA(cudaMemcpy(device, host, kBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(output.data(), device, kBytes, cudaMemcpyDeviceToHost));
    if (check_values(output, 100) != 0) {
        return 1;
    }
    CHECK_CUDA(cudaFree(device));
    CHECK_CUDA(cudaFreeHost(host));

    void *registered = nullptr;
    if (posix_memalign(&registered, 4096, kBytes) != 0) {
        return 1;
    }
    CHECK_CUDA(cudaHostRegister(registered, kBytes, cudaHostRegisterPortable));
    flags = 0;
    CHECK_CUDA(cudaHostGetFlags(
        &flags, static_cast<char *>(registered) + sizeof(int)));
    if ((flags & cudaHostRegisterPortable) == 0) {
        std::fprintf(stderr, "runtime registered host flags mismatch: %u\n", flags);
        return 1;
    }
    CHECK_CUDA(cudaHostUnregister(registered));
    std::free(registered);
    return 0;
}

static int run_driver_host_memory()
{
    constexpr int kCount = 64;
    constexpr size_t kBytes = kCount * sizeof(int);

    CHECK_DRV(cuInit(0));
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaFree(nullptr));

    int *host = nullptr;
    CHECK_DRV(cuMemHostAlloc(
        reinterpret_cast<void **>(&host), kBytes, CU_MEMHOSTALLOC_PORTABLE));
    for (int i = 0; i < kCount; ++i) {
        host[i] = 500 + i;
    }

    unsigned int flags = 0;
    CHECK_DRV(cuMemHostGetFlags(&flags, host + 5));
    if ((flags & CU_MEMHOSTALLOC_PORTABLE) == 0) {
        std::fprintf(stderr, "driver host allocation flags mismatch: %u\n", flags);
        return 1;
    }
    if (check_driver_host_device_pointer(host) != 0) {
        return 1;
    }

    CUdeviceptr device = 0;
    std::vector<int> output(kCount, 0);
    CHECK_DRV(cuMemAlloc(&device, kBytes));
    CHECK_DRV(cuMemcpyHtoD(device, host, kBytes));
    CHECK_DRV(cuMemcpyDtoH(output.data(), device, kBytes));
    if (check_values(output, 500) != 0) {
        return 1;
    }
    CHECK_DRV(cuMemFree(device));
    CHECK_DRV(cuMemFreeHost(host));

    void *allocated = nullptr;
    CHECK_DRV(cuMemAllocHost(&allocated, kBytes));
    flags = 1;
    CHECK_DRV(cuMemHostGetFlags(&flags, allocated));
    if ((flags & CU_MEMHOSTALLOC_PORTABLE) != 0) {
        std::fprintf(stderr, "driver legacy host flags mismatch: %u\n", flags);
        return 1;
    }
    CHECK_DRV(cuMemFreeHost(allocated));

    void *registered = nullptr;
    if (posix_memalign(&registered, 4096, kBytes) != 0) {
        return 1;
    }
    CHECK_DRV(cuMemHostRegister(
        registered, kBytes, CU_MEMHOSTREGISTER_PORTABLE));
    flags = 0;
    CHECK_DRV(cuMemHostGetFlags(
        &flags, static_cast<char *>(registered) + sizeof(int)));
    if ((flags & CU_MEMHOSTREGISTER_PORTABLE) == 0) {
        std::fprintf(stderr, "driver registered host flags mismatch: %u\n", flags);
        return 1;
    }
    CHECK_DRV(cuMemHostUnregister(registered));
    std::free(registered);
    return 0;
}

int main()
{
    if (run_runtime_host_memory() != 0) {
        return 1;
    }
    if (run_driver_host_memory() != 0) {
        return 1;
    }

    std::puts("host memory API test passed");
    return 0;
}
