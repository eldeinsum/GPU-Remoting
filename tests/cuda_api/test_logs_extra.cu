#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>

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

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t result = (call);                                           \
        if (result != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s failed: %s (%d)\n", #call,               \
                         cudaGetErrorString(result), static_cast<int>(result)); \
            return 1;                                                          \
        }                                                                      \
    } while (0)

int main()
{
    CHECK_DRV(cuInit(0));
    CHECK_CUDA(cudaSetDevice(0));

    CUlogIterator driver_iterator = 0;
    CHECK_DRV(cuLogsCurrent(&driver_iterator, 0));

    char driver_buffer[4096] = {};
    size_t driver_size = sizeof(driver_buffer);
    CHECK_DRV(cuLogsDumpToMemory(&driver_iterator, driver_buffer,
                                 &driver_size, 0));
    if (driver_size > sizeof(driver_buffer) - 1) {
        std::fprintf(stderr, "driver log dump size too large: %zu\n",
                     driver_size);
        return 1;
    }
    CHECK_DRV(cuLogsDumpToFile(nullptr, "/tmp/gpu_remoting_driver_logs.txt",
                               0));

    cudaLogIterator runtime_iterator = 0;
    CHECK_CUDA(cudaLogsCurrent(&runtime_iterator, 0));

    char runtime_buffer[4096] = {};
    size_t runtime_size = sizeof(runtime_buffer);
    CHECK_CUDA(cudaLogsDumpToMemory(&runtime_iterator, runtime_buffer,
                                    &runtime_size, 0));
    if (runtime_size > sizeof(runtime_buffer) - 1) {
        std::fprintf(stderr, "runtime log dump size too large: %zu\n",
                     runtime_size);
        return 1;
    }
    CHECK_CUDA(cudaLogsDumpToFile(nullptr, "/tmp/gpu_remoting_runtime_logs.txt",
                                  0));

    std::puts("logs API test passed");
    return 0;
}
