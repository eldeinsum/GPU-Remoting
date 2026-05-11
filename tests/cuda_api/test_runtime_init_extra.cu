#include <cuda.h>
#include <cudaProfiler.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <cstdio>

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

int main()
{
    int valid_devices[] = {0};
    CHECK_CUDA(cudaSetValidDevices(valid_devices, 1));
    CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleAuto));
    CHECK_CUDA(cudaInitDevice(0, cudaDeviceScheduleAuto, 0));

    unsigned int flags = 0;
    CHECK_CUDA(cudaGetDeviceFlags(&flags));
    if ((flags & cudaDeviceScheduleAuto) != cudaDeviceScheduleAuto) {
        std::fprintf(stderr, "unexpected device flags: %u\n", flags);
        return 1;
    }

    CUresult profiler_init =
        cuProfilerInitialize("unused.cfg", "unused.out", CU_OUT_KEY_VALUE_PAIR);
    if (profiler_init != CUDA_SUCCESS &&
        profiler_init != CUDA_ERROR_NOT_SUPPORTED) {
        const char *name = nullptr;
        cuGetErrorName(profiler_init, &name);
        std::fprintf(stderr, "cuProfilerInitialize failed: %s (%d)\n",
                     name == nullptr ? "unknown" : name,
                     static_cast<int>(profiler_init));
        return 1;
    }
    CHECK_DRV(cuProfilerStart());
    CHECK_DRV(cuProfilerStop());

    CHECK_CUDA(cudaProfilerStart());
    CHECK_CUDA(cudaProfilerStop());
    CHECK_CUDA(cudaDeviceReset());

    std::puts("runtime init API test passed");
    return 0;
}
