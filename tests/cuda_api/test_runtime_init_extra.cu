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

    CHECK_CUDA(cudaProfilerStart());
    CHECK_CUDA(cudaProfilerStop());
    CHECK_CUDA(cudaDeviceReset());

    std::puts("runtime init API test passed");
    return 0;
}
