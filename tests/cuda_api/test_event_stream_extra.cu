#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t result = (call);                                           \
        if (result != cudaSuccess && result != cudaErrorNotReady) {            \
            std::fprintf(stderr, "%s failed: %s (%d)\n", #call,               \
                         cudaGetErrorString(result), static_cast<int>(result)); \
            return 1;                                                          \
        }                                                                      \
    } while (0)

#define CHECK_CUDA_SUCCESS(call)                                               \
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
        if (result != CUDA_SUCCESS && result != CUDA_ERROR_NOT_READY) {        \
            const char *name = nullptr;                                        \
            cuGetErrorName(result, &name);                                     \
            std::fprintf(stderr, "%s failed: %s (%d)\n", #call,               \
                         name == nullptr ? "unknown" : name,                  \
                         static_cast<int>(result));                            \
            return 1;                                                          \
        }                                                                      \
    } while (0)

#define CHECK_DRV_SUCCESS(call)                                                \
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
    CHECK_CUDA_SUCCESS(cudaSetDevice(0));

    cudaStream_t runtime_a = nullptr;
    cudaStream_t runtime_b = nullptr;
    cudaEvent_t runtime_start = nullptr;
    cudaEvent_t runtime_end = nullptr;

    CHECK_CUDA_SUCCESS(cudaStreamCreateWithPriority(
        &runtime_a, cudaStreamNonBlocking, 0));
    CHECK_CUDA_SUCCESS(cudaStreamCreate(&runtime_b));
    CHECK_CUDA_SUCCESS(cudaStreamCopyAttributes(runtime_b, runtime_a));

    unsigned long long runtime_stream_id = 0;
    int runtime_device = -1;
    CHECK_CUDA_SUCCESS(cudaStreamGetId(runtime_a, &runtime_stream_id));
    CHECK_CUDA_SUCCESS(cudaStreamGetDevice(runtime_a, &runtime_device));
    if (runtime_device != 0) {
        return 1;
    }

    CHECK_CUDA_SUCCESS(cudaEventCreate(&runtime_start));
    CHECK_CUDA_SUCCESS(cudaEventCreate(&runtime_end));
    CHECK_CUDA_SUCCESS(cudaEventRecord(runtime_start, runtime_a));
    CHECK_CUDA_SUCCESS(cudaStreamWaitEvent(runtime_b, runtime_start, 0));
    CHECK_CUDA_SUCCESS(cudaEventRecord(runtime_end, runtime_b));
    CHECK_CUDA_SUCCESS(cudaEventSynchronize(runtime_end));
    CHECK_CUDA(cudaEventQuery(runtime_end));

    float runtime_ms = -1.0f;
    CHECK_CUDA_SUCCESS(cudaEventElapsedTime(
        &runtime_ms, runtime_start, runtime_end));
    if (runtime_ms < 0.0f) {
        return 1;
    }

    CHECK_CUDA_SUCCESS(cudaEventDestroy(runtime_start));
    CHECK_CUDA_SUCCESS(cudaEventDestroy(runtime_end));
    CHECK_CUDA_SUCCESS(cudaStreamDestroy(runtime_a));
    CHECK_CUDA_SUCCESS(cudaStreamDestroy(runtime_b));

    CHECK_DRV_SUCCESS(cuInit(0));

    CUstream driver_a = nullptr;
    CUstream driver_b = nullptr;
    CUevent driver_start = nullptr;
    CUevent driver_end = nullptr;

    CHECK_DRV_SUCCESS(cuStreamCreateWithPriority(
        &driver_a, CU_STREAM_NON_BLOCKING, 0));
    CHECK_DRV_SUCCESS(cuStreamCreate(&driver_b, CU_STREAM_DEFAULT));
    CHECK_DRV_SUCCESS(cuStreamCopyAttributes(driver_b, driver_a));

    unsigned long long driver_stream_id = 0;
    CUdevice driver_device = -1;
    CUcontext driver_context = nullptr;
    CHECK_DRV_SUCCESS(cuStreamGetId(driver_a, &driver_stream_id));
    CHECK_DRV_SUCCESS(cuStreamGetDevice(driver_a, &driver_device));
    CHECK_DRV_SUCCESS(cuStreamGetCtx(driver_a, &driver_context));
    if (driver_device != 0 || driver_context == nullptr) {
        return 1;
    }

    CHECK_DRV_SUCCESS(cuEventCreate(&driver_start, CU_EVENT_DEFAULT));
    CHECK_DRV_SUCCESS(cuEventCreate(&driver_end, CU_EVENT_DEFAULT));
    CHECK_DRV_SUCCESS(cuEventRecord(driver_start, driver_a));
    CHECK_DRV_SUCCESS(cuStreamWaitEvent(driver_b, driver_start, 0));
    CHECK_DRV_SUCCESS(cuEventRecord(driver_end, driver_b));
    CHECK_DRV_SUCCESS(cuEventSynchronize(driver_end));
    CHECK_DRV(cuEventQuery(driver_end));

    float driver_ms = -1.0f;
    CHECK_DRV_SUCCESS(cuEventElapsedTime(&driver_ms, driver_start, driver_end));
    if (driver_ms < 0.0f) {
        return 1;
    }

    CHECK_DRV_SUCCESS(cuEventDestroy(driver_start));
    CHECK_DRV_SUCCESS(cuEventDestroy(driver_end));
    CHECK_DRV_SUCCESS(cuStreamDestroy(driver_a));
    CHECK_DRV_SUCCESS(cuStreamDestroy(driver_b));

    std::puts("event/stream API test passed");
    return 0;
}
