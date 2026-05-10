#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

static int verify_bytes(const unsigned char *actual,
                        const unsigned char *expected,
                        size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "mismatch at %zu: got %u expected %u\n",
                         i, static_cast<unsigned>(actual[i]),
                         static_cast<unsigned>(expected[i]));
            return 1;
        }
    }
    return 0;
}

int main()
{
    const size_t count = 128;
    std::vector<unsigned char> input(count);
    std::vector<unsigned char> output(count, 0);
    for (size_t i = 0; i < count; ++i) {
        input[i] = static_cast<unsigned char>((i * 13) & 0xff);
    }

    const char *runtime_error = cudaGetErrorName(cudaSuccess);
    if (runtime_error == nullptr || std::strlen(runtime_error) == 0) {
        return 1;
    }

    cudaStream_t runtime_stream = nullptr;
    CHECK_CUDA_SUCCESS(cudaStreamCreateWithFlags(&runtime_stream, cudaStreamNonBlocking));
    unsigned int runtime_flags = 0;
    int runtime_priority = 0;
    CHECK_CUDA_SUCCESS(cudaStreamGetFlags(runtime_stream, &runtime_flags));
    CHECK_CUDA_SUCCESS(cudaStreamGetPriority(runtime_stream, &runtime_priority));
    CHECK_CUDA(cudaStreamQuery(runtime_stream));

    unsigned char *runtime_device = nullptr;
    CHECK_CUDA_SUCCESS(cudaMallocAsync(
        reinterpret_cast<void **>(&runtime_device), count, runtime_stream));
    CHECK_CUDA_SUCCESS(cudaMemsetAsync(runtime_device, 0, count, runtime_stream));
    CHECK_CUDA_SUCCESS(cudaMemcpyAsync(
        runtime_device, input.data(), count, cudaMemcpyDefault, runtime_stream));

    cudaEvent_t runtime_event = nullptr;
    CHECK_CUDA_SUCCESS(cudaEventCreateWithFlags(&runtime_event, cudaEventDisableTiming));
    CHECK_CUDA_SUCCESS(cudaEventRecordWithFlags(
        runtime_event, runtime_stream, cudaEventRecordDefault));
    CHECK_CUDA_SUCCESS(cudaEventSynchronize(runtime_event));
    CHECK_CUDA_SUCCESS(cudaMemcpyAsync(
        output.data(), runtime_device, count, cudaMemcpyDefault, runtime_stream));
    CHECK_CUDA_SUCCESS(cudaStreamSynchronize(runtime_stream));
    if (verify_bytes(output.data(), input.data(), count) != 0) {
        return 1;
    }
    CHECK_CUDA_SUCCESS(cudaFreeAsync(runtime_device, runtime_stream));
    CHECK_CUDA_SUCCESS(cudaStreamSynchronize(runtime_stream));
    CHECK_CUDA_SUCCESS(cudaEventDestroy(runtime_event));
    CHECK_CUDA_SUCCESS(cudaStreamDestroy(runtime_stream));
    CHECK_CUDA_SUCCESS(cudaCtxResetPersistingL2Cache());

    const size_t row_bytes = 8;
    const size_t rows = 4;
    std::vector<unsigned char> matrix(row_bytes * rows);
    std::vector<unsigned char> matrix_output(row_bytes * rows, 0);
    for (size_t i = 0; i < matrix.size(); ++i) {
        matrix[i] = static_cast<unsigned char>(0x80 + i);
    }

    unsigned char *pitched = nullptr;
    size_t pitch = 0;
    cudaStream_t copy_stream = nullptr;
    CHECK_CUDA_SUCCESS(cudaStreamCreate(&copy_stream));
    CHECK_CUDA_SUCCESS(cudaMallocPitch(
        reinterpret_cast<void **>(&pitched), &pitch, row_bytes, rows));
    CHECK_CUDA_SUCCESS(cudaMemcpy2DAsync(
        pitched, pitch, matrix.data(), row_bytes, row_bytes, rows,
        cudaMemcpyDefault, copy_stream));
    CHECK_CUDA_SUCCESS(cudaMemcpy2DAsync(
        matrix_output.data(), row_bytes, pitched, pitch, row_bytes, rows,
        cudaMemcpyDefault, copy_stream));
    CHECK_CUDA_SUCCESS(cudaStreamSynchronize(copy_stream));
    if (verify_bytes(matrix_output.data(), matrix.data(), matrix.size()) != 0) {
        return 1;
    }
    CHECK_CUDA_SUCCESS(cudaFree(pitched));
    CHECK_CUDA_SUCCESS(cudaStreamDestroy(copy_stream));

    CHECK_CUDA_SUCCESS(cudaSetDevice(0));
    void *init = nullptr;
    CHECK_CUDA_SUCCESS(cudaMalloc(&init, 1));
    CHECK_CUDA_SUCCESS(cudaFree(init));

    CHECK_DRV_SUCCESS(cuInit(0));
    CUstream driver_stream = nullptr;
    CHECK_DRV_SUCCESS(cuStreamCreateWithPriority(
        &driver_stream, CU_STREAM_NON_BLOCKING, 0));
    unsigned int driver_flags = 0;
    int driver_priority = 0;
    CHECK_DRV_SUCCESS(cuStreamGetFlags(driver_stream, &driver_flags));
    CHECK_DRV_SUCCESS(cuStreamGetPriority(driver_stream, &driver_priority));
    CHECK_DRV(cuStreamQuery(driver_stream));

    size_t free_mem = 0;
    size_t total_mem = 0;
    CHECK_DRV_SUCCESS(cuMemGetInfo(&free_mem, &total_mem));
    if (free_mem == 0 || total_mem == 0) {
        return 1;
    }

    CUdeviceptr driver_a = 0;
    CUdeviceptr driver_b = 0;
    CHECK_DRV_SUCCESS(cuMemAllocAsync(&driver_a, count, driver_stream));
    CHECK_DRV_SUCCESS(cuMemAllocAsync(&driver_b, count, driver_stream));
    CHECK_DRV_SUCCESS(cuMemcpyHtoDAsync(driver_a, input.data(), count, driver_stream));
    CHECK_DRV_SUCCESS(cuMemcpyDtoDAsync(driver_b, driver_a, count, driver_stream));
    CHECK_DRV_SUCCESS(cuMemcpyAsync(driver_a, driver_b, count, driver_stream));

    CUevent driver_event = nullptr;
    CHECK_DRV_SUCCESS(cuEventCreate(&driver_event, CU_EVENT_DISABLE_TIMING));
    CHECK_DRV_SUCCESS(cuEventRecordWithFlags(
        driver_event, driver_stream, CU_EVENT_RECORD_DEFAULT));
    CHECK_DRV_SUCCESS(cuEventSynchronize(driver_event));

    std::fill(output.begin(), output.end(), 0);
    CHECK_DRV_SUCCESS(cuMemcpyDtoHAsync(output.data(), driver_a, count, driver_stream));
    CHECK_DRV_SUCCESS(cuStreamSynchronize(driver_stream));
    if (verify_bytes(output.data(), input.data(), count) != 0) {
        return 1;
    }

    CHECK_DRV_SUCCESS(cuMemsetD8Async(driver_b, 0x5a, count, driver_stream));
    CHECK_DRV_SUCCESS(cuMemcpyDtoHAsync(output.data(), driver_b, count, driver_stream));
    CHECK_DRV_SUCCESS(cuStreamSynchronize(driver_stream));
    for (size_t i = 0; i < count; ++i) {
        if (output[i] != 0x5a) {
            return 1;
        }
    }

    CHECK_DRV_SUCCESS(cuMemsetD32Async(driver_b, 0x01020304u, count / 4, driver_stream));
    CHECK_DRV_SUCCESS(cuMemcpyDtoHAsync(output.data(), driver_b, count, driver_stream));
    CHECK_DRV_SUCCESS(cuStreamSynchronize(driver_stream));
    for (size_t i = 0; i < count; i += 4) {
        if (output[i] != 0x04 || output[i + 1] != 0x03 ||
            output[i + 2] != 0x02 || output[i + 3] != 0x01) {
            return 1;
        }
    }

    CHECK_DRV_SUCCESS(cuMemFreeAsync(driver_a, driver_stream));
    CHECK_DRV_SUCCESS(cuMemFreeAsync(driver_b, driver_stream));
    CHECK_DRV_SUCCESS(cuStreamSynchronize(driver_stream));
    CHECK_DRV_SUCCESS(cuEventDestroy(driver_event));
    CHECK_DRV_SUCCESS(cuStreamDestroy(driver_stream));

    std::puts("async allocation/copy API test passed");
    return 0;
}
