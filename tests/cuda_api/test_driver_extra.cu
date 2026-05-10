#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

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
    CHECK_CUDA(cudaSetDevice(0));
    void *runtime_context_init = nullptr;
    CHECK_CUDA(cudaMalloc(&runtime_context_init, 1));
    CHECK_CUDA(cudaFree(runtime_context_init));

    CHECK_DRV(cuInit(0));

    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));
    CHECK_DRV(cuCtxSynchronize());

    const int count = 32;
    const size_t bytes = count * sizeof(unsigned int);
    unsigned int host[count];
    unsigned int output[count];
    unsigned char byte_output[bytes];

    for (int i = 0; i < count; ++i) {
        host[i] = 0x1000u + static_cast<unsigned int>(i);
        output[i] = 0;
    }
    std::memset(byte_output, 0, sizeof(byte_output));

    CUdeviceptr device_a = 0;
    CUdeviceptr device_b = 0;
    CHECK_DRV(cuMemAlloc(&device_a, bytes));
    CHECK_DRV(cuMemAlloc(&device_b, bytes));
    CHECK_DRV(cuMemcpyHtoD(device_a, host, bytes));
    CHECK_DRV(cuMemcpyDtoD(device_b, device_a, bytes));
    CHECK_DRV(cuMemcpyDtoH(output, device_b, bytes));

    for (int i = 0; i < count; ++i) {
        if (output[i] != host[i]) {
            std::fprintf(stderr, "copy mismatch at %d\n", i);
            return 1;
        }
    }

    CHECK_DRV(cuMemsetD8(device_a, 0x7f, bytes));
    CHECK_DRV(cuMemcpyDtoH(byte_output, device_a, bytes));
    for (size_t i = 0; i < bytes; ++i) {
        if (byte_output[i] != 0x7f) {
            std::fprintf(stderr, "D8 memset mismatch at %zu\n", i);
            return 1;
        }
    }

    CHECK_DRV(cuMemsetD32(device_b, 0x01020304u, count));
    CHECK_DRV(cuMemcpyDtoH(output, device_b, bytes));
    for (int i = 0; i < count; ++i) {
        if (output[i] != 0x01020304u) {
            std::fprintf(stderr, "D32 memset mismatch at %d\n", i);
            return 1;
        }
    }

    CUstream stream = nullptr;
    CUevent event = nullptr;
    CHECK_DRV(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
    CHECK_DRV(cuEventCreate(&event, CU_EVENT_DEFAULT));
    CHECK_DRV(cuEventRecord(event, stream));
    CHECK_DRV(cuEventSynchronize(event));
    CHECK_DRV(cuStreamSynchronize(stream));
    CHECK_DRV(cuEventDestroy(event));
    CHECK_DRV(cuStreamDestroy(stream));

    CHECK_DRV(cuMemFree(device_a));
    CHECK_DRV(cuMemFree(device_b));

    std::puts("driver extra API test passed");
    return 0;
}
