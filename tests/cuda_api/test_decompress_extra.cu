#include <cuda.h>

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

int main()
{
    CHECK_DRV(cuInit(0));

    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));

    CUcontext context = nullptr;
    CHECK_DRV(cuCtxCreate(&context, nullptr, 0, device));

    int decompress_mask = 0;
    CHECK_DRV(cuDeviceGetAttribute(
        &decompress_mask, CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK,
        device));

    if (decompress_mask != 0) {
        std::puts("hardware decompress API supported; skipped");
        CHECK_DRV(cuCtxDestroy(context));
        return 0;
    }

    size_t null_error_index = 123;
    CUresult null_result = cuMemBatchDecompressAsync(
        nullptr, 0, 0, &null_error_index, nullptr);
    if (null_result != CUDA_ERROR_NOT_SUPPORTED) {
        const char *name = nullptr;
        cuGetErrorName(null_result, &name);
        std::fprintf(stderr,
                     "cuMemBatchDecompressAsync(nullptr) returned %s (%d), expected not supported\n",
                     name == nullptr ? "unknown" : name,
                     static_cast<int>(null_result));
        return 1;
    }
    if (null_error_index != 123) {
        std::fprintf(stderr,
                     "unsupported null decompress call changed error index to %zu\n",
                     null_error_index);
        return 1;
    }

    CUdeviceptr src = 0;
    CUdeviceptr dst = 0;
    CUdeviceptr actual_bytes = 0;
    CHECK_DRV(cuMemAlloc(&src, 256));
    CHECK_DRV(cuMemAlloc(&dst, 256));
    CHECK_DRV(cuMemAlloc(&actual_bytes, sizeof(cuuint32_t)));

    CUmemDecompressParams params = {};
    params.srcNumBytes = 256;
    params.dstNumBytes = 256;
    params.dstActBytes = reinterpret_cast<cuuint32_t *>(actual_bytes);
    params.src = reinterpret_cast<const void *>(src);
    params.dst = reinterpret_cast<void *>(dst);
    params.algo = CU_MEM_DECOMPRESS_ALGORITHM_DEFLATE;

    size_t error_index = 123;
    CUresult result = cuMemBatchDecompressAsync(
        &params, 1, 0, &error_index, nullptr);
    if (result != CUDA_ERROR_NOT_SUPPORTED) {
        const char *name = nullptr;
        cuGetErrorName(result, &name);
        std::fprintf(stderr,
                     "cuMemBatchDecompressAsync returned %s (%d), expected not supported\n",
                     name == nullptr ? "unknown" : name,
                     static_cast<int>(result));
        return 1;
    }
    if (error_index != 123) {
        std::fprintf(stderr,
                     "unsupported decompress call changed error index to %zu\n",
                     error_index);
        return 1;
    }

    CHECK_DRV(cuMemFree(actual_bytes));
    CHECK_DRV(cuMemFree(dst));
    CHECK_DRV(cuMemFree(src));
    CHECK_DRV(cuCtxDestroy(context));

    std::puts("hardware decompress API test passed");
    return 0;
}
