#include <cuda.h>

#include <cstdio>
#include <unistd.h>

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

static int expect_result(CUresult actual, CUresult expected, const char *call)
{
    if (actual == expected) {
        return 0;
    }

    const char *actual_name = nullptr;
    const char *expected_name = nullptr;
    cuGetErrorName(actual, &actual_name);
    cuGetErrorName(expected, &expected_name);
    std::fprintf(stderr, "%s returned %s (%d), expected %s (%d)\n", call,
                 actual_name == nullptr ? "unknown" : actual_name,
                 static_cast<int>(actual),
                 expected_name == nullptr ? "unknown" : expected_name,
                 static_cast<int>(expected));
    return 1;
}

int main()
{
    CHECK_DRV(cuInit(0));

    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));

    int dma_buf_supported = 0;
    CHECK_DRV(cuDeviceGetAttribute(&dma_buf_supported,
                                   CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED,
                                   device));

    CUcontext context = nullptr;
    CHECK_DRV(cuCtxCreate(&context, nullptr, 0, device));

    CUdeviceptr allocation = 0;
    CHECK_DRV(cuMemAlloc(&allocation, 4096));

    int status = 0;
    int fd = -1;
    CUresult result = cuMemGetHandleForAddressRange(
        &fd, allocation, 4096, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
    if (dma_buf_supported) {
        status |= expect_result(result, CUDA_SUCCESS,
                                "cuMemGetHandleForAddressRange");
        if (result == CUDA_SUCCESS) {
            if (fd < 0) {
                std::fprintf(stderr, "expected a non-negative DMA-BUF fd\n");
                status = 1;
            } else {
                close(fd);
            }
        }
    } else {
        status |= expect_result(result, CUDA_ERROR_NOT_SUPPORTED,
                                "cuMemGetHandleForAddressRange");
        if (status == 0) {
            std::puts("DMA-BUF range handles unsupported on this device");
        }
    }

    status |= expect_result(
        cuMemGetHandleForAddressRange(nullptr, allocation, 4096,
                                      CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0),
        CUDA_ERROR_INVALID_VALUE, "cuMemGetHandleForAddressRange null handle");
    status |= expect_result(
        cuMemGetHandleForAddressRange(&fd, allocation, 4096,
                                      static_cast<CUmemRangeHandleType>(0), 0),
        CUDA_ERROR_INVALID_VALUE, "cuMemGetHandleForAddressRange invalid type");

    CHECK_DRV(cuMemFree(allocation));
    CHECK_DRV(cuCtxDestroy(context));
    return status;
}
