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
    CHECK_DRV(cuDevicePrimaryCtxRetain(&context, device));
    CHECK_DRV(cuCtxSetCurrent(context));

    CUdeviceptr ptr = 0;
    CHECK_DRV(cuMemAlloc(&ptr, 4096));

    CUpointer_attribute attrs[] = {
        CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
        CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
        CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
        CU_POINTER_ATTRIBUTE_RANGE_SIZE,
        CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
    };
    CUmemorytype memory_type = CU_MEMORYTYPE_HOST;
    CUdeviceptr device_pointer = 0;
    int ordinal = -1;
    size_t range_size = 0;
    unsigned int sync_memops = 99;
    void *data[] = {
        &memory_type,
        &device_pointer,
        &ordinal,
        &range_size,
        &sync_memops,
    };

    CHECK_DRV(cuPointerGetAttributes(5, attrs, data, ptr));
    if (memory_type != CU_MEMORYTYPE_DEVICE || device_pointer != ptr ||
        ordinal != 0 || range_size < 4096 || sync_memops != 0) {
        std::fprintf(
            stderr,
            "unexpected pointer attributes: type=%u dptr=%llu ordinal=%d range=%zu sync=%u\n",
            static_cast<unsigned>(memory_type),
            static_cast<unsigned long long>(device_pointer),
            ordinal,
            range_size,
            sync_memops);
        return 1;
    }

    unsigned int sync_value = 1;
    CHECK_DRV(cuPointerSetAttribute(
        &sync_value, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, ptr));
    sync_memops = 0;
    CHECK_DRV(cuPointerGetAttribute(
        &sync_memops, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, ptr));
    if (sync_memops != 1) {
        std::fprintf(stderr, "sync memops attribute was not set\n");
        return 1;
    }

    sync_value = 0;
    CHECK_DRV(cuPointerSetAttribute(
        &sync_value, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, ptr));
    sync_memops = 1;
    CHECK_DRV(cuPointerGetAttribute(
        &sync_memops, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, ptr));
    if (sync_memops != 0) {
        std::fprintf(stderr, "sync memops attribute was not cleared\n");
        return 1;
    }

    CHECK_DRV(cuMemFree(ptr));
    CHECK_DRV(cuDevicePrimaryCtxRelease(device));

    std::puts("pointer attributes API test passed");
    return 0;
}
