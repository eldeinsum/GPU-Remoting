#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
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

int main()
{
    CHECK_CUDA(cudaSetDevice(0));
    void *runtime_context_init = nullptr;
    CHECK_CUDA(cudaMalloc(&runtime_context_init, 1));
    CHECK_CUDA(cudaFree(runtime_context_init));

    CHECK_DRV(cuInit(0));

    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));

    int vmm_supported = 0;
    CHECK_DRV(cuDeviceGetAttribute(
        &vmm_supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
        device));
    if (!vmm_supported) {
        std::puts("virtual memory management API unsupported on this device");
        return 0;
    }

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;

    size_t min_granularity = 0;
    size_t recommended_granularity = 0;
    CHECK_DRV(cuMemGetAllocationGranularity(
        &min_granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    CHECK_DRV(cuMemGetAllocationGranularity(
        &recommended_granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    if (min_granularity == 0 || recommended_granularity == 0) {
        return 1;
    }

    const size_t allocation_size = min_granularity;
    CUmemGenericAllocationHandle handle = 0;
    CHECK_DRV(cuMemCreate(&handle, allocation_size, &prop, 0));

    CUmemAllocationProp queried_prop = {};
    CHECK_DRV(cuMemGetAllocationPropertiesFromHandle(&queried_prop, handle));
    if (queried_prop.type != CU_MEM_ALLOCATION_TYPE_PINNED ||
        queried_prop.location.type != CU_MEM_LOCATION_TYPE_DEVICE ||
        queried_prop.location.id != device) {
        return 1;
    }

    CUdeviceptr va = 0;
    CHECK_DRV(cuMemAddressReserve(&va, allocation_size, 0, 0, 0));
    CHECK_DRV(cuMemMap(va, allocation_size, 0, handle, 0));

    CUmemAccessDesc access = {};
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_DRV(cuMemSetAccess(va, allocation_size, &access, 1));

    unsigned long long access_flags = 0;
    CHECK_DRV(cuMemGetAccess(&access_flags, &prop.location, va));
    if (access_flags != CU_MEM_ACCESS_FLAGS_PROT_READWRITE) {
        return 1;
    }

    CUdeviceptr range_base = 0;
    size_t range_size = 0;
    CHECK_DRV(cuMemGetAddressRange(&range_base, &range_size, va));
    if (range_base != va || range_size < allocation_size) {
        return 1;
    }

    constexpr size_t value_count = 256;
    std::vector<unsigned int> output(value_count, 0);
    CHECK_DRV(cuMemsetD32(va, 0x5aa55aa5u, value_count));
    CHECK_DRV(cuMemcpyDtoH(
        output.data(), va, value_count * sizeof(unsigned int)));
    if (!std::all_of(output.begin(), output.end(), [](unsigned int value) {
            return value == 0x5aa55aa5u;
        })) {
        return 1;
    }

    CUmemGenericAllocationHandle retained_handle = 0;
    CHECK_DRV(cuMemRetainAllocationHandle(
        &retained_handle, reinterpret_cast<void *>(va)));
    if (retained_handle == 0) {
        return 1;
    }
    CHECK_DRV(cuMemRelease(retained_handle));

    CHECK_DRV(cuMemUnmap(va, allocation_size));
    CHECK_DRV(cuMemAddressFree(va, allocation_size));
    CHECK_DRV(cuMemRelease(handle));

    std::puts("virtual memory management API test passed");
    return 0;
}
