#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <unistd.h>

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

    int fd_handles_supported = 0;
    CHECK_DRV(cuDeviceGetAttribute(
        &fd_handles_supported,
        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
        device));
    if (fd_handles_supported) {
        CUmemAllocationProp share_prop = prop;
        share_prop.requestedHandleTypes =
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

        CUmemGenericAllocationHandle share_handle = 0;
        CHECK_DRV(cuMemCreate(&share_handle, allocation_size, &share_prop, 0));

        int exported_fd = -1;
        CHECK_DRV(cuMemExportToShareableHandle(
            &exported_fd, share_handle,
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
        if (exported_fd < 0) {
            std::fprintf(stderr, "invalid exported fd %d\n", exported_fd);
            return 1;
        }

        CUmemGenericAllocationHandle imported_handle = 0;
        CHECK_DRV(cuMemImportFromShareableHandle(
            &imported_handle,
            reinterpret_cast<void *>(static_cast<intptr_t>(exported_fd)),
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
        close(exported_fd);

        CUdeviceptr shared_va = 0;
        CHECK_DRV(cuMemAddressReserve(&shared_va, allocation_size, 0, 0, 0));
        CHECK_DRV(cuMemMap(shared_va, allocation_size, 0, imported_handle, 0));
        CHECK_DRV(cuMemSetAccess(shared_va, allocation_size, &access, 1));

        std::fill(output.begin(), output.end(), 0);
        CHECK_DRV(cuMemsetD32(shared_va, 0xa55aa55au, value_count));
        CHECK_DRV(cuMemcpyDtoH(
            output.data(), shared_va, value_count * sizeof(unsigned int)));
        if (!std::all_of(output.begin(), output.end(), [](unsigned int value) {
                return value == 0xa55aa55au;
            })) {
            return 1;
        }

        CHECK_DRV(cuMemUnmap(shared_va, allocation_size));
        CHECK_DRV(cuMemAddressFree(shared_va, allocation_size));
        CHECK_DRV(cuMemRelease(imported_handle));
        CHECK_DRV(cuMemRelease(share_handle));
    } else {
        std::puts("POSIX shareable VMM handles unsupported on this device");
    }

    std::puts("virtual memory management API test passed");
    return 0;
}
