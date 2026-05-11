#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
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
    CHECK_CUDA(cudaSetDevice(0));

    int memory_pools_supported = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(
        &memory_pools_supported, cudaDevAttrMemoryPoolsSupported, 0));
    if (!memory_pools_supported) {
        std::puts("memory pool API unsupported on this device");
        return 0;
    }

    cudaMemPool_t runtime_default_pool = nullptr;
    cudaMemPool_t runtime_current_pool = nullptr;
    CHECK_CUDA(cudaDeviceGetDefaultMemPool(&runtime_default_pool, 0));
    CHECK_CUDA(cudaDeviceGetMemPool(&runtime_current_pool, 0));
    if (runtime_default_pool == nullptr || runtime_current_pool == nullptr) {
        return 1;
    }
    CHECK_CUDA(cudaDeviceSetMemPool(0, runtime_default_pool));

    std::uint64_t runtime_threshold = 0;
    std::uint64_t runtime_used = 0;
    std::uint64_t runtime_reserved = 0;
    CHECK_CUDA(cudaMemPoolGetAttribute(
        runtime_default_pool, cudaMemPoolAttrReleaseThreshold,
        &runtime_threshold));
    CHECK_CUDA(cudaMemPoolSetAttribute(
        runtime_default_pool, cudaMemPoolAttrReleaseThreshold,
        &runtime_threshold));
    CHECK_CUDA(cudaMemPoolGetAttribute(
        runtime_default_pool, cudaMemPoolAttrUsedMemCurrent, &runtime_used));
    CHECK_CUDA(cudaMemPoolGetAttribute(
        runtime_default_pool, cudaMemPoolAttrReservedMemCurrent,
        &runtime_reserved));

    cudaStream_t runtime_stream = nullptr;
    void *runtime_ptr = nullptr;
    CHECK_CUDA(cudaStreamCreate(&runtime_stream));
    CHECK_CUDA(cudaMallocFromPoolAsync(
        &runtime_ptr, 256, runtime_default_pool, runtime_stream));
    CHECK_CUDA(cudaMemsetAsync(runtime_ptr, 0x2a, 256, runtime_stream));
    CHECK_CUDA(cudaFreeAsync(runtime_ptr, runtime_stream));
    CHECK_CUDA(cudaStreamSynchronize(runtime_stream));
    CHECK_CUDA(cudaMemPoolTrimTo(runtime_default_pool, 0));
    CHECK_CUDA(cudaStreamDestroy(runtime_stream));

    cudaMemLocation runtime_location = {};
    runtime_location.type = cudaMemLocationTypeDevice;
    runtime_location.id = 0;

    cudaMemPool_t runtime_location_default_pool = nullptr;
    cudaMemPool_t runtime_location_current_pool = nullptr;
    CHECK_CUDA(cudaMemGetDefaultMemPool(
        &runtime_location_default_pool, &runtime_location,
        cudaMemAllocationTypePinned));
    CHECK_CUDA(cudaMemGetMemPool(
        &runtime_location_current_pool, &runtime_location,
        cudaMemAllocationTypePinned));
    if (runtime_location_default_pool == nullptr ||
        runtime_location_current_pool == nullptr) {
        return 1;
    }

    cudaMemPoolProps runtime_pool_props = {};
    runtime_pool_props.allocType = cudaMemAllocationTypePinned;
    runtime_pool_props.handleTypes = cudaMemHandleTypeNone;
    runtime_pool_props.location = runtime_location;

    cudaMemPool_t runtime_created_pool = nullptr;
    CHECK_CUDA(cudaMemPoolCreate(&runtime_created_pool, &runtime_pool_props));
    if (runtime_created_pool == nullptr) {
        return 1;
    }

    cudaMemAccessDesc runtime_access = {};
    runtime_access.location = runtime_location;
    runtime_access.flags = cudaMemAccessFlagsProtReadWrite;
    CHECK_CUDA(cudaMemPoolSetAccess(runtime_created_pool, &runtime_access, 1));

    cudaMemAccessFlags runtime_access_flags = cudaMemAccessFlagsProtNone;
    CHECK_CUDA(cudaMemPoolGetAccess(
        &runtime_access_flags, runtime_created_pool, &runtime_location));
    if (runtime_access_flags != cudaMemAccessFlagsProtReadWrite) {
        return 1;
    }

    CHECK_CUDA(cudaMemSetMemPool(
        &runtime_location, cudaMemAllocationTypePinned, runtime_created_pool));
    cudaMemPool_t runtime_updated_pool = nullptr;
    CHECK_CUDA(cudaMemGetMemPool(
        &runtime_updated_pool, &runtime_location, cudaMemAllocationTypePinned));
    if (runtime_updated_pool != runtime_created_pool) {
        return 1;
    }

    runtime_stream = nullptr;
    runtime_ptr = nullptr;
    CHECK_CUDA(cudaStreamCreate(&runtime_stream));
    CHECK_CUDA(cudaMallocFromPoolAsync(
        &runtime_ptr, 128, runtime_created_pool, runtime_stream));
    CHECK_CUDA(cudaMemsetAsync(runtime_ptr, 0x3b, 128, runtime_stream));
    CHECK_CUDA(cudaFreeAsync(runtime_ptr, runtime_stream));
    CHECK_CUDA(cudaStreamSynchronize(runtime_stream));
    CHECK_CUDA(cudaStreamDestroy(runtime_stream));
    CHECK_CUDA(cudaMemSetMemPool(
        &runtime_location, cudaMemAllocationTypePinned,
        runtime_location_default_pool));
    CHECK_CUDA(cudaMemPoolDestroy(runtime_created_pool));

    CHECK_DRV(cuInit(0));
    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));

    CUmemoryPool driver_default_pool = nullptr;
    CUmemoryPool driver_current_pool = nullptr;
    CHECK_DRV(cuDeviceGetDefaultMemPool(&driver_default_pool, device));
    CHECK_DRV(cuDeviceGetMemPool(&driver_current_pool, device));
    if (driver_default_pool == nullptr || driver_current_pool == nullptr) {
        return 1;
    }
    CHECK_DRV(cuDeviceSetMemPool(device, driver_default_pool));

    std::uint64_t driver_threshold = 0;
    std::uint64_t driver_used = 0;
    std::uint64_t driver_reserved = 0;
    CHECK_DRV(cuMemPoolGetAttribute(
        driver_default_pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
        &driver_threshold));
    CHECK_DRV(cuMemPoolSetAttribute(
        driver_default_pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
        &driver_threshold));
    CHECK_DRV(cuMemPoolGetAttribute(
        driver_default_pool, CU_MEMPOOL_ATTR_USED_MEM_CURRENT, &driver_used));
    CHECK_DRV(cuMemPoolGetAttribute(
        driver_default_pool, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
        &driver_reserved));

    CUstream driver_stream = nullptr;
    CUdeviceptr driver_ptr = 0;
    CHECK_DRV(cuStreamCreate(&driver_stream, CU_STREAM_DEFAULT));
    CHECK_DRV(cuMemAllocFromPoolAsync(
        &driver_ptr, 256, driver_default_pool, driver_stream));
    CHECK_DRV(cuMemsetD8Async(driver_ptr, 0x5a, 256, driver_stream));
    CHECK_DRV(cuMemFreeAsync(driver_ptr, driver_stream));
    CHECK_DRV(cuStreamSynchronize(driver_stream));
    CHECK_DRV(cuMemPoolTrimTo(driver_default_pool, 0));
    CHECK_DRV(cuStreamDestroy(driver_stream));

    CUmemLocation driver_location = {};
    driver_location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    driver_location.id = device;

    CUmemoryPool driver_location_default_pool = nullptr;
    CUmemoryPool driver_location_current_pool = nullptr;
    CHECK_DRV(cuMemGetDefaultMemPool(
        &driver_location_default_pool, &driver_location,
        CU_MEM_ALLOCATION_TYPE_PINNED));
    CHECK_DRV(cuMemGetMemPool(
        &driver_location_current_pool, &driver_location,
        CU_MEM_ALLOCATION_TYPE_PINNED));
    if (driver_location_default_pool == nullptr ||
        driver_location_current_pool == nullptr) {
        return 1;
    }

    CUmemPoolProps driver_pool_props = {};
    driver_pool_props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    driver_pool_props.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
    driver_pool_props.location = driver_location;

    CUmemoryPool driver_created_pool = nullptr;
    CHECK_DRV(cuMemPoolCreate(&driver_created_pool, &driver_pool_props));
    if (driver_created_pool == nullptr) {
        return 1;
    }

    CUmemAccessDesc driver_access = {};
    driver_access.location = driver_location;
    driver_access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_DRV(cuMemPoolSetAccess(driver_created_pool, &driver_access, 1));

    CUmemAccess_flags driver_access_flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
    CHECK_DRV(cuMemPoolGetAccess(
        &driver_access_flags, driver_created_pool, &driver_location));
    if (driver_access_flags != CU_MEM_ACCESS_FLAGS_PROT_READWRITE) {
        return 1;
    }

    CHECK_DRV(cuMemSetMemPool(
        &driver_location, CU_MEM_ALLOCATION_TYPE_PINNED, driver_created_pool));
    CUmemoryPool driver_updated_pool = nullptr;
    CHECK_DRV(cuMemGetMemPool(
        &driver_updated_pool, &driver_location, CU_MEM_ALLOCATION_TYPE_PINNED));
    if (driver_updated_pool != driver_created_pool) {
        return 1;
    }

    driver_stream = nullptr;
    driver_ptr = 0;
    CHECK_DRV(cuStreamCreate(&driver_stream, CU_STREAM_DEFAULT));
    CHECK_DRV(cuMemAllocFromPoolAsync(
        &driver_ptr, 128, driver_created_pool, driver_stream));
    CHECK_DRV(cuMemsetD8Async(driver_ptr, 0x6c, 128, driver_stream));
    CHECK_DRV(cuMemFreeAsync(driver_ptr, driver_stream));
    CHECK_DRV(cuStreamSynchronize(driver_stream));
    CHECK_DRV(cuStreamDestroy(driver_stream));
    CHECK_DRV(cuMemSetMemPool(
        &driver_location, CU_MEM_ALLOCATION_TYPE_PINNED,
        driver_location_default_pool));
    CHECK_DRV(cuMemPoolDestroy(driver_created_pool));

    std::puts("memory pool API test passed");
    return 0;
}
