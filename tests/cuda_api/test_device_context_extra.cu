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
    int runtime_version = 0;
    int driver_version = 0;
    CHECK_CUDA(cudaRuntimeGetVersion(&runtime_version));
    CHECK_CUDA(cudaDriverGetVersion(&driver_version));
    if (runtime_version <= 0 || driver_version <= 0) {
        return 1;
    }

    int runtime_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&runtime_count));
    if (runtime_count <= 0) {
        return 1;
    }

    char runtime_pci[64] = {};
    CHECK_CUDA(cudaDeviceGetPCIBusId(runtime_pci, sizeof(runtime_pci), 0));
    int runtime_device_from_pci = -1;
    CHECK_CUDA(cudaDeviceGetByPCIBusId(&runtime_device_from_pci, runtime_pci));
    if (runtime_device_from_pci != 0) {
        return 1;
    }

    int runtime_peer_access = 0;
    CHECK_CUDA(cudaDeviceCanAccessPeer(
        &runtime_peer_access, 0, runtime_count > 1 ? 1 : 0));

    unsigned int device_flags = 0;
    CHECK_CUDA(cudaGetDeviceFlags(&device_flags));

    CHECK_CUDA(cudaSetDevice(0));
    void *init = nullptr;
    CHECK_CUDA(cudaMalloc(&init, 1));
    CHECK_CUDA(cudaFree(init));

    cudaFuncCache runtime_cache = cudaFuncCachePreferNone;
    CHECK_CUDA(cudaDeviceGetCacheConfig(&runtime_cache));
    CHECK_CUDA(cudaDeviceSetCacheConfig(runtime_cache));

    cudaSharedMemConfig runtime_shared = cudaSharedMemBankSizeDefault;
    CHECK_CUDA(cudaDeviceGetSharedMemConfig(&runtime_shared));
    CHECK_CUDA(cudaDeviceSetSharedMemConfig(runtime_shared));

    size_t runtime_stack_limit = 0;
    CHECK_CUDA(cudaDeviceGetLimit(&runtime_stack_limit, cudaLimitStackSize));
    CHECK_CUDA(cudaDeviceSetLimit(cudaLimitStackSize, runtime_stack_limit));

    CHECK_DRV(cuInit(0));
    int driver_count = 0;
    CHECK_DRV(cuDeviceGetCount(&driver_count));
    if (driver_count != runtime_count) {
        return 1;
    }

    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));

    char name[256] = {};
    CHECK_DRV(cuDeviceGetName(name, sizeof(name), device));
    if (std::strlen(name) == 0) {
        return 1;
    }

    char driver_pci[64] = {};
    CHECK_DRV(cuDeviceGetPCIBusId(driver_pci, sizeof(driver_pci), device));
    if (std::strcmp(driver_pci, runtime_pci) != 0) {
        return 1;
    }

    CUdevice driver_device_from_pci = -1;
    CHECK_DRV(cuDeviceGetByPCIBusId(&driver_device_from_pci, driver_pci));
    if (driver_device_from_pci != device) {
        return 1;
    }

    int major = 0;
    int minor = 0;
    CHECK_DRV(cuDeviceComputeCapability(&major, &minor, device));
    if (major <= 0) {
        return 1;
    }

    size_t total_mem = 0;
    CHECK_DRV(cuDeviceTotalMem(&total_mem, device));
    if (total_mem == 0) {
        return 1;
    }

    CUuuid uuid = {};
    CHECK_DRV(cuDeviceGetUuid(&uuid, device));

    int driver_peer_access = 0;
    CHECK_DRV(cuDeviceCanAccessPeer(
        &driver_peer_access, device, runtime_count > 1 ? 1 : device));
    if (driver_peer_access != runtime_peer_access && runtime_count > 1) {
        return 1;
    }

    CUcontext retained = nullptr;
    CHECK_DRV(cuDevicePrimaryCtxRetain(&retained, device));
    if (retained == nullptr) {
        return 1;
    }

    CUcontext current = nullptr;
    CHECK_DRV(cuCtxGetCurrent(&current));
    if (current == nullptr) {
        return 1;
    }

    unsigned int api_version = 0;
    CHECK_DRV(cuCtxGetApiVersion(current, &api_version));
    if (api_version == 0) {
        return 1;
    }

    CUdevice context_device = -1;
    CHECK_DRV(cuCtxGetDevice(&context_device));
    if (context_device != device) {
        return 1;
    }

    CUdevice explicit_context_device = -1;
    CHECK_DRV(cuCtxGetDevice_v2(&explicit_context_device, current));
    if (explicit_context_device != device) {
        return 1;
    }

    unsigned int context_flags = 0;
    CHECK_DRV(cuCtxGetFlags(&context_flags));

    CUfunc_cache cache_config = CU_FUNC_CACHE_PREFER_NONE;
    CHECK_DRV(cuCtxGetCacheConfig(&cache_config));
    CHECK_DRV(cuCtxSetCacheConfig(cache_config));

    CUsharedconfig shared_config = CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE;
    CHECK_DRV(cuCtxGetSharedMemConfig(&shared_config));
    CHECK_DRV(cuCtxSetSharedMemConfig(shared_config));

    size_t stack_limit = 0;
    CHECK_DRV(cuCtxGetLimit(&stack_limit, CU_LIMIT_STACK_SIZE));
    CHECK_DRV(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, stack_limit));

    int least_priority = 0;
    int greatest_priority = 0;
    CHECK_DRV(cuCtxGetStreamPriorityRange(&least_priority, &greatest_priority));

    CHECK_DRV(cuCtxResetPersistingL2Cache());
    CHECK_DRV(cuDevicePrimaryCtxRelease(device));

    std::puts("device/context extra API test passed");
    return 0;
}
