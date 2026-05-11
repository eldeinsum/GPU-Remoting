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

static int check_cuda_optional(cudaError_t result, const char *call,
                               cudaError_t allowed)
{
    if (result == cudaSuccess || result == allowed) {
        return 0;
    }
    std::fprintf(stderr, "%s failed: %s (%d)\n", call,
                 cudaGetErrorString(result), static_cast<int>(result));
    return 1;
}

static int check_driver_optional(CUresult result, const char *call,
                                 CUresult allowed)
{
    if (result == CUDA_SUCCESS || result == allowed) {
        return 0;
    }
    const char *name = nullptr;
    cuGetErrorName(result, &name);
    std::fprintf(stderr, "%s failed: %s (%d)\n", call,
                 name == nullptr ? "unknown" : name, static_cast<int>(result));
    return 1;
}

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
    if (check_cuda_optional(cudaDeviceFlushGPUDirectRDMAWrites(
                                cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                cudaFlushGPUDirectRDMAWritesToOwner),
                            "cudaDeviceFlushGPUDirectRDMAWrites",
                            cudaErrorNotSupported)) {
        return 1;
    }

    CHECK_DRV(cuInit(0));
    int driver_count = 0;
    CHECK_DRV(cuDeviceGetCount(&driver_count));
    if (driver_count != runtime_count) {
        return 1;
    }

    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));
    if (driver_count > 1) {
        CUdevice secondary_device = 0;
        CHECK_DRV(cuDeviceGet(&secondary_device, 1));
        unsigned int secondary_flags = 0;
        int secondary_active = 0;
        CHECK_DRV(cuDevicePrimaryCtxGetState(
            secondary_device, &secondary_flags, &secondary_active));
        if (secondary_active == 0) {
            CHECK_DRV(cuDevicePrimaryCtxSetFlags(secondary_device,
                                                 secondary_flags));
            CUcontext secondary_context = nullptr;
            CHECK_DRV(cuDevicePrimaryCtxRetain(&secondary_context,
                                               secondary_device));
            if (secondary_context == nullptr) {
                std::fprintf(stderr,
                             "cuDevicePrimaryCtxRetain returned null\n");
                return 1;
            }
            CHECK_DRV(cuDevicePrimaryCtxRelease(secondary_device));
            CHECK_DRV(cuDevicePrimaryCtxReset(secondary_device));
            CHECK_DRV(cuDevicePrimaryCtxGetState(
                secondary_device, &secondary_flags, &secondary_active));
            if (secondary_active != 0) {
                std::fprintf(stderr,
                             "secondary primary context remained active\n");
                return 1;
            }
        }
    }

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
    char luid[8] = {};
    unsigned int device_node_mask = 0;
    if (check_driver_optional(cuDeviceGetLuid(luid, &device_node_mask, device),
                              "cuDeviceGetLuid",
                              CUDA_ERROR_NOT_SUPPORTED)) {
        return 1;
    }
    int exec_affinity_supported = 0;
    CHECK_DRV(cuDeviceGetExecAffinitySupport(
        &exec_affinity_supported, CU_EXEC_AFFINITY_TYPE_SM_COUNT, device));
    if (check_driver_optional(cuFlushGPUDirectRDMAWrites(
                                  CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
                                  CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER),
                              "cuFlushGPUDirectRDMAWrites",
                              CUDA_ERROR_NOT_SUPPORTED)) {
        return 1;
    }

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
    if (check_driver_optional(cuCtxSetFlags(context_flags), "cuCtxSetFlags",
                              CUDA_ERROR_INVALID_CONTEXT)) {
        return 1;
    }
    CUexecAffinityParam exec_affinity = {};
    CUresult exec_affinity_result = cuCtxGetExecAffinity(
        &exec_affinity, CU_EXEC_AFFINITY_TYPE_SM_COUNT);
    if (exec_affinity_result == CUDA_SUCCESS) {
        if (exec_affinity.type != CU_EXEC_AFFINITY_TYPE_SM_COUNT) {
            std::fprintf(stderr, "unexpected exec affinity type\n");
            return 1;
        }
    } else if (exec_affinity_result != CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY &&
               exec_affinity_result != CUDA_ERROR_NOT_SUPPORTED) {
        const char *name = nullptr;
        cuGetErrorName(exec_affinity_result, &name);
        std::fprintf(stderr, "cuCtxGetExecAffinity failed: %s (%d)\n",
                     name == nullptr ? "unknown" : name,
                     static_cast<int>(exec_affinity_result));
        return 1;
    }

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

    CUcontext created = nullptr;
    CHECK_DRV(cuCtxCreate(&created, nullptr, 0, device));
    if (created == nullptr) {
        std::fprintf(stderr, "cuCtxCreate returned a null context\n");
        return 1;
    }
    CUcontext attached = nullptr;
    CHECK_DRV(cuCtxAttach(&attached, 0));
    if (attached != created) {
        std::fprintf(stderr, "cuCtxAttach returned unexpected context\n");
        return 1;
    }
    CHECK_DRV(cuCtxDetach(attached));
    CHECK_DRV(cuCtxDestroy(created));
    CHECK_DRV(cuCtxSetCurrent(current));

    CHECK_DRV(cuDevicePrimaryCtxRelease(device));

    std::puts("device/context extra API test passed");
    return 0;
}
