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
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        return 1;
    }

    cudaDeviceProp runtime_prop = {};
    CHECK_CUDA(cudaGetDeviceProperties(&runtime_prop, 0));
    if (std::strlen(runtime_prop.name) == 0 || runtime_prop.multiProcessorCount <= 0) {
        return 1;
    }

    CHECK_DRV(cuInit(0));
    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));
    CUdevprop driver_prop = {};
    CHECK_DRV(cuDeviceGetProperties(&driver_prop, device));
    if (driver_prop.maxThreadsPerBlock <= 0 || driver_prop.maxGridSize[0] <= 0) {
        return 1;
    }

    if (device_count > 1) {
        int runtime_peer_attr = 0;
        CHECK_CUDA(cudaDeviceGetP2PAttribute(
            &runtime_peer_attr, cudaDevP2PAttrAccessSupported, 0, 1));

        cudaAtomicOperation runtime_ops[2] = {
            cudaAtomicOperationIntegerAdd,
            cudaAtomicOperationCAS,
        };
        unsigned int runtime_caps[2] = {};
        CHECK_CUDA(cudaDeviceGetP2PAtomicCapabilities(
            runtime_caps, runtime_ops, 2, 0, 1));

        CUdevice peer = 0;
        CHECK_DRV(cuDeviceGet(&peer, 1));
        int driver_peer_attr = 0;
        CHECK_DRV(cuDeviceGetP2PAttribute(
            &driver_peer_attr, CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED,
            device, peer));
        if (driver_peer_attr != runtime_peer_attr) {
            return 1;
        }

        CUatomicOperation driver_ops[2] = {
            CU_ATOMIC_OPERATION_INTEGER_ADD,
            CU_ATOMIC_OPERATION_CAS,
        };
        unsigned int driver_caps[2] = {};
        CHECK_DRV(cuDeviceGetP2PAtomicCapabilities(
            driver_caps, driver_ops, 2, device, peer));
    }

    std::puts("device peer API test passed");
    return 0;
}
