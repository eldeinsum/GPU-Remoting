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

        if (runtime_peer_attr) {
            constexpr int kCount = 8;
            constexpr size_t kBytes = kCount * sizeof(int);
            int input[kCount] = {0, 1, 2, 3, 4, 5, 6, 7};
            int output[kCount] = {};

            int *device0_data = nullptr;
            int *device1_data = nullptr;

            CHECK_CUDA(cudaSetDevice(0));
            cudaError_t enable_01 = cudaDeviceEnablePeerAccess(1, 0);
            if (enable_01 != cudaSuccess &&
                enable_01 != cudaErrorPeerAccessAlreadyEnabled) {
                std::fprintf(stderr, "cudaDeviceEnablePeerAccess(1) failed: %s (%d)\n",
                             cudaGetErrorString(enable_01),
                             static_cast<int>(enable_01));
                return 1;
            }
            CHECK_CUDA(cudaMalloc(&device0_data, kBytes));
            CHECK_CUDA(cudaMemcpy(device0_data, input, kBytes,
                                  cudaMemcpyHostToDevice));

            CHECK_CUDA(cudaSetDevice(1));
            int current_device = -1;
            CHECK_CUDA(cudaGetDevice(&current_device));
            if (current_device != 1) {
                std::fprintf(stderr, "cudaSetDevice(1) left current device %d\n",
                             current_device);
                return 1;
            }
            cudaError_t enable_10 = cudaDeviceEnablePeerAccess(0, 0);
            if (enable_10 != cudaSuccess &&
                enable_10 != cudaErrorPeerAccessAlreadyEnabled) {
                std::fprintf(stderr, "cudaDeviceEnablePeerAccess(0) failed: %s (%d)\n",
                             cudaGetErrorString(enable_10),
                             static_cast<int>(enable_10));
                return 1;
            }
            CHECK_CUDA(cudaMalloc(&device1_data, kBytes));
            CHECK_CUDA(cudaMemcpyPeer(device1_data, 1, device0_data, 0,
                                      kBytes));
            CHECK_CUDA(cudaMemcpy(output, device1_data, kBytes,
                                  cudaMemcpyDeviceToHost));
            if (std::memcmp(input, output, kBytes) != 0) {
                std::fprintf(stderr, "cudaMemcpyPeer copied unexpected data\n");
                return 1;
            }

            CHECK_CUDA(cudaMemset(device1_data, 0, kBytes));
            cudaStream_t peer_stream = nullptr;
            CHECK_CUDA(cudaStreamCreate(&peer_stream));
            CHECK_CUDA(cudaMemcpyPeerAsync(device1_data, 1, device0_data, 0,
                                           kBytes, peer_stream));
            CHECK_CUDA(cudaStreamSynchronize(peer_stream));
            CHECK_CUDA(cudaStreamDestroy(peer_stream));
            std::memset(output, 0, kBytes);
            CHECK_CUDA(cudaMemcpy(output, device1_data, kBytes,
                                  cudaMemcpyDeviceToHost));
            if (std::memcmp(input, output, kBytes) != 0) {
                std::fprintf(stderr,
                             "cudaMemcpyPeerAsync copied unexpected data\n");
                return 1;
            }

            CHECK_CUDA(cudaFree(device1_data));
            CHECK_CUDA(cudaSetDevice(0));
            CHECK_CUDA(cudaFree(device0_data));
        }

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
