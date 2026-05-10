#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t result = (call);                                            \
        if (result != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s failed: %s (%d)\n", #call,                \
                         cudaGetErrorString(result), static_cast<int>(result)); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

#define CHECK_DRV(call)                                                         \
    do {                                                                        \
        CUresult result = (call);                                               \
        if (result != CUDA_SUCCESS) {                                           \
            const char *name = nullptr;                                         \
            cuGetErrorName(result, &name);                                      \
            std::fprintf(stderr, "%s failed: %s (%d)\n", #call,                \
                         name ? name : "unknown", static_cast<int>(result));   \
            return 1;                                                           \
        }                                                                       \
    } while (0)

int main() {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        std::fprintf(stderr, "no CUDA devices found\n");
        return 1;
    }

    cudaDeviceProp prop = {};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    int chosen = -1;
    CHECK_CUDA(cudaChooseDevice(&chosen, &prop));
    if (chosen < 0 || chosen >= device_count) {
        std::fprintf(stderr, "cudaChooseDevice returned invalid device %d\n", chosen);
        return 1;
    }

    cudaChannelFormatDesc runtime_desc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
    size_t runtime_texture_width = 0;
    CHECK_CUDA(cudaDeviceGetTexture1DLinearMaxWidth(
        &runtime_texture_width, &runtime_desc, 0));
    if (runtime_texture_width == 0) {
        std::fprintf(stderr, "runtime texture max width is zero\n");
        return 1;
    }

    cudaAtomicOperation runtime_ops[2] = {
        cudaAtomicOperationIntegerAdd,
        cudaAtomicOperationCAS,
    };
    unsigned int runtime_caps[2] = {};
    CHECK_CUDA(cudaDeviceGetHostAtomicCapabilities(
        runtime_caps, runtime_ops, 2, 0));

    if (device_count > 1) {
        int can_access = 0;
        CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, 0, 1));
        if (can_access) {
            CHECK_CUDA(cudaSetDevice(0));
            cudaError_t enable_result = cudaDeviceEnablePeerAccess(1, 0);
            if (enable_result != cudaSuccess &&
                enable_result != cudaErrorPeerAccessAlreadyEnabled) {
                std::fprintf(stderr, "cudaDeviceEnablePeerAccess failed: %s (%d)\n",
                             cudaGetErrorString(enable_result),
                             static_cast<int>(enable_result));
                return 1;
            }
            CHECK_CUDA(cudaDeviceDisablePeerAccess(1));
        }
    }

    CHECK_DRV(cuInit(0));
    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));
    CUcontext ctx = nullptr;
    CHECK_DRV(cuCtxGetCurrent(&ctx));
    if (ctx == nullptr) {
        CHECK_DRV(cuDevicePrimaryCtxRetain(&ctx, device));
        CHECK_DRV(cuCtxSetCurrent(ctx));
    }

    unsigned long long ctx_id = 0;
    CHECK_DRV(cuCtxGetId(ctx, &ctx_id));
    if (ctx_id == 0) {
        std::fprintf(stderr, "cuCtxGetId returned zero\n");
        return 1;
    }

    CHECK_DRV(cuCtxPushCurrent(ctx));
    CUcontext popped = nullptr;
    CHECK_DRV(cuCtxPopCurrent(&popped));
    if (popped != ctx) {
        std::fprintf(stderr, "cuCtxPopCurrent returned a different context\n");
        return 1;
    }
    CHECK_DRV(cuCtxSetCurrent(ctx));

    size_t driver_texture_width = 0;
    CHECK_DRV(cuDeviceGetTexture1DLinearMaxWidth(
        &driver_texture_width, CU_AD_FORMAT_UNSIGNED_INT32, 1, device));
    if (driver_texture_width == 0) {
        std::fprintf(stderr, "driver texture max width is zero\n");
        return 1;
    }

    CUatomicOperation driver_ops[2] = {
        CU_ATOMIC_OPERATION_INTEGER_ADD,
        CU_ATOMIC_OPERATION_CAS,
    };
    unsigned int driver_caps[2] = {};
    CHECK_DRV(cuDeviceGetHostAtomicCapabilities(
        driver_caps, driver_ops, 2, device));

    CUdeviceptr dptr = 0;
    CHECK_DRV(cuMemAlloc(&dptr, 1024));
    CUmemorytype memory_type = CU_MEMORYTYPE_HOST;
    CHECK_DRV(cuPointerGetAttribute(
        &memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, dptr));
    if (memory_type != CU_MEMORYTYPE_DEVICE) {
        std::fprintf(stderr, "unexpected pointer memory type %u\n",
                     static_cast<unsigned>(memory_type));
        return 1;
    }

    CUdeviceptr device_pointer = 0;
    CHECK_DRV(cuPointerGetAttribute(
        &device_pointer, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, dptr));
    if (device_pointer != dptr) {
        std::fprintf(stderr, "device pointer attribute mismatch\n");
        return 1;
    }

    int ordinal = -1;
    CHECK_DRV(cuPointerGetAttribute(
        &ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, dptr));
    if (ordinal != 0) {
        std::fprintf(stderr, "unexpected pointer device ordinal %d\n", ordinal);
        return 1;
    }

    size_t range_size = 0;
    CHECK_DRV(cuPointerGetAttribute(
        &range_size, CU_POINTER_ATTRIBUTE_RANGE_SIZE, dptr));
    if (range_size < 1024) {
        std::fprintf(stderr, "unexpected pointer range size %zu\n", range_size);
        return 1;
    }

    int managed = -1;
    CHECK_DRV(cuPointerGetAttribute(
        &managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, dptr));
    if (managed != 0) {
        std::fprintf(stderr, "unexpected managed attribute %d\n", managed);
        return 1;
    }

    CHECK_DRV(cuMemFree(dptr));
    return 0;
}
