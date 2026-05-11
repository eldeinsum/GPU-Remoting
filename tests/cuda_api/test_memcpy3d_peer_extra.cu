#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
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

static int verify_equal(const std::vector<unsigned char> &actual,
                        const std::vector<unsigned char> &expected)
{
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "size mismatch: got %zu expected %zu\n",
                     actual.size(), expected.size());
        return 1;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "mismatch at byte %zu: got %u expected %u\n",
                         i, static_cast<unsigned>(actual[i]),
                         static_cast<unsigned>(expected[i]));
            return 1;
        }
    }
    return 0;
}

int main()
{
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        std::puts("memcpy3d peer API test skipped");
        return 0;
    }

    int peer_access = 0;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&peer_access, 1, 0));
    if (!peer_access) {
        std::puts("memcpy3d peer API test skipped");
        return 0;
    }

    constexpr size_t width = 16;
    constexpr size_t height = 4;
    constexpr size_t depth = 3;
    constexpr size_t bytes = width * height * depth;

    std::vector<unsigned char> input(bytes);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<unsigned char>((i * 11 + 5) & 0xff);
    }

    unsigned char *src = nullptr;
    unsigned char *dst = nullptr;

    CHECK_CUDA(cudaSetDevice(0));
    cudaError_t enable_01 = cudaDeviceEnablePeerAccess(1, 0);
    if (enable_01 != cudaSuccess &&
        enable_01 != cudaErrorPeerAccessAlreadyEnabled) {
        std::fprintf(stderr, "cudaDeviceEnablePeerAccess(1) failed: %s (%d)\n",
                     cudaGetErrorString(enable_01),
                     static_cast<int>(enable_01));
        return 1;
    }
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&src), bytes));
    CHECK_CUDA(cudaMemcpy(src, input.data(), bytes, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaSetDevice(1));
    cudaError_t enable_10 = cudaDeviceEnablePeerAccess(0, 0);
    if (enable_10 != cudaSuccess &&
        enable_10 != cudaErrorPeerAccessAlreadyEnabled) {
        std::fprintf(stderr, "cudaDeviceEnablePeerAccess(0) failed: %s (%d)\n",
                     cudaGetErrorString(enable_10),
                     static_cast<int>(enable_10));
        return 1;
    }
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dst), bytes));

    cudaMemcpy3DPeerParms params = {};
    params.srcPtr = make_cudaPitchedPtr(src, width, width, height);
    params.srcDevice = 0;
    params.dstPtr = make_cudaPitchedPtr(dst, width, width, height);
    params.dstDevice = 1;
    params.extent = make_cudaExtent(width, height, depth);
    CHECK_CUDA(cudaMemcpy3DPeer(&params));

    std::vector<unsigned char> output(bytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), dst, bytes, cudaMemcpyDeviceToHost));
    if (verify_equal(output, input) != 0) {
        return 1;
    }

    CHECK_CUDA(cudaSetDevice(0));
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaMemset(dst, 0, bytes));
    CHECK_CUDA(cudaMemcpy3DPeerAsync(&params, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    output.assign(bytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), dst, bytes, cudaMemcpyDeviceToHost));
    if (verify_equal(output, input) != 0) {
        return 1;
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(dst));
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaFree(src));

    CHECK_DRV(cuInit(0));
    CUdevice driver_device0 = 0;
    CUdevice driver_device1 = 0;
    CHECK_DRV(cuDeviceGet(&driver_device0, 0));
    CHECK_DRV(cuDeviceGet(&driver_device1, 1));

    CUcontext ctx0 = nullptr;
    CUcontext ctx1 = nullptr;
    CHECK_DRV(cuDevicePrimaryCtxRetain(&ctx0, driver_device0));
    CHECK_DRV(cuDevicePrimaryCtxRetain(&ctx1, driver_device1));

    CHECK_DRV(cuCtxSetCurrent(ctx0));
    CUdeviceptr driver_src = 0;
    CHECK_DRV(cuMemAlloc(&driver_src, bytes));
    CHECK_DRV(cuMemcpyHtoD(driver_src, input.data(), bytes));

    CHECK_DRV(cuCtxSetCurrent(ctx1));
    CUdeviceptr driver_dst = 0;
    CHECK_DRV(cuMemAlloc(&driver_dst, bytes));

    output.assign(bytes, 0);
    CHECK_DRV(cuMemsetD8(driver_dst, 0, bytes));
    CHECK_DRV(cuMemcpyPeer(driver_dst, ctx1, driver_src, ctx0, bytes));
    CHECK_DRV(cuMemcpyDtoH(output.data(), driver_dst, bytes));
    if (verify_equal(output, input) != 0) {
        return 1;
    }

    CUstream driver_stream = nullptr;
    CHECK_DRV(cuStreamCreate(&driver_stream, CU_STREAM_DEFAULT));
    output.assign(bytes, 0);
    CHECK_DRV(cuMemsetD8(driver_dst, 0, bytes));
    CHECK_DRV(cuMemcpyPeerAsync(
        driver_dst, ctx1, driver_src, ctx0, bytes, driver_stream));
    CHECK_DRV(cuStreamSynchronize(driver_stream));
    CHECK_DRV(cuMemcpyDtoH(output.data(), driver_dst, bytes));
    if (verify_equal(output, input) != 0) {
        return 1;
    }

    CUDA_MEMCPY3D_PEER driver_params = {};
    driver_params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    driver_params.srcDevice = driver_src;
    driver_params.srcContext = ctx0;
    driver_params.srcPitch = width;
    driver_params.srcHeight = height;
    driver_params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    driver_params.dstDevice = driver_dst;
    driver_params.dstContext = ctx1;
    driver_params.dstPitch = width;
    driver_params.dstHeight = height;
    driver_params.WidthInBytes = width;
    driver_params.Height = height;
    driver_params.Depth = depth;

    output.assign(bytes, 0);
    CHECK_DRV(cuMemsetD8(driver_dst, 0, bytes));
    CHECK_DRV(cuMemcpy3DPeer(&driver_params));
    CHECK_DRV(cuMemcpyDtoH(output.data(), driver_dst, bytes));
    if (verify_equal(output, input) != 0) {
        return 1;
    }

    output.assign(bytes, 0);
    CHECK_DRV(cuMemsetD8(driver_dst, 0, bytes));
    CHECK_DRV(cuMemcpy3DPeerAsync(&driver_params, driver_stream));
    CHECK_DRV(cuStreamSynchronize(driver_stream));
    CHECK_DRV(cuMemcpyDtoH(output.data(), driver_dst, bytes));
    if (verify_equal(output, input) != 0) {
        return 1;
    }

    CHECK_DRV(cuStreamDestroy(driver_stream));
    CHECK_DRV(cuMemFree(driver_dst));
    CHECK_DRV(cuCtxSetCurrent(ctx0));
    CHECK_DRV(cuMemFree(driver_src));

    std::puts("memcpy3d peer API test passed");
    return 0;
}
