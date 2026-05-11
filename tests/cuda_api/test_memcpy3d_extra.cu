#include <cuda.h>
#include <cuda_runtime.h>

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

static int verify_equal(const std::vector<unsigned char> &actual,
                        const std::vector<unsigned char> &expected)
{
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
    constexpr size_t width = 16;
    constexpr size_t height = 4;
    constexpr size_t depth = 3;
    constexpr size_t bytes = width * height * depth;

    std::vector<unsigned char> input(bytes);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<unsigned char>((i * 7) & 0xff);
    }

    unsigned char *src = nullptr;
    unsigned char *dst = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&src), bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dst), bytes));
    CHECK_CUDA(cudaMemcpy(src, input.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dst, 0, bytes));

    cudaMemcpy3DParms params = {};
    params.srcPtr = make_cudaPitchedPtr(src, width, width, height);
    params.dstPtr = make_cudaPitchedPtr(dst, width, width, height);
    params.extent = make_cudaExtent(width, height, depth);
    params.kind = cudaMemcpyDeviceToDevice;
    CHECK_CUDA(cudaMemcpy3D(&params));

    std::vector<unsigned char> output(bytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), dst, bytes, cudaMemcpyDeviceToHost));
    if (verify_equal(output, input) != 0) {
        return 1;
    }

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaMemset(dst, 0, bytes));
    CHECK_CUDA(cudaMemcpy3DAsync(&params, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    output.assign(bytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), dst, bytes, cudaMemcpyDeviceToHost));
    if (verify_equal(output, input) != 0) {
        return 1;
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(dst));
    CHECK_CUDA(cudaFree(src));

    CHECK_DRV(cuInit(0));
    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));
    CUcontext ctx = nullptr;
    CHECK_DRV(cuCtxGetCurrent(&ctx));
    if (ctx == nullptr) {
        CHECK_DRV(cuDevicePrimaryCtxRetain(&ctx, device));
        CHECK_DRV(cuCtxSetCurrent(ctx));
    }

    constexpr size_t width2d = 16;
    constexpr size_t height2d = 4;
    constexpr size_t src_pitch2d = 24;
    constexpr size_t dst_pitch2d = 32;
    std::vector<unsigned char> input2d(src_pitch2d * height2d, 0);
    std::vector<unsigned char> expected2d(dst_pitch2d * height2d, 0);
    for (size_t row = 0; row < height2d; ++row) {
        for (size_t col = 0; col < width2d; ++col) {
            unsigned char value =
                static_cast<unsigned char>((row * 17 + col * 3) & 0xff);
            input2d[row * src_pitch2d + col] = value;
            expected2d[row * dst_pitch2d + col] = value;
        }
    }

    CUdeviceptr driver_src2d = 0;
    CUdeviceptr driver_dst2d = 0;
    CHECK_DRV(cuMemAlloc(&driver_src2d, input2d.size()));
    CHECK_DRV(cuMemAlloc(&driver_dst2d, expected2d.size()));
    CHECK_DRV(cuMemcpyHtoD(driver_src2d, input2d.data(), input2d.size()));

    CUDA_MEMCPY2D copy2d = {};
    copy2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy2d.srcDevice = driver_src2d;
    copy2d.srcPitch = src_pitch2d;
    copy2d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy2d.dstDevice = driver_dst2d;
    copy2d.dstPitch = dst_pitch2d;
    copy2d.WidthInBytes = width2d;
    copy2d.Height = height2d;

    std::vector<unsigned char> output2d(expected2d.size(), 0);
    CHECK_DRV(cuMemsetD8(driver_dst2d, 0, expected2d.size()));
    CHECK_DRV(cuMemcpy2D(&copy2d));
    CHECK_DRV(cuMemcpyDtoH(output2d.data(), driver_dst2d, output2d.size()));
    if (verify_equal(output2d, expected2d) != 0) {
        return 1;
    }

    output2d.assign(expected2d.size(), 0);
    CHECK_DRV(cuMemsetD8(driver_dst2d, 0, expected2d.size()));
    CHECK_DRV(cuMemcpy2DUnaligned(&copy2d));
    CHECK_DRV(cuMemcpyDtoH(output2d.data(), driver_dst2d, output2d.size()));
    if (verify_equal(output2d, expected2d) != 0) {
        return 1;
    }

    CUstream driver_stream = nullptr;
    CHECK_DRV(cuStreamCreate(&driver_stream, CU_STREAM_DEFAULT));
    output2d.assign(expected2d.size(), 0);
    CHECK_DRV(cuMemsetD8(driver_dst2d, 0, expected2d.size()));
    CHECK_DRV(cuMemcpy2DAsync(&copy2d, driver_stream));
    CHECK_DRV(cuStreamSynchronize(driver_stream));
    CHECK_DRV(cuMemcpyDtoH(output2d.data(), driver_dst2d, output2d.size()));
    if (verify_equal(output2d, expected2d) != 0) {
        return 1;
    }

    CUdeviceptr driver_src3d = 0;
    CUdeviceptr driver_dst3d = 0;
    CHECK_DRV(cuMemAlloc(&driver_src3d, bytes));
    CHECK_DRV(cuMemAlloc(&driver_dst3d, bytes));
    CHECK_DRV(cuMemcpyHtoD(driver_src3d, input.data(), bytes));

    CUDA_MEMCPY3D copy3d = {};
    copy3d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy3d.srcDevice = driver_src3d;
    copy3d.srcPitch = width;
    copy3d.srcHeight = height;
    copy3d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy3d.dstDevice = driver_dst3d;
    copy3d.dstPitch = width;
    copy3d.dstHeight = height;
    copy3d.WidthInBytes = width;
    copy3d.Height = height;
    copy3d.Depth = depth;

    output.assign(bytes, 0);
    CHECK_DRV(cuMemsetD8(driver_dst3d, 0, bytes));
    CHECK_DRV(cuMemcpy3D(&copy3d));
    CHECK_DRV(cuMemcpyDtoH(output.data(), driver_dst3d, bytes));
    if (verify_equal(output, input) != 0) {
        return 1;
    }

    output.assign(bytes, 0);
    CHECK_DRV(cuMemsetD8(driver_dst3d, 0, bytes));
    CHECK_DRV(cuMemcpy3DAsync(&copy3d, driver_stream));
    CHECK_DRV(cuStreamSynchronize(driver_stream));
    CHECK_DRV(cuMemcpyDtoH(output.data(), driver_dst3d, bytes));
    if (verify_equal(output, input) != 0) {
        return 1;
    }

    CHECK_DRV(cuStreamDestroy(driver_stream));
    CHECK_DRV(cuMemFree(driver_dst3d));
    CHECK_DRV(cuMemFree(driver_src3d));
    CHECK_DRV(cuMemFree(driver_dst2d));
    CHECK_DRV(cuMemFree(driver_src2d));

    std::puts("memcpy3d API test passed");
    return 0;
}
