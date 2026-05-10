#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
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

template <size_t N>
bool equal_bytes(const std::array<unsigned char, N> &left,
                 const std::array<unsigned char, N> &right) {
    for (size_t i = 0; i < N; ++i) {
        if (left[i] != right[i]) {
            std::fprintf(stderr, "byte mismatch at %zu: %u != %u\n", i,
                         static_cast<unsigned>(left[i]),
                         static_cast<unsigned>(right[i]));
            return false;
        }
    }
    return true;
}

int main() {
    constexpr size_t kBytes = 64;
    std::array<unsigned char, kBytes> input = {};
    std::array<unsigned char, kBytes> output = {};
    std::array<unsigned char, kBytes> device_roundtrip = {};
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<unsigned char>(i + 3);
    }

    cudaChannelFormatDesc desc =
        cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaArray_t runtime_array = nullptr;
    CHECK_CUDA(cudaMallocArray(&runtime_array, &desc, kBytes, 1));

    cudaChannelFormatDesc actual_desc = {};
    cudaExtent actual_extent = {};
    unsigned int actual_flags = 0;
    CHECK_CUDA(cudaArrayGetInfo(
        &actual_desc, &actual_extent, &actual_flags, runtime_array));
    if (actual_desc.x != 8 || actual_extent.width != kBytes ||
        actual_extent.height != 1) {
        std::fprintf(stderr, "unexpected runtime array descriptor\n");
        return 1;
    }

    CHECK_CUDA(cudaMemcpyToArray(
        runtime_array, 0, 0, input.data(), kBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyFromArray(
        output.data(), runtime_array, 0, 0, kBytes, cudaMemcpyDeviceToHost));
    if (!equal_bytes(input, output)) {
        return 1;
    }

    cudaStream_t runtime_stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&runtime_stream));
    output.fill(0);
    CHECK_CUDA(cudaMemcpyToArrayAsync(
        runtime_array, 0, 0, input.data(), kBytes, cudaMemcpyHostToDevice,
        runtime_stream));
    CHECK_CUDA(cudaMemcpyFromArrayAsync(
        output.data(), runtime_array, 0, 0, kBytes, cudaMemcpyDeviceToHost,
        runtime_stream));
    CHECK_CUDA(cudaStreamSynchronize(runtime_stream));
    if (!equal_bytes(input, output)) {
        return 1;
    }

    void *runtime_device = nullptr;
    void *runtime_device_out = nullptr;
    CHECK_CUDA(cudaMalloc(&runtime_device, kBytes));
    CHECK_CUDA(cudaMalloc(&runtime_device_out, kBytes));
    CHECK_CUDA(cudaMemcpy(
        runtime_device, input.data(), kBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToArray(
        runtime_array, 0, 0, runtime_device, kBytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpyFromArray(
        runtime_device_out, runtime_array, 0, 0, kBytes,
        cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(
        device_roundtrip.data(), runtime_device_out, kBytes,
        cudaMemcpyDeviceToHost));
    if (!equal_bytes(input, device_roundtrip)) {
        return 1;
    }

    CHECK_CUDA(cudaFree(runtime_device));
    CHECK_CUDA(cudaFree(runtime_device_out));
    CHECK_CUDA(cudaFreeArray(runtime_array));

    constexpr size_t kWidth = 8;
    constexpr size_t kHeight = 4;
    constexpr size_t kPitch = 16;
    std::array<unsigned char, kPitch * kHeight> input_2d = {};
    std::array<unsigned char, kPitch * kHeight> output_2d = {};
    for (size_t row = 0; row < kHeight; ++row) {
        for (size_t col = 0; col < kWidth; ++col) {
            input_2d[row * kPitch + col] =
                static_cast<unsigned char>(row * 31 + col + 7);
        }
    }

    cudaArray_t runtime_array_2d = nullptr;
    CHECK_CUDA(cudaMallocArray(&runtime_array_2d, &desc, kWidth, kHeight));
    CHECK_CUDA(cudaMemcpy2DToArray(
        runtime_array_2d, 0, 0, input_2d.data(), kPitch, kWidth, kHeight,
        cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy2DFromArray(
        output_2d.data(), kPitch, runtime_array_2d, 0, 0, kWidth, kHeight,
        cudaMemcpyDeviceToHost));
    for (size_t row = 0; row < kHeight; ++row) {
        for (size_t col = 0; col < kWidth; ++col) {
            if (input_2d[row * kPitch + col] != output_2d[row * kPitch + col]) {
                std::fprintf(stderr, "2D runtime mismatch at row %zu col %zu\n",
                             row, col);
                return 1;
            }
        }
    }

    output_2d.fill(0);
    CHECK_CUDA(cudaMemcpy2DToArrayAsync(
        runtime_array_2d, 0, 0, input_2d.data(), kPitch, kWidth, kHeight,
        cudaMemcpyHostToDevice, runtime_stream));
    CHECK_CUDA(cudaMemcpy2DFromArrayAsync(
        output_2d.data(), kPitch, runtime_array_2d, 0, 0, kWidth, kHeight,
        cudaMemcpyDeviceToHost, runtime_stream));
    CHECK_CUDA(cudaStreamSynchronize(runtime_stream));
    for (size_t row = 0; row < kHeight; ++row) {
        for (size_t col = 0; col < kWidth; ++col) {
            if (input_2d[row * kPitch + col] != output_2d[row * kPitch + col]) {
                std::fprintf(stderr,
                             "2D runtime async mismatch at row %zu col %zu\n",
                             row, col);
                return 1;
            }
        }
    }
    CHECK_CUDA(cudaFreeArray(runtime_array_2d));
    CHECK_CUDA(cudaStreamDestroy(runtime_stream));

    CHECK_DRV(cuInit(0));
    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));
    CUcontext ctx = nullptr;
    CHECK_DRV(cuCtxGetCurrent(&ctx));
    if (ctx == nullptr) {
        CHECK_DRV(cuDevicePrimaryCtxRetain(&ctx, device));
        CHECK_DRV(cuCtxSetCurrent(ctx));
    }

    CUDA_ARRAY_DESCRIPTOR driver_desc = {};
    driver_desc.Width = kBytes;
    driver_desc.Height = 1;
    driver_desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
    driver_desc.NumChannels = 1;
    CUarray driver_array = nullptr;
    CHECK_DRV(cuArrayCreate(&driver_array, &driver_desc));

    CUDA_ARRAY_DESCRIPTOR actual_driver_desc = {};
    CHECK_DRV(cuArrayGetDescriptor(&actual_driver_desc, driver_array));
    if (actual_driver_desc.Width != kBytes || actual_driver_desc.Height != 1 ||
        actual_driver_desc.Format != CU_AD_FORMAT_UNSIGNED_INT8 ||
        actual_driver_desc.NumChannels != 1) {
        std::fprintf(stderr, "unexpected driver array descriptor\n");
        return 1;
    }

    output.fill(0);
    device_roundtrip.fill(0);
    CHECK_DRV(cuMemcpyHtoA(driver_array, 0, input.data(), kBytes));
    CHECK_DRV(cuMemcpyAtoH(output.data(), driver_array, 0, kBytes));
    if (!equal_bytes(input, output)) {
        return 1;
    }

    CUstream driver_stream = nullptr;
    CHECK_DRV(cuStreamCreate(&driver_stream, CU_STREAM_DEFAULT));
    output.fill(0);
    CHECK_DRV(cuMemcpyHtoAAsync(
        driver_array, 0, input.data(), kBytes, driver_stream));
    CHECK_DRV(cuMemcpyAtoHAsync(
        output.data(), driver_array, 0, kBytes, driver_stream));
    CHECK_DRV(cuStreamSynchronize(driver_stream));
    if (!equal_bytes(input, output)) {
        return 1;
    }

    CUdeviceptr driver_device = 0;
    CUdeviceptr driver_device_out = 0;
    CHECK_DRV(cuMemAlloc(&driver_device, kBytes));
    CHECK_DRV(cuMemAlloc(&driver_device_out, kBytes));
    CHECK_DRV(cuMemcpyHtoD(driver_device, input.data(), kBytes));
    CHECK_DRV(cuMemcpyDtoA(driver_array, 0, driver_device, kBytes));
    CHECK_DRV(cuMemcpyAtoD(driver_device_out, driver_array, 0, kBytes));
    CHECK_DRV(cuMemcpyDtoH(device_roundtrip.data(), driver_device_out, kBytes));
    if (!equal_bytes(input, device_roundtrip)) {
        return 1;
    }

    CHECK_DRV(cuMemFree(driver_device));
    CHECK_DRV(cuMemFree(driver_device_out));
    CHECK_DRV(cuStreamDestroy(driver_stream));
    CHECK_DRV(cuArrayDestroy(driver_array));

    std::puts("array API test passed");
    return 0;
}
