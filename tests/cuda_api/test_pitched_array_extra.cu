#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
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

static int verify_value(const std::vector<unsigned char> &data, size_t pitch,
                        size_t width, size_t height, size_t depth,
                        unsigned char expected)
{
    for (size_t z = 0; z < depth; ++z) {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                const size_t index = z * pitch * height + y * pitch + x;
                if (data[index] != expected) {
                    std::fprintf(stderr,
                                 "mismatch at z=%zu y=%zu x=%zu: got %u "
                                 "expected %u\n",
                                 z, y, x, static_cast<unsigned>(data[index]),
                                 static_cast<unsigned>(expected));
                    return 1;
                }
            }
        }
    }
    return 0;
}

static int verify_u16(const std::vector<unsigned char> &data, size_t pitch,
                      size_t width, size_t height, unsigned short expected)
{
    for (size_t y = 0; y < height; ++y) {
        const unsigned short *row =
            reinterpret_cast<const unsigned short *>(data.data() + y * pitch);
        for (size_t x = 0; x < width; ++x) {
            if (row[x] != expected) {
                std::fprintf(stderr,
                             "u16 mismatch at y=%zu x=%zu: got %u expected %u\n",
                             y, x, static_cast<unsigned>(row[x]),
                             static_cast<unsigned>(expected));
                return 1;
            }
        }
    }
    return 0;
}

static int verify_u32(const std::vector<unsigned char> &data, size_t pitch,
                      size_t width, size_t height, unsigned int expected)
{
    for (size_t y = 0; y < height; ++y) {
        const unsigned int *row =
            reinterpret_cast<const unsigned int *>(data.data() + y * pitch);
        for (size_t x = 0; x < width; ++x) {
            if (row[x] != expected) {
                std::fprintf(stderr,
                             "u32 mismatch at y=%zu x=%zu: got %u expected %u\n",
                             y, x, row[x], expected);
                return 1;
            }
        }
    }
    return 0;
}

int main()
{
    constexpr size_t kWidth = 16;
    constexpr size_t kHeight = 4;
    constexpr size_t kDepth = 3;

    CHECK_CUDA(cudaSetDevice(0));

    cudaPitchedPtr pitched = {};
    CHECK_CUDA(cudaMalloc3D(&pitched, make_cudaExtent(kWidth, kHeight, kDepth)));
    if (pitched.ptr == nullptr || pitched.pitch < kWidth ||
        pitched.xsize != kWidth || pitched.ysize != kHeight) {
        std::fprintf(stderr, "unexpected pitched allocation metadata\n");
        return 1;
    }

    cudaExtent memset_extent = make_cudaExtent(kWidth, kHeight, kDepth);
    CHECK_CUDA(cudaMemset3D(pitched, 0x33, memset_extent));

    std::vector<unsigned char> output(pitched.pitch * kHeight * kDepth, 0);
    for (size_t z = 0; z < kDepth; ++z) {
        const char *slice = static_cast<const char *>(pitched.ptr) +
                            z * pitched.pitch * kHeight;
        CHECK_CUDA(cudaMemcpy2D(output.data() + z * pitched.pitch * kHeight,
                                pitched.pitch, slice, pitched.pitch, kWidth,
                                kHeight, cudaMemcpyDeviceToHost));
    }
    if (verify_value(output, pitched.pitch, kWidth, kHeight, kDepth, 0x33) != 0) {
        return 1;
    }

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaMemset3DAsync(pitched, 0x44, memset_extent, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    output.assign(pitched.pitch * kHeight * kDepth, 0);
    for (size_t z = 0; z < kDepth; ++z) {
        const char *slice = static_cast<const char *>(pitched.ptr) +
                            z * pitched.pitch * kHeight;
        CHECK_CUDA(cudaMemcpy2D(output.data() + z * pitched.pitch * kHeight,
                                pitched.pitch, slice, pitched.pitch, kWidth,
                                kHeight, cudaMemcpyDeviceToHost));
    }
    if (verify_value(output, pitched.pitch, kWidth, kHeight, kDepth, 0x44) != 0) {
        return 1;
    }

    CHECK_CUDA(cudaMemset2D(pitched.ptr, pitched.pitch, 0x55, kWidth, kHeight));
    output.assign(pitched.pitch * kHeight, 0);
    CHECK_CUDA(cudaMemcpy2D(output.data(), pitched.pitch, pitched.ptr,
                            pitched.pitch, kWidth, kHeight,
                            cudaMemcpyDeviceToHost));
    if (verify_value(output, pitched.pitch, kWidth, kHeight, 1, 0x55) != 0) {
        return 1;
    }

    CHECK_CUDA(cudaMemset2DAsync(pitched.ptr, pitched.pitch, 0x66, kWidth,
                                 kHeight, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    output.assign(pitched.pitch * kHeight, 0);
    CHECK_CUDA(cudaMemcpy2D(output.data(), pitched.pitch, pitched.ptr,
                            pitched.pitch, kWidth, kHeight,
                            cudaMemcpyDeviceToHost));
    if (verify_value(output, pitched.pitch, kWidth, kHeight, 1, 0x66) != 0) {
        return 1;
    }

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
    cudaArray_t array3d = nullptr;
    CHECK_CUDA(cudaMalloc3DArray(&array3d, &desc,
                                 make_cudaExtent(kWidth, kHeight, kDepth), 0));
    cudaChannelFormatDesc queried_desc = {};
    CHECK_CUDA(cudaGetChannelDesc(&queried_desc, array3d));
    if (queried_desc.x != desc.x || queried_desc.y != desc.y ||
        queried_desc.z != desc.z || queried_desc.w != desc.w ||
        queried_desc.f != desc.f) {
        std::fprintf(stderr, "channel descriptor mismatch\n");
        return 1;
    }

    CHECK_CUDA(cudaFreeArray(array3d));

    constexpr size_t kArrayBytes = 32;
    std::array<unsigned char, kArrayBytes> array_input = {};
    std::array<unsigned char, kArrayBytes> array_output = {};
    for (size_t i = 0; i < kArrayBytes; ++i) {
        array_input[i] = static_cast<unsigned char>(i * 3 + 1);
    }

    cudaArray_t array_src = nullptr;
    cudaArray_t array_dst = nullptr;
    CHECK_CUDA(cudaMallocArray(&array_src, &desc, kArrayBytes));
    CHECK_CUDA(cudaMallocArray(&array_dst, &desc, kArrayBytes));
    CHECK_CUDA(cudaMemcpyToArray(array_src, 0, 0, array_input.data(),
                                 kArrayBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyArrayToArray(array_dst, 0, 0, array_src, 0, 0,
                                      kArrayBytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpyFromArray(array_output.data(), array_dst, 0, 0,
                                   kArrayBytes, cudaMemcpyDeviceToHost));
    if (array_input != array_output) {
        std::fprintf(stderr, "1D array-to-array copy mismatch\n");
        return 1;
    }
    CHECK_CUDA(cudaFreeArray(array_src));
    CHECK_CUDA(cudaFreeArray(array_dst));

    constexpr size_t kRows = 4;
    std::array<unsigned char, kWidth * kRows> input_2d = {};
    std::array<unsigned char, kWidth * kRows> output_2d = {};
    for (size_t i = 0; i < input_2d.size(); ++i) {
        input_2d[i] = static_cast<unsigned char>(i + 9);
    }

    CHECK_CUDA(cudaMallocArray(&array_src, &desc, kWidth, kRows));
    CHECK_CUDA(cudaMallocArray(&array_dst, &desc, kWidth, kRows));
    CHECK_CUDA(cudaMemcpy2DToArray(array_src, 0, 0, input_2d.data(), kWidth,
                                   kWidth, kRows, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy2DArrayToArray(array_dst, 0, 0, array_src, 0, 0,
                                        kWidth, kRows,
                                        cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy2DFromArray(output_2d.data(), kWidth, array_dst, 0, 0,
                                     kWidth, kRows, cudaMemcpyDeviceToHost));
    if (input_2d != output_2d) {
        std::fprintf(stderr, "2D array-to-array copy mismatch\n");
        return 1;
    }

    CHECK_CUDA(cudaFreeArray(array_src));
    CHECK_CUDA(cudaFreeArray(array_dst));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(pitched.ptr));

    CHECK_DRV(cuInit(0));
    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));
    CUcontext context = nullptr;
    CHECK_DRV(cuCtxGetCurrent(&context));
    if (context == nullptr) {
        CHECK_DRV(cuDevicePrimaryCtxRetain(&context, device));
        CHECK_DRV(cuCtxSetCurrent(context));
    }

    CUDA_ARRAY3D_DESCRIPTOR driver_array_desc = {};
    driver_array_desc.Width = kWidth;
    driver_array_desc.Height = kHeight;
    driver_array_desc.Depth = kDepth;
    driver_array_desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
    driver_array_desc.NumChannels = 1;
    driver_array_desc.Flags = 0;

    CUarray driver_array = nullptr;
    CHECK_DRV(cuArray3DCreate(&driver_array, &driver_array_desc));
    CUDA_ARRAY3D_DESCRIPTOR queried_driver_desc = {};
    CHECK_DRV(cuArray3DGetDescriptor(&queried_driver_desc, driver_array));
    if (queried_driver_desc.Width != driver_array_desc.Width ||
        queried_driver_desc.Height != driver_array_desc.Height ||
        queried_driver_desc.Depth != driver_array_desc.Depth ||
        queried_driver_desc.Format != driver_array_desc.Format ||
        queried_driver_desc.NumChannels != driver_array_desc.NumChannels) {
        std::fprintf(stderr, "driver 3D array descriptor mismatch\n");
        return 1;
    }
    CHECK_DRV(cuArrayDestroy(driver_array));

    CUdeviceptr driver_ptr = 0;
    size_t driver_pitch = 0;
    CHECK_DRV(cuMemAllocPitch(&driver_ptr, &driver_pitch, kWidth, kHeight, 4));
    if (driver_ptr == 0 || driver_pitch < kWidth) {
        std::fprintf(stderr, "unexpected driver pitched allocation metadata\n");
        return 1;
    }

    auto copy_driver_rows = [&](std::vector<unsigned char> *dst,
                                size_t row_bytes) -> int {
        dst->assign(driver_pitch * kHeight, 0);
        for (size_t row = 0; row < kHeight; ++row) {
            CHECK_DRV(cuMemcpyDtoH(dst->data() + row * driver_pitch,
                                   driver_ptr + row * driver_pitch,
                                   row_bytes));
        }
        return 0;
    };

    std::vector<unsigned char> driver_output;
    CHECK_DRV(cuMemsetD2D8(driver_ptr, driver_pitch, 0x77, kWidth, kHeight));
    if (copy_driver_rows(&driver_output, kWidth) != 0 ||
        verify_value(driver_output, driver_pitch, kWidth, kHeight, 1, 0x77) != 0) {
        return 1;
    }

    CUstream driver_stream = nullptr;
    CHECK_DRV(cuStreamCreate(&driver_stream, CU_STREAM_DEFAULT));
    CHECK_DRV(cuMemsetD2D8Async(driver_ptr, driver_pitch, 0x78, kWidth,
                                kHeight, driver_stream));
    CHECK_DRV(cuStreamSynchronize(driver_stream));
    if (copy_driver_rows(&driver_output, kWidth) != 0 ||
        verify_value(driver_output, driver_pitch, kWidth, kHeight, 1, 0x78) != 0) {
        return 1;
    }

    constexpr size_t kWidth16 = kWidth / sizeof(unsigned short);
    CHECK_DRV(cuMemsetD2D16(driver_ptr, driver_pitch, 0x1234, kWidth16,
                            kHeight));
    if (copy_driver_rows(&driver_output, kWidth) != 0 ||
        verify_u16(driver_output, driver_pitch, kWidth16, kHeight, 0x1234) != 0) {
        return 1;
    }
    CHECK_DRV(cuMemsetD2D16Async(driver_ptr, driver_pitch, 0x2345, kWidth16,
                                 kHeight, driver_stream));
    CHECK_DRV(cuStreamSynchronize(driver_stream));
    if (copy_driver_rows(&driver_output, kWidth) != 0 ||
        verify_u16(driver_output, driver_pitch, kWidth16, kHeight, 0x2345) != 0) {
        return 1;
    }

    constexpr size_t kWidth32 = kWidth / sizeof(unsigned int);
    CHECK_DRV(cuMemsetD2D32(driver_ptr, driver_pitch, 0x12345678, kWidth32,
                            kHeight));
    if (copy_driver_rows(&driver_output, kWidth) != 0 ||
        verify_u32(driver_output, driver_pitch, kWidth32, kHeight,
                   0x12345678) != 0) {
        return 1;
    }
    CHECK_DRV(cuMemsetD2D32Async(driver_ptr, driver_pitch, 0x23456789,
                                 kWidth32, kHeight, driver_stream));
    CHECK_DRV(cuStreamSynchronize(driver_stream));
    if (copy_driver_rows(&driver_output, kWidth) != 0 ||
        verify_u32(driver_output, driver_pitch, kWidth32, kHeight,
                   0x23456789) != 0) {
        return 1;
    }

    CHECK_DRV(cuStreamDestroy(driver_stream));
    CHECK_DRV(cuMemFree(driver_ptr));

    std::puts("pitched array API test passed");
    return 0;
}
