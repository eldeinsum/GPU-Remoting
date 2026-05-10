#include <cuda.h>
#include <cuda_runtime.h>

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

static int check_drv_desc(const char *name,
                          const CUDA_ARRAY3D_DESCRIPTOR &desc,
                          size_t width, size_t height, size_t depth,
                          CUarray_format format, unsigned int channels)
{
    if (desc.Width != width || desc.Height != height || desc.Depth != depth ||
        desc.Format != format || desc.NumChannels != channels) {
        std::fprintf(stderr,
                     "%s driver descriptor mismatch: got "
                     "(%zu,%zu,%zu,%d,%u), expected (%zu,%zu,%zu,%d,%u)\n",
                     name, desc.Width, desc.Height, desc.Depth,
                     static_cast<int>(desc.Format), desc.NumChannels, width,
                     height, depth, static_cast<int>(format), channels);
        return 1;
    }
    return 0;
}

static int check_desc(const char *name, const cudaChannelFormatDesc &desc,
                      int x, int y, int z, int w, cudaChannelFormatKind kind)
{
    if (desc.x != x || desc.y != y || desc.z != z || desc.w != w ||
        desc.f != kind) {
        std::fprintf(stderr,
                     "%s descriptor mismatch: got (%d,%d,%d,%d,%d), "
                     "expected (%d,%d,%d,%d,%d)\n",
                     name, desc.x, desc.y, desc.z, desc.w,
                     static_cast<int>(desc.f), x, y, z, w,
                     static_cast<int>(kind));
        return 1;
    }
    return 0;
}

static int check_extent(const char *name, const cudaExtent &extent,
                        size_t width, size_t height, size_t depth)
{
    if (extent.width != width || extent.height != height ||
        extent.depth != depth) {
        std::fprintf(stderr,
                     "%s extent mismatch: got (%zu,%zu,%zu), "
                     "expected (%zu,%zu,%zu)\n",
                     name, extent.width, extent.height, extent.depth,
                     width, height, depth);
        return 1;
    }
    return 0;
}

int main()
{
    constexpr size_t width = 16;
    constexpr size_t height = 8;

    CHECK_DRV(cuInit(0));
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaFree(nullptr));

    CUDA_ARRAY3D_DESCRIPTOR desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.Depth = 0;
    desc.Format = CU_AD_FORMAT_NV12;
    desc.NumChannels = 3;
    desc.Flags = CUDA_ARRAY3D_VIDEO_ENCODE_DECODE;

    CUarray array = nullptr;
    CHECK_DRV(cuArray3DCreate(&array, &desc));

    CUarray driver_plane0 = nullptr;
    CUarray driver_plane1 = nullptr;
    CHECK_DRV(cuArrayGetPlane(&driver_plane0, array, 0));
    CHECK_DRV(cuArrayGetPlane(&driver_plane1, array, 1));

    CUDA_ARRAY3D_DESCRIPTOR driver_desc = {};
    CHECK_DRV(cuArray3DGetDescriptor(&driver_desc, driver_plane0));
    if (check_drv_desc("driver plane0", driver_desc, width, height, 0,
                       CU_AD_FORMAT_UNSIGNED_INT8, 1) != 0) {
        return 1;
    }

    driver_desc = {};
    CHECK_DRV(cuArray3DGetDescriptor(&driver_desc, driver_plane1));
    if (check_drv_desc("driver plane1", driver_desc, width / 2, height / 2, 0,
                       CU_AD_FORMAT_UNSIGNED_INT8, 2) != 0) {
        return 1;
    }

    cudaArray_t plane0 = nullptr;
    cudaArray_t plane1 = nullptr;
    CHECK_CUDA(
        cudaArrayGetPlane(&plane0, reinterpret_cast<cudaArray_t>(array), 0));
    CHECK_CUDA(
        cudaArrayGetPlane(&plane1, reinterpret_cast<cudaArray_t>(array), 1));

    cudaChannelFormatDesc plane_desc = {};
    cudaExtent extent = {};
    unsigned int flags = 0;

    CHECK_CUDA(cudaArrayGetInfo(&plane_desc, &extent, &flags, plane0));
    if (check_desc("plane0", plane_desc, 8, 0, 0, 0,
                   cudaChannelFormatKindUnsigned) != 0) {
        return 1;
    }
    if (check_extent("plane0", extent, width, height, 0) != 0) {
        return 1;
    }

    plane_desc = {};
    extent = {};
    flags = 0;
    CHECK_CUDA(cudaArrayGetInfo(&plane_desc, &extent, &flags, plane1));
    if (check_desc("plane1", plane_desc, 8, 8, 0, 0,
                   cudaChannelFormatKindUnsigned) != 0) {
        return 1;
    }
    if (check_extent("plane1", extent, width / 2, height / 2, 0) != 0) {
        return 1;
    }

    CHECK_DRV(cuArrayDestroy(array));
    std::puts("array plane API test passed");
    return 0;
}
