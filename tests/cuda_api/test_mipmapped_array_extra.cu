#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
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

template <size_t N>
static int verify_equal(const std::array<unsigned char, N> &actual,
                        const std::array<unsigned char, N> &expected)
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

template <typename Requirements>
static int check_memory_requirements(const Requirements &requirements)
{
    if (requirements.size == 0 || requirements.alignment == 0) {
        std::fprintf(stderr,
                     "unexpected memory requirements: size=%zu alignment=%zu\n",
                     requirements.size, requirements.alignment);
        return 1;
    }
    return 0;
}

template <typename Properties>
static int check_sparse_properties(const Properties &properties)
{
    if (properties.tileExtent.width == 0 || properties.tileExtent.height == 0) {
        std::fprintf(stderr, "unexpected sparse tile extent: %u x %u x %u\n",
                     properties.tileExtent.width, properties.tileExtent.height,
                     properties.tileExtent.depth);
        return 1;
    }
    return 0;
}

int main()
{
    CHECK_CUDA(cudaSetDevice(0));

    constexpr size_t kWidth = 16;
    constexpr size_t kHeight = 8;
    constexpr size_t kBytes = kWidth * kHeight;

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
    cudaMipmappedArray_t mipmap = nullptr;
    CHECK_CUDA(cudaMallocMipmappedArray(
        &mipmap, &desc, make_cudaExtent(kWidth, kHeight, 0), 3));

    cudaArray_t level0 = nullptr;
    cudaArray_t level1 = nullptr;
    CHECK_CUDA(cudaGetMipmappedArrayLevel(&level0, mipmap, 0));
    CHECK_CUDA(cudaGetMipmappedArrayLevel(&level1, mipmap, 1));

    cudaChannelFormatDesc level_desc = {};
    cudaExtent level_extent = {};
    unsigned int level_flags = 0;
    CHECK_CUDA(cudaArrayGetInfo(&level_desc, &level_extent, &level_flags,
                                level0));
    if (level_desc.x != desc.x || level_extent.width != kWidth ||
        level_extent.height != kHeight) {
        std::fprintf(stderr, "unexpected mipmap level 0 metadata\n");
        return 1;
    }

    CHECK_CUDA(cudaArrayGetInfo(&level_desc, &level_extent, &level_flags,
                                level1));
    if (level_extent.width != kWidth / 2 ||
        level_extent.height != kHeight / 2) {
        std::fprintf(stderr, "unexpected mipmap level 1 extent\n");
        return 1;
    }

    std::array<unsigned char, kBytes> input = {};
    std::array<unsigned char, kBytes> output = {};
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<unsigned char>((i * 13 + 7) & 0xff);
    }
    CHECK_CUDA(cudaMemcpy2DToArray(level0, 0, 0, input.data(), kWidth, kWidth,
                                   kHeight, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy2DFromArray(output.data(), kWidth, level0, 0, 0,
                                     kWidth, kHeight,
                                     cudaMemcpyDeviceToHost));
    if (verify_equal(output, input) != 0) {
        return 1;
    }

    CHECK_CUDA(cudaFreeMipmappedArray(mipmap));

    CHECK_DRV(cuInit(0));
    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));
    CUcontext context = nullptr;
    CHECK_DRV(cuDevicePrimaryCtxRetain(&context, device));
    CHECK_DRV(cuCtxSetCurrent(context));

    CUDA_ARRAY3D_DESCRIPTOR driver_desc = {};
    driver_desc.Width = kWidth;
    driver_desc.Height = kHeight;
    driver_desc.Depth = 0;
    driver_desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
    driver_desc.NumChannels = 1;

    CUmipmappedArray driver_mipmap = nullptr;
    CHECK_DRV(cuMipmappedArrayCreate(&driver_mipmap, &driver_desc, 3));
    CUarray driver_level0 = nullptr;
    CUarray driver_level1 = nullptr;
    CHECK_DRV(cuMipmappedArrayGetLevel(&driver_level0, driver_mipmap, 0));
    CHECK_DRV(cuMipmappedArrayGetLevel(&driver_level1, driver_mipmap, 1));

    CUDA_ARRAY_DESCRIPTOR driver_level_desc = {};
    CHECK_DRV(cuArrayGetDescriptor(&driver_level_desc, driver_level0));
    if (driver_level_desc.Width != kWidth ||
        driver_level_desc.Height != kHeight ||
        driver_level_desc.Format != CU_AD_FORMAT_UNSIGNED_INT8) {
        std::fprintf(stderr, "unexpected driver mipmap level 0 metadata\n");
        return 1;
    }
    CHECK_DRV(cuArrayGetDescriptor(&driver_level_desc, driver_level1));
    if (driver_level_desc.Width != kWidth / 2 ||
        driver_level_desc.Height != kHeight / 2) {
        std::fprintf(stderr, "unexpected driver mipmap level 1 metadata\n");
        return 1;
    }
    CHECK_DRV(cuMipmappedArrayDestroy(driver_mipmap));

    int deferred_mapping_supported = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(
        &deferred_mapping_supported,
        cudaDevAttrDeferredMappingCudaArraySupported, 0));
    if (deferred_mapping_supported) {
        cudaArray_t deferred_array = nullptr;
        CHECK_CUDA(cudaMallocArray(&deferred_array, &desc, 64, 64,
                                   cudaArrayDeferredMapping));
        cudaArrayMemoryRequirements array_requirements = {};
        CHECK_CUDA(cudaArrayGetMemoryRequirements(&array_requirements,
                                                  deferred_array, 0));
        if (check_memory_requirements(array_requirements) != 0) {
            return 1;
        }
        CHECK_CUDA(cudaFreeArray(deferred_array));

        cudaMipmappedArray_t deferred_mipmap = nullptr;
        CHECK_CUDA(cudaMallocMipmappedArray(
            &deferred_mipmap, &desc, make_cudaExtent(64, 64, 0), 4,
            cudaArrayDeferredMapping));
        cudaArrayMemoryRequirements mipmap_requirements = {};
        CHECK_CUDA(cudaMipmappedArrayGetMemoryRequirements(
            &mipmap_requirements, deferred_mipmap, 0));
        if (check_memory_requirements(mipmap_requirements) != 0) {
            return 1;
        }
        CHECK_CUDA(cudaFreeMipmappedArray(deferred_mipmap));

        CUDA_ARRAY3D_DESCRIPTOR deferred_driver_desc = driver_desc;
        deferred_driver_desc.Width = 64;
        deferred_driver_desc.Height = 64;
        deferred_driver_desc.Flags = CUDA_ARRAY3D_DEFERRED_MAPPING;
        CUarray deferred_driver_array = nullptr;
        CHECK_DRV(cuArray3DCreate(&deferred_driver_array,
                                  &deferred_driver_desc));
        CUDA_ARRAY_MEMORY_REQUIREMENTS driver_array_requirements = {};
        CHECK_DRV(cuArrayGetMemoryRequirements(&driver_array_requirements,
                                               deferred_driver_array, device));
        if (check_memory_requirements(driver_array_requirements) != 0) {
            return 1;
        }
        CHECK_DRV(cuArrayDestroy(deferred_driver_array));

        CUmipmappedArray deferred_driver_mipmap = nullptr;
        CHECK_DRV(cuMipmappedArrayCreate(&deferred_driver_mipmap,
                                         &deferred_driver_desc, 4));
        CUDA_ARRAY_MEMORY_REQUIREMENTS driver_mipmap_requirements = {};
        CHECK_DRV(cuMipmappedArrayGetMemoryRequirements(
            &driver_mipmap_requirements, deferred_driver_mipmap, device));
        if (check_memory_requirements(driver_mipmap_requirements) != 0) {
            return 1;
        }
        CHECK_DRV(cuMipmappedArrayDestroy(deferred_driver_mipmap));
    }

    int sparse_supported = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&sparse_supported,
                                      cudaDevAttrSparseCudaArraySupported, 0));
    if (sparse_supported) {
        cudaArray_t sparse_array = nullptr;
        CHECK_CUDA(
            cudaMallocArray(&sparse_array, &desc, 64, 64, cudaArraySparse));
        cudaArraySparseProperties array_properties = {};
        CHECK_CUDA(
            cudaArrayGetSparseProperties(&array_properties, sparse_array));
        if (check_sparse_properties(array_properties) != 0) {
            return 1;
        }
        CHECK_CUDA(cudaFreeArray(sparse_array));

        cudaMipmappedArray_t sparse_mipmap = nullptr;
        CHECK_CUDA(cudaMallocMipmappedArray(
            &sparse_mipmap, &desc, make_cudaExtent(64, 64, 0), 4,
            cudaArraySparse));
        cudaArraySparseProperties mipmap_properties = {};
        CHECK_CUDA(cudaMipmappedArrayGetSparseProperties(&mipmap_properties,
                                                        sparse_mipmap));
        if (check_sparse_properties(mipmap_properties) != 0) {
            return 1;
        }
        CHECK_CUDA(cudaFreeMipmappedArray(sparse_mipmap));

        CUDA_ARRAY3D_DESCRIPTOR sparse_driver_desc = driver_desc;
        sparse_driver_desc.Width = 64;
        sparse_driver_desc.Height = 64;
        sparse_driver_desc.Flags = CUDA_ARRAY3D_SPARSE;
        CUarray sparse_driver_array = nullptr;
        CHECK_DRV(cuArray3DCreate(&sparse_driver_array, &sparse_driver_desc));
        CUDA_ARRAY_SPARSE_PROPERTIES driver_array_properties = {};
        CHECK_DRV(cuArrayGetSparseProperties(&driver_array_properties,
                                             sparse_driver_array));
        if (check_sparse_properties(driver_array_properties) != 0) {
            return 1;
        }
        CHECK_DRV(cuArrayDestroy(sparse_driver_array));

        CUmipmappedArray sparse_driver_mipmap = nullptr;
        CHECK_DRV(cuMipmappedArrayCreate(&sparse_driver_mipmap,
                                         &sparse_driver_desc, 4));
        CUDA_ARRAY_SPARSE_PROPERTIES driver_mipmap_properties = {};
        CHECK_DRV(cuMipmappedArrayGetSparseProperties(
            &driver_mipmap_properties, sparse_driver_mipmap));
        if (check_sparse_properties(driver_mipmap_properties) != 0) {
            return 1;
        }
        CHECK_DRV(cuMipmappedArrayDestroy(sparse_driver_mipmap));
    }

    CHECK_DRV(cuDevicePrimaryCtxRelease(device));

    std::puts("mipmapped array API test passed");
    return 0;
}
