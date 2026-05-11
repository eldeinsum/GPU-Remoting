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

static int require(bool condition, const char *message)
{
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        return 1;
    }
    return 0;
}

static int run_driver_objects()
{
    constexpr size_t kWidth = 16;
    constexpr size_t kHeight = 8;

    CHECK_DRV(cuInit(0));
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaFree(nullptr));

    CUDA_ARRAY3D_DESCRIPTOR array_desc = {};
    array_desc.Width = kWidth;
    array_desc.Height = kHeight;
    array_desc.Depth = 0;
    array_desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
    array_desc.NumChannels = 1;
    array_desc.Flags = CUDA_ARRAY3D_SURFACE_LDST;

    CUarray array = nullptr;
    CHECK_DRV(cuArray3DCreate(&array, &array_desc));

    CUDA_RESOURCE_DESC resource_desc = {};
    resource_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    resource_desc.res.array.hArray = array;

    CUDA_TEXTURE_DESC texture_desc = {};
    texture_desc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
    texture_desc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
    texture_desc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
    texture_desc.filterMode = CU_TR_FILTER_MODE_POINT;
    texture_desc.flags = CU_TRSF_READ_AS_INTEGER;

    CUDA_RESOURCE_VIEW_DESC view_desc = {};
    view_desc.format = CU_RES_VIEW_FORMAT_UINT_1X8;
    view_desc.width = kWidth;
    view_desc.height = kHeight;
    view_desc.depth = 0;
    view_desc.firstMipmapLevel = 0;
    view_desc.lastMipmapLevel = 0;
    view_desc.firstLayer = 0;
    view_desc.lastLayer = 0;

    CUtexObject texture = 0;
    CHECK_DRV(cuTexObjectCreate(
        &texture, &resource_desc, &texture_desc, &view_desc));
    if (require(texture != 0, "driver texture object handle is zero") != 0) {
        return 1;
    }

    CUDA_RESOURCE_DESC actual_resource_desc = {};
    CHECK_DRV(cuTexObjectGetResourceDesc(&actual_resource_desc, texture));
    if (require(actual_resource_desc.resType == CU_RESOURCE_TYPE_ARRAY,
                "driver texture resource type mismatch") != 0 ||
        require(actual_resource_desc.res.array.hArray == array,
                "driver texture resource array mismatch") != 0) {
        return 1;
    }

    CUDA_TEXTURE_DESC actual_texture_desc = {};
    CHECK_DRV(cuTexObjectGetTextureDesc(&actual_texture_desc, texture));
    if (require(actual_texture_desc.addressMode[0] == CU_TR_ADDRESS_MODE_CLAMP,
                "driver texture address mode mismatch") != 0 ||
        require(actual_texture_desc.filterMode == CU_TR_FILTER_MODE_POINT,
                "driver texture filter mode mismatch") != 0 ||
        require(actual_texture_desc.flags == CU_TRSF_READ_AS_INTEGER,
                "driver texture flags mismatch") != 0) {
        return 1;
    }

    CUDA_RESOURCE_VIEW_DESC actual_view_desc = {};
    CHECK_DRV(cuTexObjectGetResourceViewDesc(&actual_view_desc, texture));
    if (require(actual_view_desc.format == CU_RES_VIEW_FORMAT_UINT_1X8,
                "driver texture resource view format mismatch") != 0 ||
        require(actual_view_desc.width == kWidth,
                "driver texture resource view width mismatch") != 0 ||
        require(actual_view_desc.height == kHeight,
                "driver texture resource view height mismatch") != 0 ||
        require(actual_view_desc.depth == 0,
                "driver texture resource view depth mismatch") != 0) {
        return 1;
    }

    CHECK_DRV(cuTexObjectDestroy(texture));

    CUsurfObject surface = 0;
    CHECK_DRV(cuSurfObjectCreate(&surface, &resource_desc));
    if (require(surface != 0, "driver surface object handle is zero") != 0) {
        return 1;
    }

    CUDA_RESOURCE_DESC surface_resource_desc = {};
    CHECK_DRV(cuSurfObjectGetResourceDesc(&surface_resource_desc, surface));
    if (require(surface_resource_desc.resType == CU_RESOURCE_TYPE_ARRAY,
                "driver surface resource type mismatch") != 0 ||
        require(surface_resource_desc.res.array.hArray == array,
                "driver surface resource array mismatch") != 0) {
        return 1;
    }

    CHECK_DRV(cuSurfObjectDestroy(surface));
    CHECK_DRV(cuArrayDestroy(array));
    return 0;
}

int main()
{
    constexpr size_t kWidth = 16;
    constexpr size_t kHeight = 8;

    cudaChannelFormatDesc channel_desc =
        cudaCreateChannelDesc<unsigned char>();
    cudaArray_t array = nullptr;
    CHECK_CUDA(cudaMallocArray(
        &array, &channel_desc, kWidth, kHeight, cudaArraySurfaceLoadStore));

    cudaResourceDesc resource_desc = {};
    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = array;

    cudaTextureDesc texture_desc = {};
    texture_desc.addressMode[0] = cudaAddressModeClamp;
    texture_desc.addressMode[1] = cudaAddressModeClamp;
    texture_desc.addressMode[2] = cudaAddressModeClamp;
    texture_desc.filterMode = cudaFilterModePoint;
    texture_desc.readMode = cudaReadModeElementType;
    texture_desc.normalizedCoords = 0;

    cudaResourceViewDesc view_desc = {};
    view_desc.format = cudaResViewFormatUnsignedChar1;
    view_desc.width = kWidth;
    view_desc.height = kHeight;
    view_desc.depth = 0;
    view_desc.firstMipmapLevel = 0;
    view_desc.lastMipmapLevel = 0;
    view_desc.firstLayer = 0;
    view_desc.lastLayer = 0;

    cudaTextureObject_t texture = 0;
    CHECK_CUDA(cudaCreateTextureObject(
        &texture, &resource_desc, &texture_desc, &view_desc));
    if (require(texture != 0, "texture object handle is zero") != 0) {
        return 1;
    }

    cudaResourceDesc actual_resource_desc = {};
    CHECK_CUDA(cudaGetTextureObjectResourceDesc(&actual_resource_desc, texture));
    if (require(actual_resource_desc.resType == cudaResourceTypeArray,
                "texture resource type mismatch") != 0 ||
        require(actual_resource_desc.res.array.array == array,
                "texture resource array mismatch") != 0) {
        return 1;
    }

    cudaTextureDesc actual_texture_desc = {};
    CHECK_CUDA(cudaGetTextureObjectTextureDesc(&actual_texture_desc, texture));
    if (require(actual_texture_desc.addressMode[0] == cudaAddressModeClamp,
                "texture address mode mismatch") != 0 ||
        require(actual_texture_desc.filterMode == cudaFilterModePoint,
                "texture filter mode mismatch") != 0 ||
        require(actual_texture_desc.readMode == cudaReadModeElementType,
                "texture read mode mismatch") != 0) {
        return 1;
    }

    cudaResourceViewDesc actual_view_desc = {};
    CHECK_CUDA(cudaGetTextureObjectResourceViewDesc(&actual_view_desc, texture));
    if (require(actual_view_desc.format == cudaResViewFormatUnsignedChar1,
                "texture resource view format mismatch") != 0 ||
        require(actual_view_desc.width == kWidth,
                "texture resource view width mismatch") != 0 ||
        require(actual_view_desc.height == kHeight,
                "texture resource view height mismatch") != 0 ||
        require(actual_view_desc.depth == 0,
                "texture resource view depth mismatch") != 0) {
        return 1;
    }

    CHECK_CUDA(cudaDestroyTextureObject(texture));

    cudaSurfaceObject_t surface = 0;
    CHECK_CUDA(cudaCreateSurfaceObject(&surface, &resource_desc));
    if (require(surface != 0, "surface object handle is zero") != 0) {
        return 1;
    }

    cudaResourceDesc surface_resource_desc = {};
    CHECK_CUDA(cudaGetSurfaceObjectResourceDesc(
        &surface_resource_desc, surface));
    if (require(surface_resource_desc.resType == cudaResourceTypeArray,
                "surface resource type mismatch") != 0 ||
        require(surface_resource_desc.res.array.array == array,
                "surface resource array mismatch") != 0) {
        return 1;
    }

    CHECK_CUDA(cudaDestroySurfaceObject(surface));
    CHECK_CUDA(cudaFreeArray(array));

    if (run_driver_objects() != 0) {
        return 1;
    }

    std::puts("texture/surface object API test passed");
    return 0;
}
