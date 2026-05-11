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

static int close_enough(float lhs, float rhs)
{
    float diff = lhs > rhs ? lhs - rhs : rhs - lhs;
    return diff < 0.0001f;
}

static int run_driver_legacy_texture_refs()
{
    constexpr size_t kBytes = 256;
    constexpr size_t kWidth = 16;
    constexpr size_t kHeight = 4;

    CHECK_DRV(cuInit(0));
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaFree(nullptr));

    CUtexref tex = nullptr;
    CHECK_DRV(cuTexRefCreate(&tex));
    if (require(tex != nullptr, "legacy texture reference handle is null") != 0) {
        return 1;
    }

    CHECK_DRV(cuTexRefSetFlags(
        tex, CU_TRSF_READ_AS_INTEGER | CU_TRSF_NORMALIZED_COORDINATES));
    unsigned int flags = 0;
    CHECK_DRV(cuTexRefGetFlags(&flags, tex));
    if ((flags & CU_TRSF_READ_AS_INTEGER) == 0 ||
        (flags & CU_TRSF_NORMALIZED_COORDINATES) == 0) {
        std::fprintf(stderr, "legacy texture flags mismatch: %u\n", flags);
        return 1;
    }

    CHECK_DRV(cuTexRefSetAddressMode(tex, 0, CU_TR_ADDRESS_MODE_WRAP));
    CUaddress_mode address_mode = CU_TR_ADDRESS_MODE_CLAMP;
    CHECK_DRV(cuTexRefGetAddressMode(&address_mode, tex, 0));
    if (address_mode != CU_TR_ADDRESS_MODE_WRAP) {
        std::fprintf(stderr, "legacy texture address mode mismatch\n");
        return 1;
    }

    CHECK_DRV(cuTexRefSetFilterMode(tex, CU_TR_FILTER_MODE_LINEAR));
    CUfilter_mode filter_mode = CU_TR_FILTER_MODE_POINT;
    CHECK_DRV(cuTexRefGetFilterMode(&filter_mode, tex));
    if (filter_mode != CU_TR_FILTER_MODE_LINEAR) {
        std::fprintf(stderr, "legacy texture filter mode mismatch\n");
        return 1;
    }

    CHECK_DRV(cuTexRefSetFormat(tex, CU_AD_FORMAT_UNSIGNED_INT8, 1));
    CUarray_format format = CU_AD_FORMAT_FLOAT;
    int channels = 0;
    CHECK_DRV(cuTexRefGetFormat(&format, &channels, tex));
    if (format != CU_AD_FORMAT_UNSIGNED_INT8 || channels != 1) {
        std::fprintf(stderr, "legacy texture format mismatch\n");
        return 1;
    }

    CHECK_DRV(cuTexRefSetMipmapFilterMode(tex, CU_TR_FILTER_MODE_POINT));
    CUfilter_mode mip_filter_mode = CU_TR_FILTER_MODE_LINEAR;
    CHECK_DRV(cuTexRefGetMipmapFilterMode(&mip_filter_mode, tex));
    if (mip_filter_mode != CU_TR_FILTER_MODE_POINT) {
        std::fprintf(stderr, "legacy texture mip filter mismatch\n");
        return 1;
    }

    CHECK_DRV(cuTexRefSetMipmapLevelBias(tex, 0.5f));
    float mip_bias = 0.0f;
    CHECK_DRV(cuTexRefGetMipmapLevelBias(&mip_bias, tex));
    if (!close_enough(mip_bias, 0.5f)) {
        std::fprintf(stderr, "legacy texture mip bias mismatch: %f\n", mip_bias);
        return 1;
    }

    CHECK_DRV(cuTexRefSetMipmapLevelClamp(tex, 0.0f, 1.0f));
    float min_clamp = -1.0f;
    float max_clamp = -1.0f;
    CHECK_DRV(cuTexRefGetMipmapLevelClamp(&min_clamp, &max_clamp, tex));
    if (!close_enough(min_clamp, 0.0f) || !close_enough(max_clamp, 1.0f)) {
        std::fprintf(stderr, "legacy texture mip clamp mismatch\n");
        return 1;
    }

    CHECK_DRV(cuTexRefSetMaxAnisotropy(tex, 1));
    int max_anisotropy = 0;
    CHECK_DRV(cuTexRefGetMaxAnisotropy(&max_anisotropy, tex));
    if (max_anisotropy != 1) {
        std::fprintf(stderr, "legacy texture anisotropy mismatch: %d\n",
                     max_anisotropy);
        return 1;
    }

    float border_color[4] = {1.0f, 0.5f, 0.25f, 0.0f};
    CHECK_DRV(cuTexRefSetBorderColor(tex, border_color));
    float actual_border_color[4] = {};
    CHECK_DRV(cuTexRefGetBorderColor(actual_border_color, tex));
    for (int i = 0; i < 4; ++i) {
        if (!close_enough(actual_border_color[i], border_color[i])) {
            std::fprintf(stderr, "legacy texture border mismatch at %d\n", i);
            return 1;
        }
    }

    CUdeviceptr linear = 0;
    CHECK_DRV(cuMemAlloc(&linear, kBytes));
    size_t byte_offset = 1;
    CHECK_DRV(cuTexRefSetAddress(&byte_offset, tex, linear, kBytes));
    CUdeviceptr queried_linear = 0;
    CHECK_DRV(cuTexRefGetAddress(&queried_linear, tex));
    if (queried_linear == 0) {
        std::fprintf(stderr, "legacy texture linear address was not bound\n");
        return 1;
    }

    CUdeviceptr pitched = 0;
    size_t pitch = 0;
    CHECK_DRV(cuMemAllocPitch(&pitched, &pitch, kWidth, kHeight, 4));
    CUDA_ARRAY_DESCRIPTOR pitch_desc = {};
    pitch_desc.Width = kWidth;
    pitch_desc.Height = kHeight;
    pitch_desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
    pitch_desc.NumChannels = 1;
    CHECK_DRV(cuTexRefSetAddress2D(tex, &pitch_desc, pitched, pitch));

    CUDA_ARRAY_DESCRIPTOR array_desc = {};
    array_desc.Width = kWidth;
    array_desc.Height = kHeight;
    array_desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
    array_desc.NumChannels = 1;

    CUarray array = nullptr;
    CHECK_DRV(cuArrayCreate(&array, &array_desc));

    CUtexref array_tex = nullptr;
    CHECK_DRV(cuTexRefCreate(&array_tex));
    CHECK_DRV(cuTexRefSetArray(array_tex, array, CU_TRSA_OVERRIDE_FORMAT));
    CUarray queried_array = nullptr;
    CHECK_DRV(cuTexRefGetArray(&queried_array, array_tex));
    if (queried_array != array) {
        std::fprintf(stderr, "legacy texture array binding mismatch\n");
        return 1;
    }

    CUDA_ARRAY3D_DESCRIPTOR mip_desc = {};
    mip_desc.Width = kWidth;
    mip_desc.Height = kHeight;
    mip_desc.Depth = 0;
    mip_desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
    mip_desc.NumChannels = 1;

    CUmipmappedArray mipmap = nullptr;
    CHECK_DRV(cuMipmappedArrayCreate(&mipmap, &mip_desc, 2));

    CUtexref mip_tex = nullptr;
    CHECK_DRV(cuTexRefCreate(&mip_tex));
    CHECK_DRV(cuTexRefSetMipmappedArray(
        mip_tex, mipmap, CU_TRSA_OVERRIDE_FORMAT));
    CUmipmappedArray queried_mipmap = nullptr;
    CHECK_DRV(cuTexRefGetMipmappedArray(&queried_mipmap, mip_tex));
    if (queried_mipmap != mipmap) {
        std::fprintf(stderr, "legacy texture mipmap binding mismatch\n");
        return 1;
    }

    CHECK_DRV(cuTexRefDestroy(mip_tex));
    CHECK_DRV(cuMipmappedArrayDestroy(mipmap));
    CHECK_DRV(cuTexRefDestroy(array_tex));
    CHECK_DRV(cuArrayDestroy(array));
    CHECK_DRV(cuMemFree(pitched));
    CHECK_DRV(cuMemFree(linear));
    CHECK_DRV(cuTexRefDestroy(tex));
    return 0;
}

static const char kReferencePtx[] = R"ptx(
.version 7.8
.target sm_52
.address_size 64

.global .texref module_tex;
.global .surfref module_surf;

.visible .entry module_ref_kernel()
{
    ret;
}
)ptx";

static int run_driver_module_refs()
{
    constexpr size_t kWidth = 16;
    constexpr size_t kHeight = 4;

    CHECK_DRV(cuInit(0));
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaFree(nullptr));

    CUmodule module = nullptr;
    CHECK_DRV(cuModuleLoadData(&module, kReferencePtx));

    CUfunction function = nullptr;
    CHECK_DRV(cuModuleGetFunction(&function, module, "module_ref_kernel"));

    CUtexref tex = nullptr;
    CHECK_DRV(cuModuleGetTexRef(&tex, module, "module_tex"));
    if (require(tex != nullptr, "module texture reference handle is null") !=
        0) {
        return 1;
    }

    CHECK_DRV(cuTexRefSetFormat(tex, CU_AD_FORMAT_UNSIGNED_INT8, 1));
    CUDA_ARRAY_DESCRIPTOR tex_desc = {};
    tex_desc.Width = kWidth;
    tex_desc.Height = kHeight;
    tex_desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
    tex_desc.NumChannels = 1;

    CUarray tex_array = nullptr;
    CHECK_DRV(cuArrayCreate(&tex_array, &tex_desc));
    CHECK_DRV(cuTexRefSetArray(tex, tex_array, CU_TRSA_OVERRIDE_FORMAT));
    CUarray queried_tex_array = nullptr;
    CHECK_DRV(cuTexRefGetArray(&queried_tex_array, tex));
    if (queried_tex_array != tex_array) {
        std::fprintf(stderr, "module texture array binding mismatch\n");
        return 1;
    }

    CHECK_DRV(cuParamSetTexRef(function, CU_PARAM_TR_DEFAULT, tex));

    CUsurfref surf = nullptr;
    CHECK_DRV(cuModuleGetSurfRef(&surf, module, "module_surf"));
    if (require(surf != nullptr, "module surface reference handle is null") !=
        0) {
        return 1;
    }

    CUDA_ARRAY3D_DESCRIPTOR surf_desc = {};
    surf_desc.Width = kWidth;
    surf_desc.Height = kHeight;
    surf_desc.Depth = 0;
    surf_desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
    surf_desc.NumChannels = 1;
    surf_desc.Flags = CUDA_ARRAY3D_SURFACE_LDST;

    CUarray surf_array = nullptr;
    CHECK_DRV(cuArray3DCreate(&surf_array, &surf_desc));
    CHECK_DRV(cuSurfRefSetArray(surf, surf_array, 0));
    CUarray queried_surf_array = nullptr;
    CHECK_DRV(cuSurfRefGetArray(&queried_surf_array, surf));
    if (queried_surf_array != surf_array) {
        std::fprintf(stderr, "module surface array binding mismatch\n");
        return 1;
    }

    CHECK_DRV(cuArrayDestroy(surf_array));
    CHECK_DRV(cuArrayDestroy(tex_array));
    CHECK_DRV(cuModuleUnload(module));
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
    if (run_driver_legacy_texture_refs() != 0) {
        return 1;
    }
    if (run_driver_module_refs() != 0) {
        return 1;
    }

    std::puts("texture/surface object API test passed");
    return 0;
}
