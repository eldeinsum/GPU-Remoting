#include <cuda.h>

#include <cstdio>

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

static int expect_result(CUresult actual, CUresult expected, const char *call)
{
    if (actual == expected) {
        return 0;
    }

    const char *actual_name = nullptr;
    const char *expected_name = nullptr;
    cuGetErrorName(actual, &actual_name);
    cuGetErrorName(expected, &expected_name);
    std::fprintf(stderr, "%s returned %s (%d), expected %s (%d)\n", call,
                 actual_name == nullptr ? "unknown" : actual_name,
                 static_cast<int>(actual),
                 expected_name == nullptr ? "unknown" : expected_name,
                 static_cast<int>(expected));
    return 1;
}

int main()
{
    CHECK_DRV(cuInit(0));

    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));

    int major = 0;
    int minor = 0;
    CHECK_DRV(cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CHECK_DRV(cuDeviceGetAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

    CUcontext context = nullptr;
    CHECK_DRV(cuCtxCreate(&context, nullptr, 0, device));

    CUdeviceptr allocation = 0;
    CHECK_DRV(cuMemAlloc(&allocation, 1 << 20));

    alignas(128) CUtensorMap tensor_map = {};
    cuuint64_t global_dim[3] = {64, 64, 4};
    cuuint64_t global_strides[2] = {64, 64 * 64};
    cuuint32_t box_dim[3] = {16, 16, 1};
    cuuint32_t element_strides[3] = {1, 1, 1};
    int lower_corner[1] = {0};
    int upper_corner[1] = {1};

    CUresult tiled_result = cuTensorMapEncodeTiled(
        &tensor_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
        reinterpret_cast<void *>(allocation), global_dim, global_strides,
        box_dim, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    CUresult im2col_result = cuTensorMapEncodeIm2col(
        &tensor_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
        reinterpret_cast<void *>(allocation), global_dim, global_strides,
        lower_corner, upper_corner, 1, 16, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    CUresult wide_result = cuTensorMapEncodeIm2colWide(
        &tensor_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
        reinterpret_cast<void *>(allocation), global_dim, global_strides, 0, 1,
        1, 16, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_IM2COL_WIDE_MODE_W, CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    CUresult replace_result = cuTensorMapReplaceAddress(
        &tensor_map, reinterpret_cast<void *>(allocation));

    int status = 0;
    if (major < 9) {
        status |= expect_result(tiled_result, CUDA_ERROR_NOT_SUPPORTED,
                                "cuTensorMapEncodeTiled");
        status |= expect_result(im2col_result, CUDA_ERROR_NOT_SUPPORTED,
                                "cuTensorMapEncodeIm2col");
        status |= expect_result(wide_result, CUDA_ERROR_NOT_SUPPORTED,
                                "cuTensorMapEncodeIm2colWide");
        status |= expect_result(replace_result, CUDA_ERROR_NOT_SUPPORTED,
                                "cuTensorMapReplaceAddress");
        if (status == 0) {
            std::printf("tensor map API unsupported on compute capability %d.%d\n",
                        major, minor);
        }
    } else {
        status |= expect_result(tiled_result, CUDA_SUCCESS,
                                "cuTensorMapEncodeTiled");
        status |= expect_result(im2col_result, CUDA_SUCCESS,
                                "cuTensorMapEncodeIm2col");
        status |= expect_result(replace_result, CUDA_SUCCESS,
                                "cuTensorMapReplaceAddress");
        if (major < 10) {
            status |= expect_result(wide_result, CUDA_ERROR_NOT_SUPPORTED,
                                    "cuTensorMapEncodeIm2colWide");
        } else {
            status |= expect_result(wide_result, CUDA_SUCCESS,
                                    "cuTensorMapEncodeIm2colWide");
        }
        if (status == 0) {
            std::puts("tensor map API test passed");
        }
    }

    CHECK_DRV(cuMemFree(allocation));
    CHECK_DRV(cuCtxDestroy(context));
    return status;
}
