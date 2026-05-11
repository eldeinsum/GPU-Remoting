#include <cuda.h>

#include <cstdio>

static const char *result_name(CUresult result)
{
    const char *name = nullptr;
    cuGetErrorName(result, &name);
    return name ? name : "unknown";
}

static int expect_exact(CUresult result, CUresult expected, const char *label)
{
    if (result != expected) {
        std::fprintf(stderr, "%s returned %s (%d), expected %s (%d)\n", label,
                     result_name(result), static_cast<int>(result),
                     result_name(expected), static_cast<int>(expected));
        return 1;
    }
    return 0;
}

static bool is_cig_rejection(CUresult result)
{
    return result == CUDA_ERROR_NOT_SUPPORTED ||
           result == CUDA_ERROR_INVALID_CONTEXT ||
           result == CUDA_ERROR_INVALID_VALUE;
}

static int expect_cig_rejection(CUresult result, const char *label)
{
    if (!is_cig_rejection(result)) {
        std::fprintf(stderr,
                     "%s returned %s (%d), expected a documented CIG rejection\n",
                     label, result_name(result), static_cast<int>(result));
        return 1;
    }
    return 0;
}

int main()
{
    int status = 0;

    status |= expect_exact(cuInit(0), CUDA_SUCCESS, "cuInit");

    CUdevice device = 0;
    status |= expect_exact(cuDeviceGet(&device, 0), CUDA_SUCCESS, "cuDeviceGet");

    CUcontext context = nullptr;
    status |= expect_exact(cuCtxCreate_v4(&context, nullptr, 0, device), CUDA_SUCCESS,
                           "cuCtxCreate");

    CUstream stream = nullptr;
    status |= expect_exact(cuStreamCreate(&stream, CU_STREAM_DEFAULT),
                           CUDA_SUCCESS, "cuStreamCreate");

    status |= expect_exact(cuStreamBeginCaptureToCig(nullptr, nullptr),
                           CUDA_ERROR_INVALID_VALUE,
                           "cuStreamBeginCaptureToCig(null stream)");
    status |= expect_exact(cuStreamBeginCaptureToCig(stream, nullptr),
                           CUDA_ERROR_INVALID_VALUE,
                           "cuStreamBeginCaptureToCig(null params)");

    CUstreamCigCaptureParams empty_capture_params = {};
    status |= expect_exact(cuStreamBeginCaptureToCig(stream, &empty_capture_params),
                           CUDA_ERROR_INVALID_VALUE,
                           "cuStreamBeginCaptureToCig(empty params)");

    CUstreamCigParam stream_cig_param = {};
    stream_cig_param.streamSharedDataType = STREAM_CIG_DATA_TYPE_D3D12_COMMAND_LIST;

    CUstreamCigCaptureParams capture_params = {};
    capture_params.streamCigParams = &stream_cig_param;
    status |= expect_cig_rejection(
        cuStreamBeginCaptureToCig(stream, &capture_params),
        "cuStreamBeginCaptureToCig(non-CIG context)");

    status |= expect_exact(cuStreamEndCaptureToCig(stream),
                           CUDA_ERROR_INVALID_VALUE,
                           "cuStreamEndCaptureToCig(non-CIG stream)");

    CUctxCigParam context_cig_param = {};
    context_cig_param.sharedDataType = CIG_DATA_TYPE_NV_BLOB;

    CUctxCreateParams context_params = {};
    context_params.cigParams = &context_cig_param;
    CUcontext cig_context = nullptr;
    status |= expect_cig_rejection(
        cuCtxCreate_v4(&cig_context, &context_params, 0, device),
        "cuCtxCreate_v4(CIG)");
    if (cig_context != nullptr) {
        cuCtxDestroy(cig_context);
        std::fprintf(stderr, "cuCtxCreate_v4(CIG) unexpectedly created a context\n");
        status = 1;
    }

    status |= expect_exact(cuStreamDestroy(stream), CUDA_SUCCESS,
                           "cuStreamDestroy");
    status |= expect_exact(cuCtxDestroy(context), CUDA_SUCCESS, "cuCtxDestroy");

    return status;
}
