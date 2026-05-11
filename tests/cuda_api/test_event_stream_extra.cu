#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

static bool running_remoted()
{
    const char *preload = std::getenv("LD_PRELOAD");
    return preload != nullptr && std::strstr(preload, "libclient.so") != nullptr;
}

struct RuntimeCallbackState {
    int called;
    cudaError_t status;
    cudaStream_t stream;
};

static void CUDART_CB runtime_stream_callback(cudaStream_t stream,
                                              cudaError_t status,
                                              void *user_data)
{
    RuntimeCallbackState *state =
        static_cast<RuntimeCallbackState *>(user_data);
    state->called += 1;
    state->status = status;
    state->stream = stream;
}

struct DriverCallbackState {
    int called;
    CUresult status;
    CUstream stream;
};

static void CUDA_CB driver_stream_callback(CUstream stream, CUresult status,
                                           void *user_data)
{
    DriverCallbackState *state = static_cast<DriverCallbackState *>(user_data);
    state->called += 1;
    state->status = status;
    state->stream = stream;
}

static unsigned int aligned_sm_count(unsigned int sm_count,
                                     unsigned int min_sm_count,
                                     unsigned int alignment)
{
    if (min_sm_count == 0) {
        min_sm_count = 1;
    }
    if (alignment == 0) {
        alignment = min_sm_count;
    }
    unsigned int group_sm_count = min_sm_count;
    if (group_sm_count < alignment) {
        group_sm_count = alignment;
    }
    unsigned int remainder = group_sm_count % alignment;
    if (remainder != 0) {
        group_sm_count += alignment - remainder;
    }
    return group_sm_count <= sm_count ? group_sm_count : 0;
}

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t result = (call);                                           \
        if (result != cudaSuccess && result != cudaErrorNotReady) {            \
            std::fprintf(stderr, "%s failed: %s (%d)\n", #call,               \
                         cudaGetErrorString(result), static_cast<int>(result)); \
            return 1;                                                          \
        }                                                                      \
    } while (0)

#define CHECK_CUDA_SUCCESS(call)                                               \
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
        if (result != CUDA_SUCCESS && result != CUDA_ERROR_NOT_READY) {        \
            const char *name = nullptr;                                        \
            cuGetErrorName(result, &name);                                     \
            std::fprintf(stderr, "%s failed: %s (%d)\n", #call,               \
                         name == nullptr ? "unknown" : name,                  \
                         static_cast<int>(result));                            \
            return 1;                                                          \
        }                                                                      \
    } while (0)

#define CHECK_DRV_SUCCESS(call)                                                \
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

int main()
{
    CHECK_CUDA_SUCCESS(cudaSetDevice(0));

    cudaStream_t runtime_a = nullptr;
    cudaStream_t runtime_b = nullptr;
    cudaEvent_t runtime_start = nullptr;
    cudaEvent_t runtime_end = nullptr;

    CHECK_CUDA_SUCCESS(cudaStreamCreateWithPriority(
        &runtime_a, cudaStreamNonBlocking, 0));
    CHECK_CUDA_SUCCESS(cudaStreamCreate(&runtime_b));
    CHECK_CUDA_SUCCESS(cudaStreamCopyAttributes(runtime_b, runtime_a));

    unsigned long long runtime_stream_id = 0;
    int runtime_device = -1;
    CHECK_CUDA_SUCCESS(cudaStreamGetId(runtime_a, &runtime_stream_id));
    CHECK_CUDA_SUCCESS(cudaStreamGetDevice(runtime_a, &runtime_device));
    if (runtime_device != 0) {
        return 1;
    }
    cudaDevResource runtime_device_resource = {};
    CHECK_CUDA_SUCCESS(cudaDeviceGetDevResource(
        0, &runtime_device_resource, cudaDevResourceTypeSm));
    if (runtime_device_resource.type != cudaDevResourceTypeSm ||
        runtime_device_resource.sm.smCount == 0) {
        std::fprintf(stderr, "unexpected runtime device SM resource\n");
        return 1;
    }
    unsigned int runtime_min_sm_count =
        runtime_device_resource.sm.minSmPartitionSize == 0
            ? 1
            : runtime_device_resource.sm.minSmPartitionSize;
    unsigned int runtime_query_groups = 0;
    CHECK_CUDA_SUCCESS(cudaDevSmResourceSplitByCount(
        nullptr, &runtime_query_groups, &runtime_device_resource, nullptr, 0,
        runtime_min_sm_count));
    if (runtime_query_groups == 0) {
        std::fprintf(stderr, "runtime SM split query returned no groups\n");
        return 1;
    }
    cudaDevResource runtime_split_resource = {};
    cudaDevResource runtime_split_remainder = {};
    unsigned int runtime_split_groups = 1;
    CHECK_CUDA_SUCCESS(cudaDevSmResourceSplitByCount(
        &runtime_split_resource, &runtime_split_groups, &runtime_device_resource,
        &runtime_split_remainder, 0, runtime_min_sm_count));
    if (runtime_split_groups != 1 ||
        runtime_split_resource.type != cudaDevResourceTypeSm ||
        runtime_split_resource.sm.smCount == 0) {
        std::fprintf(stderr, "unexpected runtime split SM resource\n");
        return 1;
    }
    unsigned int runtime_group_sm_count = aligned_sm_count(
        runtime_device_resource.sm.smCount, runtime_min_sm_count,
        runtime_device_resource.sm.smCoscheduledAlignment);
    if (runtime_group_sm_count == 0) {
        std::fprintf(stderr, "runtime SM resource cannot be grouped\n");
        return 1;
    }
    cudaDevResource runtime_structured_resource = {};
    cudaDevResource runtime_structured_remainder = {};
    cudaDevSmResourceGroupParams runtime_group_params = {};
    runtime_group_params.smCount = runtime_group_sm_count;
    runtime_group_params.coscheduledSmCount =
        runtime_device_resource.sm.smCoscheduledAlignment == 0
            ? runtime_group_sm_count
            : runtime_device_resource.sm.smCoscheduledAlignment;
    CHECK_CUDA_SUCCESS(cudaDevSmResourceSplit(
        &runtime_structured_resource, 1, &runtime_device_resource,
        &runtime_structured_remainder, 0, &runtime_group_params));
    if (runtime_structured_resource.type != cudaDevResourceTypeSm ||
        runtime_structured_resource.sm.smCount == 0 ||
        runtime_group_params.smCount == 0) {
        std::fprintf(stderr, "unexpected runtime structured SM resource\n");
        return 1;
    }
    cudaExecutionContext_t runtime_execution_context = nullptr;
    CHECK_CUDA_SUCCESS(cudaDeviceGetExecutionCtx(&runtime_execution_context, 0));
    if (runtime_execution_context == nullptr) {
        std::fprintf(stderr, "missing runtime execution context\n");
        return 1;
    }
    int execution_context_device = -1;
    CHECK_CUDA_SUCCESS(cudaExecutionCtxGetDevice(
        &execution_context_device, runtime_execution_context));
    if (execution_context_device != 0) {
        std::fprintf(stderr, "unexpected execution context device\n");
        return 1;
    }
    unsigned long long execution_context_id = 0;
    CHECK_CUDA_SUCCESS(cudaExecutionCtxGetId(
        runtime_execution_context, &execution_context_id));
    if (execution_context_id == 0) {
        std::fprintf(stderr, "unexpected execution context id\n");
        return 1;
    }
    cudaDevResource runtime_context_resource = {};
    CHECK_CUDA_SUCCESS(cudaExecutionCtxGetDevResource(
        runtime_execution_context, &runtime_context_resource,
        cudaDevResourceTypeSm));
    if (runtime_context_resource.type != cudaDevResourceTypeSm ||
        runtime_context_resource.sm.smCount == 0) {
        std::fprintf(stderr, "unexpected runtime context SM resource\n");
        return 1;
    }
    cudaStream_t runtime_execution_stream = nullptr;
    CHECK_CUDA_SUCCESS(cudaExecutionCtxStreamCreate(
        &runtime_execution_stream, runtime_execution_context,
        cudaStreamNonBlocking, 0));
    cudaDevResource runtime_stream_resource = {};
    CHECK_CUDA_SUCCESS(cudaStreamGetDevResource(
        runtime_execution_stream, &runtime_stream_resource,
        cudaDevResourceTypeSm));
    if (runtime_stream_resource.type != cudaDevResourceTypeSm ||
        runtime_stream_resource.sm.smCount == 0) {
        std::fprintf(stderr, "unexpected runtime stream SM resource\n");
        return 1;
    }
    cudaEvent_t runtime_ctx_event = nullptr;
    CHECK_CUDA_SUCCESS(cudaEventCreate(&runtime_ctx_event));
    CHECK_CUDA_SUCCESS(cudaExecutionCtxRecordEvent(runtime_execution_context,
                                                   runtime_ctx_event));
    CHECK_CUDA_SUCCESS(cudaExecutionCtxWaitEvent(runtime_execution_context,
                                                 runtime_ctx_event));
    CHECK_CUDA_SUCCESS(cudaExecutionCtxSynchronize(runtime_execution_context));
    CHECK_CUDA(cudaEventQuery(runtime_ctx_event));
    CHECK_CUDA_SUCCESS(cudaEventDestroy(runtime_ctx_event));
    CHECK_CUDA_SUCCESS(cudaStreamDestroy(runtime_execution_stream));

    cudaDevResourceDesc_t runtime_green_desc = nullptr;
    CHECK_CUDA_SUCCESS(cudaDevResourceGenerateDesc(
        &runtime_green_desc, &runtime_split_resource, 1));
    cudaExecutionContext_t runtime_green_context = nullptr;
    CHECK_CUDA_SUCCESS(
        cudaGreenCtxCreate(&runtime_green_context, runtime_green_desc, 0, 0));
    if (runtime_green_context == nullptr) {
        std::fprintf(stderr, "missing runtime green execution context\n");
        return 1;
    }
    int runtime_green_device = -1;
    CHECK_CUDA_SUCCESS(cudaExecutionCtxGetDevice(&runtime_green_device,
                                                 runtime_green_context));
    if (runtime_green_device != 0) {
        std::fprintf(stderr, "unexpected runtime green device\n");
        return 1;
    }
    unsigned long long runtime_green_id = 0;
    CHECK_CUDA_SUCCESS(cudaExecutionCtxGetId(runtime_green_context,
                                             &runtime_green_id));
    if (runtime_green_id == 0) {
        std::fprintf(stderr, "unexpected runtime green context id\n");
        return 1;
    }
    cudaDevResource runtime_green_resource = {};
    CHECK_CUDA_SUCCESS(cudaExecutionCtxGetDevResource(
        runtime_green_context, &runtime_green_resource, cudaDevResourceTypeSm));
    if (runtime_green_resource.type != cudaDevResourceTypeSm ||
        runtime_green_resource.sm.smCount == 0) {
        std::fprintf(stderr, "unexpected runtime green SM resource\n");
        return 1;
    }
    cudaStream_t runtime_green_stream = nullptr;
    CHECK_CUDA_SUCCESS(cudaExecutionCtxStreamCreate(
        &runtime_green_stream, runtime_green_context, cudaStreamNonBlocking,
        0));
    cudaDevResource runtime_green_stream_resource = {};
    CHECK_CUDA_SUCCESS(cudaStreamGetDevResource(
        runtime_green_stream, &runtime_green_stream_resource,
        cudaDevResourceTypeSm));
    if (runtime_green_stream_resource.type != cudaDevResourceTypeSm ||
        runtime_green_stream_resource.sm.smCount == 0) {
        std::fprintf(stderr, "unexpected runtime green stream SM resource\n");
        return 1;
    }
    CHECK_CUDA_SUCCESS(cudaStreamDestroy(runtime_green_stream));
    CHECK_CUDA_SUCCESS(cudaExecutionCtxDestroy(runtime_green_context));

    cudaStreamAttrValue runtime_attr = {};
    runtime_attr.syncPolicy = cudaSyncPolicyAuto;
    CHECK_CUDA_SUCCESS(cudaStreamSetAttribute(
        runtime_a, cudaStreamAttributeSynchronizationPolicy, &runtime_attr));
    cudaStreamAttrValue runtime_attr_out = {};
    CHECK_CUDA_SUCCESS(cudaStreamGetAttribute(
        runtime_a, cudaStreamAttributeSynchronizationPolicy,
        &runtime_attr_out));
    if (runtime_attr_out.syncPolicy != cudaSyncPolicyAuto) {
        std::fprintf(stderr, "unexpected runtime stream sync policy\n");
        return 1;
    }

    CHECK_CUDA_SUCCESS(cudaEventCreate(&runtime_start));
    CHECK_CUDA_SUCCESS(cudaEventCreate(&runtime_end));
    CHECK_CUDA_SUCCESS(cudaEventRecord(runtime_start, runtime_a));
    CHECK_CUDA_SUCCESS(cudaStreamWaitEvent(runtime_b, runtime_start, 0));
    CHECK_CUDA_SUCCESS(cudaEventRecord(runtime_end, runtime_b));
    CHECK_CUDA_SUCCESS(cudaEventSynchronize(runtime_end));
    CHECK_CUDA(cudaEventQuery(runtime_end));

    RuntimeCallbackState runtime_callback_state = {};
    CHECK_CUDA_SUCCESS(cudaStreamAddCallback(
        runtime_a, runtime_stream_callback, &runtime_callback_state, 0));
    CHECK_CUDA_SUCCESS(cudaStreamSynchronize(runtime_a));
    if (runtime_callback_state.called != 1 ||
        runtime_callback_state.status != cudaSuccess ||
        runtime_callback_state.stream != runtime_a) {
        std::fprintf(stderr, "unexpected runtime stream callback state\n");
        return 1;
    }

    cudaEvent_t runtime_ipc_event = nullptr;
    cudaEvent_t runtime_ipc_opened = nullptr;
    CHECK_CUDA_SUCCESS(cudaEventCreateWithFlags(
        &runtime_ipc_event, cudaEventDisableTiming | cudaEventInterprocess));
    cudaIpcEventHandle_t runtime_ipc_handle = {};
    CHECK_CUDA_SUCCESS(cudaIpcGetEventHandle(&runtime_ipc_handle,
                                             runtime_ipc_event));
    if (running_remoted()) {
        CHECK_CUDA_SUCCESS(cudaIpcOpenEventHandle(&runtime_ipc_opened,
                                                  runtime_ipc_handle));
        CHECK_CUDA_SUCCESS(cudaEventRecord(runtime_ipc_event, runtime_a));
        CHECK_CUDA_SUCCESS(
            cudaStreamWaitEvent(runtime_b, runtime_ipc_opened, 0));
        CHECK_CUDA_SUCCESS(cudaStreamSynchronize(runtime_b));
        CHECK_CUDA_SUCCESS(cudaEventDestroy(runtime_ipc_opened));
    }
    CHECK_CUDA_SUCCESS(cudaEventDestroy(runtime_ipc_event));

    float runtime_ms = -1.0f;
    CHECK_CUDA_SUCCESS(cudaEventElapsedTime(
        &runtime_ms, runtime_start, runtime_end));
    if (runtime_ms < 0.0f) {
        return 1;
    }

    CHECK_CUDA_SUCCESS(cudaEventDestroy(runtime_start));
    CHECK_CUDA_SUCCESS(cudaEventDestroy(runtime_end));
    CHECK_CUDA_SUCCESS(cudaStreamDestroy(runtime_a));
    CHECK_CUDA_SUCCESS(cudaStreamDestroy(runtime_b));

    CHECK_DRV_SUCCESS(cuInit(0));

    CUstream driver_a = nullptr;
    CUstream driver_b = nullptr;
    CUevent driver_start = nullptr;
    CUevent driver_end = nullptr;

    CHECK_DRV_SUCCESS(cuStreamCreateWithPriority(
        &driver_a, CU_STREAM_NON_BLOCKING, 0));
    CHECK_DRV_SUCCESS(cuStreamCreate(&driver_b, CU_STREAM_DEFAULT));
    CHECK_DRV_SUCCESS(cuStreamCopyAttributes(driver_b, driver_a));

    unsigned long long driver_stream_id = 0;
    CUdevice driver_device = -1;
    CUcontext driver_context = nullptr;
    CHECK_DRV_SUCCESS(cuStreamGetId(driver_a, &driver_stream_id));
    CHECK_DRV_SUCCESS(cuStreamGetDevice(driver_a, &driver_device));
    CHECK_DRV_SUCCESS(cuStreamGetCtx(driver_a, &driver_context));
    CUdevResource driver_device_resource = {};
    CHECK_DRV_SUCCESS(cuDeviceGetDevResource(
        driver_device, &driver_device_resource, CU_DEV_RESOURCE_TYPE_SM));
    if (driver_device_resource.type != CU_DEV_RESOURCE_TYPE_SM ||
        driver_device_resource.sm.smCount == 0) {
        std::fprintf(stderr, "unexpected driver device SM resource\n");
        return 1;
    }
    unsigned int driver_min_sm_count =
        driver_device_resource.sm.minSmPartitionSize == 0
            ? 1
            : driver_device_resource.sm.minSmPartitionSize;
    unsigned int driver_query_groups = 0;
    CHECK_DRV_SUCCESS(cuDevSmResourceSplitByCount(
        nullptr, &driver_query_groups, &driver_device_resource, nullptr, 0,
        driver_min_sm_count));
    if (driver_query_groups == 0) {
        std::fprintf(stderr, "driver SM split query returned no groups\n");
        return 1;
    }
    CUdevResource driver_split_resource = {};
    CUdevResource driver_split_remainder = {};
    unsigned int driver_split_groups = 1;
    CHECK_DRV_SUCCESS(cuDevSmResourceSplitByCount(
        &driver_split_resource, &driver_split_groups, &driver_device_resource,
        &driver_split_remainder, 0, driver_min_sm_count));
    if (driver_split_groups != 1 ||
        driver_split_resource.type != CU_DEV_RESOURCE_TYPE_SM ||
        driver_split_resource.sm.smCount == 0) {
        std::fprintf(stderr, "unexpected driver split SM resource\n");
        return 1;
    }
    unsigned int driver_group_sm_count = aligned_sm_count(
        driver_device_resource.sm.smCount, driver_min_sm_count,
        driver_device_resource.sm.smCoscheduledAlignment);
    if (driver_group_sm_count == 0) {
        std::fprintf(stderr, "driver SM resource cannot be grouped\n");
        return 1;
    }
    CUdevResource driver_structured_resource = {};
    CUdevResource driver_structured_remainder = {};
    CU_DEV_SM_RESOURCE_GROUP_PARAMS driver_group_params = {};
    driver_group_params.smCount = driver_group_sm_count;
    driver_group_params.coscheduledSmCount =
        driver_device_resource.sm.smCoscheduledAlignment == 0
            ? driver_group_sm_count
            : driver_device_resource.sm.smCoscheduledAlignment;
    CHECK_DRV_SUCCESS(cuDevSmResourceSplit(
        &driver_structured_resource, 1, &driver_device_resource,
        &driver_structured_remainder, 0, &driver_group_params));
    if (driver_structured_resource.type != CU_DEV_RESOURCE_TYPE_SM ||
        driver_structured_resource.sm.smCount == 0 ||
        driver_group_params.smCount == 0) {
        std::fprintf(stderr, "unexpected driver structured SM resource\n");
        return 1;
    }
    CUdevResource driver_context_resource = {};
    CHECK_DRV_SUCCESS(cuCtxGetDevResource(
        driver_context, &driver_context_resource, CU_DEV_RESOURCE_TYPE_SM));
    if (driver_context_resource.type != CU_DEV_RESOURCE_TYPE_SM ||
        driver_context_resource.sm.smCount == 0) {
        std::fprintf(stderr, "unexpected driver context SM resource\n");
        return 1;
    }
    CUdevResource driver_stream_resource = {};
    CHECK_DRV_SUCCESS(cuStreamGetDevResource(
        driver_a, &driver_stream_resource, CU_DEV_RESOURCE_TYPE_SM));
    if (driver_stream_resource.type != CU_DEV_RESOURCE_TYPE_SM ||
        driver_stream_resource.sm.smCount == 0) {
        std::fprintf(stderr, "unexpected driver stream SM resource\n");
        return 1;
    }
    CUcontext driver_context_v2 = nullptr;
    CUgreenCtx driver_green_context = nullptr;
    CHECK_DRV_SUCCESS(cuStreamGetCtx_v2(driver_a, &driver_context_v2,
                                        &driver_green_context));
    CUgreenCtx driver_stream_green_context = nullptr;
    CHECK_DRV_SUCCESS(cuStreamGetGreenCtx(driver_a,
                                          &driver_stream_green_context));
    if (driver_device != 0 || driver_context == nullptr) {
        return 1;
    }
    if (driver_context_v2 != driver_context || driver_green_context != nullptr ||
        driver_stream_green_context != nullptr) {
        std::fprintf(stderr, "unexpected driver stream context metadata\n");
        return 1;
    }
    CUdevResourceDesc driver_green_desc = nullptr;
    CHECK_DRV_SUCCESS(cuDevResourceGenerateDesc(
        &driver_green_desc, &driver_split_resource, 1));
    CUgreenCtx driver_green = nullptr;
    CHECK_DRV_SUCCESS(cuGreenCtxCreate(&driver_green, driver_green_desc,
                                       driver_device,
                                       CU_GREEN_CTX_DEFAULT_STREAM));
    if (driver_green == nullptr) {
        std::fprintf(stderr, "missing driver green context\n");
        return 1;
    }
    unsigned long long driver_green_id = 0;
    CHECK_DRV_SUCCESS(cuGreenCtxGetId(driver_green, &driver_green_id));
    if (driver_green_id == 0) {
        std::fprintf(stderr, "unexpected driver green context id\n");
        return 1;
    }
    CUdevResource driver_green_resource = {};
    CHECK_DRV_SUCCESS(cuGreenCtxGetDevResource(
        driver_green, &driver_green_resource, CU_DEV_RESOURCE_TYPE_SM));
    if (driver_green_resource.type != CU_DEV_RESOURCE_TYPE_SM ||
        driver_green_resource.sm.smCount == 0) {
        std::fprintf(stderr, "unexpected driver green SM resource\n");
        return 1;
    }
    CUcontext driver_green_primary = nullptr;
    CHECK_DRV_SUCCESS(cuCtxFromGreenCtx(&driver_green_primary, driver_green));
    if (driver_green_primary == nullptr) {
        std::fprintf(stderr, "missing driver green primary context\n");
        return 1;
    }
    CUstream driver_green_stream = nullptr;
    CHECK_DRV_SUCCESS(cuGreenCtxStreamCreate(
        &driver_green_stream, driver_green, CU_STREAM_NON_BLOCKING, 0));
    CUgreenCtx driver_green_stream_context = nullptr;
    CHECK_DRV_SUCCESS(cuStreamGetGreenCtx(driver_green_stream,
                                          &driver_green_stream_context));
    if (driver_green_stream_context != driver_green) {
        std::fprintf(stderr, "unexpected driver green stream context\n");
        return 1;
    }
    CUdevResource driver_green_stream_resource = {};
    CHECK_DRV_SUCCESS(cuStreamGetDevResource(
        driver_green_stream, &driver_green_stream_resource,
        CU_DEV_RESOURCE_TYPE_SM));
    if (driver_green_stream_resource.type != CU_DEV_RESOURCE_TYPE_SM ||
        driver_green_stream_resource.sm.smCount == 0) {
        std::fprintf(stderr, "unexpected driver green stream SM resource\n");
        return 1;
    }
    CUevent driver_green_event = nullptr;
    CHECK_DRV_SUCCESS(cuEventCreate(&driver_green_event, CU_EVENT_DEFAULT));
    CHECK_DRV_SUCCESS(cuGreenCtxRecordEvent(driver_green, driver_green_event));
    CHECK_DRV_SUCCESS(cuGreenCtxWaitEvent(driver_green, driver_green_event));
    CHECK_DRV_SUCCESS(cuCtxSynchronize_v2(driver_green_primary));
    CHECK_DRV(cuEventQuery(driver_green_event));
    CHECK_DRV_SUCCESS(cuEventDestroy(driver_green_event));
    CHECK_DRV_SUCCESS(cuStreamDestroy(driver_green_stream));
    CHECK_DRV_SUCCESS(cuGreenCtxDestroy(driver_green));

    CUstreamAttrValue driver_attr = {};
    driver_attr.syncPolicy = CU_SYNC_POLICY_AUTO;
    CHECK_DRV_SUCCESS(cuStreamSetAttribute(
        driver_a, CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY, &driver_attr));
    CUstreamAttrValue driver_attr_out = {};
    CHECK_DRV_SUCCESS(cuStreamGetAttribute(
        driver_a, CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY,
        &driver_attr_out));
    if (driver_attr_out.syncPolicy != CU_SYNC_POLICY_AUTO) {
        std::fprintf(stderr, "unexpected driver stream sync policy\n");
        return 1;
    }
    CUstreamCaptureMode driver_capture_mode = CU_STREAM_CAPTURE_MODE_RELAXED;
    CHECK_DRV_SUCCESS(cuThreadExchangeStreamCaptureMode(&driver_capture_mode));
    if (driver_capture_mode != CU_STREAM_CAPTURE_MODE_GLOBAL) {
        std::fprintf(stderr, "unexpected driver capture mode\n");
        return 1;
    }
    CHECK_DRV_SUCCESS(cuThreadExchangeStreamCaptureMode(&driver_capture_mode));

    CHECK_DRV_SUCCESS(cuEventCreate(&driver_start, CU_EVENT_DEFAULT));
    CHECK_DRV_SUCCESS(cuEventCreate(&driver_end, CU_EVENT_DEFAULT));
    CHECK_DRV_SUCCESS(cuEventRecord(driver_start, driver_a));
    CHECK_DRV_SUCCESS(cuStreamWaitEvent(driver_b, driver_start, 0));
    CHECK_DRV_SUCCESS(cuEventRecord(driver_end, driver_b));
    CHECK_DRV_SUCCESS(cuEventSynchronize(driver_end));
    CHECK_DRV(cuEventQuery(driver_end));

    DriverCallbackState driver_callback_state = {};
    CHECK_DRV_SUCCESS(cuStreamAddCallback(
        driver_a, driver_stream_callback, &driver_callback_state, 0));
    CHECK_DRV_SUCCESS(cuStreamSynchronize(driver_a));
    if (driver_callback_state.called != 1 ||
        driver_callback_state.status != CUDA_SUCCESS ||
        driver_callback_state.stream != driver_a) {
        std::fprintf(stderr, "unexpected driver stream callback state\n");
        return 1;
    }

    CUevent driver_ipc_event = nullptr;
    CUevent driver_ipc_opened = nullptr;
    CHECK_DRV_SUCCESS(cuEventCreate(
        &driver_ipc_event, CU_EVENT_DISABLE_TIMING | CU_EVENT_INTERPROCESS));
    CUipcEventHandle driver_ipc_handle = {};
    CHECK_DRV_SUCCESS(cuIpcGetEventHandle(&driver_ipc_handle,
                                          driver_ipc_event));
    if (running_remoted()) {
        CHECK_DRV_SUCCESS(cuIpcOpenEventHandle(&driver_ipc_opened,
                                               driver_ipc_handle));
        CHECK_DRV_SUCCESS(cuEventRecord(driver_ipc_event, driver_a));
        CHECK_DRV_SUCCESS(cuStreamWaitEvent(driver_b, driver_ipc_opened, 0));
        CHECK_DRV_SUCCESS(cuStreamSynchronize(driver_b));
        CHECK_DRV_SUCCESS(cuEventDestroy(driver_ipc_opened));
    }
    CHECK_DRV_SUCCESS(cuEventDestroy(driver_ipc_event));

    float driver_ms = -1.0f;
    CHECK_DRV_SUCCESS(cuEventElapsedTime(&driver_ms, driver_start, driver_end));
    if (driver_ms < 0.0f) {
        return 1;
    }

    CUdeviceptr wait32_device = 0;
    CUdeviceptr wait64_device = 0;
    CHECK_DRV_SUCCESS(cuMemAlloc(&wait32_device, sizeof(cuuint32_t)));
    CHECK_DRV_SUCCESS(cuMemAlloc(&wait64_device, sizeof(cuuint64_t)));
    CHECK_DRV_SUCCESS(cuMemsetD32(wait32_device, 0, 1));
    CHECK_DRV_SUCCESS(cuMemsetD32(wait64_device, 0, 2));

    const cuuint32_t expected32 = 0x10203040u;
    const cuuint64_t expected64 = 0x1020304050607080ull;
    CHECK_DRV_SUCCESS(cuStreamWriteValue32(driver_a, wait32_device, expected32,
                                           CU_STREAM_WRITE_VALUE_DEFAULT));
    CHECK_DRV_SUCCESS(cuStreamWaitValue32(driver_a, wait32_device, expected32,
                                          CU_STREAM_WAIT_VALUE_EQ));
    CHECK_DRV_SUCCESS(cuStreamWriteValue64(driver_a, wait64_device, expected64,
                                           CU_STREAM_WRITE_VALUE_DEFAULT));
    CHECK_DRV_SUCCESS(cuStreamWaitValue64(driver_a, wait64_device, expected64,
                                          CU_STREAM_WAIT_VALUE_EQ));
    CHECK_DRV_SUCCESS(cuStreamSynchronize(driver_a));

    cuuint32_t output32 = 0;
    cuuint64_t output64 = 0;
    CHECK_DRV_SUCCESS(cuMemcpyDtoH(&output32, wait32_device, sizeof(output32)));
    CHECK_DRV_SUCCESS(cuMemcpyDtoH(&output64, wait64_device, sizeof(output64)));
    if (output32 != expected32 || output64 != expected64) {
        std::fprintf(stderr, "stream value output mismatch\n");
        return 1;
    }

    CHECK_DRV_SUCCESS(cuMemsetD32(wait32_device, 0, 1));
    CHECK_DRV_SUCCESS(cuMemsetD32(wait64_device, 0, 2));
    const cuuint32_t batch32 = 0xa1b2c3d4u;
    const cuuint64_t batch64 = 0x1122334455667788ull;
    CUstreamBatchMemOpParams batch_ops[4] = {};
    batch_ops[0].operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    batch_ops[0].writeValue.address = wait32_device;
    batch_ops[0].writeValue.value = batch32;
    batch_ops[0].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
    batch_ops[0].writeValue.alias = 0;
    batch_ops[1].operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    batch_ops[1].waitValue.address = wait32_device;
    batch_ops[1].waitValue.value = batch32;
    batch_ops[1].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
    batch_ops[1].waitValue.alias = 0;
    batch_ops[2].operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
    batch_ops[2].writeValue.address = wait64_device;
    batch_ops[2].writeValue.value64 = batch64;
    batch_ops[2].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
    batch_ops[2].writeValue.alias = 0;
    batch_ops[3].operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
    batch_ops[3].waitValue.address = wait64_device;
    batch_ops[3].waitValue.value64 = batch64;
    batch_ops[3].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
    batch_ops[3].waitValue.alias = 0;
    CHECK_DRV_SUCCESS(cuStreamBatchMemOp(driver_a, 4, batch_ops, 0));
    CHECK_DRV_SUCCESS(cuStreamSynchronize(driver_a));

    output32 = 0;
    output64 = 0;
    CHECK_DRV_SUCCESS(cuMemcpyDtoH(&output32, wait32_device, sizeof(output32)));
    CHECK_DRV_SUCCESS(cuMemcpyDtoH(&output64, wait64_device, sizeof(output64)));
    if (output32 != batch32 || output64 != batch64) {
        std::fprintf(stderr, "stream batch mem-op output mismatch\n");
        return 1;
    }

    CUevent driver_ctx_event = nullptr;
    CHECK_DRV_SUCCESS(cuEventCreate(&driver_ctx_event, CU_EVENT_DEFAULT));
    CHECK_DRV_SUCCESS(cuCtxRecordEvent(driver_context, driver_ctx_event));
    CHECK_DRV_SUCCESS(cuCtxWaitEvent(driver_context, driver_ctx_event));
    CHECK_DRV_SUCCESS(cuCtxSynchronize_v2(driver_context));
    CHECK_DRV(cuEventQuery(driver_ctx_event));
    CHECK_DRV_SUCCESS(cuEventDestroy(driver_ctx_event));

    CHECK_DRV_SUCCESS(cuMemFree(wait64_device));
    CHECK_DRV_SUCCESS(cuMemFree(wait32_device));

    CHECK_DRV_SUCCESS(cuEventDestroy(driver_start));
    CHECK_DRV_SUCCESS(cuEventDestroy(driver_end));
    CHECK_DRV_SUCCESS(cuStreamDestroy(driver_a));
    CHECK_DRV_SUCCESS(cuStreamDestroy(driver_b));

    std::puts("event/stream API test passed");
    return 0;
}
