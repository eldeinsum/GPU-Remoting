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
