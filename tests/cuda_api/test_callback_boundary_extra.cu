#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdio>

static const char *driver_result_name(CUresult result)
{
    const char *name = nullptr;
    cuGetErrorName(result, &name);
    return name ? name : "unknown";
}

static int expect_driver(CUresult result, CUresult expected, const char *label)
{
    if (result != expected) {
        std::fprintf(stderr, "%s returned %s (%d), expected %s (%d)\n", label,
                     driver_result_name(result), static_cast<int>(result),
                     driver_result_name(expected), static_cast<int>(expected));
        return 1;
    }
    return 0;
}

static int expect_runtime(cudaError_t result, cudaError_t expected,
                          const char *label)
{
    if (result != expected) {
        std::fprintf(stderr, "%s returned %s (%d), expected %s (%d)\n", label,
                     cudaGetErrorName(result), static_cast<int>(result),
                     cudaGetErrorName(expected), static_cast<int>(expected));
        return 1;
    }
    return 0;
}

static bool driver_registration_result(CUresult result)
{
    return result == CUDA_SUCCESS || result == CUDA_ERROR_NOT_SUPPORTED;
}

static bool runtime_registration_result(cudaError_t result)
{
    return result == cudaSuccess || result == cudaErrorNotSupported;
}

static int expect_driver_registration(CUresult result, const char *label)
{
    if (!driver_registration_result(result)) {
        std::fprintf(stderr,
                     "%s returned %s (%d), expected success or not supported\n",
                     label, driver_result_name(result), static_cast<int>(result));
        return 1;
    }
    return 0;
}

static int expect_runtime_registration(cudaError_t result, const char *label)
{
    if (!runtime_registration_result(result)) {
        std::fprintf(stderr,
                     "%s returned %s (%d), expected success or not supported\n",
                     label, cudaGetErrorName(result), static_cast<int>(result));
        return 1;
    }
    return 0;
}

static void CUDA_CB driver_async_callback(CUasyncNotificationInfo *, void *,
                                          CUasyncCallbackHandle)
{
}

static void CUDART_CB runtime_async_callback(cudaAsyncNotificationInfo_t *,
                                             void *,
                                             cudaAsyncCallbackHandle_t)
{
}

static void CUDA_CB coredump_callback(void *, int, CUdevice)
{
}

static void CUDA_CB driver_log_callback(void *, CUlogLevel, char *, size_t)
{
}

static void CUDART_CB runtime_log_callback(void *, cudaLogLevel, char *, size_t)
{
}

int main()
{
    int status = 0;

    status |= expect_driver(cuInit(0), CUDA_SUCCESS, "cuInit");

    CUdevice device = 0;
    status |= expect_driver(cuDeviceGet(&device, 0), CUDA_SUCCESS,
                            "cuDeviceGet");

    status |= expect_driver(
        cuDeviceGetNvSciSyncAttributes(nullptr, device,
                                       CUDA_NVSCISYNC_ATTR_SIGNAL),
        CUDA_ERROR_INVALID_HANDLE,
        "cuDeviceGetNvSciSyncAttributes(null)");
    status |= expect_runtime(
        cudaDeviceGetNvSciSyncAttributes(nullptr, 0, cudaNvSciSyncAttrSignal),
        cudaErrorInvalidResourceHandle,
        "cudaDeviceGetNvSciSyncAttributes(null)");

    CUasyncCallbackHandle driver_async_handle = nullptr;
    CUresult driver_async_result = cuDeviceRegisterAsyncNotification(
        device, driver_async_callback, nullptr, &driver_async_handle);
    status |= expect_driver_registration(
        driver_async_result,
        "cuDeviceRegisterAsyncNotification(valid)");
    if (driver_async_result == CUDA_SUCCESS) {
        status |= expect_driver(
            cuDeviceUnregisterAsyncNotification(device, driver_async_handle),
            CUDA_SUCCESS,
            "cuDeviceUnregisterAsyncNotification(valid)");
    }

    cudaAsyncCallbackHandle_t runtime_async_handle = nullptr;
    cudaError_t runtime_async_result = cudaDeviceRegisterAsyncNotification(
        0, runtime_async_callback, nullptr, &runtime_async_handle);
    status |= expect_runtime_registration(
        runtime_async_result,
        "cudaDeviceRegisterAsyncNotification(valid)");
    if (runtime_async_result == cudaSuccess) {
        status |= expect_runtime(
            cudaDeviceUnregisterAsyncNotification(0, runtime_async_handle),
            cudaSuccess,
            "cudaDeviceUnregisterAsyncNotification(valid)");
    }

    CUcoredumpCallbackHandle start_handle = nullptr;
    status |= expect_driver(
        cuCoredumpRegisterStartCallback(nullptr, nullptr, &start_handle),
        CUDA_ERROR_INVALID_VALUE,
        "cuCoredumpRegisterStartCallback(null)");
    CUresult start_result = cuCoredumpRegisterStartCallback(
        coredump_callback, nullptr, &start_handle);
    status |= expect_driver_registration(
        start_result,
        "cuCoredumpRegisterStartCallback(valid)");
    if (start_result == CUDA_SUCCESS) {
        status |= expect_driver(cuCoredumpDeregisterStartCallback(start_handle),
                                CUDA_SUCCESS,
                                "cuCoredumpDeregisterStartCallback(valid)");
    }
    status |= expect_driver(cuCoredumpDeregisterStartCallback(nullptr),
                            CUDA_ERROR_INVALID_VALUE,
                            "cuCoredumpDeregisterStartCallback(null)");

    CUcoredumpCallbackHandle complete_handle = nullptr;
    status |= expect_driver(
        cuCoredumpRegisterCompleteCallback(nullptr, nullptr, &complete_handle),
        CUDA_ERROR_INVALID_VALUE,
        "cuCoredumpRegisterCompleteCallback(null)");
    CUresult complete_result = cuCoredumpRegisterCompleteCallback(
        coredump_callback, nullptr, &complete_handle);
    status |= expect_driver_registration(
        complete_result,
        "cuCoredumpRegisterCompleteCallback(valid)");
    if (complete_result == CUDA_SUCCESS) {
        status |= expect_driver(
            cuCoredumpDeregisterCompleteCallback(complete_handle),
            CUDA_SUCCESS,
            "cuCoredumpDeregisterCompleteCallback(valid)");
    }
    status |= expect_driver(cuCoredumpDeregisterCompleteCallback(nullptr),
                            CUDA_ERROR_INVALID_VALUE,
                            "cuCoredumpDeregisterCompleteCallback(null)");

    CUlogsCallbackHandle driver_log_handle = nullptr;
    status |= expect_driver(cuLogsRegisterCallback(nullptr, nullptr,
                                                   &driver_log_handle),
                            CUDA_ERROR_INVALID_VALUE,
                            "cuLogsRegisterCallback(null)");
    CUresult driver_log_result = cuLogsRegisterCallback(
        driver_log_callback, nullptr, &driver_log_handle);
    status |= expect_driver_registration(driver_log_result,
                                         "cuLogsRegisterCallback(valid)");
    if (driver_log_result == CUDA_SUCCESS) {
        status |= expect_driver(cuLogsUnregisterCallback(driver_log_handle),
                                CUDA_SUCCESS,
                                "cuLogsUnregisterCallback(valid)");
    }
    status |= expect_driver(cuLogsUnregisterCallback(nullptr),
                            CUDA_ERROR_INVALID_VALUE,
                            "cuLogsUnregisterCallback(null)");

    cudaLogsCallbackHandle runtime_log_handle = nullptr;
    status |= expect_runtime(cudaLogsRegisterCallback(nullptr, nullptr,
                                                      &runtime_log_handle),
                             cudaErrorInvalidValue,
                             "cudaLogsRegisterCallback(null)");
    cudaError_t runtime_log_result = cudaLogsRegisterCallback(
        runtime_log_callback, nullptr, &runtime_log_handle);
    status |= expect_runtime_registration(runtime_log_result,
                                          "cudaLogsRegisterCallback(valid)");
    if (runtime_log_result == cudaSuccess) {
        status |= expect_runtime(cudaLogsUnregisterCallback(runtime_log_handle),
                                 cudaSuccess,
                                 "cudaLogsUnregisterCallback(valid)");
    }
    status |= expect_runtime(cudaLogsUnregisterCallback(nullptr),
                             cudaErrorInvalidValue,
                             "cudaLogsUnregisterCallback(null)");

    return status;
}
