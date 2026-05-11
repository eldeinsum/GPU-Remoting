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

struct CallbackState {
    int value;
    int calls;
};

static void CUDART_CB record_callback(void *data)
{
    CallbackState *state = static_cast<CallbackState *>(data);
    if (state->value == 42 || state->value == 84 || state->value == 126 ||
        state->value == 168) {
        state->calls += 1;
    }
}

static void CUDA_CB record_driver_callback(void *data)
{
    record_callback(data);
}

int main()
{
    CHECK_CUDA(cudaSetDevice(0));

    int *device_value = nullptr;
    cudaStream_t stream = nullptr;
    CallbackState state{0, 0};
    int host_value = 42;

    CHECK_CUDA(cudaMalloc(&device_value, sizeof(int)));
    CHECK_CUDA(cudaStreamCreate(&stream));

    CHECK_CUDA(cudaMemcpyAsync(device_value, &host_value, sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(&state.value, device_value, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaLaunchHostFunc(stream, record_callback, &state));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    if (state.value != 42 || state.calls != 1) {
        std::fprintf(stderr, "first host callback state value=%d calls=%d\n",
                     state.value, state.calls);
        return 1;
    }

#if CUDART_VERSION >= 13000
    host_value = 84;
    CHECK_CUDA(cudaMemcpyAsync(device_value, &host_value, sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(&state.value, device_value, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaLaunchHostFunc_v2(stream, record_callback, &state, 0));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    if (state.value != 84 || state.calls != 2) {
        std::fprintf(stderr, "second host callback state value=%d calls=%d\n",
                     state.value, state.calls);
        return 1;
    }
#endif

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(device_value));

    CHECK_DRV(cuInit(0));
    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));
    CUcontext context = nullptr;
    CHECK_DRV(cuDevicePrimaryCtxRetain(&context, device));
    CHECK_DRV(cuCtxSetCurrent(context));

    CUdeviceptr driver_device_value = 0;
    CUstream driver_stream = nullptr;
    CallbackState driver_state{0, 0};
    host_value = 126;

    CHECK_DRV(cuMemAlloc(&driver_device_value, sizeof(int)));
    CHECK_DRV(cuStreamCreate(&driver_stream, CU_STREAM_DEFAULT));

    CHECK_DRV(cuMemcpyHtoDAsync(driver_device_value, &host_value, sizeof(int),
                                driver_stream));
    CHECK_DRV(cuMemcpyDtoHAsync(&driver_state.value, driver_device_value,
                                sizeof(int), driver_stream));
    CHECK_DRV(cuLaunchHostFunc(driver_stream, record_driver_callback,
                               &driver_state));
    CHECK_DRV(cuStreamSynchronize(driver_stream));
    if (driver_state.value != 126 || driver_state.calls != 1) {
        std::fprintf(stderr,
                     "driver host callback state value=%d calls=%d\n",
                     driver_state.value, driver_state.calls);
        return 1;
    }

#if CUDA_VERSION >= 13000
    host_value = 168;
    CHECK_DRV(cuMemcpyHtoDAsync(driver_device_value, &host_value, sizeof(int),
                                driver_stream));
    CHECK_DRV(cuMemcpyDtoHAsync(&driver_state.value, driver_device_value,
                                sizeof(int), driver_stream));
    CHECK_DRV(cuLaunchHostFunc_v2(driver_stream, record_driver_callback,
                                  &driver_state, 0));
    CHECK_DRV(cuStreamSynchronize(driver_stream));
    if (driver_state.value != 168 || driver_state.calls != 2) {
        std::fprintf(stderr,
                     "driver host callback v2 state value=%d calls=%d\n",
                     driver_state.value, driver_state.calls);
        return 1;
    }
#endif

    CHECK_DRV(cuStreamDestroy(driver_stream));
    CHECK_DRV(cuMemFree(driver_device_value));
    CHECK_DRV(cuDevicePrimaryCtxRelease(device));

    std::printf("launch host func API test passed\n");
    return 0;
}
