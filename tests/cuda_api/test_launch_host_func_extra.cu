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

struct CallbackState {
    int value;
    int calls;
};

static void CUDART_CB record_callback(void *data)
{
    CallbackState *state = static_cast<CallbackState *>(data);
    if (state->value == 42 || state->value == 84) {
        state->calls += 1;
    }
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

    std::printf("launch host func API test passed\n");
    return 0;
}
