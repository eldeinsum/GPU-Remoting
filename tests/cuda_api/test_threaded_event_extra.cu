#include <cuda_runtime.h>

#include <cstdio>
#include <thread>

#define CHECK_RT(call)                                                         \
    do {                                                                       \
        cudaError_t result = (call);                                           \
        if (result != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s failed: %s (%d)\n", #call,               \
                         cudaGetErrorString(result),                           \
                         static_cast<int>(result));                            \
            return 1;                                                          \
        }                                                                      \
    } while (0)

int main()
{
    CHECK_RT(cudaSetDevice(0));

    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
    CHECK_RT(cudaStreamCreate(&stream));
    CHECK_RT(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    CHECK_RT(cudaEventRecord(event, stream));

    cudaError_t thread_result = cudaSuccess;
    std::thread waiter([&]() {
        thread_result = cudaEventSynchronize(event);
    });
    waiter.join();

    if (thread_result != cudaSuccess) {
        std::fprintf(stderr,
                     "cross-thread cudaEventSynchronize failed: %s (%d)\n",
                     cudaGetErrorString(thread_result),
                     static_cast<int>(thread_result));
        return 1;
    }

    CHECK_RT(cudaEventDestroy(event));
    CHECK_RT(cudaStreamDestroy(stream));

    std::puts("threaded event API test passed");
    return 0;
}
