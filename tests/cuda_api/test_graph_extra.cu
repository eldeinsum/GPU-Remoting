#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t result = (call);                                           \
        if (result != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s failed: %s (%d)\n", #call,               \
                         cudaGetErrorString(result), static_cast<int>(result)); \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static int verify_value(const std::vector<unsigned char> &data,
                        unsigned char expected)
{
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] != expected) {
            std::fprintf(stderr, "mismatch at byte %zu: got %u expected %u\n",
                         i, static_cast<unsigned>(data[i]),
                         static_cast<unsigned>(expected));
            return 1;
        }
    }
    return 0;
}

int main()
{
    constexpr size_t kBytes = 256;

    CHECK_CUDA(cudaSetDevice(0));

    unsigned char *device = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&device), kBytes));
    CHECK_CUDA(cudaMemset(device, 0, kBytes));

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    unsigned long long capture_id = 0;
    CHECK_CUDA(cudaStreamGetCaptureInfo(stream, &status, &capture_id, nullptr,
                                        nullptr, nullptr, nullptr));
    if (status != cudaStreamCaptureStatusNone) {
        std::fprintf(stderr, "unexpected initial capture status %d\n",
                     static_cast<int>(status));
        return 1;
    }

    cudaStreamCaptureMode mode = cudaStreamCaptureModeThreadLocal;
    CHECK_CUDA(cudaThreadExchangeStreamCaptureMode(&mode));
    CHECK_CUDA(cudaThreadExchangeStreamCaptureMode(&mode));

    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    CHECK_CUDA(cudaStreamGetCaptureInfo(stream, &status, &capture_id, nullptr,
                                        nullptr, nullptr, nullptr));
    if (status != cudaStreamCaptureStatusActive || capture_id == 0) {
        std::fprintf(stderr, "bad capture info: status=%d id=%llu\n",
                     static_cast<int>(status), capture_id);
        return 1;
    }

    CHECK_CUDA(cudaMemsetAsync(device, 0x5a, kBytes, stream));

    cudaGraph_t graph = nullptr;
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    if (graph == nullptr) {
        std::fprintf(stderr, "cudaStreamEndCapture returned a null graph\n");
        return 1;
    }

    size_t nodes = 0;
    CHECK_CUDA(cudaGraphGetNodes(graph, nullptr, &nodes));
    if (nodes == 0) {
        std::fprintf(stderr, "captured graph has no nodes\n");
        return 1;
    }

    cudaGraphExec_t exec = nullptr;
    CHECK_CUDA(cudaGraphInstantiate(&exec, graph, 0));
    CHECK_CUDA(cudaMemset(device, 0, kBytes));
    CHECK_CUDA(cudaGraphLaunch(exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<unsigned char> output(kBytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), device, kBytes, cudaMemcpyDeviceToHost));
    if (verify_value(output, 0x5a) != 0) {
        return 1;
    }
    CHECK_CUDA(cudaGraphExecDestroy(exec));

    exec = nullptr;
    CHECK_CUDA(cudaGraphInstantiateWithFlags(&exec, graph, 0));
    CHECK_CUDA(cudaMemset(device, 0, kBytes));
    CHECK_CUDA(cudaGraphLaunch(exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    output.assign(kBytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), device, kBytes, cudaMemcpyDeviceToHost));
    if (verify_value(output, 0x5a) != 0) {
        return 1;
    }

    CHECK_CUDA(cudaGraphExecDestroy(exec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(device));

    std::puts("graph API test passed");
    return 0;
}
