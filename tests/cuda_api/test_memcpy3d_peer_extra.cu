#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
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

static int verify_equal(const std::vector<unsigned char> &actual,
                        const std::vector<unsigned char> &expected)
{
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "size mismatch: got %zu expected %zu\n",
                     actual.size(), expected.size());
        return 1;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "mismatch at byte %zu: got %u expected %u\n",
                         i, static_cast<unsigned>(actual[i]),
                         static_cast<unsigned>(expected[i]));
            return 1;
        }
    }
    return 0;
}

int main()
{
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        std::puts("memcpy3d peer API test skipped");
        return 0;
    }

    int peer_access = 0;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&peer_access, 1, 0));
    if (!peer_access) {
        std::puts("memcpy3d peer API test skipped");
        return 0;
    }

    constexpr size_t width = 16;
    constexpr size_t height = 4;
    constexpr size_t depth = 3;
    constexpr size_t bytes = width * height * depth;

    std::vector<unsigned char> input(bytes);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<unsigned char>((i * 11 + 5) & 0xff);
    }

    unsigned char *src = nullptr;
    unsigned char *dst = nullptr;

    CHECK_CUDA(cudaSetDevice(0));
    cudaError_t enable_01 = cudaDeviceEnablePeerAccess(1, 0);
    if (enable_01 != cudaSuccess &&
        enable_01 != cudaErrorPeerAccessAlreadyEnabled) {
        std::fprintf(stderr, "cudaDeviceEnablePeerAccess(1) failed: %s (%d)\n",
                     cudaGetErrorString(enable_01),
                     static_cast<int>(enable_01));
        return 1;
    }
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&src), bytes));
    CHECK_CUDA(cudaMemcpy(src, input.data(), bytes, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaSetDevice(1));
    cudaError_t enable_10 = cudaDeviceEnablePeerAccess(0, 0);
    if (enable_10 != cudaSuccess &&
        enable_10 != cudaErrorPeerAccessAlreadyEnabled) {
        std::fprintf(stderr, "cudaDeviceEnablePeerAccess(0) failed: %s (%d)\n",
                     cudaGetErrorString(enable_10),
                     static_cast<int>(enable_10));
        return 1;
    }
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dst), bytes));

    cudaMemcpy3DPeerParms params = {};
    params.srcPtr = make_cudaPitchedPtr(src, width, width, height);
    params.srcDevice = 0;
    params.dstPtr = make_cudaPitchedPtr(dst, width, width, height);
    params.dstDevice = 1;
    params.extent = make_cudaExtent(width, height, depth);
    CHECK_CUDA(cudaMemcpy3DPeer(&params));

    std::vector<unsigned char> output(bytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), dst, bytes, cudaMemcpyDeviceToHost));
    if (verify_equal(output, input) != 0) {
        return 1;
    }

    CHECK_CUDA(cudaSetDevice(0));
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaMemset(dst, 0, bytes));
    CHECK_CUDA(cudaMemcpy3DPeerAsync(&params, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    output.assign(bytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), dst, bytes, cudaMemcpyDeviceToHost));
    if (verify_equal(output, input) != 0) {
        return 1;
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(dst));
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaFree(src));
    std::puts("memcpy3d peer API test passed");
    return 0;
}
