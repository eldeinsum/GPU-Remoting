#include <cuda_runtime.h>

#include <cstdio>
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
    constexpr size_t width = 16;
    constexpr size_t height = 4;
    constexpr size_t depth = 3;
    constexpr size_t bytes = width * height * depth;

    std::vector<unsigned char> input(bytes);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<unsigned char>((i * 7) & 0xff);
    }

    unsigned char *src = nullptr;
    unsigned char *dst = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&src), bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dst), bytes));
    CHECK_CUDA(cudaMemcpy(src, input.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dst, 0, bytes));

    cudaMemcpy3DParms params = {};
    params.srcPtr = make_cudaPitchedPtr(src, width, width, height);
    params.dstPtr = make_cudaPitchedPtr(dst, width, width, height);
    params.extent = make_cudaExtent(width, height, depth);
    params.kind = cudaMemcpyDeviceToDevice;
    CHECK_CUDA(cudaMemcpy3D(&params));

    std::vector<unsigned char> output(bytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), dst, bytes, cudaMemcpyDeviceToHost));
    if (verify_equal(output, input) != 0) {
        return 1;
    }

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaMemset(dst, 0, bytes));
    CHECK_CUDA(cudaMemcpy3DAsync(&params, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    output.assign(bytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), dst, bytes, cudaMemcpyDeviceToHost));
    if (verify_equal(output, input) != 0) {
        return 1;
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(dst));
    CHECK_CUDA(cudaFree(src));
    std::puts("memcpy3d API test passed");
    return 0;
}
