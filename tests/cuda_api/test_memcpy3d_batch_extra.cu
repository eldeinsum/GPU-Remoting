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

static cudaMemcpy3DBatchOp make_pointer_op(void *dst, const void *src,
                                           size_t width, size_t height,
                                           size_t depth)
{
    cudaMemcpy3DBatchOp op = {};
    op.src.type = cudaMemcpyOperandTypePointer;
    op.src.op.ptr.ptr = const_cast<void *>(src);
    op.src.op.ptr.rowLength = width;
    op.src.op.ptr.layerHeight = height;
    op.dst.type = cudaMemcpyOperandTypePointer;
    op.dst.op.ptr.ptr = dst;
    op.dst.op.ptr.rowLength = width;
    op.dst.op.ptr.layerHeight = height;
    op.extent = make_cudaExtent(width, height, depth);
    op.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    op.flags = cudaMemcpyFlagDefault;
    return op;
}

int main()
{
    constexpr size_t width = 16;
    constexpr size_t height = 4;
    constexpr size_t depth = 3;
    constexpr size_t bytes = width * height * depth;

    std::vector<unsigned char> input(bytes);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<unsigned char>((i * 17 + 3) & 0xff);
    }

    unsigned char *src = nullptr;
    unsigned char *dst = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&src), bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dst), bytes));
    CHECK_CUDA(cudaMemcpy(src, input.data(), bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaMemcpy3DBatchOp op =
        make_pointer_op(dst, src, width, height, depth);
    CHECK_CUDA(cudaMemset(dst, 0, bytes));
    CHECK_CUDA(cudaMemcpy3DBatchAsync(1, &op, 0, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<unsigned char> output(bytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), dst, bytes, cudaMemcpyDeviceToHost));
    if (verify_equal(output, input) != 0) {
        return 1;
    }

    CHECK_CUDA(cudaMemset(dst, 0, bytes));
    CHECK_CUDA(cudaMemcpy3DWithAttributesAsync(&op, 0, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    output.assign(bytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), dst, bytes, cudaMemcpyDeviceToHost));
    if (verify_equal(output, input) != 0) {
        return 1;
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(dst));
    CHECK_CUDA(cudaFree(src));
    std::puts("memcpy3d batch API test passed");
    return 0;
}
