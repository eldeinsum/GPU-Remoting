#include <cuda.h>
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

static CUDA_MEMCPY3D_BATCH_OP make_driver_pointer_op(CUdeviceptr dst,
                                                     CUdeviceptr src,
                                                     size_t width,
                                                     size_t height,
                                                     size_t depth)
{
    CUDA_MEMCPY3D_BATCH_OP op = {};
    op.src.type = CU_MEMCPY_OPERAND_TYPE_POINTER;
    op.src.op.ptr.ptr = src;
    op.src.op.ptr.rowLength = width;
    op.src.op.ptr.layerHeight = height;
    op.dst.type = CU_MEMCPY_OPERAND_TYPE_POINTER;
    op.dst.op.ptr.ptr = dst;
    op.dst.op.ptr.rowLength = width;
    op.dst.op.ptr.layerHeight = height;
    op.extent.width = width;
    op.extent.height = height;
    op.extent.depth = depth;
    op.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
    op.flags = CU_MEMCPY_FLAG_DEFAULT;
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
    unsigned char *dst2 = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&src), bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dst), bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dst2), bytes));
    CHECK_CUDA(cudaMemcpy(src, input.data(), bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaMemcpyAttributes runtime_attr = {};
    runtime_attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    runtime_attr.flags = cudaMemcpyFlagDefault;
    cudaMemcpyAttributes runtime_attrs[] = {runtime_attr, runtime_attr};
    runtime_attrs[1].flags = cudaMemcpyFlagPreferOverlapWithCompute;
    size_t runtime_attr_indices[] = {0, 1};
    void *runtime_dsts[] = {dst, dst2};
    const void *runtime_srcs[] = {src, src};
    size_t runtime_sizes[] = {bytes, bytes};

    CHECK_CUDA(cudaMemset(dst, 0, bytes));
    CHECK_CUDA(cudaMemset(dst2, 0, bytes));
    CHECK_CUDA(cudaMemcpyBatchAsync(
        runtime_dsts, runtime_srcs, runtime_sizes, 2, runtime_attrs,
        runtime_attr_indices, 2, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<unsigned char> output(bytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), dst, bytes, cudaMemcpyDeviceToHost));
    if (verify_equal(output, input) != 0) {
        return 1;
    }
    output.assign(bytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), dst2, bytes, cudaMemcpyDeviceToHost));
    if (verify_equal(output, input) != 0) {
        return 1;
    }

    CHECK_CUDA(cudaMemset(dst, 0, bytes));
    CHECK_CUDA(
        cudaMemcpyWithAttributesAsync(dst, src, bytes, &runtime_attr, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    output.assign(bytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), dst, bytes, cudaMemcpyDeviceToHost));
    if (verify_equal(output, input) != 0) {
        return 1;
    }

    cudaMemcpy3DBatchOp op =
        make_pointer_op(dst, src, width, height, depth);
    CHECK_CUDA(cudaMemset(dst, 0, bytes));
    CHECK_CUDA(cudaMemcpy3DBatchAsync(1, &op, 0, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    output.assign(bytes, 0);
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
    CHECK_CUDA(cudaFree(dst2));
    CHECK_CUDA(cudaFree(dst));
    CHECK_CUDA(cudaFree(src));

    CHECK_DRV(cuInit(0));
    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));
    CUcontext context = nullptr;
    CHECK_DRV(cuCtxGetCurrent(&context));
    if (context == nullptr) {
        CHECK_DRV(cuDevicePrimaryCtxRetain(&context, device));
        CHECK_DRV(cuCtxSetCurrent(context));
    }

    CUdeviceptr driver_src = 0;
    CUdeviceptr driver_dst = 0;
    CUdeviceptr driver_dst2 = 0;
    CHECK_DRV(cuMemAlloc(&driver_src, bytes));
    CHECK_DRV(cuMemAlloc(&driver_dst, bytes));
    CHECK_DRV(cuMemAlloc(&driver_dst2, bytes));
    CHECK_DRV(cuMemcpyHtoD(driver_src, input.data(), bytes));

    CUstream driver_stream = nullptr;
    CHECK_DRV(cuStreamCreate(&driver_stream, CU_STREAM_DEFAULT));
    CUmemcpyAttributes attr = {};
    attr.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
    attr.flags = CU_MEMCPY_FLAG_DEFAULT;
    CUmemcpyAttributes batch_attrs[] = {attr, attr};
    batch_attrs[1].flags = CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE;
    size_t attr_indices[] = {0, 1};
    CUdeviceptr batch_dsts[] = {driver_dst, driver_dst2};
    CUdeviceptr batch_srcs[] = {driver_src, driver_src};
    size_t batch_sizes[] = {bytes, bytes};

    output.assign(bytes, 0);
    CHECK_DRV(cuMemsetD8(driver_dst, 0, bytes));
    CHECK_DRV(cuMemsetD8(driver_dst2, 0, bytes));
    CHECK_DRV(cuMemcpyBatchAsync(
        batch_dsts, batch_srcs, batch_sizes, 2, batch_attrs, attr_indices, 2,
        driver_stream));
    CHECK_DRV(cuStreamSynchronize(driver_stream));
    CHECK_DRV(cuMemcpyDtoH(output.data(), driver_dst, bytes));
    if (verify_equal(output, input) != 0) {
        return 1;
    }
    output.assign(bytes, 0);
    CHECK_DRV(cuMemcpyDtoH(output.data(), driver_dst2, bytes));
    if (verify_equal(output, input) != 0) {
        return 1;
    }

    output.assign(bytes, 0);
    CHECK_DRV(cuMemsetD8(driver_dst, 0, bytes));
    CHECK_DRV(cuMemcpyWithAttributesAsync(
        driver_dst, driver_src, bytes, &attr, driver_stream));
    CHECK_DRV(cuStreamSynchronize(driver_stream));
    CHECK_DRV(cuMemcpyDtoH(output.data(), driver_dst, bytes));
    if (verify_equal(output, input) != 0) {
        return 1;
    }

    CUDA_MEMCPY3D_BATCH_OP driver_op =
        make_driver_pointer_op(driver_dst, driver_src, width, height, depth);
    output.assign(bytes, 0);
    CHECK_DRV(cuMemsetD8(driver_dst, 0, bytes));
    CHECK_DRV(cuMemcpy3DBatchAsync(1, &driver_op, 0, driver_stream));
    CHECK_DRV(cuStreamSynchronize(driver_stream));
    CHECK_DRV(cuMemcpyDtoH(output.data(), driver_dst, bytes));
    if (verify_equal(output, input) != 0) {
        return 1;
    }

    output.assign(bytes, 0);
    CHECK_DRV(cuMemsetD8(driver_dst, 0, bytes));
    CHECK_DRV(cuMemcpy3DWithAttributesAsync(&driver_op, 0, driver_stream));
    CHECK_DRV(cuStreamSynchronize(driver_stream));
    CHECK_DRV(cuMemcpyDtoH(output.data(), driver_dst, bytes));
    if (verify_equal(output, input) != 0) {
        return 1;
    }

    CHECK_DRV(cuStreamDestroy(driver_stream));
    CHECK_DRV(cuMemFree(driver_dst2));
    CHECK_DRV(cuMemFree(driver_dst));
    CHECK_DRV(cuMemFree(driver_src));

    std::puts("memcpy3d batch API test passed");
    return 0;
}
