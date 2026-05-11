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

__global__ void fill_kernel(int *data, int count, int value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] = value;
    }
}

__global__ void add_kernel(int *data, int count, int value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] += value;
    }
}

static int check_values(const std::vector<int> &values, int expected)
{
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i] != expected) {
            std::fprintf(stderr, "value mismatch at %zu: got %d expected %d\n",
                         i, values[i], expected);
            return 1;
        }
    }
    return 0;
}

static int has_concurrent_managed_access()
{
    int value = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(
        &value, cudaDevAttrConcurrentManagedAccess, 0));
    return value != 0;
}

static int run_runtime_batch()
{
    constexpr int kCount = 128;
    constexpr int kHalf = kCount / 2;
    constexpr size_t kBytes = kCount * sizeof(int);

    CHECK_CUDA(cudaSetDevice(0));
    if (!has_concurrent_managed_access()) {
        std::puts("runtime managed batch API test skipped");
        return 0;
    }

    int *managed = nullptr;
    CHECK_CUDA(cudaMallocManaged(
        reinterpret_cast<void **>(&managed), kBytes, cudaMemAttachGlobal));

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));

    std::vector<int> input(kCount, 10);
    std::vector<int> output(kCount, 0);
    CHECK_CUDA(cudaMemcpy(managed, input.data(), kBytes, cudaMemcpyHostToDevice));

    void *ptrs[] = {
        managed,
        managed + kHalf,
    };
    size_t sizes[] = {
        kHalf * sizeof(int),
        kHalf * sizeof(int),
    };
    cudaMemLocation location = {};
    location.type = cudaMemLocationTypeDevice;
    location.id = 0;
    size_t location_indices[] = {0};

    CHECK_CUDA(cudaMemPrefetchBatchAsync(
        ptrs, sizes, 2, &location, location_indices, 1, 0, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    add_kernel<<<1, kCount, 0, stream>>>(managed, kCount, 3);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(output.data(), managed, kBytes, cudaMemcpyDeviceToHost));
    if (check_values(output, 13) != 0) {
        return 1;
    }

    CHECK_CUDA(cudaMemDiscardBatchAsync(ptrs, sizes, 2, 0, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemDiscardAndPrefetchBatchAsync(
        ptrs, sizes, 2, &location, location_indices, 1, 0, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    fill_kernel<<<1, kCount, 0, stream>>>(managed, kCount, 29);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(output.data(), managed, kBytes, cudaMemcpyDeviceToHost));
    if (check_values(output, 29) != 0) {
        return 1;
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(managed));
    return 0;
}

static int run_driver_batch()
{
    constexpr int kCount = 128;
    constexpr int kHalf = kCount / 2;
    constexpr size_t kBytes = kCount * sizeof(int);
    constexpr size_t kHalfBytes = kHalf * sizeof(int);

    CHECK_DRV(cuInit(0));
    CHECK_CUDA(cudaSetDevice(0));
    if (!has_concurrent_managed_access()) {
        std::puts("driver managed batch API test skipped");
        return 0;
    }
    CHECK_CUDA(cudaFree(nullptr));

    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));

    CUdeviceptr managed = 0;
    CHECK_DRV(cuMemAllocManaged(&managed, kBytes, CU_MEM_ATTACH_GLOBAL));

    CUstream stream = nullptr;
    CHECK_DRV(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

    std::vector<int> input(kCount, 5);
    std::vector<int> output(kCount, 0);
    CHECK_DRV(cuMemcpyHtoD(managed, input.data(), kBytes));

    CUdeviceptr ptrs[] = {
        managed,
        managed + kHalfBytes,
    };
    size_t sizes[] = {
        kHalfBytes,
        kHalfBytes,
    };
    CUmemLocation location = {};
    location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    location.id = device;
    size_t location_indices[] = {0};

    CHECK_DRV(cuMemPrefetchBatchAsync(
        ptrs, sizes, 2, &location, location_indices, 1, 0, stream));
    CHECK_DRV(cuStreamSynchronize(stream));

    CHECK_DRV(cuMemsetD32Async(managed, 17, kCount, stream));
    CHECK_DRV(cuStreamSynchronize(stream));
    CHECK_DRV(cuMemcpyDtoH(output.data(), managed, kBytes));
    if (check_values(output, 17) != 0) {
        return 1;
    }

    CHECK_DRV(cuMemDiscardBatchAsync(ptrs, sizes, 2, 0, stream));
    CHECK_DRV(cuStreamSynchronize(stream));
    CHECK_DRV(cuMemDiscardAndPrefetchBatchAsync(
        ptrs, sizes, 2, &location, location_indices, 1, 0, stream));
    CHECK_DRV(cuStreamSynchronize(stream));

    CHECK_DRV(cuMemsetD32Async(managed, 31, kCount, stream));
    CHECK_DRV(cuStreamSynchronize(stream));
    CHECK_DRV(cuMemcpyDtoH(output.data(), managed, kBytes));
    if (check_values(output, 31) != 0) {
        return 1;
    }

    CHECK_DRV(cuStreamDestroy(stream));
    CHECK_DRV(cuMemFree(managed));
    return 0;
}

int main()
{
    if (run_runtime_batch() != 0) {
        return 1;
    }
    if (run_driver_batch() != 0) {
        return 1;
    }

    std::puts("managed batch API test passed");
    return 0;
}
