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

__global__ void add_kernel(int *data, int count, int delta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] += delta;
    }
}

static int require(bool condition, const char *message)
{
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        return 1;
    }
    return 0;
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

static int run_runtime_managed()
{
    constexpr int kCount = 64;
    constexpr size_t kBytes = kCount * sizeof(int);

    CHECK_CUDA(cudaSetDevice(0));

    int *managed = nullptr;
    CHECK_CUDA(cudaMallocManaged(
        reinterpret_cast<void **>(&managed), kBytes, cudaMemAttachGlobal));

    std::vector<int> input(kCount, 37);
    std::vector<int> output(kCount, 0);
    CHECK_CUDA(cudaMemcpy(
        managed, input.data(), kBytes, cudaMemcpyHostToDevice));

    cudaMemLocation location = {};
    location.type = cudaMemLocationTypeDevice;
    location.id = 0;

    CHECK_CUDA(cudaMemAdvise(
        managed, kBytes, cudaMemAdviseSetReadMostly, location));
    CHECK_CUDA(cudaMemAdvise(
        managed, kBytes, cudaMemAdviseSetPreferredLocation, location));
    CHECK_CUDA(cudaMemPrefetchAsync(managed, kBytes, location, 0, nullptr));
    CHECK_CUDA(cudaDeviceSynchronize());

    add_kernel<<<1, kCount>>>(managed, kCount, 5);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(
        output.data(), managed, kBytes, cudaMemcpyDeviceToHost));
    if (check_values(output, 42) != 0) {
        return 1;
    }

    int read_mostly = 0;
    CHECK_CUDA(cudaMemRangeGetAttribute(
        &read_mostly, sizeof(read_mostly),
        cudaMemRangeAttributeReadMostly, managed, kBytes));
    if (require(read_mostly == 1, "runtime read-mostly attribute mismatch") != 0) {
        return 1;
    }

    int read_mostly_multi = 0;
    cudaMemLocationType preferred_type = cudaMemLocationTypeInvalid;
    int preferred_id = -1;
    void *data[] = {&read_mostly_multi, &preferred_type, &preferred_id};
    size_t data_sizes[] = {
        sizeof(read_mostly_multi),
        sizeof(preferred_type),
        sizeof(preferred_id),
    };
    cudaMemRangeAttribute attributes[] = {
        cudaMemRangeAttributeReadMostly,
        cudaMemRangeAttributePreferredLocationType,
        cudaMemRangeAttributePreferredLocationId,
    };
    CHECK_CUDA(cudaMemRangeGetAttributes(
        data, data_sizes, attributes, 3, managed, kBytes));
    if (require(read_mostly_multi == 1,
                "runtime multi read-mostly attribute mismatch") != 0 ||
        require(preferred_type == cudaMemLocationTypeDevice,
                "runtime preferred location type mismatch") != 0 ||
        require(preferred_id == 0,
                "runtime preferred location id mismatch") != 0) {
        return 1;
    }

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaStreamAttachMemAsync(
        stream, managed, kBytes, cudaMemAttachSingle));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(managed));
    return 0;
}

static int run_driver_managed()
{
    constexpr int kCount = 64;
    constexpr size_t kBytes = kCount * sizeof(int);

    CHECK_DRV(cuInit(0));
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaFree(nullptr));

    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));

    CUdeviceptr managed = 0;
    CHECK_DRV(cuMemAllocManaged(&managed, kBytes, CU_MEM_ATTACH_GLOBAL));

    std::vector<int> input(kCount, 11);
    std::vector<int> output(kCount, 0);
    CHECK_DRV(cuMemcpyHtoD(managed, input.data(), kBytes));

    CUmemLocation location = {};
    location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    location.id = device;

    CHECK_DRV(cuMemAdvise(
        managed, kBytes, CU_MEM_ADVISE_SET_READ_MOSTLY, location));
    CHECK_DRV(cuMemAdvise(
        managed, kBytes, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, location));

    CUstream stream = nullptr;
    CHECK_DRV(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
    CHECK_DRV(cuMemPrefetchAsync(managed, kBytes, location, 0, stream));
    CHECK_DRV(cuStreamSynchronize(stream));

    CHECK_DRV(cuMemsetD32(managed, 23, kCount));
    CHECK_DRV(cuMemcpyDtoH(output.data(), managed, kBytes));
    if (check_values(output, 23) != 0) {
        return 1;
    }

    int read_mostly = 0;
    CHECK_DRV(cuMemRangeGetAttribute(
        &read_mostly, sizeof(read_mostly),
        CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY, managed, kBytes));
    if (require(read_mostly == 1, "driver read-mostly attribute mismatch") != 0) {
        return 1;
    }

    int read_mostly_multi = 0;
    CUmemLocationType preferred_type = CU_MEM_LOCATION_TYPE_INVALID;
    int preferred_id = -1;
    void *data[] = {&read_mostly_multi, &preferred_type, &preferred_id};
    size_t data_sizes[] = {
        sizeof(read_mostly_multi),
        sizeof(preferred_type),
        sizeof(preferred_id),
    };
    CUmem_range_attribute attributes[] = {
        CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY,
        CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_TYPE,
        CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_ID,
    };
    CHECK_DRV(cuMemRangeGetAttributes(
        data, data_sizes, attributes, 3, managed, kBytes));
    if (require(read_mostly_multi == 1,
                "driver multi read-mostly attribute mismatch") != 0 ||
        require(preferred_type == CU_MEM_LOCATION_TYPE_DEVICE,
                "driver preferred location type mismatch") != 0 ||
        require(preferred_id == device,
                "driver preferred location id mismatch") != 0) {
        return 1;
    }

    CHECK_DRV(cuStreamAttachMemAsync(
        stream, managed, kBytes, CU_MEM_ATTACH_SINGLE));
    CHECK_DRV(cuStreamSynchronize(stream));
    CHECK_DRV(cuStreamDestroy(stream));
    CHECK_DRV(cuMemFree(managed));
    return 0;
}

int main()
{
    if (run_runtime_managed() != 0) {
        return 1;
    }
    if (run_driver_managed() != 0) {
        return 1;
    }

    std::puts("managed memory API test passed");
    return 0;
}
