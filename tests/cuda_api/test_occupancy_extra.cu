#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

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

static const char kPtx[] = R"ptx(
.version 7.8
.target sm_52
.address_size 64

.visible .entry occupancy_kernel()
{
    ret;
}
)ptx";

__global__ void runtime_occupancy_kernel(int *out)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = blockDim.x;
    }
}

static bool is_expected_driver_cluster_result(CUresult result)
{
    return result == CUDA_ERROR_INVALID_VALUE ||
           result == CUDA_ERROR_NOT_SUPPORTED ||
           result == CUDA_ERROR_INVALID_CLUSTER_SIZE ||
           result == CUDA_ERROR_INVALID_HANDLE;
}

static bool is_expected_runtime_cluster_result(cudaError_t result)
{
    return result == cudaErrorInvalidValue ||
           result == cudaErrorNotSupported ||
           result == cudaErrorInvalidClusterSize ||
           result == cudaErrorInvalidDeviceFunction;
}

static int check_driver_cluster_result(const char *name, CUresult result,
                                       int value, bool require_success)
{
    if (result == CUDA_SUCCESS) {
        if (value <= 0) {
            std::fprintf(stderr, "%s returned non-positive value %d\n", name,
                         value);
            return 1;
        }
        return 0;
    }
    if (!require_success && is_expected_driver_cluster_result(result)) {
        return 0;
    }

    const char *error_name = nullptr;
    cuGetErrorName(result, &error_name);
    std::fprintf(stderr, "%s returned unexpected result: %s (%d)\n", name,
                 error_name == nullptr ? "unknown" : error_name,
                 static_cast<int>(result));
    return 1;
}

static int check_runtime_cluster_result(const char *name, cudaError_t result,
                                        int value, bool require_success)
{
    if (result == cudaSuccess) {
        if (value <= 0) {
            std::fprintf(stderr, "%s returned non-positive value %d\n", name,
                         value);
            return 1;
        }
        return 0;
    }
    if (!require_success && is_expected_runtime_cluster_result(result)) {
        return 0;
    }

    std::fprintf(stderr, "%s returned unexpected result: %s (%d)\n", name,
                 cudaGetErrorString(result), static_cast<int>(result));
    return 1;
}

int main()
{
    CHECK_DRV(cuInit(0));

    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));
    int major = 0;
    CHECK_DRV(cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    const bool require_cluster_success = major >= 9;

    CUcontext context = nullptr;
    CHECK_DRV(cuDevicePrimaryCtxRetain(&context, device));
    CHECK_DRV(cuCtxSetCurrent(context));

    CUmodule module = nullptr;
    CHECK_DRV(cuModuleLoadData(&module, kPtx));
    CUfunction function = nullptr;
    CHECK_DRV(cuModuleGetFunction(&function, module, "occupancy_kernel"));

    int min_grid = 0;
    int block_size = 0;
    CHECK_DRV(cuOccupancyMaxPotentialBlockSize(
        &min_grid, &block_size, function, nullptr, 0, 0));
    if (min_grid <= 0 || block_size <= 0) {
        std::fprintf(stderr,
                     "invalid potential occupancy result: grid=%d block=%d\n",
                     min_grid, block_size);
        return 1;
    }

    int limited_grid = 0;
    int limited_block = 0;
    CHECK_DRV(cuOccupancyMaxPotentialBlockSizeWithFlags(
        &limited_grid, &limited_block, function, nullptr, 0, 128, 0));
    if (limited_grid <= 0 || limited_block <= 0 || limited_block > 128) {
        std::fprintf(stderr,
                     "invalid limited occupancy result: grid=%d block=%d\n",
                     limited_grid, limited_block);
        return 1;
    }

    CUlaunchAttribute driver_attr;
    std::memset(&driver_attr, 0, sizeof(driver_attr));
    driver_attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    driver_attr.value.clusterDim.x = 1;
    driver_attr.value.clusterDim.y = 1;
    driver_attr.value.clusterDim.z = 1;

    CUlaunchConfig driver_config;
    std::memset(&driver_config, 0, sizeof(driver_config));
    driver_config.gridDimX = 1;
    driver_config.gridDimY = 1;
    driver_config.gridDimZ = 1;
    driver_config.blockDimX = 64;
    driver_config.blockDimY = 1;
    driver_config.blockDimZ = 1;
    driver_config.attrs = &driver_attr;
    driver_config.numAttrs = 1;

    int cluster_size = 0;
    CUresult driver_cluster_result = cuOccupancyMaxPotentialClusterSize(
        &cluster_size, function, &driver_config);
    if (check_driver_cluster_result("cuOccupancyMaxPotentialClusterSize",
                                    driver_cluster_result, cluster_size,
                                    require_cluster_success) != 0) {
        return 1;
    }

    int active_clusters = 0;
    driver_cluster_result =
        cuOccupancyMaxActiveClusters(&active_clusters, function, &driver_config);
    if (check_driver_cluster_result("cuOccupancyMaxActiveClusters",
                                    driver_cluster_result, active_clusters,
                                    require_cluster_success) != 0) {
        return 1;
    }

    cudaLaunchAttribute runtime_attr;
    std::memset(&runtime_attr, 0, sizeof(runtime_attr));
    runtime_attr.id = cudaLaunchAttributeClusterDimension;
    runtime_attr.val.clusterDim.x = 1;
    runtime_attr.val.clusterDim.y = 1;
    runtime_attr.val.clusterDim.z = 1;

    cudaLaunchConfig_t runtime_config;
    std::memset(&runtime_config, 0, sizeof(runtime_config));
    runtime_config.gridDim = dim3(1, 1, 1);
    runtime_config.blockDim = dim3(64, 1, 1);
    runtime_config.attrs = &runtime_attr;
    runtime_config.numAttrs = 1;

    cluster_size = 0;
    cudaError_t runtime_cluster_result = cudaOccupancyMaxPotentialClusterSize(
        &cluster_size, reinterpret_cast<const void *>(runtime_occupancy_kernel),
        &runtime_config);
    if (check_runtime_cluster_result("cudaOccupancyMaxPotentialClusterSize",
                                     runtime_cluster_result, cluster_size,
                                     require_cluster_success) != 0) {
        return 1;
    }

    active_clusters = 0;
    runtime_cluster_result = cudaOccupancyMaxActiveClusters(
        &active_clusters,
        reinterpret_cast<const void *>(runtime_occupancy_kernel),
        &runtime_config);
    if (check_runtime_cluster_result("cudaOccupancyMaxActiveClusters",
                                     runtime_cluster_result, active_clusters,
                                     require_cluster_success) != 0) {
        return 1;
    }

    CHECK_DRV(cuModuleUnload(module));
    CHECK_DRV(cuDevicePrimaryCtxRelease(device));

    std::puts("occupancy API test passed");
    return 0;
}
