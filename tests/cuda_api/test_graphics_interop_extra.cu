#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdio>

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

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t result = (call);                                           \
        if (result != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s failed: %s (%d)\n", #call,               \
                         cudaGetErrorName(result), static_cast<int>(result));  \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static int expect_driver(CUresult actual, CUresult expected, const char *call)
{
    if (actual == expected) {
        return 0;
    }

    const char *actual_name = nullptr;
    const char *expected_name = nullptr;
    cuGetErrorName(actual, &actual_name);
    cuGetErrorName(expected, &expected_name);
    std::fprintf(stderr, "%s returned %s (%d), expected %s (%d)\n", call,
                 actual_name == nullptr ? "unknown" : actual_name,
                 static_cast<int>(actual),
                 expected_name == nullptr ? "unknown" : expected_name,
                 static_cast<int>(expected));
    return 1;
}

static int expect_runtime(cudaError_t actual, cudaError_t expected,
                          const char *call)
{
    if (actual == expected) {
        return 0;
    }

    std::fprintf(stderr, "%s returned %s (%d), expected %s (%d)\n", call,
                 cudaGetErrorName(actual), static_cast<int>(actual),
                 cudaGetErrorName(expected), static_cast<int>(expected));
    return 1;
}

int main()
{
    CHECK_DRV(cuInit(0));
    CHECK_CUDA(cudaFree(nullptr));

    int status = 0;

    CUgraphicsResource driver_resource = nullptr;
    CUgraphicsResource driver_resources[1] = {driver_resource};
    CUarray driver_array = nullptr;
    CUmipmappedArray driver_mipmapped_array = nullptr;
    CUdeviceptr driver_ptr = 0;
    size_t driver_size = 0;

    status |= expect_driver(
        cuGraphicsResourceSetMapFlags(driver_resource, 0),
        CUDA_ERROR_INVALID_HANDLE, "cuGraphicsResourceSetMapFlags");
    status |= expect_driver(
        cuGraphicsMapResources(1, driver_resources, nullptr),
        CUDA_ERROR_INVALID_HANDLE, "cuGraphicsMapResources");
    status |= expect_driver(
        cuGraphicsResourceGetMappedPointer(&driver_ptr, &driver_size,
                                           driver_resource),
        CUDA_ERROR_INVALID_HANDLE, "cuGraphicsResourceGetMappedPointer");
    status |= expect_driver(
        cuGraphicsSubResourceGetMappedArray(&driver_array, driver_resource, 0,
                                            0),
        CUDA_ERROR_INVALID_HANDLE, "cuGraphicsSubResourceGetMappedArray");
    status |= expect_driver(
        cuGraphicsResourceGetMappedMipmappedArray(&driver_mipmapped_array,
                                                  driver_resource),
        CUDA_ERROR_INVALID_HANDLE,
        "cuGraphicsResourceGetMappedMipmappedArray");
    status |= expect_driver(
        cuGraphicsUnmapResources(1, driver_resources, nullptr),
        CUDA_ERROR_INVALID_HANDLE, "cuGraphicsUnmapResources");
    status |= expect_driver(
        cuGraphicsUnregisterResource(driver_resource),
        CUDA_ERROR_INVALID_HANDLE, "cuGraphicsUnregisterResource");

    cudaGraphicsResource_t runtime_resource = nullptr;
    cudaGraphicsResource_t runtime_resources[1] = {runtime_resource};
    cudaArray_t runtime_array = nullptr;
    cudaMipmappedArray_t runtime_mipmapped_array = nullptr;
    void *runtime_ptr = nullptr;
    size_t runtime_size = 0;

    status |= expect_runtime(
        cudaGraphicsResourceSetMapFlags(runtime_resource, 0),
        cudaErrorInvalidResourceHandle,
        "cudaGraphicsResourceSetMapFlags");
    status |= expect_runtime(
        cudaGraphicsMapResources(1, runtime_resources, nullptr),
        cudaErrorInvalidResourceHandle, "cudaGraphicsMapResources");
    status |= expect_runtime(
        cudaGraphicsResourceGetMappedPointer(&runtime_ptr, &runtime_size,
                                             runtime_resource),
        cudaErrorInvalidResourceHandle,
        "cudaGraphicsResourceGetMappedPointer");
    status |= expect_runtime(
        cudaGraphicsSubResourceGetMappedArray(&runtime_array, runtime_resource,
                                              0, 0),
        cudaErrorInvalidResourceHandle,
        "cudaGraphicsSubResourceGetMappedArray");
    status |= expect_runtime(
        cudaGraphicsResourceGetMappedMipmappedArray(&runtime_mipmapped_array,
                                                    runtime_resource),
        cudaErrorInvalidResourceHandle,
        "cudaGraphicsResourceGetMappedMipmappedArray");
    status |= expect_runtime(
        cudaGraphicsUnmapResources(1, runtime_resources, nullptr),
        cudaErrorInvalidResourceHandle, "cudaGraphicsUnmapResources");
    status |= expect_runtime(
        cudaGraphicsUnregisterResource(runtime_resource),
        cudaErrorInvalidResourceHandle, "cudaGraphicsUnregisterResource");

    if (status == 0) {
        std::puts("graphics interop invalid-handle checks passed");
    }
    return status;
}
