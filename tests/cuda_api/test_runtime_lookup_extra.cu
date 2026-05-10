#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#define CHECK_CUDA(expr)                                                                       \
    do {                                                                                       \
        cudaError_t err = (expr);                                                              \
        if (err != cudaSuccess) {                                                              \
            std::cerr << #expr << " failed: " << cudaGetErrorString(err) << " (" << err       \
                      << ")" << std::endl;                                                    \
            return 1;                                                                          \
        }                                                                                      \
    } while (0)

#define CHECK_DRV(expr)                                                                        \
    do {                                                                                       \
        CUresult err = (expr);                                                                 \
        if (err != CUDA_SUCCESS) {                                                             \
            const char *name = nullptr;                                                        \
            cuGetErrorName(err, &name);                                                        \
            std::cerr << #expr << " failed: " << (name ? name : "unknown") << std::endl;      \
            return 1;                                                                          \
        }                                                                                      \
    } while (0)

__device__ int device_value;

__global__ void read_device_value(int *out)
{
    *out = device_value;
}

int main()
{
    CHECK_CUDA(cudaFree(nullptr));

    void *entry = nullptr;
    cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSymbolNotFound;
    CHECK_CUDA(cudaGetDriverEntryPointByVersion(
        "cuInit", &entry, CUDART_VERSION, cudaEnableDefault, &status));
    if (entry == nullptr || status != cudaDriverEntryPointSuccess) {
        std::cerr << "driver entry point lookup failed" << std::endl;
        return 1;
    }

    CUmoduleLoadingMode mode;
    CHECK_DRV(cuModuleGetLoadingMode(&mode));
    if (mode != CU_MODULE_EAGER_LOADING && mode != CU_MODULE_LAZY_LOADING) {
        std::cerr << "unexpected module loading mode: " << static_cast<int>(mode) << std::endl;
        return 1;
    }

    void *symbol_ptr = nullptr;
    CHECK_CUDA(cudaGetSymbolAddress(&symbol_ptr, device_value));
    if (symbol_ptr == nullptr) {
        std::cerr << "cudaGetSymbolAddress returned null" << std::endl;
        return 1;
    }

    int input = 12345;
    CHECK_CUDA(cudaMemcpy(symbol_ptr, &input, sizeof(input), cudaMemcpyHostToDevice));

    int *device_out = nullptr;
    CHECK_CUDA(cudaMalloc(&device_out, sizeof(int)));
    read_device_value<<<1, 1>>>(device_out);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    int output = 0;
    CHECK_CUDA(cudaMemcpy(&output, device_out, sizeof(output), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(device_out));

    if (output != input) {
        std::cerr << "symbol value mismatch: got " << output << " expected " << input
                  << std::endl;
        return 1;
    }

    std::cout << "runtime lookup test passed" << std::endl;
    return 0;
}
