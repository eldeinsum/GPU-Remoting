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
__device__ int device_values[4];

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
    size_t symbol_size = 0;
    CHECK_CUDA(cudaGetSymbolSize(&symbol_size, device_value));
    if (symbol_size != sizeof(int)) {
        std::cerr << "cudaGetSymbolSize returned " << symbol_size << std::endl;
        return 1;
    }

    int input = 12345;
    CHECK_CUDA(cudaMemcpyToSymbol(device_value, &input, sizeof(input), 0,
                                  cudaMemcpyDefault));

    int symbol_output = 0;
    CHECK_CUDA(cudaMemcpyFromSymbol(&symbol_output, device_value,
                                    sizeof(symbol_output), 0,
                                    cudaMemcpyDefault));
    if (symbol_output != input) {
        std::cerr << "cudaMemcpyFromSymbol mismatch: got " << symbol_output
                  << " expected " << input << std::endl;
        return 1;
    }

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

    int array_input[4] = {7, 11, 13, 17};
    int array_output[4] = {};
    CHECK_CUDA(cudaGetSymbolSize(&symbol_size, device_values));
    if (symbol_size != sizeof(array_input)) {
        std::cerr << "cudaGetSymbolSize array returned " << symbol_size
                  << std::endl;
        return 1;
    }
    CHECK_CUDA(cudaMemcpyToSymbol(device_values, array_input,
                                  sizeof(array_input), 0,
                                  cudaMemcpyHostToDevice));
    int replacement = 23;
    CHECK_CUDA(cudaMemcpyToSymbol(device_values, &replacement,
                                  sizeof(replacement), 2 * sizeof(int),
                                  cudaMemcpyDefault));
    CHECK_CUDA(cudaMemcpyFromSymbol(array_output, device_values,
                                    sizeof(array_output), 0,
                                    cudaMemcpyDeviceToHost));
    if (array_output[0] != 7 || array_output[1] != 11 ||
        array_output[2] != replacement || array_output[3] != 17) {
        std::cerr << "symbol array host copy mismatch" << std::endl;
        return 1;
    }

    int *device_in = nullptr;
    int *device_copy = nullptr;
    CHECK_CUDA(cudaMalloc(&device_in, sizeof(array_input)));
    CHECK_CUDA(cudaMalloc(&device_copy, sizeof(array_input)));
    CHECK_CUDA(cudaMemcpy(device_in, array_input, sizeof(array_input),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(device_values, device_in,
                                  sizeof(array_input), 0,
                                  cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpyFromSymbol(device_copy, device_values,
                                    sizeof(array_input), 0,
                                    cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(array_output, device_copy, sizeof(array_output),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < 4; ++i) {
        if (array_output[i] != array_input[i]) {
            std::cerr << "symbol array device copy mismatch at " << i
                      << std::endl;
            return 1;
        }
    }

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));
    int async_value = 54321;
    symbol_output = 0;
    CHECK_CUDA(cudaMemcpyToSymbolAsync(device_value, &async_value,
                                       sizeof(async_value), 0,
                                       cudaMemcpyDefault, stream));
    CHECK_CUDA(cudaMemcpyFromSymbolAsync(&symbol_output, device_value,
                                         sizeof(symbol_output), 0,
                                         cudaMemcpyDefault, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(device_copy));
    CHECK_CUDA(cudaFree(device_in));
    if (symbol_output != async_value) {
        std::cerr << "async symbol copy mismatch: got " << symbol_output
                  << " expected " << async_value << std::endl;
        return 1;
    }

    std::cout << "runtime lookup test passed" << std::endl;
    return 0;
}
