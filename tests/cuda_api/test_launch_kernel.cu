#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <iostream>
#include <string>

#define CHECK_CUDA(expr)                                                                                               \
    do {                                                                                                               \
        cudaError_t result = (expr);                                                                                   \
        if (result != cudaSuccess) {                                                                                   \
            std::cerr << #expr << " failed: " << cudaGetErrorString(result) << " (" << static_cast<int>(result)        \
                      << ")" << std::endl;                                                                            \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

__global__ void addKernel(int *c, const int *a, const int *b, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + 1;
    }
}

int main(int argc, char **argv)
{
    const int size = 1000000;
    const int iterations = 10000;
    int *a = new int[size];
    int *dev_a = nullptr;

    for (int i = 0; i < size; i++) {
        a[i] = i;
    }
    for (int i = 0; i < 10; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;

    CHECK_CUDA(cudaFuncSetCacheConfig(addKernel, cudaFuncCachePreferNone));
    CHECK_CUDA(cudaFuncSetSharedMemConfig(addKernel, cudaSharedMemBankSizeDefault));

    int active_blocks = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, addKernel, 128, 0));
    if (active_blocks <= 0) {
        std::cerr << "invalid occupancy block count: " << active_blocks << std::endl;
        return 1;
    }

    int active_blocks_with_flags = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&active_blocks_with_flags, addKernel, 128, 0, 0));
    if (active_blocks_with_flags <= 0) {
        std::cerr << "invalid occupancy block count with flags: " << active_blocks_with_flags << std::endl;
        return 1;
    }

    size_t dynamic_smem = 0;
    CHECK_CUDA(cudaOccupancyAvailableDynamicSMemPerBlock(&dynamic_smem, addKernel, active_blocks, 128));

    const void *kernel_ptr = reinterpret_cast<const void *>(addKernel);
    const char *function_name = nullptr;
    CHECK_CUDA(cudaFuncGetName(&function_name, kernel_ptr));
    if (function_name == nullptr || std::string(function_name).find("addKernel") == std::string::npos) {
        std::cerr << "unexpected runtime function name: "
                  << (function_name ? function_name : "(null)") << std::endl;
        return 1;
    }
    size_t param_count = 0;
    CHECK_CUDA(cudaFuncGetParamCount(kernel_ptr, &param_count));
    if (param_count != 4) {
        std::cerr << "unexpected runtime function param count: " << param_count << std::endl;
        return 1;
    }
    size_t param_offset = 0;
    size_t param_size = 0;
    CHECK_CUDA(cudaFuncGetParamInfo(kernel_ptr, 0, &param_offset, &param_size));
    if (param_offset != 0 || param_size != sizeof(int *)) {
        std::cerr << "unexpected runtime function param info: offset="
                  << param_offset << " size=" << param_size << std::endl;
        return 1;
    }

    // Allocate GPU buffers for three vectors (two input, one output)
    CHECK_CUDA(cudaMalloc((void **)&dev_a, size * sizeof(int)));

    // Copy input vectors from host memory to GPU buffers.
    CHECK_CUDA(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));

    // remove initial overhead
    for (int i = 0; i < 10; i++) {
        addKernel<<<(size + 1023) / 1024, 1024>>>(dev_a, dev_a, dev_a, size);
    }

    if (auto err = cudaGetLastError()) {
        std::cerr << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Launch a kernel on the GPU with one thread for each element.
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        addKernel<<<(size + 1023) / 1024, 1024>>>(dev_a, dev_a, dev_a, size);
    }
    // cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaGetLastError());
    // Calculate the elapsed time in milliseconds
    std::chrono::duration<double, std::milli> elapsed = end - start;

    double totalElapsedTime = elapsed.count();

    //  Calculate the average elapsed time
    double averageElapsedTime = totalElapsedTime / iterations;

    std::cout << "Total elapsed time: " << totalElapsedTime << " ms" << std::endl;
    std::cout << "Average elapsed time: " << averageElapsedTime << " ms" << std::endl;

    // Copy output vector from GPU buffer to host memory.
    CHECK_CUDA(cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(dev_a));

    for (int i = 0; i < 10; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
