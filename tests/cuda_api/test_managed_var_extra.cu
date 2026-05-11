#include <cuda_runtime.h>

#include <iostream>

#define CHECK_CUDA(expr)                                                                                              \
    do {                                                                                                              \
        cudaError_t result = (expr);                                                                                  \
        if (result != cudaSuccess) {                                                                                  \
            std::cerr << #expr << " failed: " << cudaGetErrorString(result) << " (" << static_cast<int>(result)      \
                      << ")" << std::endl;                                                                           \
            return 1;                                                                                                 \
        }                                                                                                             \
    } while (0)

__managed__ int managed_value = 5;

__global__ void read_managed(int *out)
{
    out[0] = managed_value;
}

__global__ void write_managed(int value)
{
    managed_value = value;
}

static int launch_and_check(int *d_output, int expected)
{
    CHECK_CUDA(cudaMemset(d_output, 0, sizeof(int)));
    read_managed<<<1, 1>>>(d_output);
    CHECK_CUDA(cudaDeviceSynchronize());

    int output = 0;
    CHECK_CUDA(cudaMemcpy(&output, d_output, sizeof(output), cudaMemcpyDeviceToHost));
    if (output != expected) {
        std::cerr << "managed variable produced " << output << " expected " << expected << std::endl;
        return 1;
    }
    return 0;
}

int main()
{
    int *d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(int)));

    if (managed_value != 5) {
        std::cerr << "initial host managed value was " << managed_value << std::endl;
        return 1;
    }

    if (launch_and_check(d_output, 5) != 0) {
        return 1;
    }

    managed_value = 11;
    if (launch_and_check(d_output, managed_value) != 0) {
        return 1;
    }

    write_managed<<<1, 1>>>(13);
    CHECK_CUDA(cudaDeviceSynchronize());
    if (managed_value != 13) {
        std::cerr << "cudaDeviceSynchronize left host managed value at " << managed_value << std::endl;
        return 1;
    }

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));
    write_managed<<<1, 1, 0, stream>>>(19);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    if (managed_value != 19) {
        std::cerr << "cudaStreamSynchronize left host managed value at " << managed_value << std::endl;
        return 1;
    }

    cudaEvent_t event = nullptr;
    CHECK_CUDA(cudaEventCreate(&event));
    write_managed<<<1, 1, 0, stream>>>(29);
    CHECK_CUDA(cudaEventRecord(event, stream));
    CHECK_CUDA(cudaEventSynchronize(event));
    if (managed_value != 29) {
        std::cerr << "cudaEventSynchronize left host managed value at " << managed_value << std::endl;
        return 1;
    }
    CHECK_CUDA(cudaEventDestroy(event));
    CHECK_CUDA(cudaStreamDestroy(stream));

    size_t symbol_size = 0;
    CHECK_CUDA(cudaGetSymbolSize(&symbol_size, managed_value));
    if (symbol_size != sizeof(int)) {
        std::cerr << "unexpected managed symbol size " << symbol_size << std::endl;
        return 1;
    }

    void *symbol_address = nullptr;
    CHECK_CUDA(cudaGetSymbolAddress(&symbol_address, managed_value));
    if (symbol_address == nullptr) {
        std::cerr << "cudaGetSymbolAddress returned null" << std::endl;
        return 1;
    }

    managed_value = 31;
    int host_dirty_symbol_copy = 0;
    CHECK_CUDA(cudaMemcpyFromSymbol(&host_dirty_symbol_copy, managed_value, sizeof(host_dirty_symbol_copy)));
    if (host_dirty_symbol_copy != managed_value) {
        std::cerr << "cudaMemcpyFromSymbol saw " << host_dirty_symbol_copy << " after host write " << managed_value
                  << std::endl;
        return 1;
    }

    int updated = 17;
    CHECK_CUDA(cudaMemcpyToSymbol(managed_value, &updated, sizeof(updated)));
    CHECK_CUDA(cudaDeviceSynchronize());
    if (managed_value != updated) {
        std::cerr << "cudaMemcpyToSymbol left host managed value at " << managed_value << std::endl;
        return 1;
    }
    if (launch_and_check(d_output, updated) != 0) {
        return 1;
    }

    int copied_back = 0;
    CHECK_CUDA(cudaMemcpyFromSymbol(&copied_back, managed_value, sizeof(copied_back)));
    if (copied_back != updated) {
        std::cerr << "cudaMemcpyFromSymbol returned " << copied_back << " expected " << updated << std::endl;
        return 1;
    }

    int via_address = 23;
    CHECK_CUDA(cudaMemcpy(symbol_address, &via_address, sizeof(via_address), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    if (managed_value != via_address) {
        std::cerr << "cudaMemcpy through symbol address left host managed value at " << managed_value << std::endl;
        return 1;
    }
    if (launch_and_check(d_output, via_address) != 0) {
        return 1;
    }

    cudaStream_t async_stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&async_stream));

    managed_value = 37;
    int async_symbol_copy = 0;
    CHECK_CUDA(cudaMemcpyFromSymbolAsync(&async_symbol_copy, managed_value, sizeof(async_symbol_copy), 0,
                                         cudaMemcpyDeviceToHost, async_stream));
    CHECK_CUDA(cudaStreamSynchronize(async_stream));
    if (async_symbol_copy != managed_value) {
        std::cerr << "cudaMemcpyFromSymbolAsync saw " << async_symbol_copy << " after host write " << managed_value
                  << std::endl;
        return 1;
    }

    int async_address_copy = 0;
    managed_value = 41;
    CHECK_CUDA(cudaMemcpyAsync(&async_address_copy, symbol_address, sizeof(async_address_copy), cudaMemcpyDeviceToHost,
                               async_stream));
    CHECK_CUDA(cudaStreamSynchronize(async_stream));
    if (async_address_copy != managed_value) {
        std::cerr << "cudaMemcpyAsync from managed address saw " << async_address_copy << " after host write "
                  << managed_value << std::endl;
        return 1;
    }

    int async_to_symbol = 43;
    CHECK_CUDA(cudaMemcpyToSymbolAsync(managed_value, &async_to_symbol, sizeof(async_to_symbol), 0,
                                       cudaMemcpyHostToDevice, async_stream));
    CHECK_CUDA(cudaStreamSynchronize(async_stream));
    if (managed_value != async_to_symbol) {
        std::cerr << "cudaMemcpyToSymbolAsync left host managed value at " << managed_value << std::endl;
        return 1;
    }
    if (launch_and_check(d_output, async_to_symbol) != 0) {
        return 1;
    }

    int *d_async_source = nullptr;
    CHECK_CUDA(cudaMalloc(&d_async_source, sizeof(int)));
    int async_d2d_value = 47;
    CHECK_CUDA(cudaMemcpyAsync(d_async_source, &async_d2d_value, sizeof(async_d2d_value), cudaMemcpyHostToDevice,
                               async_stream));
    CHECK_CUDA(cudaMemcpyAsync(symbol_address, d_async_source, sizeof(async_d2d_value), cudaMemcpyDeviceToDevice,
                               async_stream));
    CHECK_CUDA(cudaStreamSynchronize(async_stream));
    if (managed_value != async_d2d_value) {
        std::cerr << "cudaMemcpyAsync device-to-device left host managed value at " << managed_value << std::endl;
        return 1;
    }
    if (launch_and_check(d_output, async_d2d_value) != 0) {
        return 1;
    }

    CHECK_CUDA(cudaFree(d_async_source));
    CHECK_CUDA(cudaStreamDestroy(async_stream));
    CHECK_CUDA(cudaFree(d_output));
    std::cout << "managed variable test passed" << std::endl;
    return 0;
}
