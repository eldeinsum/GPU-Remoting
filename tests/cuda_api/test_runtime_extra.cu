#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t result = (call);                                           \
        if (result != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s failed: %s (%d)\n", #call,               \
                         cudaGetErrorString(result), static_cast<int>(result)); \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static int verify_ints(const int *actual, const int *expected, int count)
{
    for (int i = 0; i < count; ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "mismatch at %d: got %d expected %d\n",
                         i, actual[i], expected[i]);
            return 1;
        }
    }
    return 0;
}

int main()
{
    const int count = 32;
    const size_t bytes = count * sizeof(int);

    int *host = nullptr;
    int *device = nullptr;
    int *output = static_cast<int *>(std::calloc(count, sizeof(int)));
    if (output == nullptr) {
        return 1;
    }

    CHECK(cudaMallocHost(reinterpret_cast<void **>(&host), bytes));
    for (int i = 0; i < count; ++i) {
        host[i] = i * 7 + 3;
    }
    CHECK(cudaHostRegister(host, bytes, cudaHostRegisterDefault));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&device), bytes));

    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaEventCreate(&event));
    CHECK(cudaMemcpyAsync(device, host, bytes, cudaMemcpyDefault, stream));
    CHECK(cudaEventRecord(event, stream));
    CHECK(cudaEventSynchronize(event));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(output, device, bytes, cudaMemcpyDefault));
    if (verify_ints(output, host, count) != 0) {
        return 1;
    }

    CHECK(cudaHostUnregister(host));
    CHECK(cudaFree(device));
    CHECK(cudaEventDestroy(event));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFreeHost(host));

    std::vector<int> source(20);
    std::vector<int> copied(20, 0);
    for (int i = 0; i < 20; ++i) {
        source[i] = 1000 + i;
    }

    int *pitched = nullptr;
    size_t pitch = 0;
    const size_t width = 5 * sizeof(int);
    const size_t height = 4;
    CHECK(cudaMallocPitch(reinterpret_cast<void **>(&pitched), &pitch, width, height));
    CHECK(cudaMemcpy2D(pitched, pitch, source.data(), width, width, height, cudaMemcpyDefault));
    CHECK(cudaMemcpy2D(copied.data(), width, pitched, pitch, width, height, cudaMemcpyDefault));
    if (verify_ints(copied.data(), source.data(), 20) != 0) {
        return 1;
    }
    CHECK(cudaFree(pitched));

    std::free(output);
    std::puts("runtime extra API test passed");
    return 0;
}
