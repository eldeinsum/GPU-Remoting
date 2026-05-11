#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(err__));              \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

#define CHECK_CUBLAS(call)                                                   \
    do {                                                                     \
        cublasStatus_t err__ = (call);                                       \
        if (err__ != CUBLAS_STATUS_SUCCESS) {                                \
            std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__,      \
                         __LINE__, static_cast<int>(err__));                 \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

static void expect_close(const std::vector<float> &got,
                         const std::vector<float> &want,
                         const char *label) {
    if (got.size() != want.size()) {
        std::fprintf(stderr, "%s size mismatch\n", label);
        std::exit(1);
    }
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(got[i] - want[i]) > 1e-5f) {
            std::fprintf(stderr, "%s mismatch at %zu: got %.6f want %.6f\n",
                         label, i, got[i], want[i]);
            std::exit(1);
        }
    }
}

static void test_vector_32() {
    const int n = 3;
    std::vector<float> host = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f};
    float *device = nullptr;
    CHECK_CUDA(cudaMalloc(&device, n * sizeof(float)));

    CHECK_CUBLAS(cublasSetVector(n, sizeof(float), host.data(), 2, device, 1));

    std::vector<float> dense(n, 0.0f);
    CHECK_CUBLAS(cublasGetVector(n, sizeof(float), device, 1, dense.data(), 1));
    expect_close(dense, {1.0f, 2.0f, 3.0f}, "dense getVector");

    std::vector<float> strided = {-9.0f, -9.0f, -9.0f, -9.0f, -9.0f};
    CHECK_CUBLAS(cublasGetVector(n, sizeof(float), device, 1, strided.data(), 2));
    expect_close(strided, {1.0f, -9.0f, 2.0f, -9.0f, 3.0f},
                 "strided getVector");

    CHECK_CUDA(cudaFree(device));
}

static void test_vector_64() {
    const long long n = 3;
    std::vector<float> host = {4.0f, -1.0f, 5.0f, -2.0f, 6.0f};
    float *device = nullptr;
    CHECK_CUDA(cudaMalloc(&device, static_cast<size_t>(n) * sizeof(float)));

    CHECK_CUBLAS(cublasSetVector_64(n, sizeof(float), host.data(), 2, device, 1));

    std::vector<float> dense(static_cast<size_t>(n), 0.0f);
    CHECK_CUBLAS(cublasGetVector_64(n, sizeof(float), device, 1, dense.data(),
                                    1));
    expect_close(dense, {4.0f, 5.0f, 6.0f}, "dense getVector_64");

    CHECK_CUDA(cudaFree(device));
}

int main() {
    test_vector_32();
    test_vector_64();
    std::puts("cuBLAS vector API test passed");
    return 0;
}
