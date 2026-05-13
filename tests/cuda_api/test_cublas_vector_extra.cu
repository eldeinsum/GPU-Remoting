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

static void expect_matrix_close(const std::vector<float> &got, int ld, int rows,
                                int cols, const std::vector<float> &want,
                                const char *label) {
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            const size_t got_index = static_cast<size_t>(col * ld + row);
            const size_t want_index = static_cast<size_t>(col * rows + row);
            if (std::fabs(got[got_index] - want[want_index]) > 1e-5f) {
                std::fprintf(stderr,
                             "%s mismatch at row %d col %d: got %.6f want %.6f\n",
                             label, row, col, got[got_index], want[want_index]);
                std::exit(1);
            }
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

static void test_matrix_32() {
    const int rows = 2;
    const int cols = 3;
    const int lda = 4;
    const int device_ld = 3;
    const int host_ld = 5;
    const std::vector<float> values = {1.0f, 2.0f, 3.0f,
                                       4.0f, 5.0f, 6.0f};
    std::vector<float> host(static_cast<size_t>(lda * cols), -7.0f);
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            host[static_cast<size_t>(col * lda + row)] =
                values[static_cast<size_t>(col * rows + row)];
        }
    }

    float *device = nullptr;
    CHECK_CUDA(cudaMalloc(&device, static_cast<size_t>(device_ld * cols) *
                                       sizeof(float)));
    CHECK_CUBLAS(cublasSetMatrix(rows, cols, sizeof(float), host.data(), lda,
                                 device, device_ld));

    std::vector<float> out(static_cast<size_t>(host_ld * cols), -9.0f);
    CHECK_CUBLAS(cublasGetMatrix(rows, cols, sizeof(float), device, device_ld,
                                 out.data(), host_ld));
    expect_matrix_close(out, host_ld, rows, cols, values, "getMatrix");

    CHECK_CUDA(cudaFree(device));
}

static void test_vector_async_32() {
    const int n = 3;
    std::vector<float> host = {11.0f, -1.0f, 12.0f, -2.0f, 13.0f};
    float *device = nullptr;
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaMalloc(&device, n * sizeof(float)));
    CHECK_CUDA(cudaStreamCreate(&stream));

    CHECK_CUBLAS(
        cublasSetVectorAsync(n, sizeof(float), host.data(), 2, device, 1,
                             stream));

    std::vector<float> strided = {-9.0f, -9.0f, -9.0f, -9.0f, -9.0f};
    CHECK_CUBLAS(
        cublasGetVectorAsync(n, sizeof(float), device, 1, strided.data(), 2,
                             stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    expect_close(strided, {11.0f, -9.0f, 12.0f, -9.0f, 13.0f},
                 "strided getVectorAsync");

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(device));
}

static void test_matrix_async_32() {
    const int rows = 2;
    const int cols = 2;
    const int lda = 3;
    const int device_ld = 4;
    const int host_ld = 3;
    const std::vector<float> values = {21.0f, 22.0f, 23.0f, 24.0f};
    std::vector<float> host(static_cast<size_t>(lda * cols), -7.0f);
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            host[static_cast<size_t>(col * lda + row)] =
                values[static_cast<size_t>(col * rows + row)];
        }
    }

    float *device = nullptr;
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaMalloc(&device, static_cast<size_t>(device_ld * cols) *
                                       sizeof(float)));
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUBLAS(cublasSetMatrixAsync(rows, cols, sizeof(float), host.data(),
                                      lda, device, device_ld, stream));

    std::vector<float> out(static_cast<size_t>(host_ld * cols), -9.0f);
    CHECK_CUBLAS(cublasGetMatrixAsync(rows, cols, sizeof(float), device,
                                      device_ld, out.data(), host_ld, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    expect_matrix_close(out, host_ld, rows, cols, values, "getMatrixAsync");

    CHECK_CUDA(cudaStreamDestroy(stream));
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

static void test_vector_async_64() {
    const long long n = 3;
    std::vector<float> host = {14.0f, -1.0f, 15.0f, -2.0f, 16.0f};
    float *device = nullptr;
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaMalloc(&device, static_cast<size_t>(n) * sizeof(float)));
    CHECK_CUDA(cudaStreamCreate(&stream));

    CHECK_CUBLAS(cublasSetVectorAsync_64(n, sizeof(float), host.data(), 2,
                                         device, 1, stream));

    std::vector<float> dense(static_cast<size_t>(n), 0.0f);
    CHECK_CUBLAS(cublasGetVectorAsync_64(n, sizeof(float), device, 1,
                                         dense.data(), 1, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    expect_close(dense, {14.0f, 15.0f, 16.0f}, "dense getVectorAsync_64");

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(device));
}

static void test_matrix_64() {
    const long long rows = 2;
    const long long cols = 2;
    const long long lda = 3;
    const long long device_ld = 4;
    const long long host_ld = 3;
    const std::vector<float> values = {7.0f, 8.0f, 9.0f, 10.0f};
    std::vector<float> host(static_cast<size_t>(lda * cols), -7.0f);
    for (long long col = 0; col < cols; ++col) {
        for (long long row = 0; row < rows; ++row) {
            host[static_cast<size_t>(col * lda + row)] =
                values[static_cast<size_t>(col * rows + row)];
        }
    }

    float *device = nullptr;
    CHECK_CUDA(cudaMalloc(&device, static_cast<size_t>(device_ld * cols) *
                                       sizeof(float)));
    CHECK_CUBLAS(cublasSetMatrix_64(rows, cols, sizeof(float), host.data(), lda,
                                    device, device_ld));

    std::vector<float> out(static_cast<size_t>(host_ld * cols), -9.0f);
    CHECK_CUBLAS(cublasGetMatrix_64(rows, cols, sizeof(float), device,
                                    device_ld, out.data(), host_ld));
    expect_matrix_close(out, static_cast<int>(host_ld), static_cast<int>(rows),
                        static_cast<int>(cols), values, "getMatrix_64");

    CHECK_CUDA(cudaFree(device));
}

static void test_matrix_async_64() {
    const long long rows = 2;
    const long long cols = 3;
    const long long lda = 4;
    const long long device_ld = 3;
    const long long host_ld = 5;
    const std::vector<float> values = {31.0f, 32.0f, 33.0f,
                                       34.0f, 35.0f, 36.0f};
    std::vector<float> host(static_cast<size_t>(lda * cols), -7.0f);
    for (long long col = 0; col < cols; ++col) {
        for (long long row = 0; row < rows; ++row) {
            host[static_cast<size_t>(col * lda + row)] =
                values[static_cast<size_t>(col * rows + row)];
        }
    }

    float *device = nullptr;
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaMalloc(&device, static_cast<size_t>(device_ld * cols) *
                                       sizeof(float)));
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUBLAS(cublasSetMatrixAsync_64(rows, cols, sizeof(float),
                                         host.data(), lda, device, device_ld,
                                         stream));

    std::vector<float> out(static_cast<size_t>(host_ld * cols), -9.0f);
    CHECK_CUBLAS(cublasGetMatrixAsync_64(rows, cols, sizeof(float), device,
                                         device_ld, out.data(), host_ld,
                                         stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    expect_matrix_close(out, static_cast<int>(host_ld), static_cast<int>(rows),
                        static_cast<int>(cols), values, "getMatrixAsync_64");

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(device));
}

int main() {
    test_vector_32();
    test_matrix_32();
    test_vector_async_32();
    test_matrix_async_32();
    test_vector_64();
    test_matrix_64();
    test_vector_async_64();
    test_matrix_async_64();
    std::puts("cuBLAS vector API test passed");
    return 0;
}
