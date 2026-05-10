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
        if (std::fabs(got[i] - want[i]) > 1e-4f) {
            std::fprintf(stderr, "%s mismatch at %zu: got %.6f want %.6f\n",
                         label, i, got[i], want[i]);
            std::exit(1);
        }
    }
}

static void expect_pointer_mode(cublasHandle_t handle,
                                cublasPointerMode_t expected) {
    cublasPointerMode_t actual;
    CHECK_CUBLAS(cublasGetPointerMode(handle, &actual));
    if (actual != expected) {
        std::fprintf(stderr, "pointer mode mismatch: got %d want %d\n",
                     static_cast<int>(actual), static_cast<int>(expected));
        std::exit(1);
    }
}

static void test_scal(cublasHandle_t handle) {
    std::vector<float> host = {1.0f, 2.0f, 3.0f, 4.0f};
    float *dev = nullptr;
    CHECK_CUDA(cudaMalloc(&dev, host.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dev, host.data(), host.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    expect_pointer_mode(handle, CUBLAS_POINTER_MODE_HOST);
    float host_alpha = 2.0f;
    CHECK_CUBLAS(cublasSscal(handle, static_cast<int>(host.size()),
                             &host_alpha, dev, 1));
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> out(host.size());
    CHECK_CUDA(cudaMemcpy(out.data(), dev, out.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    expect_close(out, {2.0f, 4.0f, 6.0f, 8.0f}, "host scal");

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    expect_pointer_mode(handle, CUBLAS_POINTER_MODE_DEVICE);

    float *dev_alpha = nullptr;
    float device_alpha = 0.5f;
    CHECK_CUDA(cudaMalloc(&dev_alpha, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dev_alpha, &device_alpha, sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUBLAS(cublasSscal(handle, static_cast<int>(host.size()),
                             dev_alpha, dev, 1));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(out.data(), dev, out.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    expect_close(out, host, "device scal");

    CHECK_CUDA(cudaFree(dev_alpha));
    CHECK_CUDA(cudaFree(dev));
}

static void test_sgemm(cublasHandle_t handle) {
    const int n = 2;
    const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> identity = {1.0f, 0.0f, 0.0f, 1.0f};
    const std::vector<float> zero = {0.0f, 0.0f, 0.0f, 0.0f};
    float *dev_a = nullptr;
    float *dev_b = nullptr;
    float *dev_c = nullptr;
    float *dev_alpha = nullptr;
    float *dev_beta = nullptr;

    CHECK_CUDA(cudaMalloc(&dev_a, a.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dev_b, identity.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dev_c, zero.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dev_alpha, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dev_beta, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dev_a, a.data(), a.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_b, identity.data(), identity.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_c, zero.data(), zero.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    float alpha = 2.0f;
    float beta = 0.0f;
    CHECK_CUDA(cudaMemcpy(dev_alpha, &alpha, sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_beta, &beta, sizeof(float),
                          cudaMemcpyHostToDevice));

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                             dev_alpha, dev_a, n, dev_b, n, dev_beta,
                             dev_c, n));
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> out(zero.size());
    CHECK_CUDA(cudaMemcpy(out.data(), dev_c, out.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    expect_close(out, {2.0f, 4.0f, 6.0f, 8.0f}, "device sgemm");

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    expect_pointer_mode(handle, CUBLAS_POINTER_MODE_HOST);

    CHECK_CUDA(cudaFree(dev_beta));
    CHECK_CUDA(cudaFree(dev_alpha));
    CHECK_CUDA(cudaFree(dev_c));
    CHECK_CUDA(cudaFree(dev_b));
    CHECK_CUDA(cudaFree(dev_a));
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    test_scal(handle);
    test_sgemm(handle);
    CHECK_CUBLAS(cublasDestroy(handle));
    std::puts("PASS");
    return 0;
}
