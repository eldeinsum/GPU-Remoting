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

static bool close_value(float got, float want) {
    return std::fabs(got - want) <= 1e-5f;
}

static bool close_value(double got, double want) {
    return std::fabs(got - want) <= 1e-10;
}

static bool close_value(cuComplex got, cuComplex want) {
    return close_value(got.x, want.x) && close_value(got.y, want.y);
}

static bool close_value(cuDoubleComplex got, cuDoubleComplex want) {
    return close_value(got.x, want.x) && close_value(got.y, want.y);
}

template <typename T>
static T *to_device(const std::vector<T> &host) {
    T *device = nullptr;
    CHECK_CUDA(cudaMalloc(&device, host.size() * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
    return device;
}

template <typename T>
static std::vector<T> from_device(const T *device, size_t count) {
    std::vector<T> host(count);
    CHECK_CUDA(cudaMemcpy(host.data(), device, count * sizeof(T),
                          cudaMemcpyDeviceToHost));
    return host;
}

template <typename T>
static void expect_vector(const std::vector<T> &got, const std::vector<T> &want,
                          const char *label) {
    if (got.size() != want.size()) {
        std::fprintf(stderr, "%s size mismatch\n", label);
        std::exit(1);
    }
    for (size_t i = 0; i < got.size(); ++i) {
        if (!close_value(got[i], want[i])) {
            std::fprintf(stderr, "%s mismatch at %zu\n", label, i);
            std::exit(1);
        }
    }
}

template <typename T, typename CopyFn, typename Copy64Fn>
static void run_copy_case(cublasHandle_t handle, const std::vector<T> &src,
                          CopyFn copy_fn, Copy64Fn copy64_fn,
                          const char *label) {
    const int n = static_cast<int>(src.size());
    std::vector<T> empty(src.size());
    T *x = to_device(src);
    T *y = to_device(empty);

    CHECK_CUBLAS(copy_fn(handle, n, x, 1, y, 1));
    expect_vector(from_device(y, src.size()), src, label);

    CHECK_CUDA(cudaMemset(y, 0, src.size() * sizeof(T)));
    CHECK_CUBLAS(copy64_fn(handle, static_cast<long long>(n), x, 1, y, 1));
    expect_vector(from_device(y, src.size()), src, label);

    CHECK_CUDA(cudaFree(x));
    CHECK_CUDA(cudaFree(y));
}

template <typename T, typename SwapFn, typename Swap64Fn>
static void run_swap_case(cublasHandle_t handle, const std::vector<T> &left,
                          const std::vector<T> &right, SwapFn swap_fn,
                          Swap64Fn swap64_fn, const char *label) {
    const int n = static_cast<int>(left.size());
    T *x = to_device(left);
    T *y = to_device(right);

    CHECK_CUBLAS(swap_fn(handle, n, x, 1, y, 1));
    expect_vector(from_device(x, left.size()), right, label);
    expect_vector(from_device(y, right.size()), left, label);

    CHECK_CUBLAS(swap64_fn(handle, static_cast<long long>(n), x, 1, y, 1));
    expect_vector(from_device(x, left.size()), left, label);
    expect_vector(from_device(y, right.size()), right, label);

    CHECK_CUDA(cudaFree(x));
    CHECK_CUDA(cudaFree(y));
}

static void test_copy(cublasHandle_t handle) {
    run_copy_case<float>(
        handle, {1.0f, 2.0f, 3.0f}, cublasScopy_v2, cublasScopy_v2_64,
        "Scopy");
    run_copy_case<double>(
        handle, {1.0, 2.0, 3.0}, cublasDcopy_v2, cublasDcopy_v2_64,
        "Dcopy");
    run_copy_case<cuComplex>(
        handle, {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}}, cublasCcopy_v2,
        cublasCcopy_v2_64, "Ccopy");
    run_copy_case<cuDoubleComplex>(
        handle, {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}, cublasZcopy_v2,
        cublasZcopy_v2_64, "Zcopy");
}

static void test_swap(cublasHandle_t handle) {
    run_swap_case<float>(
        handle, {1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, cublasSswap_v2,
        cublasSswap_v2_64, "Sswap");
    run_swap_case<double>(
        handle, {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, cublasDswap_v2,
        cublasDswap_v2_64, "Dswap");
    run_swap_case<cuComplex>(
        handle, {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
        {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}}, cublasCswap_v2,
        cublasCswap_v2_64, "Cswap");
    run_swap_case<cuDoubleComplex>(
        handle, {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}},
        {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}}, cublasZswap_v2,
        cublasZswap_v2_64, "Zswap");
}

int main() {
    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&handle));
    test_copy(handle);
    test_swap(handle);
    CHECK_CUBLAS(cublasDestroy(handle));
    std::puts("cuBLAS Level-1 copy/swap test passed");
    return 0;
}
