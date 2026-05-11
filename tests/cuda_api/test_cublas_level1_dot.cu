#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
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
    return std::fabs(got - want) <= 1e-4f;
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
static cudaDataType_t cuda_type();

template <>
cudaDataType_t cuda_type<float>() {
    return CUDA_R_32F;
}

template <>
cudaDataType_t cuda_type<double>() {
    return CUDA_R_64F;
}

template <>
cudaDataType_t cuda_type<cuComplex>() {
    return CUDA_C_32F;
}

template <>
cudaDataType_t cuda_type<cuDoubleComplex>() {
    return CUDA_C_64F;
}

template <typename T>
static T *to_device(const std::vector<T> &host) {
    T *device = nullptr;
    CHECK_CUDA(cudaMalloc(&device, host.size() * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
    return device;
}

template <typename ResultT, typename VecT, typename Fn, typename Fn64>
static void run_dot_case(cublasHandle_t handle, const std::vector<VecT> &x,
                         const std::vector<VecT> &y, ResultT expected, Fn fn,
                         Fn64 fn64, const char *label) {
    const int n = static_cast<int>(x.size());
    VecT *device_x = to_device(x);
    VecT *device_y = to_device(y);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    ResultT host_result = ResultT{};
    CHECK_CUBLAS(fn(handle, n, device_x, 1, device_y, 1, &host_result));
    if (!close_value(host_result, expected)) {
        std::fprintf(stderr, "%s host result mismatch\n", label);
        std::exit(1);
    }

    host_result = ResultT{};
    CHECK_CUBLAS(fn64(handle, static_cast<int64_t>(n), device_x, 1, device_y,
                     1, &host_result));
    if (!close_value(host_result, expected)) {
        std::fprintf(stderr, "%s host 64-bit result mismatch\n", label);
        std::exit(1);
    }

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    ResultT zero = ResultT{};
    ResultT *device_result = nullptr;
    CHECK_CUDA(cudaMalloc(&device_result, sizeof(ResultT)));
    CHECK_CUDA(cudaMemcpy(device_result, &zero, sizeof(ResultT),
                          cudaMemcpyHostToDevice));

    CHECK_CUBLAS(fn(handle, n, device_x, 1, device_y, 1, device_result));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&host_result, device_result, sizeof(ResultT),
                          cudaMemcpyDeviceToHost));
    if (!close_value(host_result, expected)) {
        std::fprintf(stderr, "%s device result mismatch\n", label);
        std::exit(1);
    }

    CHECK_CUDA(cudaMemcpy(device_result, &zero, sizeof(ResultT),
                          cudaMemcpyHostToDevice));
    CHECK_CUBLAS(fn64(handle, static_cast<int64_t>(n), device_x, 1, device_y,
                     1, device_result));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&host_result, device_result, sizeof(ResultT),
                          cudaMemcpyDeviceToHost));
    if (!close_value(host_result, expected)) {
        std::fprintf(stderr, "%s device 64-bit result mismatch\n", label);
        std::exit(1);
    }

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUDA(cudaFree(device_result));
    CHECK_CUDA(cudaFree(device_y));
    CHECK_CUDA(cudaFree(device_x));
}

template <typename ResultT, typename VecT, typename Fn, typename Fn64>
static void run_dot_ex_case(cublasHandle_t handle, const std::vector<VecT> &x,
                            const std::vector<VecT> &y, ResultT expected,
                            Fn fn, Fn64 fn64, const char *label) {
    const int n = static_cast<int>(x.size());
    VecT *device_x = to_device(x);
    VecT *device_y = to_device(y);
    const cudaDataType_t value_type = cuda_type<VecT>();
    const cudaDataType_t result_type = cuda_type<ResultT>();

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    ResultT host_result = ResultT{};
    CHECK_CUBLAS(fn(handle, n, device_x, value_type, 1, device_y, value_type,
                    1, &host_result, result_type, result_type));
    if (!close_value(host_result, expected)) {
        std::fprintf(stderr, "%s Ex host result mismatch\n", label);
        std::exit(1);
    }

    host_result = ResultT{};
    CHECK_CUBLAS(fn64(handle, static_cast<int64_t>(n), device_x, value_type, 1,
                     device_y, value_type, 1, &host_result, result_type,
                     result_type));
    if (!close_value(host_result, expected)) {
        std::fprintf(stderr, "%s Ex host 64-bit result mismatch\n", label);
        std::exit(1);
    }

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    ResultT zero = ResultT{};
    ResultT *device_result = nullptr;
    CHECK_CUDA(cudaMalloc(&device_result, sizeof(ResultT)));
    CHECK_CUDA(cudaMemcpy(device_result, &zero, sizeof(ResultT),
                          cudaMemcpyHostToDevice));

    CHECK_CUBLAS(fn(handle, n, device_x, value_type, 1, device_y, value_type,
                    1, device_result, result_type, result_type));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&host_result, device_result, sizeof(ResultT),
                          cudaMemcpyDeviceToHost));
    if (!close_value(host_result, expected)) {
        std::fprintf(stderr, "%s Ex device result mismatch\n", label);
        std::exit(1);
    }

    CHECK_CUDA(cudaMemcpy(device_result, &zero, sizeof(ResultT),
                          cudaMemcpyHostToDevice));
    CHECK_CUBLAS(fn64(handle, static_cast<int64_t>(n), device_x, value_type, 1,
                     device_y, value_type, 1, device_result, result_type,
                     result_type));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&host_result, device_result, sizeof(ResultT),
                          cudaMemcpyDeviceToHost));
    if (!close_value(host_result, expected)) {
        std::fprintf(stderr, "%s Ex device 64-bit result mismatch\n", label);
        std::exit(1);
    }

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUDA(cudaFree(device_result));
    CHECK_CUDA(cudaFree(device_y));
    CHECK_CUDA(cudaFree(device_x));
}

static void test_real_dot(cublasHandle_t handle) {
    run_dot_case<float>(
        handle, std::vector<float>{1.0f, 2.0f, -3.0f},
        std::vector<float>{4.0f, -5.0f, 6.0f}, -24.0f, cublasSdot_v2,
        cublasSdot_v2_64, "Sdot");
    run_dot_case<double>(
        handle, std::vector<double>{1.0, 2.0, -3.0},
        std::vector<double>{4.0, -5.0, 6.0}, -24.0, cublasDdot_v2,
        cublasDdot_v2_64, "Ddot");
    run_dot_ex_case<float>(
        handle, std::vector<float>{1.0f, 2.0f, -3.0f},
        std::vector<float>{4.0f, -5.0f, 6.0f}, -24.0f, cublasDotEx,
        cublasDotEx_64, "DotEx float");
    run_dot_ex_case<double>(
        handle, std::vector<double>{1.0, 2.0, -3.0},
        std::vector<double>{4.0, -5.0, 6.0}, -24.0, cublasDotEx,
        cublasDotEx_64, "DotEx double");
}

static void test_complex_dot(cublasHandle_t handle) {
    const std::vector<cuComplex> cx{{1.0f, 2.0f}, {3.0f, -4.0f}};
    const std::vector<cuComplex> cy{{-2.0f, 1.0f}, {0.5f, 3.0f}};
    run_dot_case<cuComplex>(
        handle, cx, cy, cuComplex{9.5f, 4.0f}, cublasCdotu_v2,
        cublasCdotu_v2_64, "Cdotu");
    run_dot_case<cuComplex>(
        handle, cx, cy, cuComplex{-10.5f, 16.0f}, cublasCdotc_v2,
        cublasCdotc_v2_64, "Cdotc");
    run_dot_ex_case<cuComplex>(
        handle, cx, cy, cuComplex{9.5f, 4.0f}, cublasDotEx, cublasDotEx_64,
        "DotEx complex-float");
    run_dot_ex_case<cuComplex>(
        handle, cx, cy, cuComplex{-10.5f, 16.0f}, cublasDotcEx,
        cublasDotcEx_64, "DotcEx complex-float");

    const std::vector<cuDoubleComplex> zx{{1.0, 2.0}, {3.0, -4.0}};
    const std::vector<cuDoubleComplex> zy{{-2.0, 1.0}, {0.5, 3.0}};
    run_dot_case<cuDoubleComplex>(
        handle, zx, zy, cuDoubleComplex{9.5, 4.0}, cublasZdotu_v2,
        cublasZdotu_v2_64, "Zdotu");
    run_dot_case<cuDoubleComplex>(
        handle, zx, zy, cuDoubleComplex{-10.5, 16.0}, cublasZdotc_v2,
        cublasZdotc_v2_64, "Zdotc");
    run_dot_ex_case<cuDoubleComplex>(
        handle, zx, zy, cuDoubleComplex{9.5, 4.0}, cublasDotEx,
        cublasDotEx_64, "DotEx complex-double");
    run_dot_ex_case<cuDoubleComplex>(
        handle, zx, zy, cuDoubleComplex{-10.5, 16.0}, cublasDotcEx,
        cublasDotcEx_64, "DotcEx complex-double");
}

int main() {
    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&handle));
    test_real_dot(handle);
    test_complex_dot(handle);
    CHECK_CUBLAS(cublasDestroy(handle));
    std::puts("cuBLAS Level-1 dot test passed");
    return 0;
}
