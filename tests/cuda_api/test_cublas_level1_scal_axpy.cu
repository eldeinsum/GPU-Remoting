#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
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

static float scale_value(float value, float alpha) {
    return value * alpha;
}

static double scale_value(double value, double alpha) {
    return value * alpha;
}

static cuComplex scale_value(cuComplex value, float alpha) {
    return cuComplex{value.x * alpha, value.y * alpha};
}

static cuComplex scale_value(cuComplex value, cuComplex alpha) {
    return cuComplex{value.x * alpha.x - value.y * alpha.y,
                     value.x * alpha.y + value.y * alpha.x};
}

static cuDoubleComplex scale_value(cuDoubleComplex value, double alpha) {
    return cuDoubleComplex{value.x * alpha, value.y * alpha};
}

static cuDoubleComplex scale_value(cuDoubleComplex value,
                                   cuDoubleComplex alpha) {
    return cuDoubleComplex{value.x * alpha.x - value.y * alpha.y,
                           value.x * alpha.y + value.y * alpha.x};
}

static float add_value(float left, float right) {
    return left + right;
}

static double add_value(double left, double right) {
    return left + right;
}

static cuComplex add_value(cuComplex left, cuComplex right) {
    return cuComplex{left.x + right.x, left.y + right.y};
}

static cuDoubleComplex add_value(cuDoubleComplex left, cuDoubleComplex right) {
    return cuDoubleComplex{left.x + right.x, left.y + right.y};
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
static void copy_to_device(T *device, const std::vector<T> &host) {
    CHECK_CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
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

template <typename VecT, typename AlphaT>
static std::vector<VecT> scaled_expected(const std::vector<VecT> &input,
                                         AlphaT alpha) {
    std::vector<VecT> expected;
    expected.reserve(input.size());
    for (const auto &value : input) {
        expected.push_back(scale_value(value, alpha));
    }
    return expected;
}

template <typename VecT, typename AlphaT>
static std::vector<VecT> axpy_expected(const std::vector<VecT> &x,
                                       const std::vector<VecT> &y,
                                       AlphaT alpha) {
    std::vector<VecT> expected;
    expected.reserve(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        expected.push_back(add_value(scale_value(x[i], alpha), y[i]));
    }
    return expected;
}

template <typename VecT, typename AlphaT, typename ScalFn, typename Scal64Fn>
static void run_scal_case(cublasHandle_t handle, const std::vector<VecT> &input,
                          AlphaT host_alpha, AlphaT device_alpha,
                          ScalFn scal_fn, Scal64Fn scal64_fn,
                          const char *label) {
    const int n = static_cast<int>(input.size());
    VecT *device = to_device(input);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUBLAS(scal_fn(handle, n, &host_alpha, device, 1));
    expect_vector(from_device(device, input.size()),
                  scaled_expected(input, host_alpha), label);

    copy_to_device(device, input);
    CHECK_CUBLAS(scal64_fn(handle, static_cast<int64_t>(n), &host_alpha,
                           device, 1));
    expect_vector(from_device(device, input.size()),
                  scaled_expected(input, host_alpha), label);

    AlphaT *device_alpha_ptr = to_device(std::vector<AlphaT>{device_alpha});
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    copy_to_device(device, input);
    CHECK_CUBLAS(scal_fn(handle, n, device_alpha_ptr, device, 1));
    expect_vector(from_device(device, input.size()),
                  scaled_expected(input, device_alpha), label);

    copy_to_device(device, input);
    CHECK_CUBLAS(scal64_fn(handle, static_cast<int64_t>(n), device_alpha_ptr,
                           device, 1));
    expect_vector(from_device(device, input.size()),
                  scaled_expected(input, device_alpha), label);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUDA(cudaFree(device_alpha_ptr));
    CHECK_CUDA(cudaFree(device));
}

template <typename VecT, typename AlphaT, typename ScalFn, typename Scal64Fn>
static void run_scal_ex_case(cublasHandle_t handle,
                             const std::vector<VecT> &input,
                             AlphaT host_alpha, AlphaT device_alpha,
                             ScalFn scal_fn, Scal64Fn scal64_fn,
                             const char *label) {
    const int n = static_cast<int>(input.size());
    const cudaDataType_t alpha_type = cuda_type<AlphaT>();
    const cudaDataType_t x_type = cuda_type<VecT>();
    VecT *device = to_device(input);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUBLAS(scal_fn(handle, n, &host_alpha, alpha_type, device, x_type, 1,
                         x_type));
    expect_vector(from_device(device, input.size()),
                  scaled_expected(input, host_alpha), label);

    copy_to_device(device, input);
    CHECK_CUBLAS(scal64_fn(handle, static_cast<int64_t>(n), &host_alpha,
                           alpha_type, device, x_type, 1, x_type));
    expect_vector(from_device(device, input.size()),
                  scaled_expected(input, host_alpha), label);

    AlphaT *device_alpha_ptr = to_device(std::vector<AlphaT>{device_alpha});
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    copy_to_device(device, input);
    CHECK_CUBLAS(scal_fn(handle, n, device_alpha_ptr, alpha_type, device,
                         x_type, 1, x_type));
    expect_vector(from_device(device, input.size()),
                  scaled_expected(input, device_alpha), label);

    copy_to_device(device, input);
    CHECK_CUBLAS(scal64_fn(handle, static_cast<int64_t>(n), device_alpha_ptr,
                           alpha_type, device, x_type, 1, x_type));
    expect_vector(from_device(device, input.size()),
                  scaled_expected(input, device_alpha), label);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUDA(cudaFree(device_alpha_ptr));
    CHECK_CUDA(cudaFree(device));
}

template <typename VecT, typename AlphaT, typename AxpyFn, typename Axpy64Fn>
static void run_axpy_case(cublasHandle_t handle, const std::vector<VecT> &x,
                          const std::vector<VecT> &y, AlphaT host_alpha,
                          AlphaT device_alpha, AxpyFn axpy_fn,
                          Axpy64Fn axpy64_fn, const char *label) {
    const int n = static_cast<int>(x.size());
    VecT *device_x = to_device(x);
    VecT *device_y = to_device(y);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUBLAS(axpy_fn(handle, n, &host_alpha, device_x, 1, device_y, 1));
    expect_vector(from_device(device_y, y.size()),
                  axpy_expected(x, y, host_alpha), label);

    copy_to_device(device_y, y);
    CHECK_CUBLAS(axpy64_fn(handle, static_cast<int64_t>(n), &host_alpha,
                           device_x, 1, device_y, 1));
    expect_vector(from_device(device_y, y.size()),
                  axpy_expected(x, y, host_alpha), label);

    AlphaT *device_alpha_ptr = to_device(std::vector<AlphaT>{device_alpha});
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    copy_to_device(device_y, y);
    CHECK_CUBLAS(
        axpy_fn(handle, n, device_alpha_ptr, device_x, 1, device_y, 1));
    expect_vector(from_device(device_y, y.size()),
                  axpy_expected(x, y, device_alpha), label);

    copy_to_device(device_y, y);
    CHECK_CUBLAS(axpy64_fn(handle, static_cast<int64_t>(n), device_alpha_ptr,
                           device_x, 1, device_y, 1));
    expect_vector(from_device(device_y, y.size()),
                  axpy_expected(x, y, device_alpha), label);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUDA(cudaFree(device_alpha_ptr));
    CHECK_CUDA(cudaFree(device_y));
    CHECK_CUDA(cudaFree(device_x));
}

template <typename VecT, typename AlphaT, typename AxpyFn, typename Axpy64Fn>
static void run_axpy_ex_case(cublasHandle_t handle, const std::vector<VecT> &x,
                             const std::vector<VecT> &y, AlphaT host_alpha,
                             AlphaT device_alpha, AxpyFn axpy_fn,
                             Axpy64Fn axpy64_fn, const char *label) {
    const int n = static_cast<int>(x.size());
    const cudaDataType_t alpha_type = cuda_type<AlphaT>();
    const cudaDataType_t value_type = cuda_type<VecT>();
    VecT *device_x = to_device(x);
    VecT *device_y = to_device(y);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUBLAS(axpy_fn(handle, n, &host_alpha, alpha_type, device_x,
                         value_type, 1, device_y, value_type, 1,
                         value_type));
    expect_vector(from_device(device_y, y.size()),
                  axpy_expected(x, y, host_alpha), label);

    copy_to_device(device_y, y);
    CHECK_CUBLAS(axpy64_fn(handle, static_cast<int64_t>(n), &host_alpha,
                           alpha_type, device_x, value_type, 1, device_y,
                           value_type, 1, value_type));
    expect_vector(from_device(device_y, y.size()),
                  axpy_expected(x, y, host_alpha), label);

    AlphaT *device_alpha_ptr = to_device(std::vector<AlphaT>{device_alpha});
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    copy_to_device(device_y, y);
    CHECK_CUBLAS(axpy_fn(handle, n, device_alpha_ptr, alpha_type, device_x,
                         value_type, 1, device_y, value_type, 1,
                         value_type));
    expect_vector(from_device(device_y, y.size()),
                  axpy_expected(x, y, device_alpha), label);

    copy_to_device(device_y, y);
    CHECK_CUBLAS(axpy64_fn(handle, static_cast<int64_t>(n), device_alpha_ptr,
                           alpha_type, device_x, value_type, 1, device_y,
                           value_type, 1, value_type));
    expect_vector(from_device(device_y, y.size()),
                  axpy_expected(x, y, device_alpha), label);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUDA(cudaFree(device_alpha_ptr));
    CHECK_CUDA(cudaFree(device_y));
    CHECK_CUDA(cudaFree(device_x));
}

static void test_scal(cublasHandle_t handle) {
    run_scal_case<float>(
        handle, {1.0f, -2.0f, 3.0f}, 2.0f, -0.5f, cublasSscal_v2,
        cublasSscal_v2_64, "Sscal");
    run_scal_case<double>(
        handle, {1.0, -2.0, 3.0}, 2.0, -0.5, cublasDscal_v2,
        cublasDscal_v2_64, "Dscal");
    run_scal_case<cuComplex>(
        handle, {{1.0f, 2.0f}, {-3.0f, 4.0f}, {5.0f, -6.0f}},
        cuComplex{0.5f, -1.0f}, cuComplex{-1.5f, 0.25f}, cublasCscal_v2,
        cublasCscal_v2_64, "Cscal");
    run_scal_case<cuComplex, float>(
        handle, {{1.0f, 2.0f}, {-3.0f, 4.0f}, {5.0f, -6.0f}}, 2.0f,
        -0.5f, cublasCsscal_v2, cublasCsscal_v2_64, "Csscal");
    run_scal_case<cuDoubleComplex>(
        handle, {{1.0, 2.0}, {-3.0, 4.0}, {5.0, -6.0}},
        cuDoubleComplex{0.5, -1.0}, cuDoubleComplex{-1.5, 0.25},
        cublasZscal_v2, cublasZscal_v2_64, "Zscal");
    run_scal_case<cuDoubleComplex, double>(
        handle, {{1.0, 2.0}, {-3.0, 4.0}, {5.0, -6.0}}, 2.0, -0.5,
        cublasZdscal_v2, cublasZdscal_v2_64, "Zdscal");
    run_scal_ex_case<float>(
        handle, {1.0f, -2.0f, 3.0f}, 2.0f, -0.5f, cublasScalEx,
        cublasScalEx_64, "ScalEx float");
    run_scal_ex_case<double>(
        handle, {1.0, -2.0, 3.0}, 2.0, -0.5, cublasScalEx,
        cublasScalEx_64, "ScalEx double");
    run_scal_ex_case<cuComplex>(
        handle, {{1.0f, 2.0f}, {-3.0f, 4.0f}, {5.0f, -6.0f}},
        cuComplex{0.5f, -1.0f}, cuComplex{-1.5f, 0.25f}, cublasScalEx,
        cublasScalEx_64, "ScalEx complex-float");
    run_scal_ex_case<cuDoubleComplex>(
        handle, {{1.0, 2.0}, {-3.0, 4.0}, {5.0, -6.0}},
        cuDoubleComplex{0.5, -1.0}, cuDoubleComplex{-1.5, 0.25},
        cublasScalEx, cublasScalEx_64, "ScalEx complex-double");
}

static void test_axpy(cublasHandle_t handle) {
    run_axpy_case<float>(
        handle, {1.0f, -2.0f, 3.0f}, {4.0f, 5.0f, -6.0f}, 2.0f, -0.5f,
        cublasSaxpy_v2, cublasSaxpy_v2_64, "Saxpy");
    run_axpy_case<double>(
        handle, {1.0, -2.0, 3.0}, {4.0, 5.0, -6.0}, 2.0, -0.5,
        cublasDaxpy_v2, cublasDaxpy_v2_64, "Daxpy");
    run_axpy_case<cuComplex>(
        handle, {{1.0f, 2.0f}, {-3.0f, 4.0f}, {5.0f, -6.0f}},
        {{7.0f, -8.0f}, {9.0f, 10.0f}, {-11.0f, 12.0f}},
        cuComplex{0.5f, -1.0f}, cuComplex{-1.5f, 0.25f}, cublasCaxpy_v2,
        cublasCaxpy_v2_64, "Caxpy");
    run_axpy_case<cuDoubleComplex>(
        handle, {{1.0, 2.0}, {-3.0, 4.0}, {5.0, -6.0}},
        {{7.0, -8.0}, {9.0, 10.0}, {-11.0, 12.0}},
        cuDoubleComplex{0.5, -1.0}, cuDoubleComplex{-1.5, 0.25},
        cublasZaxpy_v2, cublasZaxpy_v2_64, "Zaxpy");
    run_axpy_ex_case<float>(
        handle, {1.0f, -2.0f, 3.0f}, {4.0f, 5.0f, -6.0f}, 2.0f, -0.5f,
        cublasAxpyEx, cublasAxpyEx_64, "AxpyEx float");
    run_axpy_ex_case<double>(
        handle, {1.0, -2.0, 3.0}, {4.0, 5.0, -6.0}, 2.0, -0.5,
        cublasAxpyEx, cublasAxpyEx_64, "AxpyEx double");
    run_axpy_ex_case<cuComplex>(
        handle, {{1.0f, 2.0f}, {-3.0f, 4.0f}, {5.0f, -6.0f}},
        {{7.0f, -8.0f}, {9.0f, 10.0f}, {-11.0f, 12.0f}},
        cuComplex{0.5f, -1.0f}, cuComplex{-1.5f, 0.25f}, cublasAxpyEx,
        cublasAxpyEx_64, "AxpyEx complex-float");
    run_axpy_ex_case<cuDoubleComplex>(
        handle, {{1.0, 2.0}, {-3.0, 4.0}, {5.0, -6.0}},
        {{7.0, -8.0}, {9.0, 10.0}, {-11.0, 12.0}},
        cuDoubleComplex{0.5, -1.0}, cuDoubleComplex{-1.5, 0.25},
        cublasAxpyEx, cublasAxpyEx_64, "AxpyEx complex-double");
}

int main() {
    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&handle));
    test_scal(handle);
    test_axpy(handle);
    CHECK_CUBLAS(cublasDestroy(handle));
    std::puts("cuBLAS Level-1 scal/axpy test passed");
    return 0;
}
