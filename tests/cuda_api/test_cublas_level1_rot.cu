#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

static bool close_value(__half got, __half want) {
    return std::fabs(__half2float(got) - __half2float(want)) <= 2e-2f;
}

static bool close_value(__nv_bfloat16 got, __nv_bfloat16 want) {
    return std::fabs(__bfloat162float(got) - __bfloat162float(want)) <= 3e-2f;
}

static bool close_value(cuComplex got, cuComplex want) {
    return close_value(got.x, want.x) && close_value(got.y, want.y);
}

static bool close_value(cuDoubleComplex got, cuDoubleComplex want) {
    return close_value(got.x, want.x) && close_value(got.y, want.y);
}

static float add_value(float left, float right) {
    return left + right;
}

static double add_value(double left, double right) {
    return left + right;
}

static __half add_value(__half left, __half right) {
    return __float2half(__half2float(left) + __half2float(right));
}

static __nv_bfloat16 add_value(__nv_bfloat16 left, __nv_bfloat16 right) {
    return __float2bfloat16(__bfloat162float(left) + __bfloat162float(right));
}

static cuComplex add_value(cuComplex left, cuComplex right) {
    return cuComplex{left.x + right.x, left.y + right.y};
}

static cuDoubleComplex add_value(cuDoubleComplex left, cuDoubleComplex right) {
    return cuDoubleComplex{left.x + right.x, left.y + right.y};
}

static float sub_value(float left, float right) {
    return left - right;
}

static double sub_value(double left, double right) {
    return left - right;
}

static __half sub_value(__half left, __half right) {
    return __float2half(__half2float(left) - __half2float(right));
}

static __nv_bfloat16 sub_value(__nv_bfloat16 left, __nv_bfloat16 right) {
    return __float2bfloat16(__bfloat162float(left) - __bfloat162float(right));
}

static cuComplex sub_value(cuComplex left, cuComplex right) {
    return cuComplex{left.x - right.x, left.y - right.y};
}

static cuDoubleComplex sub_value(cuDoubleComplex left, cuDoubleComplex right) {
    return cuDoubleComplex{left.x - right.x, left.y - right.y};
}

static float scale_value(float value, float scalar) {
    return value * scalar;
}

static double scale_value(double value, double scalar) {
    return value * scalar;
}

static __half scale_value(__half value, __half scalar) {
    return __float2half(__half2float(value) * __half2float(scalar));
}

static __nv_bfloat16 scale_value(__nv_bfloat16 value,
                                 __nv_bfloat16 scalar) {
    return __float2bfloat16(__bfloat162float(value) *
                            __bfloat162float(scalar));
}

static cuComplex scale_value(cuComplex value, float scalar) {
    return cuComplex{value.x * scalar, value.y * scalar};
}

static cuDoubleComplex scale_value(cuDoubleComplex value, double scalar) {
    return cuDoubleComplex{value.x * scalar, value.y * scalar};
}

static cuComplex mul_value(cuComplex left, cuComplex right) {
    return cuComplex{left.x * right.x - left.y * right.y,
                     left.x * right.y + left.y * right.x};
}

static cuDoubleComplex mul_value(cuDoubleComplex left, cuDoubleComplex right) {
    return cuDoubleComplex{left.x * right.x - left.y * right.y,
                           left.x * right.y + left.y * right.x};
}

static cuComplex conj_value(cuComplex value) {
    return cuComplex{value.x, -value.y};
}

static cuDoubleComplex conj_value(cuDoubleComplex value) {
    return cuDoubleComplex{value.x, -value.y};
}

static float real_component(cuComplex value) { return value.x; }

static double real_component(cuDoubleComplex value) { return value.x; }

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
cudaDataType_t cuda_type<__half>() {
    return CUDA_R_16F;
}

template <>
cudaDataType_t cuda_type<__nv_bfloat16>() {
    return CUDA_R_16BF;
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

template <typename VecT, typename ScalarT>
static std::vector<VecT> real_rot_x_expected(const std::vector<VecT> &x,
                                             const std::vector<VecT> &y,
                                             ScalarT c, ScalarT s) {
    std::vector<VecT> expected;
    expected.reserve(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        expected.push_back(
            add_value(scale_value(x[i], c), scale_value(y[i], s)));
    }
    return expected;
}

template <typename VecT, typename ScalarT>
static std::vector<VecT> real_rot_y_expected(const std::vector<VecT> &x,
                                             const std::vector<VecT> &y,
                                             ScalarT c, ScalarT s) {
    std::vector<VecT> expected;
    expected.reserve(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        expected.push_back(
            sub_value(scale_value(y[i], c), scale_value(x[i], s)));
    }
    return expected;
}

template <typename VecT, typename RealT, typename ScalarT>
static std::vector<VecT> complex_rot_x_expected(const std::vector<VecT> &x,
                                                const std::vector<VecT> &y,
                                                RealT c, ScalarT s) {
    std::vector<VecT> expected;
    expected.reserve(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        expected.push_back(add_value(scale_value(x[i], c), mul_value(s, y[i])));
    }
    return expected;
}

template <typename VecT, typename RealT, typename ScalarT>
static std::vector<VecT> complex_rot_y_expected(const std::vector<VecT> &x,
                                                const std::vector<VecT> &y,
                                                RealT c, ScalarT s) {
    std::vector<VecT> expected;
    expected.reserve(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        expected.push_back(
            sub_value(scale_value(y[i], c), mul_value(conj_value(s), x[i])));
    }
    return expected;
}

template <typename VecT, typename ScalarT, typename Fn, typename Fn64>
static void run_real_rot_case(cublasHandle_t handle, const std::vector<VecT> &x,
                              const std::vector<VecT> &y, ScalarT host_c,
                              ScalarT host_s, ScalarT device_c,
                              ScalarT device_s, Fn fn, Fn64 fn64,
                              const char *label) {
    const int n = static_cast<int>(x.size());
    VecT *device_x = to_device(x);
    VecT *device_y = to_device(y);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    CHECK_CUBLAS(fn(handle, n, device_x, 1, device_y, 1, &host_c, &host_s));
    expect_vector(from_device(device_x, x.size()),
                  real_rot_x_expected(x, y, host_c, host_s), label);
    expect_vector(from_device(device_y, y.size()),
                  real_rot_y_expected(x, y, host_c, host_s), label);

    copy_to_device(device_x, x);
    copy_to_device(device_y, y);
    CHECK_CUBLAS(fn64(handle, static_cast<int64_t>(n), device_x, 1, device_y, 1,
                     &host_c, &host_s));
    expect_vector(from_device(device_x, x.size()),
                  real_rot_x_expected(x, y, host_c, host_s), label);
    expect_vector(from_device(device_y, y.size()),
                  real_rot_y_expected(x, y, host_c, host_s), label);

    ScalarT *device_c_ptr = to_device(std::vector<ScalarT>{device_c});
    ScalarT *device_s_ptr = to_device(std::vector<ScalarT>{device_s});
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    copy_to_device(device_x, x);
    copy_to_device(device_y, y);
    CHECK_CUBLAS(
        fn(handle, n, device_x, 1, device_y, 1, device_c_ptr, device_s_ptr));
    expect_vector(from_device(device_x, x.size()),
                  real_rot_x_expected(x, y, device_c, device_s), label);
    expect_vector(from_device(device_y, y.size()),
                  real_rot_y_expected(x, y, device_c, device_s), label);

    copy_to_device(device_x, x);
    copy_to_device(device_y, y);
    CHECK_CUBLAS(fn64(handle, static_cast<int64_t>(n), device_x, 1, device_y, 1,
                     device_c_ptr, device_s_ptr));
    expect_vector(from_device(device_x, x.size()),
                  real_rot_x_expected(x, y, device_c, device_s), label);
    expect_vector(from_device(device_y, y.size()),
                  real_rot_y_expected(x, y, device_c, device_s), label);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUDA(cudaFree(device_s_ptr));
    CHECK_CUDA(cudaFree(device_c_ptr));
    CHECK_CUDA(cudaFree(device_y));
    CHECK_CUDA(cudaFree(device_x));
}

template <typename VecT, typename RealT, typename ScalarT, typename Fn,
          typename Fn64>
static void run_complex_rot_case(cublasHandle_t handle,
                                 const std::vector<VecT> &x,
                                 const std::vector<VecT> &y, RealT host_c,
                                 ScalarT host_s, RealT device_c,
                                 ScalarT device_s, Fn fn, Fn64 fn64,
                                 const char *label) {
    const int n = static_cast<int>(x.size());
    VecT *device_x = to_device(x);
    VecT *device_y = to_device(y);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    CHECK_CUBLAS(fn(handle, n, device_x, 1, device_y, 1, &host_c, &host_s));
    expect_vector(from_device(device_x, x.size()),
                  complex_rot_x_expected(x, y, host_c, host_s), label);
    expect_vector(from_device(device_y, y.size()),
                  complex_rot_y_expected(x, y, host_c, host_s), label);

    copy_to_device(device_x, x);
    copy_to_device(device_y, y);
    CHECK_CUBLAS(fn64(handle, static_cast<int64_t>(n), device_x, 1, device_y, 1,
                     &host_c, &host_s));
    expect_vector(from_device(device_x, x.size()),
                  complex_rot_x_expected(x, y, host_c, host_s), label);
    expect_vector(from_device(device_y, y.size()),
                  complex_rot_y_expected(x, y, host_c, host_s), label);

    RealT *device_c_ptr = to_device(std::vector<RealT>{device_c});
    ScalarT *device_s_ptr = to_device(std::vector<ScalarT>{device_s});
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    copy_to_device(device_x, x);
    copy_to_device(device_y, y);
    CHECK_CUBLAS(
        fn(handle, n, device_x, 1, device_y, 1, device_c_ptr, device_s_ptr));
    expect_vector(from_device(device_x, x.size()),
                  complex_rot_x_expected(x, y, device_c, device_s), label);
    expect_vector(from_device(device_y, y.size()),
                  complex_rot_y_expected(x, y, device_c, device_s), label);

    copy_to_device(device_x, x);
    copy_to_device(device_y, y);
    CHECK_CUBLAS(fn64(handle, static_cast<int64_t>(n), device_x, 1, device_y, 1,
                     device_c_ptr, device_s_ptr));
    expect_vector(from_device(device_x, x.size()),
                  complex_rot_x_expected(x, y, device_c, device_s), label);
    expect_vector(from_device(device_y, y.size()),
                  complex_rot_y_expected(x, y, device_c, device_s), label);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUDA(cudaFree(device_s_ptr));
    CHECK_CUDA(cudaFree(device_c_ptr));
    CHECK_CUDA(cudaFree(device_y));
    CHECK_CUDA(cudaFree(device_x));
}

template <typename VecT, typename ScalarT>
static void run_real_rot_ex_case(cublasHandle_t handle,
                                 const std::vector<VecT> &x,
                                 const std::vector<VecT> &y, ScalarT host_c,
                                 ScalarT host_s, ScalarT device_c,
                                 ScalarT device_s,
                                 cudaDataType_t execution_type,
                                 const char *label) {
    const int n = static_cast<int>(x.size());
    const cudaDataType_t value_type = cuda_type<VecT>();
    const cudaDataType_t cs_type = cuda_type<ScalarT>();
    VecT *device_x = to_device(x);
    VecT *device_y = to_device(y);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    CHECK_CUBLAS(cublasRotEx(handle, n, device_x, value_type, 1, device_y,
                             value_type, 1, &host_c, &host_s, cs_type,
                             execution_type));
    expect_vector(from_device(device_x, x.size()),
                  real_rot_x_expected(x, y, host_c, host_s), label);
    expect_vector(from_device(device_y, y.size()),
                  real_rot_y_expected(x, y, host_c, host_s), label);

    copy_to_device(device_x, x);
    copy_to_device(device_y, y);
    CHECK_CUBLAS(cublasRotEx_64(handle, static_cast<int64_t>(n), device_x,
                                value_type, int64_t{1}, device_y, value_type,
                                int64_t{1}, &host_c, &host_s, cs_type,
                                execution_type));
    expect_vector(from_device(device_x, x.size()),
                  real_rot_x_expected(x, y, host_c, host_s), label);
    expect_vector(from_device(device_y, y.size()),
                  real_rot_y_expected(x, y, host_c, host_s), label);

    ScalarT *device_c_ptr = to_device(std::vector<ScalarT>{device_c});
    ScalarT *device_s_ptr = to_device(std::vector<ScalarT>{device_s});
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    copy_to_device(device_x, x);
    copy_to_device(device_y, y);
    CHECK_CUBLAS(cublasRotEx(handle, n, device_x, value_type, 1, device_y,
                             value_type, 1, device_c_ptr, device_s_ptr,
                             cs_type, execution_type));
    expect_vector(from_device(device_x, x.size()),
                  real_rot_x_expected(x, y, device_c, device_s), label);
    expect_vector(from_device(device_y, y.size()),
                  real_rot_y_expected(x, y, device_c, device_s), label);

    copy_to_device(device_x, x);
    copy_to_device(device_y, y);
    CHECK_CUBLAS(cublasRotEx_64(handle, static_cast<int64_t>(n), device_x,
                                value_type, int64_t{1}, device_y, value_type,
                                int64_t{1}, device_c_ptr, device_s_ptr,
                                cs_type, execution_type));
    expect_vector(from_device(device_x, x.size()),
                  real_rot_x_expected(x, y, device_c, device_s), label);
    expect_vector(from_device(device_y, y.size()),
                  real_rot_y_expected(x, y, device_c, device_s), label);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUDA(cudaFree(device_s_ptr));
    CHECK_CUDA(cudaFree(device_c_ptr));
    CHECK_CUDA(cudaFree(device_y));
    CHECK_CUDA(cudaFree(device_x));
}

template <typename VecT, typename RealT, typename ScalarT>
static void run_complex_rot_ex_case(cublasHandle_t handle,
                                    const std::vector<VecT> &x,
                                    const std::vector<VecT> &y,
                                    ScalarT host_c, ScalarT host_s,
                                    ScalarT device_c, ScalarT device_s,
                                    cudaDataType_t execution_type,
                                    const char *label) {
    const int n = static_cast<int>(x.size());
    const cudaDataType_t value_type = cuda_type<VecT>();
    const cudaDataType_t cs_type = cuda_type<ScalarT>();
    VecT *device_x = to_device(x);
    VecT *device_y = to_device(y);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    CHECK_CUBLAS(cublasRotEx(handle, n, device_x, value_type, 1, device_y,
                             value_type, 1, &host_c, &host_s, cs_type,
                             execution_type));
    expect_vector(from_device(device_x, x.size()),
                  complex_rot_x_expected<VecT, RealT, ScalarT>(
                      x, y, real_component(host_c), host_s),
                  label);
    expect_vector(from_device(device_y, y.size()),
                  complex_rot_y_expected<VecT, RealT, ScalarT>(
                      x, y, real_component(host_c), host_s),
                  label);

    copy_to_device(device_x, x);
    copy_to_device(device_y, y);
    CHECK_CUBLAS(cublasRotEx_64(handle, static_cast<int64_t>(n), device_x,
                                value_type, int64_t{1}, device_y, value_type,
                                int64_t{1}, &host_c, &host_s, cs_type,
                                execution_type));
    expect_vector(from_device(device_x, x.size()),
                  complex_rot_x_expected<VecT, RealT, ScalarT>(
                      x, y, real_component(host_c), host_s),
                  label);
    expect_vector(from_device(device_y, y.size()),
                  complex_rot_y_expected<VecT, RealT, ScalarT>(
                      x, y, real_component(host_c), host_s),
                  label);

    ScalarT *device_c_ptr = to_device(std::vector<ScalarT>{device_c});
    ScalarT *device_s_ptr = to_device(std::vector<ScalarT>{device_s});
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    copy_to_device(device_x, x);
    copy_to_device(device_y, y);
    CHECK_CUBLAS(cublasRotEx(handle, n, device_x, value_type, 1, device_y,
                             value_type, 1, device_c_ptr, device_s_ptr,
                             cs_type, execution_type));
    expect_vector(from_device(device_x, x.size()),
                  complex_rot_x_expected<VecT, RealT, ScalarT>(
                      x, y, real_component(device_c), device_s),
                  label);
    expect_vector(from_device(device_y, y.size()),
                  complex_rot_y_expected<VecT, RealT, ScalarT>(
                      x, y, real_component(device_c), device_s),
                  label);

    copy_to_device(device_x, x);
    copy_to_device(device_y, y);
    CHECK_CUBLAS(cublasRotEx_64(handle, static_cast<int64_t>(n), device_x,
                                value_type, int64_t{1}, device_y, value_type,
                                int64_t{1}, device_c_ptr, device_s_ptr,
                                cs_type, execution_type));
    expect_vector(from_device(device_x, x.size()),
                  complex_rot_x_expected<VecT, RealT, ScalarT>(
                      x, y, real_component(device_c), device_s),
                  label);
    expect_vector(from_device(device_y, y.size()),
                  complex_rot_y_expected<VecT, RealT, ScalarT>(
                      x, y, real_component(device_c), device_s),
                  label);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUDA(cudaFree(device_s_ptr));
    CHECK_CUDA(cudaFree(device_c_ptr));
    CHECK_CUDA(cudaFree(device_y));
    CHECK_CUDA(cudaFree(device_x));
}

static void test_real_rot(cublasHandle_t handle) {
    run_real_rot_case<float>(
        handle, {1.0f, 2.0f}, {3.0f, 4.0f}, 0.5f, 0.25f, 0.75f, -0.5f,
        cublasSrot_v2, cublasSrot_v2_64, "Srot");
    run_real_rot_case<double>(
        handle, {1.0, 2.0}, {3.0, 4.0}, 0.5, 0.25, 0.75, -0.5,
        cublasDrot_v2, cublasDrot_v2_64, "Drot");
    run_real_rot_ex_case<float, float>(
        handle, {1.0f, 2.0f}, {3.0f, 4.0f}, 0.5f, 0.25f, 0.75f, -0.5f,
        CUDA_R_32F, "RotEx float");
    run_real_rot_ex_case<double, double>(
        handle, {1.0, 2.0}, {3.0, 4.0}, 0.5, 0.25, 0.75, -0.5, CUDA_R_64F,
        "RotEx double");
    run_real_rot_ex_case<__half, __half>(
        handle, {__float2half(1.0f), __float2half(2.0f)},
        {__float2half(3.0f), __float2half(4.0f)}, __float2half(0.5f),
        __float2half(0.25f), __float2half(0.75f), __float2half(-0.5f),
        CUDA_R_32F, "RotEx half");
    run_real_rot_ex_case<__nv_bfloat16, __nv_bfloat16>(
        handle, {__float2bfloat16(1.0f), __float2bfloat16(2.0f)},
        {__float2bfloat16(3.0f), __float2bfloat16(4.0f)},
        __float2bfloat16(0.5f), __float2bfloat16(0.25f),
        __float2bfloat16(0.75f), __float2bfloat16(-0.5f), CUDA_R_32F,
        "RotEx bfloat16");
}

static void test_complex_rot(cublasHandle_t handle) {
    const std::vector<cuComplex> cx{{1.0f, 2.0f}, {-3.0f, 0.5f}};
    const std::vector<cuComplex> cy{{0.5f, -1.0f}, {2.0f, 3.0f}};
    run_complex_rot_case<cuComplex, float, cuComplex>(
        handle, cx, cy, 0.5f, cuComplex{0.25f, -0.5f}, 0.75f,
        cuComplex{-0.25f, 0.5f}, cublasCrot_v2, cublasCrot_v2_64, "Crot");
    run_real_rot_case<cuComplex, float>(
        handle, cx, cy, 0.5f, 0.25f, 0.75f, -0.5f, cublasCsrot_v2,
        cublasCsrot_v2_64, "Csrot");
    run_complex_rot_ex_case<cuComplex, float, cuComplex>(
        handle, cx, cy, cuComplex{0.5f, 0.0f}, cuComplex{0.25f, -0.5f},
        cuComplex{0.75f, 0.0f}, cuComplex{-0.25f, 0.5f}, CUDA_C_32F,
        "RotEx complex-float");
    run_real_rot_ex_case<cuComplex, float>(
        handle, cx, cy, 0.5f, 0.25f, 0.75f, -0.5f, CUDA_C_32F,
        "RotEx complex-float real cs");

    const std::vector<cuDoubleComplex> zx{{1.0, 2.0}, {-3.0, 0.5}};
    const std::vector<cuDoubleComplex> zy{{0.5, -1.0}, {2.0, 3.0}};
    run_complex_rot_case<cuDoubleComplex, double, cuDoubleComplex>(
        handle, zx, zy, 0.5, cuDoubleComplex{0.25, -0.5}, 0.75,
        cuDoubleComplex{-0.25, 0.5}, cublasZrot_v2, cublasZrot_v2_64,
        "Zrot");
    run_real_rot_case<cuDoubleComplex, double>(
        handle, zx, zy, 0.5, 0.25, 0.75, -0.5, cublasZdrot_v2,
        cublasZdrot_v2_64, "Zdrot");
    run_complex_rot_ex_case<cuDoubleComplex, double, cuDoubleComplex>(
        handle, zx, zy, cuDoubleComplex{0.5, 0.0},
        cuDoubleComplex{0.25, -0.5}, cuDoubleComplex{0.75, 0.0},
        cuDoubleComplex{-0.25, 0.5}, CUDA_C_64F, "RotEx complex-double");
    run_real_rot_ex_case<cuDoubleComplex, double>(
        handle, zx, zy, 0.5, 0.25, 0.75, -0.5, CUDA_C_64F,
        "RotEx complex-double real cs");
}

int main() {
    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&handle));
    test_real_rot(handle);
    test_complex_rot(handle);
    CHECK_CUBLAS(cublasDestroy(handle));
    std::puts("cuBLAS Level-1 rot test passed");
    return 0;
}
