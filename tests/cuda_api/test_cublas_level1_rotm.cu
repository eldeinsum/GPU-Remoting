#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
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

template <typename T>
static void expect_close(T got, T want, const char *label) {
    if (!close_value(got, want)) {
        std::fprintf(stderr, "%s mismatch\n", label);
        std::exit(1);
    }
}

template <typename T>
static T make_value(double value);

template <>
float make_value<float>(double value) {
    return static_cast<float>(value);
}

template <>
double make_value<double>(double value) {
    return value;
}

template <>
__half make_value<__half>(double value) {
    return __float2half(static_cast<float>(value));
}

template <>
__nv_bfloat16 make_value<__nv_bfloat16>(double value) {
    return __float2bfloat16(static_cast<float>(value));
}

static double scalar_value(float value) { return value; }

static double scalar_value(double value) { return value; }

static double scalar_value(__half value) { return __half2float(value); }

static double scalar_value(__nv_bfloat16 value) {
    return __bfloat162float(value);
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
cudaDataType_t cuda_type<__half>() {
    return CUDA_R_16F;
}

template <>
cudaDataType_t cuda_type<__nv_bfloat16>() {
    return CUDA_R_16BF;
}

template <typename T>
static T *to_device_scalar(T value) {
    T *device = nullptr;
    CHECK_CUDA(cudaMalloc(&device, sizeof(T)));
    CHECK_CUDA(cudaMemcpy(device, &value, sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

template <typename T>
static T from_device_scalar(const T *device) {
    T host{};
    CHECK_CUDA(cudaMemcpy(&host, device, sizeof(T), cudaMemcpyDeviceToHost));
    return host;
}

template <typename T>
static T *to_device_vector(const std::vector<T> &host) {
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
static std::vector<T> from_device_vector(const T *device, size_t count) {
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

template <typename T, typename Fn>
static void run_real_rotg_case(cublasHandle_t handle, T a0, T b0, Fn fn,
                               const char *label) {
    T host_a = a0;
    T host_b = b0;
    T host_c = 0;
    T host_s = 0;
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUBLAS(fn(handle, &host_a, &host_b, &host_c, &host_s));

    T *device_a = to_device_scalar(a0);
    T *device_b = to_device_scalar(b0);
    T *device_c = to_device_scalar(static_cast<T>(0));
    T *device_s = to_device_scalar(static_cast<T>(0));
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    CHECK_CUBLAS(fn(handle, device_a, device_b, device_c, device_s));

    expect_close(from_device_scalar(device_a), host_a, label);
    expect_close(from_device_scalar(device_b), host_b, label);
    expect_close(from_device_scalar(device_c), host_c, label);
    expect_close(from_device_scalar(device_s), host_s, label);
    expect_close(host_c * b0 - host_s * a0, static_cast<T>(0), label);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUDA(cudaFree(device_s));
    CHECK_CUDA(cudaFree(device_c));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_a));
}

template <typename VecT, typename RealT, typename Fn>
static void run_complex_rotg_case(cublasHandle_t handle, VecT a0, VecT b0,
                                  Fn fn, const char *label) {
    VecT host_a = a0;
    VecT host_b = b0;
    RealT host_c = 0;
    VecT host_s{};
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUBLAS(fn(handle, &host_a, &host_b, &host_c, &host_s));

    VecT *device_a = to_device_scalar(a0);
    VecT *device_b = to_device_scalar(b0);
    RealT *device_c = to_device_scalar(static_cast<RealT>(0));
    VecT *device_s = to_device_scalar(VecT{});
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    CHECK_CUBLAS(fn(handle, device_a, device_b, device_c, device_s));

    expect_close(from_device_scalar(device_a), host_a, label);
    expect_close(from_device_scalar(device_b), host_b, label);
    expect_close(from_device_scalar(device_c), host_c, label);
    expect_close(from_device_scalar(device_s), host_s, label);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUDA(cudaFree(device_s));
    CHECK_CUDA(cudaFree(device_c));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_a));
}

template <typename T>
static void run_real_rotg_ex_case(cublasHandle_t handle, T a0, T b0,
                                  cudaDataType_t execution_type,
                                  const char *label) {
    const cudaDataType_t value_type = cuda_type<T>();
    T host_a = a0;
    T host_b = b0;
    T host_c = make_value<T>(0);
    T host_s = make_value<T>(0);
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUBLAS(cublasRotgEx(handle, &host_a, &host_b, value_type, &host_c,
                              &host_s, value_type, execution_type));

    T *device_a = to_device_scalar(a0);
    T *device_b = to_device_scalar(b0);
    T *device_c = to_device_scalar(make_value<T>(0));
    T *device_s = to_device_scalar(make_value<T>(0));
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    CHECK_CUBLAS(cublasRotgEx(handle, device_a, device_b, value_type, device_c,
                              device_s, value_type, execution_type));

    expect_close(from_device_scalar(device_a), host_a, label);
    expect_close(from_device_scalar(device_b), host_b, label);
    expect_close(from_device_scalar(device_c), host_c, label);
    expect_close(from_device_scalar(device_s), host_s, label);
    if (std::fabs(scalar_value(host_c) * scalar_value(b0) -
                  scalar_value(host_s) * scalar_value(a0)) > 5e-2) {
        std::fprintf(stderr, "%s rotation invariant mismatch\n", label);
        std::exit(1);
    }

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUDA(cudaFree(device_s));
    CHECK_CUDA(cudaFree(device_c));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_a));
}

template <typename T>
static void rotm_expected(const std::vector<T> &x, const std::vector<T> &y,
                          const std::array<T, 5> &param, std::vector<T> *out_x,
                          std::vector<T> *out_y) {
    out_x->clear();
    out_y->clear();
    out_x->reserve(x.size());
    out_y->reserve(y.size());
    const double flag = scalar_value(param[0]);
    for (size_t i = 0; i < x.size(); ++i) {
        double h11 = 0;
        double h12 = 0;
        double h21 = 0;
        double h22 = 0;
        if (flag == -2.0) {
            h11 = 1;
            h22 = 1;
        } else if (flag == -1.0) {
            h11 = scalar_value(param[1]);
            h21 = scalar_value(param[2]);
            h12 = scalar_value(param[3]);
            h22 = scalar_value(param[4]);
        } else if (flag == 0.0) {
            h11 = 1;
            h21 = scalar_value(param[2]);
            h12 = scalar_value(param[3]);
            h22 = 1;
        } else if (flag == 1.0) {
            h11 = scalar_value(param[1]);
            h21 = -1;
            h12 = 1;
            h22 = scalar_value(param[4]);
        }
        out_x->push_back(
            make_value<T>(h11 * scalar_value(x[i]) + h12 * scalar_value(y[i])));
        out_y->push_back(
            make_value<T>(h21 * scalar_value(x[i]) + h22 * scalar_value(y[i])));
    }
}

template <typename T, typename Fn>
static void run_rotm_once(cublasHandle_t handle, const std::vector<T> &x,
                          const std::vector<T> &y,
                          const std::array<T, 5> &param, Fn fn,
                          const char *label) {
    const int n = static_cast<int>(x.size());
    T *host_x = to_device_vector(x);
    T *host_y = to_device_vector(y);
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUBLAS(fn(handle, n, host_x, 1, host_y, 1, param.data()));
    std::vector<T> got_host_x = from_device_vector(host_x, x.size());
    std::vector<T> got_host_y = from_device_vector(host_y, y.size());

    T *device_x = to_device_vector(x);
    T *device_y = to_device_vector(y);
    std::vector<T> param_vec(param.begin(), param.end());
    T *device_param = to_device_vector(param_vec);
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    CHECK_CUBLAS(fn(handle, n, device_x, 1, device_y, 1, device_param));
    std::vector<T> got_device_x = from_device_vector(device_x, x.size());
    std::vector<T> got_device_y = from_device_vector(device_y, y.size());

    std::vector<T> expected_x;
    std::vector<T> expected_y;
    rotm_expected(x, y, param, &expected_x, &expected_y);
    expect_vector(got_host_x, expected_x, label);
    expect_vector(got_host_y, expected_y, label);
    expect_vector(got_device_x, got_host_x, label);
    expect_vector(got_device_y, got_host_y, label);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUDA(cudaFree(device_param));
    CHECK_CUDA(cudaFree(device_y));
    CHECK_CUDA(cudaFree(device_x));
    CHECK_CUDA(cudaFree(host_y));
    CHECK_CUDA(cudaFree(host_x));
}

template <typename T, typename Fn64>
static void run_rotm64_once(cublasHandle_t handle, const std::vector<T> &x,
                            const std::vector<T> &y,
                            const std::array<T, 5> &param, Fn64 fn,
                            const char *label) {
    const int64_t n = static_cast<int64_t>(x.size());
    T *host_x = to_device_vector(x);
    T *host_y = to_device_vector(y);
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUBLAS(fn(handle, n, host_x, int64_t{1}, host_y, int64_t{1},
                    param.data()));
    std::vector<T> got_host_x = from_device_vector(host_x, x.size());
    std::vector<T> got_host_y = from_device_vector(host_y, y.size());

    T *device_x = to_device_vector(x);
    T *device_y = to_device_vector(y);
    std::vector<T> param_vec(param.begin(), param.end());
    T *device_param = to_device_vector(param_vec);
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    CHECK_CUBLAS(fn(handle, n, device_x, int64_t{1}, device_y, int64_t{1},
                    device_param));
    std::vector<T> got_device_x = from_device_vector(device_x, x.size());
    std::vector<T> got_device_y = from_device_vector(device_y, y.size());

    expect_vector(got_device_x, got_host_x, label);
    expect_vector(got_device_y, got_host_y, label);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUDA(cudaFree(device_param));
    CHECK_CUDA(cudaFree(device_y));
    CHECK_CUDA(cudaFree(device_x));
    CHECK_CUDA(cudaFree(host_y));
    CHECK_CUDA(cudaFree(host_x));
}

template <typename T, typename Fn, typename Fn64>
static void run_rotm_case(cublasHandle_t handle, Fn fn, Fn64 fn64,
                          const char *label) {
    std::vector<T> x{make_value<T>(1), make_value<T>(-2), make_value<T>(3)};
    std::vector<T> y{make_value<T>(4), make_value<T>(5), make_value<T>(-6)};
    std::vector<std::array<T, 5>> params{
        std::array<T, 5>{make_value<T>(-1), make_value<T>(1.25),
                         make_value<T>(-0.5), make_value<T>(0.75),
                         make_value<T>(0.5)},
        std::array<T, 5>{make_value<T>(0), make_value<T>(0),
                         make_value<T>(0.25), make_value<T>(-0.5),
                         make_value<T>(0)},
        std::array<T, 5>{make_value<T>(1), make_value<T>(1.5),
                         make_value<T>(0), make_value<T>(0),
                         make_value<T>(0.25)},
        std::array<T, 5>{make_value<T>(-2), make_value<T>(0),
                         make_value<T>(0), make_value<T>(0),
                         make_value<T>(0)}};

    for (size_t i = 0; i < params.size(); ++i) {
        run_rotm_once(handle, x, y, params[i], fn, label);
        run_rotm64_once(handle, x, y, params[i], fn64, label);
    }
}

template <typename T>
static void run_rotm_ex_once(cublasHandle_t handle, const std::vector<T> &x,
                             const std::vector<T> &y,
                             const std::array<T, 5> &param,
                             cudaDataType_t execution_type,
                             const char *label) {
    const int n = static_cast<int>(x.size());
    const cudaDataType_t value_type = cuda_type<T>();
    T *host_x = to_device_vector(x);
    T *host_y = to_device_vector(y);
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUBLAS(cublasRotmEx(handle, n, host_x, value_type, 1, host_y,
                              value_type, 1, param.data(), value_type,
                              execution_type));
    std::vector<T> got_host_x = from_device_vector(host_x, x.size());
    std::vector<T> got_host_y = from_device_vector(host_y, y.size());

    T *device_x = to_device_vector(x);
    T *device_y = to_device_vector(y);
    std::vector<T> param_vec(param.begin(), param.end());
    T *device_param = to_device_vector(param_vec);
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    CHECK_CUBLAS(cublasRotmEx(handle, n, device_x, value_type, 1, device_y,
                              value_type, 1, device_param, value_type,
                              execution_type));
    std::vector<T> got_device_x = from_device_vector(device_x, x.size());
    std::vector<T> got_device_y = from_device_vector(device_y, y.size());

    std::vector<T> expected_x;
    std::vector<T> expected_y;
    rotm_expected(x, y, param, &expected_x, &expected_y);
    expect_vector(got_host_x, expected_x, label);
    expect_vector(got_host_y, expected_y, label);
    expect_vector(got_device_x, got_host_x, label);
    expect_vector(got_device_y, got_host_y, label);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUDA(cudaFree(device_param));
    CHECK_CUDA(cudaFree(device_y));
    CHECK_CUDA(cudaFree(device_x));
    CHECK_CUDA(cudaFree(host_y));
    CHECK_CUDA(cudaFree(host_x));
}

template <typename T>
static void run_rotm_ex64_once(cublasHandle_t handle, const std::vector<T> &x,
                               const std::vector<T> &y,
                               const std::array<T, 5> &param,
                               cudaDataType_t execution_type,
                               const char *label) {
    const int64_t n = static_cast<int64_t>(x.size());
    const cudaDataType_t value_type = cuda_type<T>();
    T *host_x = to_device_vector(x);
    T *host_y = to_device_vector(y);
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUBLAS(cublasRotmEx_64(handle, n, host_x, value_type, int64_t{1},
                                 host_y, value_type, int64_t{1}, param.data(),
                                 value_type, execution_type));
    std::vector<T> got_host_x = from_device_vector(host_x, x.size());
    std::vector<T> got_host_y = from_device_vector(host_y, y.size());

    T *device_x = to_device_vector(x);
    T *device_y = to_device_vector(y);
    std::vector<T> param_vec(param.begin(), param.end());
    T *device_param = to_device_vector(param_vec);
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    CHECK_CUBLAS(cublasRotmEx_64(handle, n, device_x, value_type, int64_t{1},
                                 device_y, value_type, int64_t{1},
                                 device_param, value_type, execution_type));
    std::vector<T> got_device_x = from_device_vector(device_x, x.size());
    std::vector<T> got_device_y = from_device_vector(device_y, y.size());

    expect_vector(got_device_x, got_host_x, label);
    expect_vector(got_device_y, got_host_y, label);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUDA(cudaFree(device_param));
    CHECK_CUDA(cudaFree(device_y));
    CHECK_CUDA(cudaFree(device_x));
    CHECK_CUDA(cudaFree(host_y));
    CHECK_CUDA(cudaFree(host_x));
}

template <typename T>
static void run_rotm_ex_case(cublasHandle_t handle,
                             cudaDataType_t execution_type,
                             const char *label) {
    std::vector<T> x{make_value<T>(1), make_value<T>(-2), make_value<T>(3)};
    std::vector<T> y{make_value<T>(4), make_value<T>(5), make_value<T>(-6)};
    std::vector<std::array<T, 5>> params{
        std::array<T, 5>{make_value<T>(-1), make_value<T>(1.25),
                         make_value<T>(-0.5), make_value<T>(0.75),
                         make_value<T>(0.5)},
        std::array<T, 5>{make_value<T>(0), make_value<T>(0),
                         make_value<T>(0.25), make_value<T>(-0.5),
                         make_value<T>(0)},
        std::array<T, 5>{make_value<T>(1), make_value<T>(1.5),
                         make_value<T>(0), make_value<T>(0),
                         make_value<T>(0.25)},
        std::array<T, 5>{make_value<T>(-2), make_value<T>(0),
                         make_value<T>(0), make_value<T>(0),
                         make_value<T>(0)}};

    for (size_t i = 0; i < params.size(); ++i) {
        run_rotm_ex_once(handle, x, y, params[i], execution_type, label);
        run_rotm_ex64_once(handle, x, y, params[i], execution_type, label);
    }
}

template <typename T, typename Fn>
static void run_rotmg_case(cublasHandle_t handle, T d10, T d20, T x10, T y10,
                           Fn fn, const char *label) {
    T host_d1 = d10;
    T host_d2 = d20;
    T host_x1 = x10;
    T host_y1 = y10;
    std::array<T, 5> host_param{};
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUBLAS(fn(handle, &host_d1, &host_d2, &host_x1, &host_y1,
                    host_param.data()));

    T *device_d1 = to_device_scalar(d10);
    T *device_d2 = to_device_scalar(d20);
    T *device_x1 = to_device_scalar(x10);
    T *device_y1 = to_device_scalar(y10);
    std::vector<T> zero_param(5, static_cast<T>(0));
    T *device_param = to_device_vector(zero_param);
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    CHECK_CUBLAS(fn(handle, device_d1, device_d2, device_x1, device_y1,
                    device_param));

    expect_close(from_device_scalar(device_d1), host_d1, label);
    expect_close(from_device_scalar(device_d2), host_d2, label);
    expect_close(from_device_scalar(device_x1), host_x1, label);
    std::vector<T> device_param_host = from_device_vector(device_param, 5);
    for (size_t i = 0; i < host_param.size(); ++i) {
        expect_close(device_param_host[i], host_param[i], label);
    }

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUDA(cudaFree(device_param));
    CHECK_CUDA(cudaFree(device_y1));
    CHECK_CUDA(cudaFree(device_x1));
    CHECK_CUDA(cudaFree(device_d2));
    CHECK_CUDA(cudaFree(device_d1));
}

template <typename T>
static void run_rotmg_ex_case(cublasHandle_t handle, T d10, T d20, T x10,
                              T y10, cudaDataType_t execution_type,
                              const char *label) {
    const cudaDataType_t value_type = cuda_type<T>();
    T host_d1 = d10;
    T host_d2 = d20;
    T host_x1 = x10;
    T host_y1 = y10;
    std::array<T, 5> host_param{};
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUBLAS(cublasRotmgEx(handle, &host_d1, value_type, &host_d2,
                               value_type, &host_x1, value_type, &host_y1,
                               value_type, host_param.data(), value_type,
                               execution_type));

    T *device_d1 = to_device_scalar(d10);
    T *device_d2 = to_device_scalar(d20);
    T *device_x1 = to_device_scalar(x10);
    T *device_y1 = to_device_scalar(y10);
    std::vector<T> zero_param(5, make_value<T>(0));
    T *device_param = to_device_vector(zero_param);
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    CHECK_CUBLAS(cublasRotmgEx(handle, device_d1, value_type, device_d2,
                               value_type, device_x1, value_type, device_y1,
                               value_type, device_param, value_type,
                               execution_type));

    expect_close(from_device_scalar(device_d1), host_d1, label);
    expect_close(from_device_scalar(device_d2), host_d2, label);
    expect_close(from_device_scalar(device_x1), host_x1, label);
    std::vector<T> host_param_vec(host_param.begin(), host_param.end());
    expect_vector(from_device_vector(device_param, 5), host_param_vec, label);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUDA(cudaFree(device_param));
    CHECK_CUDA(cudaFree(device_y1));
    CHECK_CUDA(cudaFree(device_x1));
    CHECK_CUDA(cudaFree(device_d2));
    CHECK_CUDA(cudaFree(device_d1));
}

int main() {
    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&handle));

    run_real_rotg_case<float>(handle, 3.0f, 4.0f, cublasSrotg_v2,
                              "cublasSrotg_v2");
    run_real_rotg_case<double>(handle, 6.0, -8.0, cublasDrotg_v2,
                               "cublasDrotg_v2");
    run_complex_rotg_case<cuComplex, float>(
        handle, cuComplex{3.0f, 1.0f}, cuComplex{2.0f, -4.0f},
        cublasCrotg_v2, "cublasCrotg_v2");
    run_complex_rotg_case<cuDoubleComplex, double>(
        handle, cuDoubleComplex{2.0, -1.0}, cuDoubleComplex{-3.0, 4.0},
        cublasZrotg_v2, "cublasZrotg_v2");
    run_real_rotg_ex_case<float>(handle, 3.0f, 4.0f, CUDA_R_32F,
                                 "cublasRotgEx float");
    run_real_rotg_ex_case<double>(handle, 6.0, -8.0, CUDA_R_64F,
                                  "cublasRotgEx double");
    run_real_rotg_ex_case<__half>(handle, make_value<__half>(3),
                                  make_value<__half>(4), CUDA_R_32F,
                                  "cublasRotgEx half");
    run_real_rotg_ex_case<__nv_bfloat16>(
        handle, make_value<__nv_bfloat16>(3), make_value<__nv_bfloat16>(4),
        CUDA_R_32F, "cublasRotgEx bfloat16");

    run_rotm_case<float>(handle, cublasSrotm_v2, cublasSrotm_v2_64,
                         "cublasSrotm_v2");
    run_rotm_case<double>(handle, cublasDrotm_v2, cublasDrotm_v2_64,
                          "cublasDrotm_v2");
    run_rotm_ex_case<float>(handle, CUDA_R_32F, "cublasRotmEx float");
    run_rotm_ex_case<double>(handle, CUDA_R_64F, "cublasRotmEx double");
    run_rotm_ex_case<__half>(handle, CUDA_R_32F, "cublasRotmEx half");
    run_rotm_ex_case<__nv_bfloat16>(handle, CUDA_R_32F,
                                    "cublasRotmEx bfloat16");

    run_rotmg_case<float>(handle, 2.0f, 3.0f, 4.0f, -1.5f, cublasSrotmg_v2,
                          "cublasSrotmg_v2");
    run_rotmg_case<double>(handle, 2.5, 1.5, -3.0, 4.0, cublasDrotmg_v2,
                           "cublasDrotmg_v2");
    run_rotmg_ex_case<float>(handle, 2.0f, 3.0f, 4.0f, -1.5f, CUDA_R_32F,
                             "cublasRotmgEx float");
    run_rotmg_ex_case<double>(handle, 2.5, 1.5, -3.0, 4.0, CUDA_R_64F,
                              "cublasRotmgEx double");
    run_rotmg_ex_case<__half>(handle, make_value<__half>(2),
                              make_value<__half>(3), make_value<__half>(4),
                              make_value<__half>(-1.5), CUDA_R_32F,
                              "cublasRotmgEx half");
    run_rotmg_ex_case<__nv_bfloat16>(
        handle, make_value<__nv_bfloat16>(2), make_value<__nv_bfloat16>(3),
        make_value<__nv_bfloat16>(4), make_value<__nv_bfloat16>(-1.5),
        CUDA_R_32F, "cublasRotmgEx bfloat16");

    CHECK_CUBLAS(cublasDestroy(handle));
    std::puts("cuBLAS Level-1 rotm/rotg test passed");
    return 0;
}
