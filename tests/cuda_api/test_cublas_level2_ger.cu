#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call)                                                        \
  do {                                                                          \
    cudaError_t err__ = (call);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      std::printf("CUDA failure %s:%d: %s\n", __FILE__, __LINE__,              \
                  cudaGetErrorString(err__));                                   \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

#define CHECK_CUBLAS(call)                                                      \
  do {                                                                          \
    cublasStatus_t st__ = (call);                                               \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                        \
      std::printf("cuBLAS failure %s:%d: %d\n", __FILE__, __LINE__,            \
                  static_cast<int>(st__));                                      \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

template <typename T> struct Ops;

template <> struct Ops<float> {
  static float value(float real, float) { return real; }
  static float add(float a, float b) { return a + b; }
  static float mul(float a, float b) { return a * b; }
  static float conj(float a) { return a; }
  static float alpha() { return 0.75f; }
  static bool near(float a, float b) { return std::fabs(a - b) < 1.0e-4f; }
};

template <> struct Ops<double> {
  static double value(float real, float) { return static_cast<double>(real); }
  static double add(double a, double b) { return a + b; }
  static double mul(double a, double b) { return a * b; }
  static double conj(double a) { return a; }
  static double alpha() { return 0.75; }
  static bool near(double a, double b) { return std::fabs(a - b) < 1.0e-9; }
};

template <> struct Ops<cuComplex> {
  static cuComplex value(float real, float imag) {
    return make_cuComplex(real, imag);
  }
  static cuComplex add(cuComplex a, cuComplex b) { return cuCaddf(a, b); }
  static cuComplex mul(cuComplex a, cuComplex b) { return cuCmulf(a, b); }
  static cuComplex conj(cuComplex a) { return cuConjf(a); }
  static cuComplex alpha() { return make_cuComplex(0.75f, -0.25f); }
  static bool near(cuComplex a, cuComplex b) {
    return std::fabs(a.x - b.x) < 1.0e-3f &&
           std::fabs(a.y - b.y) < 1.0e-3f;
  }
};

template <> struct Ops<cuDoubleComplex> {
  static cuDoubleComplex value(float real, float imag) {
    return make_cuDoubleComplex(static_cast<double>(real),
                                static_cast<double>(imag));
  }
  static cuDoubleComplex add(cuDoubleComplex a, cuDoubleComplex b) {
    return cuCadd(a, b);
  }
  static cuDoubleComplex mul(cuDoubleComplex a, cuDoubleComplex b) {
    return cuCmul(a, b);
  }
  static cuDoubleComplex conj(cuDoubleComplex a) { return cuConj(a); }
  static cuDoubleComplex alpha() { return make_cuDoubleComplex(0.75, -0.25); }
  static bool near(cuDoubleComplex a, cuDoubleComplex b) {
    return std::fabs(a.x - b.x) < 1.0e-9 &&
           std::fabs(a.y - b.y) < 1.0e-9;
  }
};

template <typename T, bool Conjugate> struct Ger;

template <> struct Ger<float, false> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n,
                               const float *alpha, const float *x, int incx,
                               const float *y, int incy, float *a, int lda) {
    return cublasSger_v2(handle, m, n, alpha, x, incx, y, incy, a, lda);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               const float *alpha, const float *x,
                               int64_t incx, const float *y, int64_t incy,
                               float *a, int64_t lda) {
    return cublasSger_v2_64(handle, m, n, alpha, x, incx, y, incy, a, lda);
  }
};

template <> struct Ger<double, false> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n,
                               const double *alpha, const double *x, int incx,
                               const double *y, int incy, double *a, int lda) {
    return cublasDger_v2(handle, m, n, alpha, x, incx, y, incy, a, lda);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               const double *alpha, const double *x,
                               int64_t incx, const double *y, int64_t incy,
                               double *a, int64_t lda) {
    return cublasDger_v2_64(handle, m, n, alpha, x, incx, y, incy, a, lda);
  }
};

template <> struct Ger<cuComplex, false> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n,
                               const cuComplex *alpha, const cuComplex *x,
                               int incx, const cuComplex *y, int incy,
                               cuComplex *a, int lda) {
    return cublasCgeru_v2(handle, m, n, alpha, x, incx, y, incy, a, lda);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               const cuComplex *alpha, const cuComplex *x,
                               int64_t incx, const cuComplex *y, int64_t incy,
                               cuComplex *a, int64_t lda) {
    return cublasCgeru_v2_64(handle, m, n, alpha, x, incx, y, incy, a, lda);
  }
};

template <> struct Ger<cuComplex, true> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n,
                               const cuComplex *alpha, const cuComplex *x,
                               int incx, const cuComplex *y, int incy,
                               cuComplex *a, int lda) {
    return cublasCgerc_v2(handle, m, n, alpha, x, incx, y, incy, a, lda);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               const cuComplex *alpha, const cuComplex *x,
                               int64_t incx, const cuComplex *y, int64_t incy,
                               cuComplex *a, int64_t lda) {
    return cublasCgerc_v2_64(handle, m, n, alpha, x, incx, y, incy, a, lda);
  }
};

template <> struct Ger<cuDoubleComplex, false> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *x, int incx,
                               const cuDoubleComplex *y, int incy,
                               cuDoubleComplex *a, int lda) {
    return cublasZgeru_v2(handle, m, n, alpha, x, incx, y, incy, a, lda);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *x, int64_t incx,
                               const cuDoubleComplex *y, int64_t incy,
                               cuDoubleComplex *a, int64_t lda) {
    return cublasZgeru_v2_64(handle, m, n, alpha, x, incx, y, incy, a, lda);
  }
};

template <> struct Ger<cuDoubleComplex, true> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *x, int incx,
                               const cuDoubleComplex *y, int incy,
                               cuDoubleComplex *a, int lda) {
    return cublasZgerc_v2(handle, m, n, alpha, x, incx, y, incy, a, lda);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *x, int64_t incx,
                               const cuDoubleComplex *y, int64_t incy,
                               cuDoubleComplex *a, int64_t lda) {
    return cublasZgerc_v2_64(handle, m, n, alpha, x, incx, y, incy, a, lda);
  }
};

template <typename T>
static std::vector<T> expected_ger(const std::vector<T> &a,
                                   const std::vector<T> &x,
                                   const std::vector<T> &y, int m, int n,
                                   T alpha, bool conjugate) {
  std::vector<T> out = a;
  for (int col = 0; col < n; ++col) {
    T y_value = conjugate ? Ops<T>::conj(y[col]) : y[col];
    for (int row = 0; row < m; ++row) {
      out[row + col * m] = Ops<T>::add(
          out[row + col * m], Ops<T>::mul(alpha, Ops<T>::mul(x[row], y_value)));
    }
  }
  return out;
}

template <typename T>
static void fill_inputs(std::vector<T> &a, std::vector<T> &x,
                        std::vector<T> &y, int m, int n) {
  for (int i = 0; i < m * n; ++i) {
    a[i] = Ops<T>::value(static_cast<float>(i + 1),
                         static_cast<float>((i % 3) - 1) * 0.125f);
  }
  for (int i = 0; i < m; ++i) {
    x[i] = Ops<T>::value(static_cast<float>(i + 2),
                         static_cast<float>(i + 1) * -0.25f);
  }
  for (int i = 0; i < n; ++i) {
    y[i] = Ops<T>::value(static_cast<float>(i - 1),
                         static_cast<float>(i + 2) * 0.375f);
  }
}

template <typename T, bool Conjugate>
static void run_case(cublasHandle_t handle, const char *name, bool use_64,
                     bool device_scalar) {
  const int m = 3;
  const int n = 2;
  std::vector<T> a(m * n);
  std::vector<T> x(m);
  std::vector<T> y(n);
  fill_inputs(a, x, y, m, n);
  T alpha = Ops<T>::alpha();
  std::vector<T> expected = expected_ger(a, x, y, m, n, alpha, Conjugate);

  T *d_a = nullptr;
  T *d_x = nullptr;
  T *d_y = nullptr;
  T *d_alpha = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, a.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_y, y.size() * sizeof(T)));
  CHECK_CUDA(
      cudaMemcpy(d_a, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_x, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_y, y.data(), y.size() * sizeof(T), cudaMemcpyHostToDevice));

  const T *alpha_arg = &alpha;
  if (device_scalar) {
    CHECK_CUDA(cudaMalloc(&d_alpha, sizeof(T)));
    CHECK_CUDA(cudaMemcpy(d_alpha, &alpha, sizeof(T), cudaMemcpyHostToDevice));
    alpha_arg = d_alpha;
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  } else {
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  }

  if (use_64) {
    CHECK_CUBLAS(
        (Ger<T, Conjugate>::call64(handle, m, n, alpha_arg, d_x, 1, d_y, 1,
                                   d_a, m)));
  } else {
    CHECK_CUBLAS(
        (Ger<T, Conjugate>::call32(handle, m, n, alpha_arg, d_x, 1, d_y, 1,
                                   d_a, m)));
  }
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<T> out(a.size());
  CHECK_CUDA(cudaMemcpy(out.data(), d_a, out.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < out.size(); ++i) {
    if (!Ops<T>::near(out[i], expected[i])) {
      std::printf("%s mismatch index %zu use_64=%d device_scalar=%d\n", name,
                  i, use_64 ? 1 : 0, device_scalar ? 1 : 0);
      std::exit(EXIT_FAILURE);
    }
  }

  cudaFree(d_alpha);
  cudaFree(d_y);
  cudaFree(d_x);
  cudaFree(d_a);
}

template <typename T, bool Conjugate>
static void run_type(cublasHandle_t handle, const char *name) {
  run_case<T, Conjugate>(handle, name, false, false);
  run_case<T, Conjugate>(handle, name, false, true);
  run_case<T, Conjugate>(handle, name, true, false);
  run_case<T, Conjugate>(handle, name, true, true);
}

int main() {
  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));
  run_type<float, false>(handle, "sger");
  run_type<double, false>(handle, "dger");
  run_type<cuComplex, false>(handle, "cgeru");
  run_type<cuComplex, true>(handle, "cgerc");
  run_type<cuDoubleComplex, false>(handle, "zgeru");
  run_type<cuDoubleComplex, true>(handle, "zgerc");
  CHECK_CUBLAS(cublasDestroy(handle));
  return 0;
}
