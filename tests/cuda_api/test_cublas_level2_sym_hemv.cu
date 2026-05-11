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
  static float zero() { return 0.0f; }
  static float add(float a, float b) { return a + b; }
  static float mul(float a, float b) { return a * b; }
  static float conj(float a) { return a; }
  static float alpha() { return 0.75f; }
  static float beta() { return -0.5f; }
  static bool near(float a, float b) { return std::fabs(a - b) < 1.0e-4f; }
};

template <> struct Ops<double> {
  static double value(float real, float) { return static_cast<double>(real); }
  static double zero() { return 0.0; }
  static double add(double a, double b) { return a + b; }
  static double mul(double a, double b) { return a * b; }
  static double conj(double a) { return a; }
  static double alpha() { return 0.75; }
  static double beta() { return -0.5; }
  static bool near(double a, double b) { return std::fabs(a - b) < 1.0e-9; }
};

template <> struct Ops<cuComplex> {
  static cuComplex value(float real, float imag) {
    return make_cuComplex(real, imag);
  }
  static cuComplex zero() { return make_cuComplex(0.0f, 0.0f); }
  static cuComplex add(cuComplex a, cuComplex b) { return cuCaddf(a, b); }
  static cuComplex mul(cuComplex a, cuComplex b) { return cuCmulf(a, b); }
  static cuComplex conj(cuComplex a) { return cuConjf(a); }
  static cuComplex alpha() { return make_cuComplex(0.75f, -0.25f); }
  static cuComplex beta() { return make_cuComplex(-0.5f, 0.125f); }
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
  static cuDoubleComplex zero() { return make_cuDoubleComplex(0.0, 0.0); }
  static cuDoubleComplex add(cuDoubleComplex a, cuDoubleComplex b) {
    return cuCadd(a, b);
  }
  static cuDoubleComplex mul(cuDoubleComplex a, cuDoubleComplex b) {
    return cuCmul(a, b);
  }
  static cuDoubleComplex conj(cuDoubleComplex a) { return cuConj(a); }
  static cuDoubleComplex alpha() { return make_cuDoubleComplex(0.75, -0.25); }
  static cuDoubleComplex beta() { return make_cuDoubleComplex(-0.5, 0.125); }
  static bool near(cuDoubleComplex a, cuDoubleComplex b) {
    return std::fabs(a.x - b.x) < 1.0e-9 &&
           std::fabs(a.y - b.y) < 1.0e-9;
  }
};

template <typename T, bool Hermitian> struct SymHemv;

template <> struct SymHemv<float, false> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const float *alpha, const float *a,
                               int lda, const float *x, int incx,
                               const float *beta, float *y, int incy) {
    return cublasSsymv_v2(handle, uplo, n, alpha, a, lda, x, incx, beta, y,
                          incy);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const float *alpha, const float *a,
                               int64_t lda, const float *x, int64_t incx,
                               const float *beta, float *y, int64_t incy) {
    return cublasSsymv_v2_64(handle, uplo, n, alpha, a, lda, x, incx, beta, y,
                             incy);
  }
};

template <> struct SymHemv<double, false> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const double *alpha, const double *a,
                               int lda, const double *x, int incx,
                               const double *beta, double *y, int incy) {
    return cublasDsymv_v2(handle, uplo, n, alpha, a, lda, x, incx, beta, y,
                          incy);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const double *alpha, const double *a,
                               int64_t lda, const double *x, int64_t incx,
                               const double *beta, double *y, int64_t incy) {
    return cublasDsymv_v2_64(handle, uplo, n, alpha, a, lda, x, incx, beta, y,
                             incy);
  }
};

template <> struct SymHemv<cuComplex, false> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuComplex *alpha,
                               const cuComplex *a, int lda,
                               const cuComplex *x, int incx,
                               const cuComplex *beta, cuComplex *y, int incy) {
    return cublasCsymv_v2(handle, uplo, n, alpha, a, lda, x, incx, beta, y,
                          incy);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const cuComplex *alpha,
                               const cuComplex *a, int64_t lda,
                               const cuComplex *x, int64_t incx,
                               const cuComplex *beta, cuComplex *y,
                               int64_t incy) {
    return cublasCsymv_v2_64(handle, uplo, n, alpha, a, lda, x, incx, beta, y,
                             incy);
  }
};

template <> struct SymHemv<cuDoubleComplex, false> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuDoubleComplex *alpha,
                               const cuDoubleComplex *a, int lda,
                               const cuDoubleComplex *x, int incx,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *y, int incy) {
    return cublasZsymv_v2(handle, uplo, n, alpha, a, lda, x, incx, beta, y,
                          incy);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const cuDoubleComplex *alpha,
                               const cuDoubleComplex *a, int64_t lda,
                               const cuDoubleComplex *x, int64_t incx,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *y, int64_t incy) {
    return cublasZsymv_v2_64(handle, uplo, n, alpha, a, lda, x, incx, beta, y,
                             incy);
  }
};

template <> struct SymHemv<cuComplex, true> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuComplex *alpha,
                               const cuComplex *a, int lda,
                               const cuComplex *x, int incx,
                               const cuComplex *beta, cuComplex *y, int incy) {
    return cublasChemv_v2(handle, uplo, n, alpha, a, lda, x, incx, beta, y,
                          incy);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const cuComplex *alpha,
                               const cuComplex *a, int64_t lda,
                               const cuComplex *x, int64_t incx,
                               const cuComplex *beta, cuComplex *y,
                               int64_t incy) {
    return cublasChemv_v2_64(handle, uplo, n, alpha, a, lda, x, incx, beta, y,
                             incy);
  }
};

template <> struct SymHemv<cuDoubleComplex, true> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuDoubleComplex *alpha,
                               const cuDoubleComplex *a, int lda,
                               const cuDoubleComplex *x, int incx,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *y, int incy) {
    return cublasZhemv_v2(handle, uplo, n, alpha, a, lda, x, incx, beta, y,
                          incy);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const cuDoubleComplex *alpha,
                               const cuDoubleComplex *a, int64_t lda,
                               const cuDoubleComplex *x, int64_t incx,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *y, int64_t incy) {
    return cublasZhemv_v2_64(handle, uplo, n, alpha, a, lda, x, incx, beta, y,
                             incy);
  }
};

template <typename T>
static T matrix_value(const std::vector<T> &a, cublasFillMode_t uplo, int lda,
                      int row, int col, bool hermitian) {
  if (row == col) {
    return a[row + col * lda];
  }
  if (uplo == CUBLAS_FILL_MODE_UPPER) {
    if (row < col) {
      return a[row + col * lda];
    }
    T stored = a[col + row * lda];
    return hermitian ? Ops<T>::conj(stored) : stored;
  }
  if (row > col) {
    return a[row + col * lda];
  }
  T stored = a[col + row * lda];
  return hermitian ? Ops<T>::conj(stored) : stored;
}

template <typename T>
static std::vector<T> expected_mv(const std::vector<T> &a,
                                  const std::vector<T> &x,
                                  const std::vector<T> &y,
                                  cublasFillMode_t uplo, int n, int lda,
                                  T alpha, T beta, bool hermitian) {
  std::vector<T> out(n);
  for (int row = 0; row < n; ++row) {
    T sum = Ops<T>::zero();
    for (int col = 0; col < n; ++col) {
      T aval = matrix_value(a, uplo, lda, row, col, hermitian);
      sum = Ops<T>::add(sum, Ops<T>::mul(aval, x[col]));
    }
    out[row] = Ops<T>::add(Ops<T>::mul(alpha, sum), Ops<T>::mul(beta, y[row]));
  }
  return out;
}

template <typename T>
static void fill_inputs(std::vector<T> &a, std::vector<T> &x,
                        std::vector<T> &y, int n, int lda, bool hermitian) {
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < lda; ++row) {
      float imag = hermitian && row == col ? 0.0f
                                           : static_cast<float>((row - col) % 5) *
                                                 0.125f;
      a[row + col * lda] =
          Ops<T>::value(static_cast<float>(10 * col + row + 1), imag);
    }
  }
  for (int i = 0; i < n; ++i) {
    x[i] = Ops<T>::value(static_cast<float>(i + 2),
                         static_cast<float>(i - 1) * 0.25f);
    y[i] = Ops<T>::value(static_cast<float>(i - 3),
                         static_cast<float>(i + 1) * -0.125f);
  }
}

template <typename T, bool Hermitian>
static void run_case(cublasHandle_t handle, const char *name,
                     cublasFillMode_t uplo, bool use_64, bool device_scalar) {
  const int n = 4;
  const int lda = 6;

  std::vector<T> a(lda * n);
  std::vector<T> x(n);
  std::vector<T> y(n);
  fill_inputs(a, x, y, n, lda, Hermitian);
  T alpha = Ops<T>::alpha();
  T beta = Ops<T>::beta();
  std::vector<T> expected =
      expected_mv(a, x, y, uplo, n, lda, alpha, beta, Hermitian);

  T *d_a = nullptr;
  T *d_x = nullptr;
  T *d_y = nullptr;
  T *d_alpha = nullptr;
  T *d_beta = nullptr;
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
  const T *beta_arg = &beta;
  if (device_scalar) {
    CHECK_CUDA(cudaMalloc(&d_alpha, sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_beta, sizeof(T)));
    CHECK_CUDA(cudaMemcpy(d_alpha, &alpha, sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_beta, &beta, sizeof(T), cudaMemcpyHostToDevice));
    alpha_arg = d_alpha;
    beta_arg = d_beta;
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  } else {
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  }

  if (use_64) {
    CHECK_CUBLAS((SymHemv<T, Hermitian>::call64(
        handle, uplo, n, alpha_arg, d_a, lda, d_x, 1, beta_arg, d_y, 1)));
  } else {
    CHECK_CUBLAS((SymHemv<T, Hermitian>::call32(
        handle, uplo, n, alpha_arg, d_a, lda, d_x, 1, beta_arg, d_y, 1)));
  }

  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<T> out(n);
  CHECK_CUDA(
      cudaMemcpy(out.data(), d_y, out.size() * sizeof(T), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    if (!Ops<T>::near(out[i], expected[i])) {
      std::printf("%s mismatch index=%d uplo=%d use64=%d device_scalar=%d\n",
                  name, i, static_cast<int>(uplo), use_64 ? 1 : 0,
                  device_scalar ? 1 : 0);
      std::exit(EXIT_FAILURE);
    }
  }

  cudaFree(d_beta);
  cudaFree(d_alpha);
  cudaFree(d_y);
  cudaFree(d_x);
  cudaFree(d_a);
}

template <typename T, bool Hermitian>
static void run_type(cublasHandle_t handle, const char *name) {
  cublasFillMode_t uplos[] = {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER};
  for (int u = 0; u < 2; ++u) {
    for (int use_64 = 0; use_64 < 2; ++use_64) {
      for (int device_scalar = 0; device_scalar < 2; ++device_scalar) {
        run_case<T, Hermitian>(handle, name, uplos[u], use_64 != 0,
                               device_scalar != 0);
      }
    }
  }
}

int main() {
  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));

  run_type<float, false>(handle, "ssymv");
  run_type<double, false>(handle, "dsymv");
  run_type<cuComplex, false>(handle, "csymv");
  run_type<cuDoubleComplex, false>(handle, "zsymv");
  run_type<cuComplex, true>(handle, "chemv");
  run_type<cuDoubleComplex, true>(handle, "zhemv");

  CHECK_CUBLAS(cublasDestroy(handle));
  return EXIT_SUCCESS;
}
