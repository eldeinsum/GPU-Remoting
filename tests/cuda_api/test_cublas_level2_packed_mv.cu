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
  static float alpha() { return 1.25f; }
  static float beta() { return -0.25f; }
  static bool near(float a, float b) { return std::fabs(a - b) < 1.0e-4f; }
};

template <> struct Ops<double> {
  static double value(float real, float) { return static_cast<double>(real); }
  static double zero() { return 0.0; }
  static double add(double a, double b) { return a + b; }
  static double mul(double a, double b) { return a * b; }
  static double conj(double a) { return a; }
  static double alpha() { return 1.25; }
  static double beta() { return -0.25; }
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
  static cuComplex alpha() { return make_cuComplex(0.625f, -0.375f); }
  static cuComplex beta() { return make_cuComplex(-0.125f, 0.25f); }
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
  static cuDoubleComplex alpha() { return make_cuDoubleComplex(0.625, -0.375); }
  static cuDoubleComplex beta() { return make_cuDoubleComplex(-0.125, 0.25); }
  static bool near(cuDoubleComplex a, cuDoubleComplex b) {
    return std::fabs(a.x - b.x) < 1.0e-9 &&
           std::fabs(a.y - b.y) < 1.0e-9;
  }
};

template <typename T, bool Hermitian> struct PackedMv;

template <> struct PackedMv<float, false> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const float *alpha, const float *ap,
                               const float *x, int incx, const float *beta,
                               float *y, int incy) {
    return cublasSspmv_v2(handle, uplo, n, alpha, ap, x, incx, beta, y, incy);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const float *alpha, const float *ap,
                               const float *x, int64_t incx,
                               const float *beta, float *y, int64_t incy) {
    return cublasSspmv_v2_64(handle, uplo, n, alpha, ap, x, incx, beta, y,
                             incy);
  }
};

template <> struct PackedMv<double, false> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const double *alpha, const double *ap,
                               const double *x, int incx, const double *beta,
                               double *y, int incy) {
    return cublasDspmv_v2(handle, uplo, n, alpha, ap, x, incx, beta, y, incy);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const double *alpha,
                               const double *ap, const double *x,
                               int64_t incx, const double *beta, double *y,
                               int64_t incy) {
    return cublasDspmv_v2_64(handle, uplo, n, alpha, ap, x, incx, beta, y,
                             incy);
  }
};

template <> struct PackedMv<cuComplex, true> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuComplex *alpha,
                               const cuComplex *ap, const cuComplex *x,
                               int incx, const cuComplex *beta, cuComplex *y,
                               int incy) {
    return cublasChpmv_v2(handle, uplo, n, alpha, ap, x, incx, beta, y, incy);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const cuComplex *alpha,
                               const cuComplex *ap, const cuComplex *x,
                               int64_t incx, const cuComplex *beta,
                               cuComplex *y, int64_t incy) {
    return cublasChpmv_v2_64(handle, uplo, n, alpha, ap, x, incx, beta, y,
                             incy);
  }
};

template <> struct PackedMv<cuDoubleComplex, true> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuDoubleComplex *alpha,
                               const cuDoubleComplex *ap,
                               const cuDoubleComplex *x, int incx,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *y, int incy) {
    return cublasZhpmv_v2(handle, uplo, n, alpha, ap, x, incx, beta, y, incy);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const cuDoubleComplex *alpha,
                               const cuDoubleComplex *ap,
                               const cuDoubleComplex *x, int64_t incx,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *y, int64_t incy) {
    return cublasZhpmv_v2_64(handle, uplo, n, alpha, ap, x, incx, beta, y,
                             incy);
  }
};

static int packed_index(cublasFillMode_t uplo, int n, int row, int col) {
  if (uplo == CUBLAS_FILL_MODE_UPPER) {
    return col * (col + 1) / 2 + row;
  }
  return col * n - col * (col - 1) / 2 + (row - col);
}

static bool in_triangle(cublasFillMode_t uplo, int row, int col) {
  return uplo == CUBLAS_FILL_MODE_UPPER ? row <= col : row >= col;
}

template <typename T>
static T matrix_value(const std::vector<T> &ap, cublasFillMode_t uplo, int n,
                      int row, int col, bool hermitian) {
  if (in_triangle(uplo, row, col)) {
    return ap[packed_index(uplo, n, row, col)];
  }
  T stored = ap[packed_index(uplo, n, col, row)];
  return hermitian ? Ops<T>::conj(stored) : stored;
}

template <typename T>
static std::vector<T> expected_mv(const std::vector<T> &ap,
                                  const std::vector<T> &x,
                                  const std::vector<T> &y,
                                  cublasFillMode_t uplo, int n, T alpha,
                                  T beta, bool hermitian) {
  std::vector<T> out(n);
  for (int row = 0; row < n; ++row) {
    T sum = Ops<T>::zero();
    for (int col = 0; col < n; ++col) {
      T aval = matrix_value(ap, uplo, n, row, col, hermitian);
      sum = Ops<T>::add(sum, Ops<T>::mul(aval, x[col]));
    }
    out[row] = Ops<T>::add(Ops<T>::mul(alpha, sum), Ops<T>::mul(beta, y[row]));
  }
  return out;
}

template <typename T>
static void fill_inputs(std::vector<T> &ap, std::vector<T> &x,
                        std::vector<T> &y, cublasFillMode_t uplo, int n,
                        bool hermitian) {
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      if (!in_triangle(uplo, row, col)) {
        continue;
      }
      float imag = hermitian && row == col ? 0.0f
                                           : static_cast<float>((row - col) % 5) *
                                                 0.125f;
      ap[packed_index(uplo, n, row, col)] =
          Ops<T>::value(static_cast<float>(10 * col + row + 1), imag);
    }
  }
  for (int i = 0; i < n; ++i) {
    x[i] = Ops<T>::value(static_cast<float>(i + 3),
                         static_cast<float>(i - 2) * 0.25f);
    y[i] = Ops<T>::value(static_cast<float>(i - 4),
                         static_cast<float>(i + 1) * -0.125f);
  }
}

template <typename T, bool Hermitian>
static void run_case(cublasHandle_t handle, const char *name,
                     cublasFillMode_t uplo, bool use_64, bool device_scalar) {
  const int n = 4;
  const int packed_len = n * (n + 1) / 2;

  std::vector<T> ap(packed_len);
  std::vector<T> x(n);
  std::vector<T> y(n);
  fill_inputs(ap, x, y, uplo, n, Hermitian);
  T alpha = Ops<T>::alpha();
  T beta = Ops<T>::beta();
  std::vector<T> expected =
      expected_mv(ap, x, y, uplo, n, alpha, beta, Hermitian);

  T *d_ap = nullptr;
  T *d_x = nullptr;
  T *d_y = nullptr;
  T *d_alpha = nullptr;
  T *d_beta = nullptr;
  CHECK_CUDA(cudaMalloc(&d_ap, ap.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_y, y.size() * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_ap, ap.data(), ap.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
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
    CHECK_CUBLAS((PackedMv<T, Hermitian>::call64(
        handle, uplo, n, alpha_arg, d_ap, d_x, 1, beta_arg, d_y, 1)));
  } else {
    CHECK_CUBLAS((PackedMv<T, Hermitian>::call32(
        handle, uplo, n, alpha_arg, d_ap, d_x, 1, beta_arg, d_y, 1)));
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
  cudaFree(d_ap);
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

  run_type<float, false>(handle, "sspmv");
  run_type<double, false>(handle, "dspmv");
  run_type<cuComplex, true>(handle, "chpmv");
  run_type<cuDoubleComplex, true>(handle, "zhpmv");

  CHECK_CUBLAS(cublasDestroy(handle));
  return EXIT_SUCCESS;
}
