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
  static float alpha() { return 1.125f; }
  static float beta() { return -0.25f; }
  static bool near(float a, float b) { return std::fabs(a - b) < 1.0e-4f; }
};

template <> struct Ops<double> {
  static double value(float real, float) { return static_cast<double>(real); }
  static double zero() { return 0.0; }
  static double add(double a, double b) { return a + b; }
  static double mul(double a, double b) { return a * b; }
  static double conj(double a) { return a; }
  static double alpha() { return 1.125; }
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
  static cuComplex alpha() { return make_cuComplex(0.75f, -0.375f); }
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
  static cuDoubleComplex alpha() { return make_cuDoubleComplex(0.75, -0.375); }
  static cuDoubleComplex beta() { return make_cuDoubleComplex(-0.125, 0.25); }
  static bool near(cuDoubleComplex a, cuDoubleComplex b) {
    return std::fabs(a.x - b.x) < 1.0e-9 &&
           std::fabs(a.y - b.y) < 1.0e-9;
  }
};

template <typename T> struct Gbmv;

template <> struct Gbmv<float> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasOperation_t trans,
                               int m, int n, int kl, int ku,
                               const float *alpha, const float *a, int lda,
                               const float *x, int incx, const float *beta,
                               float *y, int incy) {
    return cublasSgbmv_v2(handle, trans, m, n, kl, ku, alpha, a, lda, x, incx,
                          beta, y, incy);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasOperation_t trans,
                               int64_t m, int64_t n, int64_t kl, int64_t ku,
                               const float *alpha, const float *a, int64_t lda,
                               const float *x, int64_t incx,
                               const float *beta, float *y, int64_t incy) {
    return cublasSgbmv_v2_64(handle, trans, m, n, kl, ku, alpha, a, lda, x,
                             incx, beta, y, incy);
  }
};

template <> struct Gbmv<double> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasOperation_t trans,
                               int m, int n, int kl, int ku,
                               const double *alpha, const double *a, int lda,
                               const double *x, int incx, const double *beta,
                               double *y, int incy) {
    return cublasDgbmv_v2(handle, trans, m, n, kl, ku, alpha, a, lda, x, incx,
                          beta, y, incy);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasOperation_t trans,
                               int64_t m, int64_t n, int64_t kl, int64_t ku,
                               const double *alpha, const double *a,
                               int64_t lda, const double *x, int64_t incx,
                               const double *beta, double *y, int64_t incy) {
    return cublasDgbmv_v2_64(handle, trans, m, n, kl, ku, alpha, a, lda, x,
                             incx, beta, y, incy);
  }
};

template <> struct Gbmv<cuComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasOperation_t trans,
                               int m, int n, int kl, int ku,
                               const cuComplex *alpha, const cuComplex *a,
                               int lda, const cuComplex *x, int incx,
                               const cuComplex *beta, cuComplex *y, int incy) {
    return cublasCgbmv_v2(handle, trans, m, n, kl, ku, alpha, a, lda, x, incx,
                          beta, y, incy);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasOperation_t trans,
                               int64_t m, int64_t n, int64_t kl, int64_t ku,
                               const cuComplex *alpha, const cuComplex *a,
                               int64_t lda, const cuComplex *x, int64_t incx,
                               const cuComplex *beta, cuComplex *y,
                               int64_t incy) {
    return cublasCgbmv_v2_64(handle, trans, m, n, kl, ku, alpha, a, lda, x,
                             incx, beta, y, incy);
  }
};

template <> struct Gbmv<cuDoubleComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasOperation_t trans,
                               int m, int n, int kl, int ku,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *a, int lda,
                               const cuDoubleComplex *x, int incx,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *y, int incy) {
    return cublasZgbmv_v2(handle, trans, m, n, kl, ku, alpha, a, lda, x, incx,
                          beta, y, incy);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasOperation_t trans,
                               int64_t m, int64_t n, int64_t kl, int64_t ku,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *a, int64_t lda,
                               const cuDoubleComplex *x, int64_t incx,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *y, int64_t incy) {
    return cublasZgbmv_v2_64(handle, trans, m, n, kl, ku, alpha, a, lda, x,
                             incx, beta, y, incy);
  }
};

static bool in_band(int row, int col, int kl, int ku) {
  return row - col <= kl && col - row <= ku;
}

static int band_index(int row, int col, int ku, int lda) {
  return (ku + row - col) + col * lda;
}

template <typename T>
static T matrix_value(const std::vector<T> &a, int row, int col, int kl,
                      int ku, int lda) {
  if (!in_band(row, col, kl, ku)) {
    return Ops<T>::zero();
  }
  return a[band_index(row, col, ku, lda)];
}

template <typename T>
static std::vector<T> expected_mv(const std::vector<T> &a,
                                  const std::vector<T> &x,
                                  const std::vector<T> &y,
                                  cublasOperation_t trans, int m, int n,
                                  int kl, int ku, int lda, T alpha, T beta) {
  const bool no_trans = trans == CUBLAS_OP_N;
  const int out_len = no_trans ? m : n;
  std::vector<T> out(out_len);
  for (int out_idx = 0; out_idx < out_len; ++out_idx) {
    T sum = Ops<T>::zero();
    if (no_trans) {
      int row = out_idx;
      for (int col = 0; col < n; ++col) {
        T aval = matrix_value(a, row, col, kl, ku, lda);
        sum = Ops<T>::add(sum, Ops<T>::mul(aval, x[col]));
      }
    } else {
      int col = out_idx;
      for (int row = 0; row < m; ++row) {
        T aval = matrix_value(a, row, col, kl, ku, lda);
        if (trans == CUBLAS_OP_C) {
          aval = Ops<T>::conj(aval);
        }
        sum = Ops<T>::add(sum, Ops<T>::mul(aval, x[row]));
      }
    }
    out[out_idx] =
        Ops<T>::add(Ops<T>::mul(alpha, sum), Ops<T>::mul(beta, y[out_idx]));
  }
  return out;
}

template <typename T>
static void fill_matrix(std::vector<T> &a, int m, int n, int kl, int ku,
                        int lda) {
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < m; ++row) {
      if (!in_band(row, col, kl, ku)) {
        continue;
      }
      a[band_index(row, col, ku, lda)] =
          Ops<T>::value(static_cast<float>(10 * col + row + 1),
                        static_cast<float>((row - col) % 5) * 0.125f);
    }
  }
}

template <typename T>
static void fill_vector(std::vector<T> &v, int bias) {
  for (int i = 0; i < static_cast<int>(v.size()); ++i) {
    v[i] = Ops<T>::value(static_cast<float>(i + bias),
                         static_cast<float>(bias - i) * 0.125f);
  }
}

template <typename T>
static void run_case(cublasHandle_t handle, const char *name,
                     cublasOperation_t trans, bool use_64,
                     bool device_scalar) {
  const int m = 5;
  const int n = 4;
  const int kl = 1;
  const int ku = 2;
  const int lda = kl + ku + 1;
  const int x_len = trans == CUBLAS_OP_N ? n : m;
  const int y_len = trans == CUBLAS_OP_N ? m : n;

  std::vector<T> a(lda * n, Ops<T>::zero());
  std::vector<T> x(x_len);
  std::vector<T> y(y_len);
  fill_matrix(a, m, n, kl, ku, lda);
  fill_vector(x, 2);
  fill_vector(y, -3);
  T alpha = Ops<T>::alpha();
  T beta = Ops<T>::beta();
  std::vector<T> expected =
      expected_mv(a, x, y, trans, m, n, kl, ku, lda, alpha, beta);

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
    CHECK_CUBLAS((Gbmv<T>::call64(handle, trans, m, n, kl, ku, alpha_arg, d_a,
                                  lda, d_x, 1, beta_arg, d_y, 1)));
  } else {
    CHECK_CUBLAS((Gbmv<T>::call32(handle, trans, m, n, kl, ku, alpha_arg, d_a,
                                  lda, d_x, 1, beta_arg, d_y, 1)));
  }

  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<T> out(y_len);
  CHECK_CUDA(
      cudaMemcpy(out.data(), d_y, out.size() * sizeof(T), cudaMemcpyDeviceToHost));
  for (int i = 0; i < y_len; ++i) {
    if (!Ops<T>::near(out[i], expected[i])) {
      std::printf("%s mismatch index=%d trans=%d use64=%d device_scalar=%d\n",
                  name, i, static_cast<int>(trans), use_64 ? 1 : 0,
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

template <typename T>
static void run_type(cublasHandle_t handle, const char *name,
                     const cublasOperation_t *ops, int op_count) {
  for (int op = 0; op < op_count; ++op) {
    for (int use_64 = 0; use_64 < 2; ++use_64) {
      for (int device_scalar = 0; device_scalar < 2; ++device_scalar) {
        run_case<T>(handle, name, ops[op], use_64 != 0, device_scalar != 0);
      }
    }
  }
}

int main() {
  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));

  cublasOperation_t real_ops[] = {CUBLAS_OP_N, CUBLAS_OP_T};
  cublasOperation_t complex_ops[] = {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};

  run_type<float>(handle, "sgbmv", real_ops, 2);
  run_type<double>(handle, "dgbmv", real_ops, 2);
  run_type<cuComplex>(handle, "cgbmv", complex_ops, 3);
  run_type<cuDoubleComplex>(handle, "zgbmv", complex_ops, 3);

  CHECK_CUBLAS(cublasDestroy(handle));
  return EXIT_SUCCESS;
}
