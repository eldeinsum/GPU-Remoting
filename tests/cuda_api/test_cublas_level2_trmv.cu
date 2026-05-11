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
  static float one() { return 1.0f; }
  static float add(float a, float b) { return a + b; }
  static float mul(float a, float b) { return a * b; }
  static float conj(float a) { return a; }
  static bool near(float a, float b) { return std::fabs(a - b) < 1.0e-4f; }
};

template <> struct Ops<double> {
  static double value(float real, float) { return static_cast<double>(real); }
  static double zero() { return 0.0; }
  static double one() { return 1.0; }
  static double add(double a, double b) { return a + b; }
  static double mul(double a, double b) { return a * b; }
  static double conj(double a) { return a; }
  static bool near(double a, double b) { return std::fabs(a - b) < 1.0e-9; }
};

template <> struct Ops<cuComplex> {
  static cuComplex value(float real, float imag) {
    return make_cuComplex(real, imag);
  }
  static cuComplex zero() { return make_cuComplex(0.0f, 0.0f); }
  static cuComplex one() { return make_cuComplex(1.0f, 0.0f); }
  static cuComplex add(cuComplex a, cuComplex b) { return cuCaddf(a, b); }
  static cuComplex mul(cuComplex a, cuComplex b) { return cuCmulf(a, b); }
  static cuComplex conj(cuComplex a) { return cuConjf(a); }
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
  static cuDoubleComplex one() { return make_cuDoubleComplex(1.0, 0.0); }
  static cuDoubleComplex add(cuDoubleComplex a, cuDoubleComplex b) {
    return cuCadd(a, b);
  }
  static cuDoubleComplex mul(cuDoubleComplex a, cuDoubleComplex b) {
    return cuCmul(a, b);
  }
  static cuDoubleComplex conj(cuDoubleComplex a) { return cuConj(a); }
  static bool near(cuDoubleComplex a, cuDoubleComplex b) {
    return std::fabs(a.x - b.x) < 1.0e-9 &&
           std::fabs(a.y - b.y) < 1.0e-9;
  }
};

template <typename T> struct Trmv;

template <> struct Trmv<float> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const float *a, int lda, float *x,
                               int incx) {
    return cublasStrmv_v2(handle, uplo, trans, diag, n, a, lda, x, incx);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int64_t n, const float *a, int64_t lda,
                               float *x, int64_t incx) {
    return cublasStrmv_v2_64(handle, uplo, trans, diag, n, a, lda, x, incx);
  }
};

template <> struct Trmv<double> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const double *a, int lda, double *x,
                               int incx) {
    return cublasDtrmv_v2(handle, uplo, trans, diag, n, a, lda, x, incx);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int64_t n, const double *a, int64_t lda,
                               double *x, int64_t incx) {
    return cublasDtrmv_v2_64(handle, uplo, trans, diag, n, a, lda, x, incx);
  }
};

template <> struct Trmv<cuComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const cuComplex *a, int lda,
                               cuComplex *x, int incx) {
    return cublasCtrmv_v2(handle, uplo, trans, diag, n, a, lda, x, incx);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int64_t n, const cuComplex *a, int64_t lda,
                               cuComplex *x, int64_t incx) {
    return cublasCtrmv_v2_64(handle, uplo, trans, diag, n, a, lda, x, incx);
  }
};

template <> struct Trmv<cuDoubleComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const cuDoubleComplex *a, int lda,
                               cuDoubleComplex *x, int incx) {
    return cublasZtrmv_v2(handle, uplo, trans, diag, n, a, lda, x, incx);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int64_t n, const cuDoubleComplex *a,
                               int64_t lda, cuDoubleComplex *x, int64_t incx) {
    return cublasZtrmv_v2_64(handle, uplo, trans, diag, n, a, lda, x, incx);
  }
};

static bool in_triangle(cublasFillMode_t uplo, int row, int col) {
  return uplo == CUBLAS_FILL_MODE_UPPER ? row <= col : row >= col;
}

template <typename T>
static T matrix_value(const std::vector<T> &a, cublasFillMode_t uplo,
                      cublasDiagType_t diag, int lda, int row, int col) {
  if (row == col && diag == CUBLAS_DIAG_UNIT) {
    return Ops<T>::one();
  }
  if (!in_triangle(uplo, row, col)) {
    return Ops<T>::zero();
  }
  return a[row + col * lda];
}

template <typename T>
static std::vector<T> expected_mv(const std::vector<T> &a,
                                  const std::vector<T> &x,
                                  cublasFillMode_t uplo,
                                  cublasOperation_t trans,
                                  cublasDiagType_t diag, int n, int lda) {
  std::vector<T> out(n);
  for (int row = 0; row < n; ++row) {
    T sum = Ops<T>::zero();
    for (int col = 0; col < n; ++col) {
      T aval;
      if (trans == CUBLAS_OP_N) {
        aval = matrix_value(a, uplo, diag, lda, row, col);
      } else {
        aval = matrix_value(a, uplo, diag, lda, col, row);
        if (trans == CUBLAS_OP_C) {
          aval = Ops<T>::conj(aval);
        }
      }
      sum = Ops<T>::add(sum, Ops<T>::mul(aval, x[col]));
    }
    out[row] = sum;
  }
  return out;
}

template <typename T>
static void fill_inputs(std::vector<T> &a, std::vector<T> &x,
                        cublasFillMode_t uplo, int n, int lda) {
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      if (!in_triangle(uplo, row, col)) {
        continue;
      }
      a[row + col * lda] =
          Ops<T>::value(static_cast<float>(10 * col + row + 1),
                        static_cast<float>((row - col) % 7) * 0.125f);
    }
  }
  for (int i = 0; i < n; ++i) {
    x[i] = Ops<T>::value(static_cast<float>(i + 2),
                         static_cast<float>(i - 2) * 0.25f);
  }
}

template <typename T>
static void run_case(cublasHandle_t handle, const char *name,
                     cublasFillMode_t uplo, cublasOperation_t trans,
                     cublasDiagType_t diag, bool use_64) {
  const int n = 5;
  const int lda = n;

  std::vector<T> a(lda * n, Ops<T>::zero());
  std::vector<T> x(n);
  fill_inputs(a, x, uplo, n, lda);
  std::vector<T> expected = expected_mv(a, x, uplo, trans, diag, n, lda);

  T *d_a = nullptr;
  T *d_x = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, a.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(T)));
  CHECK_CUDA(
      cudaMemcpy(d_a, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_x, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));

  if (use_64) {
    CHECK_CUBLAS(
        (Trmv<T>::call64(handle, uplo, trans, diag, n, d_a, lda, d_x, 1)));
  } else {
    CHECK_CUBLAS(
        (Trmv<T>::call32(handle, uplo, trans, diag, n, d_a, lda, d_x, 1)));
  }

  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<T> out(n);
  CHECK_CUDA(
      cudaMemcpy(out.data(), d_x, out.size() * sizeof(T), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    if (!Ops<T>::near(out[i], expected[i])) {
      std::printf("%s mismatch index=%d uplo=%d trans=%d diag=%d use64=%d\n",
                  name, i, static_cast<int>(uplo), static_cast<int>(trans),
                  static_cast<int>(diag), use_64 ? 1 : 0);
      std::exit(EXIT_FAILURE);
    }
  }

  cudaFree(d_x);
  cudaFree(d_a);
}

template <typename T>
static void run_type(cublasHandle_t handle, const char *name,
                     const cublasOperation_t *ops, int op_count) {
  cublasFillMode_t uplos[] = {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER};
  cublasDiagType_t diags[] = {CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT};
  for (int u = 0; u < 2; ++u) {
    for (int op = 0; op < op_count; ++op) {
      for (int d = 0; d < 2; ++d) {
        for (int use_64 = 0; use_64 < 2; ++use_64) {
          run_case<T>(handle, name, uplos[u], ops[op], diags[d], use_64 != 0);
        }
      }
    }
  }
}

int main() {
  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));

  cublasOperation_t real_ops[] = {CUBLAS_OP_N, CUBLAS_OP_T};
  cublasOperation_t complex_ops[] = {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};

  run_type<float>(handle, "strmv", real_ops, 2);
  run_type<double>(handle, "dtrmv", real_ops, 2);
  run_type<cuComplex>(handle, "ctrmv", complex_ops, 3);
  run_type<cuDoubleComplex>(handle, "ztrmv", complex_ops, 3);

  CHECK_CUBLAS(cublasDestroy(handle));
  return EXIT_SUCCESS;
}
