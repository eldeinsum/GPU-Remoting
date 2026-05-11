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
  static bool near(float a, float b) { return std::fabs(a - b) < 1.0e-3f; }
};

template <> struct Ops<double> {
  static double value(float real, float) { return static_cast<double>(real); }
  static double zero() { return 0.0; }
  static double one() { return 1.0; }
  static double add(double a, double b) { return a + b; }
  static double mul(double a, double b) { return a * b; }
  static double conj(double a) { return a; }
  static bool near(double a, double b) { return std::fabs(a - b) < 1.0e-10; }
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
    return std::fabs(a.x - b.x) < 1.0e-10 &&
           std::fabs(a.y - b.y) < 1.0e-10;
  }
};

template <typename T> struct Tpmv;

template <> struct Tpmv<float> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const float *ap, float *x, int incx) {
    return cublasStpmv_v2(handle, uplo, trans, diag, n, ap, x, incx);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int64_t n, const float *ap, float *x,
                               int64_t incx) {
    return cublasStpmv_v2_64(handle, uplo, trans, diag, n, ap, x, incx);
  }
};

template <> struct Tpmv<double> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const double *ap, double *x, int incx) {
    return cublasDtpmv_v2(handle, uplo, trans, diag, n, ap, x, incx);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int64_t n, const double *ap, double *x,
                               int64_t incx) {
    return cublasDtpmv_v2_64(handle, uplo, trans, diag, n, ap, x, incx);
  }
};

template <> struct Tpmv<cuComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const cuComplex *ap, cuComplex *x,
                               int incx) {
    return cublasCtpmv_v2(handle, uplo, trans, diag, n, ap, x, incx);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int64_t n, const cuComplex *ap, cuComplex *x,
                               int64_t incx) {
    return cublasCtpmv_v2_64(handle, uplo, trans, diag, n, ap, x, incx);
  }
};

template <> struct Tpmv<cuDoubleComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const cuDoubleComplex *ap,
                               cuDoubleComplex *x, int incx) {
    return cublasZtpmv_v2(handle, uplo, trans, diag, n, ap, x, incx);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int64_t n, const cuDoubleComplex *ap,
                               cuDoubleComplex *x, int64_t incx) {
    return cublasZtpmv_v2_64(handle, uplo, trans, diag, n, ap, x, incx);
  }
};

template <typename T> struct Tpsv;

template <> struct Tpsv<float> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const float *ap, float *x, int incx) {
    return cublasStpsv_v2(handle, uplo, trans, diag, n, ap, x, incx);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int64_t n, const float *ap, float *x,
                               int64_t incx) {
    return cublasStpsv_v2_64(handle, uplo, trans, diag, n, ap, x, incx);
  }
};

template <> struct Tpsv<double> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const double *ap, double *x, int incx) {
    return cublasDtpsv_v2(handle, uplo, trans, diag, n, ap, x, incx);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int64_t n, const double *ap, double *x,
                               int64_t incx) {
    return cublasDtpsv_v2_64(handle, uplo, trans, diag, n, ap, x, incx);
  }
};

template <> struct Tpsv<cuComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const cuComplex *ap, cuComplex *x,
                               int incx) {
    return cublasCtpsv_v2(handle, uplo, trans, diag, n, ap, x, incx);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int64_t n, const cuComplex *ap, cuComplex *x,
                               int64_t incx) {
    return cublasCtpsv_v2_64(handle, uplo, trans, diag, n, ap, x, incx);
  }
};

template <> struct Tpsv<cuDoubleComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const cuDoubleComplex *ap,
                               cuDoubleComplex *x, int incx) {
    return cublasZtpsv_v2(handle, uplo, trans, diag, n, ap, x, incx);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int64_t n, const cuDoubleComplex *ap,
                               cuDoubleComplex *x, int64_t incx) {
    return cublasZtpsv_v2_64(handle, uplo, trans, diag, n, ap, x, incx);
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
static T matrix_value(const std::vector<T> &ap, cublasFillMode_t uplo,
                      cublasDiagType_t diag, int n, int row, int col) {
  if (row == col && diag == CUBLAS_DIAG_UNIT) {
    return Ops<T>::one();
  }
  if (!in_triangle(uplo, row, col)) {
    return Ops<T>::zero();
  }
  return ap[packed_index(uplo, n, row, col)];
}

template <typename T>
static std::vector<T> apply_op(const std::vector<T> &ap,
                               const std::vector<T> &x,
                               cublasFillMode_t uplo, cublasOperation_t trans,
                               cublasDiagType_t diag, int n) {
  std::vector<T> out(n);
  for (int row = 0; row < n; ++row) {
    T sum = Ops<T>::zero();
    for (int col = 0; col < n; ++col) {
      T aval;
      if (trans == CUBLAS_OP_N) {
        aval = matrix_value(ap, uplo, diag, n, row, col);
      } else {
        aval = matrix_value(ap, uplo, diag, n, col, row);
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
static void fill_inputs(std::vector<T> &ap, std::vector<T> &x,
                        cublasFillMode_t uplo, int n) {
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      if (!in_triangle(uplo, row, col)) {
        continue;
      }
      const float diag_bias = row == col ? 1.5f : 0.0f;
      ap[packed_index(uplo, n, row, col)] =
          Ops<T>::value(diag_bias + 0.05f * static_cast<float>(row + col + 1),
                        0.01f * static_cast<float>(row - col));
    }
  }
  for (int i = 0; i < n; ++i) {
    x[i] = Ops<T>::value(0.25f * static_cast<float>(i + 1),
                         0.125f * static_cast<float>((i % 3) - 1));
  }
}

template <typename T>
static void run_tpmv_case(cublasHandle_t handle, const char *name,
                          cublasFillMode_t uplo, cublasOperation_t trans,
                          cublasDiagType_t diag, bool use_64) {
  const int n = 5;
  const int packed_len = n * (n + 1) / 2;

  std::vector<T> ap(packed_len);
  std::vector<T> x(n);
  fill_inputs(ap, x, uplo, n);
  std::vector<T> expected = apply_op(ap, x, uplo, trans, diag, n);

  T *d_ap = nullptr;
  T *d_x = nullptr;
  CHECK_CUDA(cudaMalloc(&d_ap, ap.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_ap, ap.data(), ap.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_x, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));

  if (use_64) {
    CHECK_CUBLAS((Tpmv<T>::call64(handle, uplo, trans, diag, n, d_ap, d_x, 1)));
  } else {
    CHECK_CUBLAS((Tpmv<T>::call32(handle, uplo, trans, diag, n, d_ap, d_x, 1)));
  }

  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<T> out(n);
  CHECK_CUDA(
      cudaMemcpy(out.data(), d_x, out.size() * sizeof(T), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    if (!Ops<T>::near(out[i], expected[i])) {
      std::printf("%s tpmv mismatch index=%d uplo=%d trans=%d diag=%d use64=%d\n",
                  name, i, static_cast<int>(uplo), static_cast<int>(trans),
                  static_cast<int>(diag), use_64 ? 1 : 0);
      std::exit(EXIT_FAILURE);
    }
  }

  cudaFree(d_x);
  cudaFree(d_ap);
}

template <typename T>
static void run_tpsv_case(cublasHandle_t handle, const char *name,
                          cublasFillMode_t uplo, cublasOperation_t trans,
                          cublasDiagType_t diag, bool use_64) {
  const int n = 5;
  const int packed_len = n * (n + 1) / 2;

  std::vector<T> ap(packed_len);
  std::vector<T> solution(n);
  fill_inputs(ap, solution, uplo, n);
  std::vector<T> rhs = apply_op(ap, solution, uplo, trans, diag, n);

  T *d_ap = nullptr;
  T *d_x = nullptr;
  CHECK_CUDA(cudaMalloc(&d_ap, ap.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_x, rhs.size() * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_ap, ap.data(), ap.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_x, rhs.data(), rhs.size() * sizeof(T),
                        cudaMemcpyHostToDevice));

  if (use_64) {
    CHECK_CUBLAS((Tpsv<T>::call64(handle, uplo, trans, diag, n, d_ap, d_x, 1)));
  } else {
    CHECK_CUBLAS((Tpsv<T>::call32(handle, uplo, trans, diag, n, d_ap, d_x, 1)));
  }

  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<T> out(n);
  CHECK_CUDA(
      cudaMemcpy(out.data(), d_x, out.size() * sizeof(T), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    if (!Ops<T>::near(out[i], solution[i])) {
      std::printf("%s tpsv mismatch index=%d uplo=%d trans=%d diag=%d use64=%d\n",
                  name, i, static_cast<int>(uplo), static_cast<int>(trans),
                  static_cast<int>(diag), use_64 ? 1 : 0);
      std::exit(EXIT_FAILURE);
    }
  }

  cudaFree(d_x);
  cudaFree(d_ap);
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
          run_tpmv_case<T>(handle, name, uplos[u], ops[op], diags[d],
                           use_64 != 0);
          run_tpsv_case<T>(handle, name, uplos[u], ops[op], diags[d],
                           use_64 != 0);
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

  run_type<float>(handle, "s", real_ops, 2);
  run_type<double>(handle, "d", real_ops, 2);
  run_type<cuComplex>(handle, "c", complex_ops, 3);
  run_type<cuDoubleComplex>(handle, "z", complex_ops, 3);

  CHECK_CUBLAS(cublasDestroy(handle));
  return EXIT_SUCCESS;
}
