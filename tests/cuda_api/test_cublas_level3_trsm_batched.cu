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
  static float zero() { return 0.0f; }
  static float one() { return 1.0f; }
  static float value(float real, float) { return real; }
  static float add(float a, float b) { return a + b; }
  static float mul(float a, float b) { return a * b; }
  static float div(float a, float b) { return a / b; }
  static float conj(float a) { return a; }
  static float alpha() { return -0.625f; }
  static bool near(float a, float b) { return std::fabs(a - b) < 1.0e-3f; }
};

template <> struct Ops<double> {
  static double zero() { return 0.0; }
  static double one() { return 1.0; }
  static double value(float real, float) { return static_cast<double>(real); }
  static double add(double a, double b) { return a + b; }
  static double mul(double a, double b) { return a * b; }
  static double div(double a, double b) { return a / b; }
  static double conj(double a) { return a; }
  static double alpha() { return -0.625; }
  static bool near(double a, double b) { return std::fabs(a - b) < 1.0e-10; }
};

template <> struct Ops<cuComplex> {
  static cuComplex zero() { return make_cuComplex(0.0f, 0.0f); }
  static cuComplex one() { return make_cuComplex(1.0f, 0.0f); }
  static cuComplex value(float real, float imag) { return make_cuComplex(real, imag); }
  static cuComplex add(cuComplex a, cuComplex b) { return cuCaddf(a, b); }
  static cuComplex mul(cuComplex a, cuComplex b) { return cuCmulf(a, b); }
  static cuComplex div(cuComplex a, cuComplex b) {
    const float denom = b.x * b.x + b.y * b.y;
    return make_cuComplex((a.x * b.x + a.y * b.y) / denom,
                          (a.y * b.x - a.x * b.y) / denom);
  }
  static cuComplex conj(cuComplex a) { return cuConjf(a); }
  static cuComplex alpha() { return make_cuComplex(-0.625f, 0.25f); }
  static bool near(cuComplex a, cuComplex b) {
    return std::fabs(a.x - b.x) < 1.0e-3f && std::fabs(a.y - b.y) < 1.0e-3f;
  }
};

template <> struct Ops<cuDoubleComplex> {
  static cuDoubleComplex zero() { return make_cuDoubleComplex(0.0, 0.0); }
  static cuDoubleComplex one() { return make_cuDoubleComplex(1.0, 0.0); }
  static cuDoubleComplex value(float real, float imag) {
    return make_cuDoubleComplex(static_cast<double>(real), static_cast<double>(imag));
  }
  static cuDoubleComplex add(cuDoubleComplex a, cuDoubleComplex b) { return cuCadd(a, b); }
  static cuDoubleComplex mul(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }
  static cuDoubleComplex div(cuDoubleComplex a, cuDoubleComplex b) {
    const double denom = b.x * b.x + b.y * b.y;
    return make_cuDoubleComplex((a.x * b.x + a.y * b.y) / denom,
                                (a.y * b.x - a.x * b.y) / denom);
  }
  static cuDoubleComplex conj(cuDoubleComplex a) { return cuConj(a); }
  static cuDoubleComplex alpha() { return make_cuDoubleComplex(-0.625, 0.25); }
  static bool near(cuDoubleComplex a, cuDoubleComplex b) {
    return std::fabs(a.x - b.x) < 1.0e-10 && std::fabs(a.y - b.y) < 1.0e-10;
  }
};

template <typename T> struct TrsmBatched;

template <> struct TrsmBatched<float> {
  typedef float Scalar;
  static cublasStatus_t call32(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float **a, int lda, float **b, int ldb, int batch_count) {
    return cublasStrsmBatched(h, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, batch_count);
  }
  static cublasStatus_t call64(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const float *alpha, const float **a, int64_t lda, float **b, int64_t ldb, int64_t batch_count) {
    return cublasStrsmBatched_64(h, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, batch_count);
  }
};

template <> struct TrsmBatched<double> {
  typedef double Scalar;
  static cublasStatus_t call32(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double **a, int lda, double **b, int ldb, int batch_count) {
    return cublasDtrsmBatched(h, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, batch_count);
  }
  static cublasStatus_t call64(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const double *alpha, const double **a, int64_t lda, double **b, int64_t ldb, int64_t batch_count) {
    return cublasDtrsmBatched_64(h, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, batch_count);
  }
};

template <> struct TrsmBatched<cuComplex> {
  typedef cuComplex Scalar;
  static cublasStatus_t call32(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex **a, int lda, cuComplex **b, int ldb, int batch_count) {
    return cublasCtrsmBatched(h, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, batch_count);
  }
  static cublasStatus_t call64(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuComplex *alpha, const cuComplex **a, int64_t lda, cuComplex **b, int64_t ldb, int64_t batch_count) {
    return cublasCtrsmBatched_64(h, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, batch_count);
  }
};

template <> struct TrsmBatched<cuDoubleComplex> {
  typedef cuDoubleComplex Scalar;
  static cublasStatus_t call32(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex **a, int lda, cuDoubleComplex **b, int ldb, int batch_count) {
    return cublasZtrsmBatched(h, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, batch_count);
  }
  static cublasStatus_t call64(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuDoubleComplex *alpha, const cuDoubleComplex **a, int64_t lda, cuDoubleComplex **b, int64_t ldb, int64_t batch_count) {
    return cublasZtrsmBatched_64(h, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, batch_count);
  }
};

static bool in_triangle(cublasFillMode_t uplo, int row, int col) {
  return uplo == CUBLAS_FILL_MODE_UPPER ? row <= col : row >= col;
}

template <typename T>
static T matrix_value(const std::vector<T> &a, int offset, cublasFillMode_t uplo, cublasDiagType_t diag, int lda, int row, int col) {
  if (row == col && diag == CUBLAS_DIAG_UNIT) {
    return Ops<T>::one();
  }
  if (!in_triangle(uplo, row, col)) {
    return Ops<T>::zero();
  }
  return a[offset + row + col * lda];
}

template <typename T>
static T op_a(const std::vector<T> &a, int offset, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int lda, int row, int col) {
  if (trans == CUBLAS_OP_N) {
    return matrix_value(a, offset, uplo, diag, lda, row, col);
  }
  T v = matrix_value(a, offset, uplo, diag, lda, col, row);
  return trans == CUBLAS_OP_C ? Ops<T>::conj(v) : v;
}

template <typename T>
static void fill_a(std::vector<T> &a, int offset, cublasFillMode_t uplo, int dim, int lda, int batch) {
  for (int col = 0; col < dim; ++col) {
    for (int row = 0; row < lda; ++row) {
      if (row < dim && in_triangle(uplo, row, col)) {
        const float diag_bias = row == col ? 1.75f : 0.0f;
        const float batch_bias = 0.0625f * static_cast<float>(batch);
        a[offset + row + col * lda] = Ops<T>::value(diag_bias + batch_bias + 0.125f * static_cast<float>(row + col + 1), 0.0625f * static_cast<float>(row - col + batch));
      } else {
        a[offset + row + col * lda] = Ops<T>::value(-100.0f - static_cast<float>(batch + row + col), 0.0f);
      }
    }
  }
}

template <typename T>
static void fill_solution(std::vector<T> &x, int offset, int m, int n, int ldx, int batch) {
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < ldx; ++row) {
      x[offset + row + col * ldx] = row < m ? Ops<T>::value(0.25f * static_cast<float>(batch + 2 * col + row + 1), 0.125f * static_cast<float>(row - col + batch)) : Ops<T>::value(-200.0f - static_cast<float>(batch + row + col), 0.0f);
    }
  }
}

template <typename T>
static void fill_rhs(std::vector<T> &b, const std::vector<T> &a, const std::vector<T> &x, int a_offset, int x_offset, int b_offset, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, int lda, int ldx, int ldb, T alpha, int batch) {
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < ldb; ++row) {
      if (row >= m) {
        b[b_offset + row + col * ldb] = Ops<T>::value(-300.0f - static_cast<float>(batch + row + col), 0.0f);
        continue;
      }
      T sum = Ops<T>::zero();
      if (side == CUBLAS_SIDE_LEFT) {
        for (int inner = 0; inner < m; ++inner) {
          sum = Ops<T>::add(sum, Ops<T>::mul(op_a(a, a_offset, uplo, trans, diag, lda, row, inner), x[x_offset + inner + col * ldx]));
        }
      } else {
        for (int inner = 0; inner < n; ++inner) {
          sum = Ops<T>::add(sum, Ops<T>::mul(x[x_offset + row + inner * ldx], op_a(a, a_offset, uplo, trans, diag, lda, inner, col)));
        }
      }
      b[b_offset + row + col * ldb] = Ops<T>::div(sum, alpha);
    }
  }
}

template <typename T>
static void run_case(cublasHandle_t handle, const char *name, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, bool use_64, bool device_scalar) {
  const int m = 4;
  const int n = 3;
  const int batch_count = 3;
  const int adim = side == CUBLAS_SIDE_LEFT ? m : n;
  const int lda = adim + 2;
  const int ldb = m + 2;
  const int stride_a = lda * adim;
  const int stride_b = ldb * n;

  std::vector<T> a(batch_count * stride_a);
  std::vector<T> x(batch_count * stride_b);
  std::vector<T> b(batch_count * stride_b);
  const typename TrsmBatched<T>::Scalar alpha = Ops<T>::alpha();
  for (int batch = 0; batch < batch_count; ++batch) {
    const int a_offset = batch * stride_a;
    const int b_offset = batch * stride_b;
    fill_a(a, a_offset, uplo, adim, lda, batch);
    fill_solution(x, b_offset, m, n, ldb, batch);
    fill_rhs(b, a, x, a_offset, b_offset, b_offset, side, uplo, trans, diag, m, n, lda, ldb, ldb, alpha, batch);
  }

  T *d_a = nullptr;
  T *d_b = nullptr;
  const T **d_a_array = nullptr;
  T **d_b_array = nullptr;
  typename TrsmBatched<T>::Scalar *d_alpha = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, a.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_b, b.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_a_array, batch_count * sizeof(T *)));
  CHECK_CUDA(cudaMalloc(&d_b_array, batch_count * sizeof(T *)));
  CHECK_CUDA(cudaMemcpy(d_a, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice));

  std::vector<const T *> a_ptrs(batch_count);
  std::vector<T *> b_ptrs(batch_count);
  for (int batch = 0; batch < batch_count; ++batch) {
    a_ptrs[batch] = d_a + batch * stride_a;
    b_ptrs[batch] = d_b + batch * stride_b;
  }
  CHECK_CUDA(cudaMemcpy(d_a_array, a_ptrs.data(), a_ptrs.size() * sizeof(a_ptrs[0]), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b_array, b_ptrs.data(), b_ptrs.size() * sizeof(b_ptrs[0]), cudaMemcpyHostToDevice));

  const typename TrsmBatched<T>::Scalar *alpha_arg = &alpha;
  if (device_scalar) {
    CHECK_CUDA(cudaMalloc(&d_alpha, sizeof(typename TrsmBatched<T>::Scalar)));
    CHECK_CUDA(cudaMemcpy(d_alpha, &alpha, sizeof(typename TrsmBatched<T>::Scalar), cudaMemcpyHostToDevice));
    alpha_arg = d_alpha;
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  } else {
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  }

  if (use_64) {
    CHECK_CUBLAS(TrsmBatched<T>::call64(handle, side, uplo, trans, diag, m, n, alpha_arg, d_a_array, lda, d_b_array, ldb, batch_count));
  } else {
    CHECK_CUBLAS(TrsmBatched<T>::call32(handle, side, uplo, trans, diag, m, n, alpha_arg, d_a_array, lda, d_b_array, ldb, batch_count));
  }
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<T> out(b.size());
  CHECK_CUDA(cudaMemcpy(out.data(), d_b, out.size() * sizeof(T), cudaMemcpyDeviceToHost));
  for (int batch = 0; batch < batch_count; ++batch) {
    const int offset = batch * stride_b;
    for (int col = 0; col < n; ++col) {
      for (int row = 0; row < m; ++row) {
        if (!Ops<T>::near(out[offset + row + col * ldb], x[offset + row + col * ldb])) {
          std::printf("%s mismatch batch=%d row=%d col=%d side=%d uplo=%d trans=%d diag=%d use64=%d device=%d\n", name, batch, row, col, static_cast<int>(side), static_cast<int>(uplo), static_cast<int>(trans), static_cast<int>(diag), use_64 ? 1 : 0, device_scalar ? 1 : 0);
          std::exit(EXIT_FAILURE);
        }
      }
    }
  }

  if (d_alpha != nullptr) cudaFree(d_alpha);
  cudaFree(d_b_array);
  cudaFree(d_a_array);
  cudaFree(d_b);
  cudaFree(d_a);
}

template <typename T>
static void run_type(cublasHandle_t handle, const char *name, const cublasOperation_t *ops, int op_count) {
  cublasSideMode_t sides[] = {CUBLAS_SIDE_LEFT, CUBLAS_SIDE_RIGHT};
  cublasFillMode_t uplos[] = {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER};
  cublasDiagType_t diags[] = {CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT};
  for (int s = 0; s < 2; ++s) {
    for (int u = 0; u < 2; ++u) {
      for (int op = 0; op < op_count; ++op) {
        for (int d = 0; d < 2; ++d) {
          for (int use64 = 0; use64 < 2; ++use64) {
            for (int device = 0; device < 2; ++device) {
              run_case<T>(handle, name, sides[s], uplos[u], ops[op], diags[d], use64 != 0, device != 0);
            }
          }
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

  run_type<float>(handle, "strsm_batched", real_ops, 2);
  run_type<double>(handle, "dtrsm_batched", real_ops, 2);
  run_type<cuComplex>(handle, "ctrsm_batched", complex_ops, 3);
  run_type<cuDoubleComplex>(handle, "ztrsm_batched", complex_ops, 3);

  CHECK_CUBLAS(cublasDestroy(handle));
  return EXIT_SUCCESS;
}
