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
  static float value(float real, float) { return real; }
  static float add(float a, float b) { return a + b; }
  static float mul(float a, float b) { return a * b; }
  static float conj(float a) { return a; }
  static float alpha() { return -0.75f; }
  static float beta() { return 0.5f; }
  static bool near(float a, float b) { return std::fabs(a - b) < 1.0e-4f; }
};

template <> struct Ops<double> {
  static double zero() { return 0.0; }
  static double value(float real, float) { return static_cast<double>(real); }
  static double add(double a, double b) { return a + b; }
  static double mul(double a, double b) { return a * b; }
  static double conj(double a) { return a; }
  static double alpha() { return -0.75; }
  static double beta() { return 0.5; }
  static bool near(double a, double b) { return std::fabs(a - b) < 1.0e-9; }
};

template <> struct Ops<cuComplex> {
  static cuComplex zero() { return make_cuComplex(0.0f, 0.0f); }
  static cuComplex value(float real, float imag) {
    return make_cuComplex(real, imag);
  }
  static cuComplex add(cuComplex a, cuComplex b) { return cuCaddf(a, b); }
  static cuComplex mul(cuComplex a, cuComplex b) { return cuCmulf(a, b); }
  static cuComplex conj(cuComplex a) { return cuConjf(a); }
  static cuComplex alpha() { return make_cuComplex(-0.625f, 0.25f); }
  static cuComplex beta() { return make_cuComplex(0.5f, -0.125f); }
  static bool near(cuComplex a, cuComplex b) {
    return std::fabs(a.x - b.x) < 1.0e-3f &&
           std::fabs(a.y - b.y) < 1.0e-3f;
  }
};

template <> struct Ops<cuDoubleComplex> {
  static cuDoubleComplex zero() { return make_cuDoubleComplex(0.0, 0.0); }
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
  static cuDoubleComplex alpha() { return make_cuDoubleComplex(-0.625, 0.25); }
  static cuDoubleComplex beta() { return make_cuDoubleComplex(0.5, -0.125); }
  static bool near(cuDoubleComplex a, cuDoubleComplex b) {
    return std::fabs(a.x - b.x) < 1.0e-9 &&
           std::fabs(a.y - b.y) < 1.0e-9;
  }
};

template <typename T> struct Symm;

template <> struct Symm<float> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasSideMode_t side,
                               cublasFillMode_t uplo, int m, int n,
                               const float *alpha, const float *a, int lda,
                               const float *b, int ldb, const float *beta,
                               float *c, int ldc) {
    return cublasSsymm_v2(handle, side, uplo, m, n, alpha, a, lda, b, ldb,
                          beta, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasSideMode_t side,
                               cublasFillMode_t uplo, int64_t m, int64_t n,
                               const float *alpha, const float *a,
                               int64_t lda, const float *b, int64_t ldb,
                               const float *beta, float *c, int64_t ldc) {
    return cublasSsymm_v2_64(handle, side, uplo, m, n, alpha, a, lda, b, ldb,
                             beta, c, ldc);
  }
};

template <> struct Symm<double> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasSideMode_t side,
                               cublasFillMode_t uplo, int m, int n,
                               const double *alpha, const double *a, int lda,
                               const double *b, int ldb, const double *beta,
                               double *c, int ldc) {
    return cublasDsymm_v2(handle, side, uplo, m, n, alpha, a, lda, b, ldb,
                          beta, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasSideMode_t side,
                               cublasFillMode_t uplo, int64_t m, int64_t n,
                               const double *alpha, const double *a,
                               int64_t lda, const double *b, int64_t ldb,
                               const double *beta, double *c, int64_t ldc) {
    return cublasDsymm_v2_64(handle, side, uplo, m, n, alpha, a, lda, b, ldb,
                             beta, c, ldc);
  }
};

template <> struct Symm<cuComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasSideMode_t side,
                               cublasFillMode_t uplo, int m, int n,
                               const cuComplex *alpha, const cuComplex *a,
                               int lda, const cuComplex *b, int ldb,
                               const cuComplex *beta, cuComplex *c, int ldc) {
    return cublasCsymm_v2(handle, side, uplo, m, n, alpha, a, lda, b, ldb,
                          beta, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasSideMode_t side,
                               cublasFillMode_t uplo, int64_t m, int64_t n,
                               const cuComplex *alpha, const cuComplex *a,
                               int64_t lda, const cuComplex *b, int64_t ldb,
                               const cuComplex *beta, cuComplex *c,
                               int64_t ldc) {
    return cublasCsymm_v2_64(handle, side, uplo, m, n, alpha, a, lda, b, ldb,
                             beta, c, ldc);
  }
};

template <> struct Symm<cuDoubleComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasSideMode_t side,
                               cublasFillMode_t uplo, int m, int n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *a, int lda,
                               const cuDoubleComplex *b, int ldb,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *c, int ldc) {
    return cublasZsymm_v2(handle, side, uplo, m, n, alpha, a, lda, b, ldb,
                          beta, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasSideMode_t side,
                               cublasFillMode_t uplo, int64_t m, int64_t n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *a, int64_t lda,
                               const cuDoubleComplex *b, int64_t ldb,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *c, int64_t ldc) {
    return cublasZsymm_v2_64(handle, side, uplo, m, n, alpha, a, lda, b, ldb,
                             beta, c, ldc);
  }
};

template <typename T> struct Hemm;

template <> struct Hemm<cuComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasSideMode_t side,
                               cublasFillMode_t uplo, int m, int n,
                               const cuComplex *alpha, const cuComplex *a,
                               int lda, const cuComplex *b, int ldb,
                               const cuComplex *beta, cuComplex *c, int ldc) {
    return cublasChemm_v2(handle, side, uplo, m, n, alpha, a, lda, b, ldb,
                          beta, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasSideMode_t side,
                               cublasFillMode_t uplo, int64_t m, int64_t n,
                               const cuComplex *alpha, const cuComplex *a,
                               int64_t lda, const cuComplex *b, int64_t ldb,
                               const cuComplex *beta, cuComplex *c,
                               int64_t ldc) {
    return cublasChemm_v2_64(handle, side, uplo, m, n, alpha, a, lda, b, ldb,
                             beta, c, ldc);
  }
};

template <> struct Hemm<cuDoubleComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasSideMode_t side,
                               cublasFillMode_t uplo, int m, int n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *a, int lda,
                               const cuDoubleComplex *b, int ldb,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *c, int ldc) {
    return cublasZhemm_v2(handle, side, uplo, m, n, alpha, a, lda, b, ldb,
                          beta, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasSideMode_t side,
                               cublasFillMode_t uplo, int64_t m, int64_t n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *a, int64_t lda,
                               const cuDoubleComplex *b, int64_t ldb,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *c, int64_t ldc) {
    return cublasZhemm_v2_64(handle, side, uplo, m, n, alpha, a, lda, b, ldb,
                             beta, c, ldc);
  }
};

template <typename T>
static T matrix_value(int row, int col, bool hermitian) {
  float imag = hermitian && row == col ? 0.0f
                                       : static_cast<float>(row - col) * 0.125f;
  return Ops<T>::value(static_cast<float>(3 * col + row + 1), imag);
}

template <typename T> static T b_value(int row, int col) {
  return Ops<T>::value(static_cast<float>(2 * col + row + 1),
                       static_cast<float>(row - 2 * col) * 0.0625f);
}

template <typename T> static T c_value(int row, int col) {
  return Ops<T>::value(static_cast<float>(col - row - 2),
                       static_cast<float>(row + col) * 0.03125f);
}

template <typename T>
static void fill_a(std::vector<T> &a, int dim, int lda, bool hermitian) {
  for (int col = 0; col < dim; ++col) {
    for (int row = 0; row < lda; ++row) {
      a[row + col * lda] =
          row < dim ? matrix_value<T>(row, col, hermitian)
                    : Ops<T>::value(static_cast<float>(-200 - row - col),
                                    0.0f);
    }
  }
}

template <typename T>
static void fill_general(std::vector<T> &m, int rows, int cols, int ld,
                         bool is_c) {
  for (int col = 0; col < cols; ++col) {
    for (int row = 0; row < ld; ++row) {
      if (row < rows) {
        m[row + col * ld] = is_c ? c_value<T>(row, col) : b_value<T>(row, col);
      } else {
        m[row + col * ld] =
            Ops<T>::value(static_cast<float>(-100 - row - col), 0.0f);
      }
    }
  }
}

template <typename T>
static T structured_at(const std::vector<T> &a, int lda, int row, int col,
                       cublasFillMode_t uplo, bool hermitian) {
  if (row == col) {
    return a[row + col * lda];
  }
  bool stored = uplo == CUBLAS_FILL_MODE_UPPER ? row < col : row > col;
  if (stored) {
    return a[row + col * lda];
  }
  T v = a[col + row * lda];
  return hermitian ? Ops<T>::conj(v) : v;
}

template <typename T>
static void expected_result(std::vector<T> &c, const std::vector<T> &a,
                            const std::vector<T> &b, cublasSideMode_t side,
                            cublasFillMode_t uplo, int m, int n, int lda,
                            int ldb, int ldc, T alpha, T beta,
                            bool hermitian) {
  std::vector<T> base = c;
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < m; ++row) {
      T sum = Ops<T>::zero();
      if (side == CUBLAS_SIDE_LEFT) {
        for (int k = 0; k < m; ++k) {
          T aval = structured_at(a, lda, row, k, uplo, hermitian);
          sum = Ops<T>::add(sum, Ops<T>::mul(aval, b[k + col * ldb]));
        }
      } else {
        for (int k = 0; k < n; ++k) {
          T aval = structured_at(a, lda, k, col, uplo, hermitian);
          sum = Ops<T>::add(sum, Ops<T>::mul(b[row + k * ldb], aval));
        }
      }
      c[row + col * ldc] =
          Ops<T>::add(Ops<T>::mul(alpha, sum),
                      Ops<T>::mul(beta, base[row + col * ldc]));
    }
  }
}

template <typename Api, typename T>
static void run_case(cublasHandle_t handle, const char *name,
                     cublasSideMode_t side, cublasFillMode_t uplo, bool use_64,
                     bool device_scalars, bool hermitian) {
  const int m = 3;
  const int n = 4;
  const int dim = side == CUBLAS_SIDE_LEFT ? m : n;
  const int lda = dim + 2;
  const int ldb = m + 2;
  const int ldc = m + 3;

  std::vector<T> a(lda * dim);
  std::vector<T> b(ldb * n);
  std::vector<T> c(ldc * n);
  fill_a(a, dim, lda, hermitian);
  fill_general(b, m, n, ldb, false);
  fill_general(c, m, n, ldc, true);

  T alpha = Ops<T>::alpha();
  T beta = Ops<T>::beta();
  std::vector<T> expected = c;
  expected_result(expected, a, b, side, uplo, m, n, lda, ldb, ldc, alpha, beta,
                  hermitian);

  T *d_a = nullptr;
  T *d_b = nullptr;
  T *d_c = nullptr;
  T *d_alpha = nullptr;
  T *d_beta = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, a.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_b, b.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_c, c.size() * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_a, a.data(), a.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b.data(), b.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_c, c.data(), c.size() * sizeof(T),
                        cudaMemcpyHostToDevice));

  const T *alpha_arg = &alpha;
  const T *beta_arg = &beta;
  if (device_scalars) {
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
    CHECK_CUBLAS(Api::call64(handle, side, uplo, m, n, alpha_arg, d_a, lda,
                             d_b, ldb, beta_arg, d_c, ldc));
  } else {
    CHECK_CUBLAS(Api::call32(handle, side, uplo, m, n, alpha_arg, d_a, lda,
                             d_b, ldb, beta_arg, d_c, ldc));
  }
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<T> out(c.size());
  CHECK_CUDA(cudaMemcpy(out.data(), d_c, out.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < m; ++row) {
      int idx = row + col * ldc;
      if (!Ops<T>::near(out[idx], expected[idx])) {
        std::printf("%s mismatch row=%d col=%d side=%d uplo=%d use64=%d "
                    "device=%d\n",
                    name, row, col, static_cast<int>(side),
                    static_cast<int>(uplo), use_64 ? 1 : 0,
                    device_scalars ? 1 : 0);
        std::exit(EXIT_FAILURE);
      }
    }
  }

  if (d_beta != nullptr) {
    cudaFree(d_beta);
  }
  if (d_alpha != nullptr) {
    cudaFree(d_alpha);
  }
  cudaFree(d_c);
  cudaFree(d_b);
  cudaFree(d_a);
}

template <typename Api, typename T>
static void run_all(cublasHandle_t handle, const char *name, bool hermitian) {
  cublasSideMode_t sides[] = {CUBLAS_SIDE_LEFT, CUBLAS_SIDE_RIGHT};
  cublasFillMode_t uplos[] = {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER};
  for (int s = 0; s < 2; ++s) {
    for (int u = 0; u < 2; ++u) {
      for (int use64 = 0; use64 < 2; ++use64) {
        for (int device = 0; device < 2; ++device) {
          run_case<Api, T>(handle, name, sides[s], uplos[u], use64 != 0,
                           device != 0, hermitian);
        }
      }
    }
  }
}

int main() {
  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));

  run_all<Symm<float>, float>(handle, "Ssymm", false);
  run_all<Symm<double>, double>(handle, "Dsymm", false);
  run_all<Symm<cuComplex>, cuComplex>(handle, "Csymm", false);
  run_all<Symm<cuDoubleComplex>, cuDoubleComplex>(handle, "Zsymm", false);
  run_all<Hemm<cuComplex>, cuComplex>(handle, "Chemm", true);
  run_all<Hemm<cuDoubleComplex>, cuDoubleComplex>(handle, "Zhemm", true);

  CHECK_CUBLAS(cublasDestroy(handle));
  return 0;
}
