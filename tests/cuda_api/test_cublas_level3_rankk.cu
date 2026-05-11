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
  static float scale(float a, float x) { return a * x; }
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
  static double scale(double a, double x) { return a * x; }
  static double alpha() { return -0.75; }
  static double beta() { return 0.5; }
  static bool near(double a, double b) { return std::fabs(a - b) < 1.0e-9; }
};

template <> struct Ops<cuComplex> {
  static cuComplex zero() { return make_cuComplex(0.0f, 0.0f); }
  static cuComplex value(float real, float imag) { return make_cuComplex(real, imag); }
  static cuComplex add(cuComplex a, cuComplex b) { return cuCaddf(a, b); }
  static cuComplex mul(cuComplex a, cuComplex b) { return cuCmulf(a, b); }
  static cuComplex conj(cuComplex a) { return cuConjf(a); }
  static cuComplex scale(cuComplex a, cuComplex x) { return cuCmulf(a, x); }
  static cuComplex scale(float a, cuComplex x) { return make_cuComplex(a * x.x, a * x.y); }
  static cuComplex alpha() { return make_cuComplex(-0.625f, 0.25f); }
  static cuComplex beta() { return make_cuComplex(0.5f, -0.125f); }
  static bool near(cuComplex a, cuComplex b) {
    return std::fabs(a.x - b.x) < 1.0e-3f && std::fabs(a.y - b.y) < 1.0e-3f;
  }
};

template <> struct Ops<cuDoubleComplex> {
  static cuDoubleComplex zero() { return make_cuDoubleComplex(0.0, 0.0); }
  static cuDoubleComplex value(float real, float imag) {
    return make_cuDoubleComplex(static_cast<double>(real), static_cast<double>(imag));
  }
  static cuDoubleComplex add(cuDoubleComplex a, cuDoubleComplex b) { return cuCadd(a, b); }
  static cuDoubleComplex mul(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }
  static cuDoubleComplex conj(cuDoubleComplex a) { return cuConj(a); }
  static cuDoubleComplex scale(cuDoubleComplex a, cuDoubleComplex x) { return cuCmul(a, x); }
  static cuDoubleComplex scale(double a, cuDoubleComplex x) {
    return make_cuDoubleComplex(a * x.x, a * x.y);
  }
  static cuDoubleComplex alpha() { return make_cuDoubleComplex(-0.625, 0.25); }
  static cuDoubleComplex beta() { return make_cuDoubleComplex(0.5, -0.125); }
  static bool near(cuDoubleComplex a, cuDoubleComplex b) {
    return std::fabs(a.x - b.x) < 1.0e-9 && std::fabs(a.y - b.y) < 1.0e-9;
  }
};

template <typename T> struct Syrk;

template <> struct Syrk<float> {
  typedef float Scalar;
  static cublasStatus_t call32(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, int n, int k, const float *alpha, const float *a, int lda, const float *beta, float *c, int ldc) {
    return cublasSsyrk_v2(h, u, t, n, k, alpha, a, lda, beta, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, int64_t n, int64_t k, const float *alpha, const float *a, int64_t lda, const float *beta, float *c, int64_t ldc) {
    return cublasSsyrk_v2_64(h, u, t, n, k, alpha, a, lda, beta, c, ldc);
  }
};

template <> struct Syrk<double> {
  typedef double Scalar;
  static cublasStatus_t call32(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, int n, int k, const double *alpha, const double *a, int lda, const double *beta, double *c, int ldc) {
    return cublasDsyrk_v2(h, u, t, n, k, alpha, a, lda, beta, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, int64_t n, int64_t k, const double *alpha, const double *a, int64_t lda, const double *beta, double *c, int64_t ldc) {
    return cublasDsyrk_v2_64(h, u, t, n, k, alpha, a, lda, beta, c, ldc);
  }
};

template <> struct Syrk<cuComplex> {
  typedef cuComplex Scalar;
  static cublasStatus_t call32(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, int n, int k, const cuComplex *alpha, const cuComplex *a, int lda, const cuComplex *beta, cuComplex *c, int ldc) {
    return cublasCsyrk_v2(h, u, t, n, k, alpha, a, lda, beta, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, int64_t n, int64_t k, const cuComplex *alpha, const cuComplex *a, int64_t lda, const cuComplex *beta, cuComplex *c, int64_t ldc) {
    return cublasCsyrk_v2_64(h, u, t, n, k, alpha, a, lda, beta, c, ldc);
  }
};

template <> struct Syrk<cuDoubleComplex> {
  typedef cuDoubleComplex Scalar;
  static cublasStatus_t call32(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *a, int lda, const cuDoubleComplex *beta, cuDoubleComplex *c, int ldc) {
    return cublasZsyrk_v2(h, u, t, n, k, alpha, a, lda, beta, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, int64_t n, int64_t k, const cuDoubleComplex *alpha, const cuDoubleComplex *a, int64_t lda, const cuDoubleComplex *beta, cuDoubleComplex *c, int64_t ldc) {
    return cublasZsyrk_v2_64(h, u, t, n, k, alpha, a, lda, beta, c, ldc);
  }
};

template <typename T> struct Herk;

template <> struct Herk<cuComplex> {
  typedef float Scalar;
  static cublasStatus_t call32(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, int n, int k, const float *alpha, const cuComplex *a, int lda, const float *beta, cuComplex *c, int ldc) {
    return cublasCherk_v2(h, u, t, n, k, alpha, a, lda, beta, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, int64_t n, int64_t k, const float *alpha, const cuComplex *a, int64_t lda, const float *beta, cuComplex *c, int64_t ldc) {
    return cublasCherk_v2_64(h, u, t, n, k, alpha, a, lda, beta, c, ldc);
  }
};

template <> struct Herk<cuDoubleComplex> {
  typedef double Scalar;
  static cublasStatus_t call32(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, int n, int k, const double *alpha, const cuDoubleComplex *a, int lda, const double *beta, cuDoubleComplex *c, int ldc) {
    return cublasZherk_v2(h, u, t, n, k, alpha, a, lda, beta, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, int64_t n, int64_t k, const double *alpha, const cuDoubleComplex *a, int64_t lda, const double *beta, cuDoubleComplex *c, int64_t ldc) {
    return cublasZherk_v2_64(h, u, t, n, k, alpha, a, lda, beta, c, ldc);
  }
};

template <typename T> static T a_value(int row, int col) {
  return Ops<T>::value(static_cast<float>(2 * col + row + 1), static_cast<float>(row - col) * 0.125f);
}

template <typename T> static T c_value(int row, int col, bool hermitian) {
  float imag = hermitian && row == col ? 0.0f : static_cast<float>(col - row) * 0.0625f;
  return Ops<T>::value(static_cast<float>(3 * col - row + 2), imag);
}

template <typename T>
static void fill_a(std::vector<T> &a, int rows, int cols, int lda) {
  for (int col = 0; col < cols; ++col) {
    for (int row = 0; row < lda; ++row) {
      a[row + col * lda] = row < rows ? a_value<T>(row, col) : Ops<T>::value(-100.0f - row - col, 0.0f);
    }
  }
}

template <typename T>
static void fill_c(std::vector<T> &c, int n, int ldc, bool hermitian) {
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < ldc; ++row) {
      c[row + col * ldc] = row < n ? c_value<T>(row, col, hermitian) : Ops<T>::value(-200.0f - row - col, 0.0f);
    }
  }
}

template <typename T>
static T op_a(const std::vector<T> &a, int lda, cublasOperation_t trans, int row, int inner) {
  if (trans == CUBLAS_OP_N) {
    return a[row + inner * lda];
  }
  T v = a[inner + row * lda];
  return trans == CUBLAS_OP_C ? Ops<T>::conj(v) : v;
}

template <typename T, typename Scalar>
static void expected_rankk(std::vector<T> &c, const std::vector<T> &a, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, int lda, int ldc, Scalar alpha, Scalar beta, bool hermitian) {
  std::vector<T> base = c;
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      bool stored = uplo == CUBLAS_FILL_MODE_UPPER ? row <= col : row >= col;
      if (!stored) {
        continue;
      }
      T sum = Ops<T>::zero();
      for (int inner = 0; inner < k; ++inner) {
        T left = op_a(a, lda, trans, row, inner);
        T right = op_a(a, lda, trans, col, inner);
        if (hermitian) {
          right = Ops<T>::conj(right);
        }
        sum = Ops<T>::add(sum, Ops<T>::mul(left, right));
      }
      c[row + col * ldc] = Ops<T>::add(Ops<T>::scale(alpha, sum), Ops<T>::scale(beta, base[row + col * ldc]));
    }
  }
}

template <typename Api, typename T>
static void run_case(cublasHandle_t handle, const char *name, cublasFillMode_t uplo, cublasOperation_t trans, bool use_64, bool device_scalars, bool hermitian, typename Api::Scalar alpha, typename Api::Scalar beta) {
  const int n = 4;
  const int k = 3;
  const int a_rows = trans == CUBLAS_OP_N ? n : k;
  const int a_cols = trans == CUBLAS_OP_N ? k : n;
  const int lda = a_rows + 2;
  const int ldc = n + 2;
  std::vector<T> a(lda * a_cols);
  std::vector<T> c(ldc * n);
  fill_a(a, a_rows, a_cols, lda);
  fill_c(c, n, ldc, hermitian);
  std::vector<T> expected = c;
  expected_rankk(expected, a, uplo, trans, n, k, lda, ldc, alpha, beta, hermitian);

  T *d_a = nullptr;
  T *d_c = nullptr;
  typename Api::Scalar *d_alpha = nullptr;
  typename Api::Scalar *d_beta = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, a.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_c, c.size() * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_a, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_c, c.data(), c.size() * sizeof(T), cudaMemcpyHostToDevice));

  const typename Api::Scalar *alpha_arg = &alpha;
  const typename Api::Scalar *beta_arg = &beta;
  if (device_scalars) {
    CHECK_CUDA(cudaMalloc(&d_alpha, sizeof(typename Api::Scalar)));
    CHECK_CUDA(cudaMalloc(&d_beta, sizeof(typename Api::Scalar)));
    CHECK_CUDA(cudaMemcpy(d_alpha, &alpha, sizeof(typename Api::Scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_beta, &beta, sizeof(typename Api::Scalar), cudaMemcpyHostToDevice));
    alpha_arg = d_alpha;
    beta_arg = d_beta;
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  } else {
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  }

  if (use_64) {
    CHECK_CUBLAS(Api::call64(handle, uplo, trans, n, k, alpha_arg, d_a, lda, beta_arg, d_c, ldc));
  } else {
    CHECK_CUBLAS(Api::call32(handle, uplo, trans, n, k, alpha_arg, d_a, lda, beta_arg, d_c, ldc));
  }
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<T> out(c.size());
  CHECK_CUDA(cudaMemcpy(out.data(), d_c, out.size() * sizeof(T), cudaMemcpyDeviceToHost));
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      bool stored = uplo == CUBLAS_FILL_MODE_UPPER ? row <= col : row >= col;
      if (stored && !Ops<T>::near(out[row + col * ldc], expected[row + col * ldc])) {
        std::printf("%s mismatch row=%d col=%d uplo=%d trans=%d use64=%d device=%d\n", name, row, col, static_cast<int>(uplo), static_cast<int>(trans), use_64 ? 1 : 0, device_scalars ? 1 : 0);
        std::exit(EXIT_FAILURE);
      }
    }
  }

  if (d_beta != nullptr) cudaFree(d_beta);
  if (d_alpha != nullptr) cudaFree(d_alpha);
  cudaFree(d_c);
  cudaFree(d_a);
}

template <typename Api, typename T>
static void run_all(cublasHandle_t handle, const char *name, bool hermitian, typename Api::Scalar alpha, typename Api::Scalar beta) {
  cublasFillMode_t uplos[] = {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER};
  cublasOperation_t transes[] = {CUBLAS_OP_N, hermitian ? CUBLAS_OP_C : CUBLAS_OP_T};
  for (int u = 0; u < 2; ++u) {
    for (int tr = 0; tr < 2; ++tr) {
      for (int use64 = 0; use64 < 2; ++use64) {
        for (int device = 0; device < 2; ++device) {
          run_case<Api, T>(handle, name, uplos[u], transes[tr], use64 != 0, device != 0, hermitian, alpha, beta);
        }
      }
    }
  }
}

int main() {
  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));

  run_all<Syrk<float>, float>(handle, "Ssyrk", false, Ops<float>::alpha(), Ops<float>::beta());
  run_all<Syrk<double>, double>(handle, "Dsyrk", false, Ops<double>::alpha(), Ops<double>::beta());
  run_all<Syrk<cuComplex>, cuComplex>(handle, "Csyrk", false, Ops<cuComplex>::alpha(), Ops<cuComplex>::beta());
  run_all<Syrk<cuDoubleComplex>, cuDoubleComplex>(handle, "Zsyrk", false, Ops<cuDoubleComplex>::alpha(), Ops<cuDoubleComplex>::beta());
  run_all<Herk<cuComplex>, cuComplex>(handle, "Cherk", true, -0.75f, 0.5f);
  run_all<Herk<cuDoubleComplex>, cuDoubleComplex>(handle, "Zherk", true, -0.75, 0.5);

  CHECK_CUBLAS(cublasDestroy(handle));
  return 0;
}
