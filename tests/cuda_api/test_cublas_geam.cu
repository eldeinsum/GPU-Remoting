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
  static float alpha() { return 1.25f; }
  static float beta() { return -0.5f; }
  static bool near(float a, float b) { return std::fabs(a - b) < 1.0e-4f; }
};

template <> struct Ops<double> {
  static double value(float real, float) { return static_cast<double>(real); }
  static double add(double a, double b) { return a + b; }
  static double mul(double a, double b) { return a * b; }
  static double alpha() { return 1.25; }
  static double beta() { return -0.5; }
  static bool near(double a, double b) { return std::fabs(a - b) < 1.0e-9; }
};

template <> struct Ops<cuComplex> {
  static cuComplex value(float real, float imag) {
    return make_cuComplex(real, imag);
  }
  static cuComplex add(cuComplex a, cuComplex b) { return cuCaddf(a, b); }
  static cuComplex mul(cuComplex a, cuComplex b) { return cuCmulf(a, b); }
  static cuComplex alpha() { return make_cuComplex(1.25f, -0.25f); }
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
  static cuDoubleComplex add(cuDoubleComplex a, cuDoubleComplex b) {
    return cuCadd(a, b);
  }
  static cuDoubleComplex mul(cuDoubleComplex a, cuDoubleComplex b) {
    return cuCmul(a, b);
  }
  static cuDoubleComplex alpha() {
    return make_cuDoubleComplex(1.25, -0.25);
  }
  static cuDoubleComplex beta() { return make_cuDoubleComplex(-0.5, 0.125); }
  static bool near(cuDoubleComplex a, cuDoubleComplex b) {
    return std::fabs(a.x - b.x) < 1.0e-9 &&
           std::fabs(a.y - b.y) < 1.0e-9;
  }
};

template <typename T> struct Geam;

template <> struct Geam<float> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n,
                               const float *alpha, const float *a, int lda,
                               const float *beta, const float *b, int ldb,
                               float *c, int ldc) {
    return cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, alpha, a, lda,
                       beta, b, ldb, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               const float *alpha, const float *a,
                               int64_t lda, const float *beta, const float *b,
                               int64_t ldb, float *c, int64_t ldc) {
    return cublasSgeam_64(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, alpha, a,
                          lda, beta, b, ldb, c, ldc);
  }
};

template <> struct Geam<double> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n,
                               const double *alpha, const double *a, int lda,
                               const double *beta, const double *b, int ldb,
                               double *c, int ldc) {
    return cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, alpha, a, lda,
                       beta, b, ldb, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               const double *alpha, const double *a,
                               int64_t lda, const double *beta,
                               const double *b, int64_t ldb, double *c,
                               int64_t ldc) {
    return cublasDgeam_64(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, alpha, a,
                          lda, beta, b, ldb, c, ldc);
  }
};

template <> struct Geam<cuComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n,
                               const cuComplex *alpha, const cuComplex *a,
                               int lda, const cuComplex *beta,
                               const cuComplex *b, int ldb, cuComplex *c,
                               int ldc) {
    return cublasCgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, alpha, a, lda,
                       beta, b, ldb, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               const cuComplex *alpha, const cuComplex *a,
                               int64_t lda, const cuComplex *beta,
                               const cuComplex *b, int64_t ldb, cuComplex *c,
                               int64_t ldc) {
    return cublasCgeam_64(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, alpha, a,
                          lda, beta, b, ldb, c, ldc);
  }
};

template <> struct Geam<cuDoubleComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *a, int lda,
                               const cuDoubleComplex *beta,
                               const cuDoubleComplex *b, int ldb,
                               cuDoubleComplex *c, int ldc) {
    return cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, alpha, a, lda,
                       beta, b, ldb, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *a, int64_t lda,
                               const cuDoubleComplex *beta,
                               const cuDoubleComplex *b, int64_t ldb,
                               cuDoubleComplex *c, int64_t ldc) {
    return cublasZgeam_64(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, alpha, a,
                          lda, beta, b, ldb, c, ldc);
  }
};

template <typename T>
static std::vector<T> expected_geam(const std::vector<T> &a,
                                    const std::vector<T> &b, int m, int n,
                                    T alpha, T beta) {
  std::vector<T> out(m * n);
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < m; ++row) {
      T a_value = a[row + col * m];
      T b_value = b[col + row * n];
      out[row + col * m] =
          Ops<T>::add(Ops<T>::mul(alpha, a_value), Ops<T>::mul(beta, b_value));
    }
  }
  return out;
}

template <typename T>
static void run_case(cublasHandle_t handle, const char *name, bool use_64,
                     bool device_scalars) {
  const int m = 2;
  const int n = 3;
  std::vector<T> a(m * n);
  std::vector<T> b(n * m);
  std::vector<T> c(m * n);
  for (int i = 0; i < m * n; ++i) {
    a[i] = Ops<T>::value(static_cast<float>(i + 1),
                         static_cast<float>((i % 2) - 1) * 0.25f);
    c[i] = Ops<T>::value(0.0f, 0.0f);
  }
  for (int i = 0; i < n * m; ++i) {
    b[i] = Ops<T>::value(static_cast<float>(i - 2),
                         static_cast<float>((i % 3) + 1) * -0.125f);
  }

  T alpha = Ops<T>::alpha();
  T beta = Ops<T>::beta();
  std::vector<T> expected = expected_geam(a, b, m, n, alpha, beta);

  T *d_a = nullptr;
  T *d_b = nullptr;
  T *d_c = nullptr;
  T *d_alpha = nullptr;
  T *d_beta = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, a.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_b, b.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_c, c.size() * sizeof(T)));
  CHECK_CUDA(
      cudaMemcpy(d_a, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_b, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_c, c.data(), c.size() * sizeof(T), cudaMemcpyHostToDevice));

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
    CHECK_CUBLAS(Geam<T>::call64(handle, m, n, alpha_arg, d_a, m, beta_arg,
                                 d_b, n, d_c, m));
  } else {
    CHECK_CUBLAS(Geam<T>::call32(handle, m, n, alpha_arg, d_a, m, beta_arg,
                                 d_b, n, d_c, m));
  }
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<T> out(c.size());
  CHECK_CUDA(cudaMemcpy(out.data(), d_c, out.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < out.size(); ++i) {
    if (!Ops<T>::near(out[i], expected[i])) {
      std::printf("%s geam mismatch index %zu use_64=%d device_scalars=%d\n",
                  name, i, use_64 ? 1 : 0, device_scalars ? 1 : 0);
      std::exit(EXIT_FAILURE);
    }
  }

  cudaFree(d_beta);
  cudaFree(d_alpha);
  cudaFree(d_c);
  cudaFree(d_b);
  cudaFree(d_a);
}

template <typename T>
static void run_type(cublasHandle_t handle, const char *name) {
  run_case<T>(handle, name, false, false);
  run_case<T>(handle, name, false, true);
  run_case<T>(handle, name, true, false);
  run_case<T>(handle, name, true, true);
}

int main() {
  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));
  run_type<float>(handle, "sgeam");
  run_type<double>(handle, "dgeam");
  run_type<cuComplex>(handle, "cgeam");
  run_type<cuDoubleComplex>(handle, "zgeam");
  CHECK_CUBLAS(cublasDestroy(handle));
  return 0;
}
