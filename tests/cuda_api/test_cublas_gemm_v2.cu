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
  static float alpha() { return 1.25f; }
  static float beta() { return -0.5f; }
  static bool near(float a, float b) { return std::fabs(a - b) < 1.0e-4f; }
};

template <> struct Ops<double> {
  static double value(float real, float) { return static_cast<double>(real); }
  static double zero() { return 0.0; }
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
  static cuComplex zero() { return make_cuComplex(0.0f, 0.0f); }
  static cuComplex add(cuComplex a, cuComplex b) { return cuCaddf(a, b); }
  static cuComplex mul(cuComplex a, cuComplex b) { return cuCmulf(a, b); }
  static cuComplex alpha() { return make_cuComplex(0.75f, 0.5f); }
  static cuComplex beta() { return make_cuComplex(-0.25f, 0.125f); }
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
  static cuDoubleComplex alpha() { return make_cuDoubleComplex(0.75, 0.5); }
  static cuDoubleComplex beta() { return make_cuDoubleComplex(-0.25, 0.125); }
  static bool near(cuDoubleComplex a, cuDoubleComplex b) {
    return std::fabs(a.x - b.x) < 1.0e-9 &&
           std::fabs(a.y - b.y) < 1.0e-9;
  }
};

template <typename T> struct Gemm;
template <typename T> struct StridedGemm;

template <> struct Gemm<float> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n, int k,
                               const float *alpha, const float *a, int lda,
                               const float *b, int ldb, const float *beta,
                               float *c, int ldc) {
    return cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, a,
                          lda, b, ldb, beta, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               int64_t k, const float *alpha, const float *a,
                               int64_t lda, const float *b, int64_t ldb,
                               const float *beta, float *c, int64_t ldc) {
    return cublasSgemm_v2_64(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha,
                             a, lda, b, ldb, beta, c, ldc);
  }
};

template <> struct StridedGemm<float> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n, int k,
                               const float *alpha, const float *a, int lda,
                               long long stride_a, const float *b, int ldb,
                               long long stride_b, const float *beta, float *c,
                               int ldc, long long stride_c, int batch_count) {
    return cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, a, lda, stride_a, b,
        ldb, stride_b, beta, c, ldc, stride_c, batch_count);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               int64_t k, const float *alpha, const float *a,
                               int64_t lda, long long stride_a, const float *b,
                               int64_t ldb, long long stride_b,
                               const float *beta, float *c, int64_t ldc,
                               long long stride_c, int64_t batch_count) {
    return cublasSgemmStridedBatched_64(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, a, lda, stride_a, b,
        ldb, stride_b, beta, c, ldc, stride_c, batch_count);
  }
};

template <> struct Gemm<double> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n, int k,
                               const double *alpha, const double *a, int lda,
                               const double *b, int ldb, const double *beta,
                               double *c, int ldc) {
    return cublasDgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, a,
                          lda, b, ldb, beta, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               int64_t k, const double *alpha, const double *a,
                               int64_t lda, const double *b, int64_t ldb,
                               const double *beta, double *c, int64_t ldc) {
    return cublasDgemm_v2_64(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha,
                             a, lda, b, ldb, beta, c, ldc);
  }
};

template <> struct StridedGemm<double> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n, int k,
                               const double *alpha, const double *a, int lda,
                               long long stride_a, const double *b, int ldb,
                               long long stride_b, const double *beta,
                               double *c, int ldc, long long stride_c,
                               int batch_count) {
    return cublasDgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, a, lda, stride_a, b,
        ldb, stride_b, beta, c, ldc, stride_c, batch_count);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               int64_t k, const double *alpha, const double *a,
                               int64_t lda, long long stride_a,
                               const double *b, int64_t ldb,
                               long long stride_b, const double *beta,
                               double *c, int64_t ldc, long long stride_c,
                               int64_t batch_count) {
    return cublasDgemmStridedBatched_64(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, a, lda, stride_a, b,
        ldb, stride_b, beta, c, ldc, stride_c, batch_count);
  }
};

template <> struct Gemm<cuComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n, int k,
                               const cuComplex *alpha, const cuComplex *a,
                               int lda, const cuComplex *b, int ldb,
                               const cuComplex *beta, cuComplex *c, int ldc) {
    return cublasCgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, a,
                          lda, b, ldb, beta, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               int64_t k, const cuComplex *alpha,
                               const cuComplex *a, int64_t lda,
                               const cuComplex *b, int64_t ldb,
                               const cuComplex *beta, cuComplex *c,
                               int64_t ldc) {
    return cublasCgemm_v2_64(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha,
                             a, lda, b, ldb, beta, c, ldc);
  }
};

template <> struct StridedGemm<cuComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n, int k,
                               const cuComplex *alpha, const cuComplex *a,
                               int lda, long long stride_a, const cuComplex *b,
                               int ldb, long long stride_b,
                               const cuComplex *beta, cuComplex *c, int ldc,
                               long long stride_c, int batch_count) {
    return cublasCgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, a, lda, stride_a, b,
        ldb, stride_b, beta, c, ldc, stride_c, batch_count);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               int64_t k, const cuComplex *alpha,
                               const cuComplex *a, int64_t lda,
                               long long stride_a, const cuComplex *b,
                               int64_t ldb, long long stride_b,
                               const cuComplex *beta, cuComplex *c,
                               int64_t ldc, long long stride_c,
                               int64_t batch_count) {
    return cublasCgemmStridedBatched_64(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, a, lda, stride_a, b,
        ldb, stride_b, beta, c, ldc, stride_c, batch_count);
  }
};

template <> struct Gemm<cuDoubleComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n, int k,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *a, int lda,
                               const cuDoubleComplex *b, int ldb,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *c, int ldc) {
    return cublasZgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, a,
                          lda, b, ldb, beta, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               int64_t k, const cuDoubleComplex *alpha,
                               const cuDoubleComplex *a, int64_t lda,
                               const cuDoubleComplex *b, int64_t ldb,
                               const cuDoubleComplex *beta, cuDoubleComplex *c,
                               int64_t ldc) {
    return cublasZgemm_v2_64(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha,
                             a, lda, b, ldb, beta, c, ldc);
  }
};

template <> struct StridedGemm<cuDoubleComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n, int k,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *a, int lda,
                               long long stride_a, const cuDoubleComplex *b,
                               int ldb, long long stride_b,
                               const cuDoubleComplex *beta, cuDoubleComplex *c,
                               int ldc, long long stride_c, int batch_count) {
    return cublasZgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, a, lda, stride_a, b,
        ldb, stride_b, beta, c, ldc, stride_c, batch_count);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               int64_t k, const cuDoubleComplex *alpha,
                               const cuDoubleComplex *a, int64_t lda,
                               long long stride_a, const cuDoubleComplex *b,
                               int64_t ldb, long long stride_b,
                               const cuDoubleComplex *beta, cuDoubleComplex *c,
                               int64_t ldc, long long stride_c,
                               int64_t batch_count) {
    return cublasZgemmStridedBatched_64(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, a, lda, stride_a, b,
        ldb, stride_b, beta, c, ldc, stride_c, batch_count);
  }
};

template <typename T>
static std::vector<T> expected_gemm(const std::vector<T> &a,
                                    const std::vector<T> &b,
                                    const std::vector<T> &c, int m, int n,
                                    int k, T alpha, T beta) {
  std::vector<T> out(m * n);
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < m; ++row) {
      T sum = Ops<T>::zero();
      for (int inner = 0; inner < k; ++inner) {
        sum = Ops<T>::add(sum, Ops<T>::mul(a[row + inner * m],
                                           b[inner + col * k]));
      }
      out[row + col * m] =
          Ops<T>::add(Ops<T>::mul(alpha, sum),
                      Ops<T>::mul(beta, c[row + col * m]));
    }
  }
  return out;
}

template <typename T>
static std::vector<T>
expected_strided_gemm(const std::vector<T> &a, const std::vector<T> &b,
                      const std::vector<T> &c, int m, int n, int k,
                      long long stride_a, long long stride_b,
                      long long stride_c, int batch_count, T alpha, T beta) {
  std::vector<T> out(c.size());
  for (int batch = 0; batch < batch_count; ++batch) {
    const long long a_base = batch * stride_a;
    const long long b_base = batch * stride_b;
    const long long c_base = batch * stride_c;
    for (int col = 0; col < n; ++col) {
      for (int row = 0; row < m; ++row) {
        T sum = Ops<T>::zero();
        for (int inner = 0; inner < k; ++inner) {
          sum = Ops<T>::add(
              sum, Ops<T>::mul(a[a_base + row + inner * m],
                               b[b_base + inner + col * k]));
        }
        out[c_base + row + col * m] =
            Ops<T>::add(Ops<T>::mul(alpha, sum),
                        Ops<T>::mul(beta, c[c_base + row + col * m]));
      }
    }
  }
  return out;
}

template <typename T>
static void run_case(cublasHandle_t handle, const char *name, bool use_64,
                     bool device_scalars) {
  const int m = 2;
  const int n = 2;
  const int k = 3;
  std::vector<T> a(m * k);
  std::vector<T> b(k * n);
  std::vector<T> c = {
      Ops<T>::value(0.5f, -0.25f),
      Ops<T>::value(-1.0f, 0.125f),
      Ops<T>::value(1.5f, 0.75f),
      Ops<T>::value(-2.0f, -0.5f),
  };
  for (int i = 0; i < m * k; ++i) {
    a[i] = Ops<T>::value(static_cast<float>(i + 1),
                         static_cast<float>((i % 3) - 1) * 0.25f);
  }
  for (int i = 0; i < k * n; ++i) {
    b[i] = Ops<T>::value(static_cast<float>(i + 7),
                         static_cast<float>((i % 2) + 1) * -0.125f);
  }

  T alpha = Ops<T>::alpha();
  T beta = Ops<T>::beta();
  std::vector<T> expected = expected_gemm(a, b, c, m, n, k, alpha, beta);

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
    CHECK_CUBLAS(Gemm<T>::call64(handle, m, n, k, alpha_arg, d_a, m, d_b, k,
                                 beta_arg, d_c, m));
  } else {
    CHECK_CUBLAS(Gemm<T>::call32(handle, m, n, k, alpha_arg, d_a, m, d_b, k,
                                 beta_arg, d_c, m));
  }
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<T> out(c.size());
  CHECK_CUDA(cudaMemcpy(out.data(), d_c, out.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < out.size(); ++i) {
    if (!Ops<T>::near(out[i], expected[i])) {
      std::printf("%s mismatch index %zu use_64=%d device_scalars=%d\n", name,
                  i, use_64 ? 1 : 0, device_scalars ? 1 : 0);
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
static void run_strided_case(cublasHandle_t handle, const char *name,
                             bool use_64, bool device_scalars) {
  const int m = 2;
  const int n = 2;
  const int k = 3;
  const int batch_count = 2;
  const long long stride_a = m * k;
  const long long stride_b = k * n;
  const long long stride_c = m * n;
  std::vector<T> a(batch_count * stride_a);
  std::vector<T> b(batch_count * stride_b);
  std::vector<T> c(batch_count * stride_c);
  for (size_t i = 0; i < a.size(); ++i) {
    int phase = static_cast<int>(i % 4) - 1;
    a[i] = Ops<T>::value(static_cast<float>(i + 1),
                         static_cast<float>(phase) * 0.125f);
  }
  for (size_t i = 0; i < b.size(); ++i) {
    int phase = static_cast<int>(i % 3) - 1;
    b[i] = Ops<T>::value(static_cast<float>(i + 3),
                         static_cast<float>(phase) * -0.25f);
  }
  for (size_t i = 0; i < c.size(); ++i) {
    int phase = static_cast<int>(i % 5) - 2;
    c[i] = Ops<T>::value(static_cast<float>(phase),
                         static_cast<float>(i % 2) * 0.5f);
  }

  T alpha = Ops<T>::alpha();
  T beta = Ops<T>::beta();
  std::vector<T> expected = expected_strided_gemm(
      a, b, c, m, n, k, stride_a, stride_b, stride_c, batch_count, alpha, beta);

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
    CHECK_CUBLAS(StridedGemm<T>::call64(
        handle, m, n, k, alpha_arg, d_a, m, stride_a, d_b, k, stride_b,
        beta_arg, d_c, m, stride_c, batch_count));
  } else {
    CHECK_CUBLAS(StridedGemm<T>::call32(
        handle, m, n, k, alpha_arg, d_a, m, stride_a, d_b, k, stride_b,
        beta_arg, d_c, m, stride_c, batch_count));
  }
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<T> out(c.size());
  CHECK_CUDA(cudaMemcpy(out.data(), d_c, out.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < out.size(); ++i) {
    if (!Ops<T>::near(out[i], expected[i])) {
      std::printf("%s strided mismatch index %zu use_64=%d device_scalars=%d\n",
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

template <typename T> static void run_type(cublasHandle_t handle,
                                           const char *name) {
  run_case<T>(handle, name, false, false);
  run_case<T>(handle, name, false, true);
  run_case<T>(handle, name, true, false);
  run_case<T>(handle, name, true, true);
  run_strided_case<T>(handle, name, false, false);
  run_strided_case<T>(handle, name, false, true);
  run_strided_case<T>(handle, name, true, false);
  run_strided_case<T>(handle, name, true, true);
}

int main() {
  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));
  run_type<float>(handle, "sgemm");
  run_type<double>(handle, "dgemm");
  run_type<cuComplex>(handle, "cgemm");
  run_type<cuDoubleComplex>(handle, "zgemm");
  CHECK_CUBLAS(cublasDestroy(handle));
  return EXIT_SUCCESS;
}
