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

template <typename T> struct Gemv;

template <> struct Gemv<float> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n,
                               const float *alpha, const float *a, int lda,
                               const float *x, int incx, const float *beta,
                               float *y, int incy) {
    return cublasSgemv_v2(handle, CUBLAS_OP_N, m, n, alpha, a, lda, x, incx,
                          beta, y, incy);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               const float *alpha, const float *a,
                               int64_t lda, const float *x, int64_t incx,
                               const float *beta, float *y, int64_t incy) {
    return cublasSgemv_v2_64(handle, CUBLAS_OP_N, m, n, alpha, a, lda, x, incx,
                             beta, y, incy);
  }
};

template <> struct Gemv<double> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n,
                               const double *alpha, const double *a, int lda,
                               const double *x, int incx, const double *beta,
                               double *y, int incy) {
    return cublasDgemv_v2(handle, CUBLAS_OP_N, m, n, alpha, a, lda, x, incx,
                          beta, y, incy);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               const double *alpha, const double *a,
                               int64_t lda, const double *x, int64_t incx,
                               const double *beta, double *y, int64_t incy) {
    return cublasDgemv_v2_64(handle, CUBLAS_OP_N, m, n, alpha, a, lda, x, incx,
                             beta, y, incy);
  }
};

template <> struct Gemv<cuComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n,
                               const cuComplex *alpha, const cuComplex *a,
                               int lda, const cuComplex *x, int incx,
                               const cuComplex *beta, cuComplex *y, int incy) {
    return cublasCgemv_v2(handle, CUBLAS_OP_N, m, n, alpha, a, lda, x, incx,
                          beta, y, incy);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               const cuComplex *alpha, const cuComplex *a,
                               int64_t lda, const cuComplex *x, int64_t incx,
                               const cuComplex *beta, cuComplex *y,
                               int64_t incy) {
    return cublasCgemv_v2_64(handle, CUBLAS_OP_N, m, n, alpha, a, lda, x,
                             incx, beta, y, incy);
  }
};

template <> struct Gemv<cuDoubleComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *a, int lda,
                               const cuDoubleComplex *x, int incx,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *y, int incy) {
    return cublasZgemv_v2(handle, CUBLAS_OP_N, m, n, alpha, a, lda, x, incx,
                          beta, y, incy);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *a, int64_t lda,
                               const cuDoubleComplex *x, int64_t incx,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *y, int64_t incy) {
    return cublasZgemv_v2_64(handle, CUBLAS_OP_N, m, n, alpha, a, lda, x,
                             incx, beta, y, incy);
  }
};

template <typename T> struct BatchedGemv;

template <> struct BatchedGemv<float> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n,
                               const float *alpha,
                               const float *const a_array[], int lda,
                               const float *const x_array[], int incx,
                               const float *beta, float *const y_array[],
                               int incy, int batch_count) {
    return cublasSgemvBatched(handle, CUBLAS_OP_N, m, n, alpha, a_array, lda,
                              x_array, incx, beta, y_array, incy,
                              batch_count);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               const float *alpha,
                               const float *const a_array[], int64_t lda,
                               const float *const x_array[], int64_t incx,
                               const float *beta, float *const y_array[],
                               int64_t incy, int64_t batch_count) {
    return cublasSgemvBatched_64(handle, CUBLAS_OP_N, m, n, alpha, a_array,
                                 lda, x_array, incx, beta, y_array, incy,
                                 batch_count);
  }
};

template <> struct BatchedGemv<double> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n,
                               const double *alpha,
                               const double *const a_array[], int lda,
                               const double *const x_array[], int incx,
                               const double *beta, double *const y_array[],
                               int incy, int batch_count) {
    return cublasDgemvBatched(handle, CUBLAS_OP_N, m, n, alpha, a_array, lda,
                              x_array, incx, beta, y_array, incy,
                              batch_count);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               const double *alpha,
                               const double *const a_array[], int64_t lda,
                               const double *const x_array[], int64_t incx,
                               const double *beta, double *const y_array[],
                               int64_t incy, int64_t batch_count) {
    return cublasDgemvBatched_64(handle, CUBLAS_OP_N, m, n, alpha, a_array,
                                 lda, x_array, incx, beta, y_array, incy,
                                 batch_count);
  }
};

template <> struct BatchedGemv<cuComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n,
                               const cuComplex *alpha,
                               const cuComplex *const a_array[], int lda,
                               const cuComplex *const x_array[], int incx,
                               const cuComplex *beta,
                               cuComplex *const y_array[], int incy,
                               int batch_count) {
    return cublasCgemvBatched(handle, CUBLAS_OP_N, m, n, alpha, a_array, lda,
                              x_array, incx, beta, y_array, incy,
                              batch_count);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               const cuComplex *alpha,
                               const cuComplex *const a_array[], int64_t lda,
                               const cuComplex *const x_array[], int64_t incx,
                               const cuComplex *beta,
                               cuComplex *const y_array[], int64_t incy,
                               int64_t batch_count) {
    return cublasCgemvBatched_64(handle, CUBLAS_OP_N, m, n, alpha, a_array,
                                 lda, x_array, incx, beta, y_array, incy,
                                 batch_count);
  }
};

template <> struct BatchedGemv<cuDoubleComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *const a_array[], int lda,
                               const cuDoubleComplex *const x_array[], int incx,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *const y_array[], int incy,
                               int batch_count) {
    return cublasZgemvBatched(handle, CUBLAS_OP_N, m, n, alpha, a_array, lda,
                              x_array, incx, beta, y_array, incy,
                              batch_count);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *const a_array[],
                               int64_t lda,
                               const cuDoubleComplex *const x_array[],
                               int64_t incx, const cuDoubleComplex *beta,
                               cuDoubleComplex *const y_array[], int64_t incy,
                               int64_t batch_count) {
    return cublasZgemvBatched_64(handle, CUBLAS_OP_N, m, n, alpha, a_array,
                                 lda, x_array, incx, beta, y_array, incy,
                                 batch_count);
  }
};

template <typename T> struct StridedBatchedGemv;

template <> struct StridedBatchedGemv<float> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n,
                               const float *alpha, const float *a, int lda,
                               long long stride_a, const float *x, int incx,
                               long long stride_x, const float *beta, float *y,
                               int incy, long long stride_y,
                               int batch_count) {
    return cublasSgemvStridedBatched(handle, CUBLAS_OP_N, m, n, alpha, a, lda,
                                     stride_a, x, incx, stride_x, beta, y,
                                     incy, stride_y, batch_count);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               const float *alpha, const float *a, int64_t lda,
                               long long stride_a, const float *x,
                               int64_t incx, long long stride_x,
                               const float *beta, float *y, int64_t incy,
                               long long stride_y, int64_t batch_count) {
    return cublasSgemvStridedBatched_64(
        handle, CUBLAS_OP_N, m, n, alpha, a, lda, stride_a, x, incx, stride_x,
        beta, y, incy, stride_y, batch_count);
  }
};

template <> struct StridedBatchedGemv<double> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n,
                               const double *alpha, const double *a, int lda,
                               long long stride_a, const double *x, int incx,
                               long long stride_x, const double *beta,
                               double *y, int incy, long long stride_y,
                               int batch_count) {
    return cublasDgemvStridedBatched(handle, CUBLAS_OP_N, m, n, alpha, a, lda,
                                     stride_a, x, incx, stride_x, beta, y,
                                     incy, stride_y, batch_count);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               const double *alpha, const double *a,
                               int64_t lda, long long stride_a,
                               const double *x, int64_t incx,
                               long long stride_x, const double *beta,
                               double *y, int64_t incy, long long stride_y,
                               int64_t batch_count) {
    return cublasDgemvStridedBatched_64(
        handle, CUBLAS_OP_N, m, n, alpha, a, lda, stride_a, x, incx, stride_x,
        beta, y, incy, stride_y, batch_count);
  }
};

template <> struct StridedBatchedGemv<cuComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n,
                               const cuComplex *alpha, const cuComplex *a,
                               int lda, long long stride_a,
                               const cuComplex *x, int incx,
                               long long stride_x, const cuComplex *beta,
                               cuComplex *y, int incy, long long stride_y,
                               int batch_count) {
    return cublasCgemvStridedBatched(handle, CUBLAS_OP_N, m, n, alpha, a, lda,
                                     stride_a, x, incx, stride_x, beta, y,
                                     incy, stride_y, batch_count);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               const cuComplex *alpha, const cuComplex *a,
                               int64_t lda, long long stride_a,
                               const cuComplex *x, int64_t incx,
                               long long stride_x, const cuComplex *beta,
                               cuComplex *y, int64_t incy,
                               long long stride_y, int64_t batch_count) {
    return cublasCgemvStridedBatched_64(
        handle, CUBLAS_OP_N, m, n, alpha, a, lda, stride_a, x, incx, stride_x,
        beta, y, incy, stride_y, batch_count);
  }
};

template <> struct StridedBatchedGemv<cuDoubleComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, int m, int n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *a, int lda,
                               long long stride_a,
                               const cuDoubleComplex *x, int incx,
                               long long stride_x,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *y, int incy,
                               long long stride_y, int batch_count) {
    return cublasZgemvStridedBatched(handle, CUBLAS_OP_N, m, n, alpha, a, lda,
                                     stride_a, x, incx, stride_x, beta, y,
                                     incy, stride_y, batch_count);
  }
  static cublasStatus_t call64(cublasHandle_t handle, int64_t m, int64_t n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *a, int64_t lda,
                               long long stride_a,
                               const cuDoubleComplex *x, int64_t incx,
                               long long stride_x,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *y, int64_t incy,
                               long long stride_y, int64_t batch_count) {
    return cublasZgemvStridedBatched_64(
        handle, CUBLAS_OP_N, m, n, alpha, a, lda, stride_a, x, incx, stride_x,
        beta, y, incy, stride_y, batch_count);
  }
};

template <typename T>
static std::vector<T> expected_gemv(const std::vector<T> &a,
                                    const std::vector<T> &x,
                                    const std::vector<T> &y, int m, int n,
                                    T alpha, T beta) {
  std::vector<T> out(m);
  for (int row = 0; row < m; ++row) {
    T sum = Ops<T>::zero();
    for (int col = 0; col < n; ++col) {
      sum = Ops<T>::add(sum, Ops<T>::mul(a[row + col * m], x[col]));
    }
    out[row] = Ops<T>::add(Ops<T>::mul(alpha, sum), Ops<T>::mul(beta, y[row]));
  }
  return out;
}

template <typename T>
static void run_case(cublasHandle_t handle, const char *name, bool use_64,
                     bool device_scalars) {
  const int m = 3;
  const int n = 2;
  std::vector<T> a(m * n);
  std::vector<T> x(n);
  std::vector<T> y(m);
  for (int i = 0; i < m * n; ++i) {
    a[i] = Ops<T>::value(static_cast<float>(i + 1),
                         static_cast<float>((i % 3) - 1) * 0.25f);
  }
  for (int i = 0; i < n; ++i) {
    x[i] = Ops<T>::value(static_cast<float>(i + 2),
                         static_cast<float>(i + 1) * -0.125f);
  }
  for (int i = 0; i < m; ++i) {
    y[i] = Ops<T>::value(static_cast<float>(i - 1),
                         static_cast<float>(i % 2) * 0.5f);
  }

  T alpha = Ops<T>::alpha();
  T beta = Ops<T>::beta();
  std::vector<T> expected = expected_gemv(a, x, y, m, n, alpha, beta);

  T *d_a = nullptr;
  T *d_x = nullptr;
  T *d_y = nullptr;
  T *d_alpha = nullptr;
  T *d_beta = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, a.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_y, y.size() * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_a, a.data(), a.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_x, x.data(), x.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_y, y.data(), y.size() * sizeof(T),
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
    CHECK_CUBLAS(
        Gemv<T>::call64(handle, m, n, alpha_arg, d_a, m, d_x, 1, beta_arg,
                        d_y, 1));
  } else {
    CHECK_CUBLAS(
        Gemv<T>::call32(handle, m, n, alpha_arg, d_a, m, d_x, 1, beta_arg,
                        d_y, 1));
  }
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<T> out(y.size());
  CHECK_CUDA(cudaMemcpy(out.data(), d_y, out.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < out.size(); ++i) {
    if (!Ops<T>::near(out[i], expected[i])) {
      std::printf("%s gemv mismatch index %zu use_64=%d device_scalars=%d\n",
                  name, i, use_64 ? 1 : 0, device_scalars ? 1 : 0);
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
static void fill_batch_inputs(std::vector<T> &a, std::vector<T> &x,
                              std::vector<T> &y, int m, int n, int batch) {
  for (int i = 0; i < m * n; ++i) {
    a[i] = Ops<T>::value(static_cast<float>(i + 1 + batch * 2),
                         static_cast<float>((i + batch) % 3 - 1) * 0.25f);
  }
  for (int i = 0; i < n; ++i) {
    x[i] = Ops<T>::value(static_cast<float>(i + 2 + batch),
                         static_cast<float>(i + batch + 1) * -0.125f);
  }
  for (int i = 0; i < m; ++i) {
    y[i] = Ops<T>::value(static_cast<float>(i - 1 + batch),
                         static_cast<float>((i + batch) % 2) * 0.5f);
  }
}

template <typename T>
static void run_batched_case(cublasHandle_t handle, const char *name,
                             bool use_64, bool device_scalars) {
  const int m = 3;
  const int n = 2;
  const int batch_count = 2;
  std::vector<std::vector<T>> a(batch_count);
  std::vector<std::vector<T>> x(batch_count);
  std::vector<std::vector<T>> y(batch_count);
  std::vector<std::vector<T>> expected(batch_count);
  for (int batch = 0; batch < batch_count; ++batch) {
    a[batch].resize(m * n);
    x[batch].resize(n);
    y[batch].resize(m);
    fill_batch_inputs(a[batch], x[batch], y[batch], m, n, batch);
    expected[batch] =
        expected_gemv(a[batch], x[batch], y[batch], m, n, Ops<T>::alpha(),
                      Ops<T>::beta());
  }

  std::vector<T *> d_a(batch_count, nullptr);
  std::vector<T *> d_x(batch_count, nullptr);
  std::vector<T *> d_y(batch_count, nullptr);
  std::vector<const T *> h_a_array(batch_count, nullptr);
  std::vector<const T *> h_x_array(batch_count, nullptr);
  std::vector<T *> h_y_array(batch_count, nullptr);
  for (int batch = 0; batch < batch_count; ++batch) {
    CHECK_CUDA(cudaMalloc(&d_a[batch], a[batch].size() * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_x[batch], x[batch].size() * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_y[batch], y[batch].size() * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(d_a[batch], a[batch].data(),
                          a[batch].size() * sizeof(T),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x[batch], x[batch].data(),
                          x[batch].size() * sizeof(T),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y[batch], y[batch].data(),
                          y[batch].size() * sizeof(T),
                          cudaMemcpyHostToDevice));
    h_a_array[batch] = d_a[batch];
    h_x_array[batch] = d_x[batch];
    h_y_array[batch] = d_y[batch];
  }

  const T **d_a_array = nullptr;
  const T **d_x_array = nullptr;
  T **d_y_array = nullptr;
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_a_array),
                        batch_count * sizeof(T *)));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_x_array),
                        batch_count * sizeof(T *)));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_y_array),
                        batch_count * sizeof(T *)));
  CHECK_CUDA(cudaMemcpy(d_a_array, h_a_array.data(),
                        batch_count * sizeof(T *), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_x_array, h_x_array.data(),
                        batch_count * sizeof(T *), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_y_array, h_y_array.data(),
                        batch_count * sizeof(T *), cudaMemcpyHostToDevice));

  T alpha = Ops<T>::alpha();
  T beta = Ops<T>::beta();
  T *d_alpha = nullptr;
  T *d_beta = nullptr;
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
    CHECK_CUBLAS(BatchedGemv<T>::call64(
        handle, m, n, alpha_arg, d_a_array, m, d_x_array, 1, beta_arg,
        d_y_array, 1, batch_count));
  } else {
    CHECK_CUBLAS(BatchedGemv<T>::call32(
        handle, m, n, alpha_arg, d_a_array, m, d_x_array, 1, beta_arg,
        d_y_array, 1, batch_count));
  }
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  CHECK_CUDA(cudaDeviceSynchronize());

  for (int batch = 0; batch < batch_count; ++batch) {
    std::vector<T> out(y[batch].size());
    CHECK_CUDA(cudaMemcpy(out.data(), d_y[batch], out.size() * sizeof(T),
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < out.size(); ++i) {
      if (!Ops<T>::near(out[i], expected[batch][i])) {
        std::printf(
            "%s batched gemv mismatch batch %d index %zu use_64=%d "
            "device_scalars=%d\n",
            name, batch, i, use_64 ? 1 : 0, device_scalars ? 1 : 0);
        std::exit(EXIT_FAILURE);
      }
    }
  }

  cudaFree(d_beta);
  cudaFree(d_alpha);
  cudaFree(d_y_array);
  cudaFree(d_x_array);
  cudaFree(d_a_array);
  for (int batch = 0; batch < batch_count; ++batch) {
    cudaFree(d_y[batch]);
    cudaFree(d_x[batch]);
    cudaFree(d_a[batch]);
  }
}

template <typename T>
static void run_strided_batched_case(cublasHandle_t handle, const char *name,
                                     bool use_64, bool device_scalars) {
  const int m = 3;
  const int n = 2;
  const int batch_count = 2;
  const long long stride_a = m * n + 1;
  const long long stride_x = n + 1;
  const long long stride_y = m + 1;
  std::vector<T> a(batch_count * stride_a, Ops<T>::zero());
  std::vector<T> x(batch_count * stride_x, Ops<T>::zero());
  std::vector<T> y(batch_count * stride_y, Ops<T>::zero());
  std::vector<std::vector<T>> expected(batch_count);
  for (int batch = 0; batch < batch_count; ++batch) {
    std::vector<T> batch_a(m * n);
    std::vector<T> batch_x(n);
    std::vector<T> batch_y(m);
    fill_batch_inputs(batch_a, batch_x, batch_y, m, n, batch);
    for (int i = 0; i < m * n; ++i) {
      a[batch * stride_a + i] = batch_a[i];
    }
    for (int i = 0; i < n; ++i) {
      x[batch * stride_x + i] = batch_x[i];
    }
    for (int i = 0; i < m; ++i) {
      y[batch * stride_y + i] = batch_y[i];
    }
    expected[batch] =
        expected_gemv(batch_a, batch_x, batch_y, m, n, Ops<T>::alpha(),
                      Ops<T>::beta());
  }

  T *d_a = nullptr;
  T *d_x = nullptr;
  T *d_y = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, a.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_y, y.size() * sizeof(T)));
  CHECK_CUDA(
      cudaMemcpy(d_a, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_x, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_y, y.data(), y.size() * sizeof(T), cudaMemcpyHostToDevice));

  T alpha = Ops<T>::alpha();
  T beta = Ops<T>::beta();
  T *d_alpha = nullptr;
  T *d_beta = nullptr;
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
    CHECK_CUBLAS(StridedBatchedGemv<T>::call64(
        handle, m, n, alpha_arg, d_a, m, stride_a, d_x, 1, stride_x, beta_arg,
        d_y, 1, stride_y, batch_count));
  } else {
    CHECK_CUBLAS(StridedBatchedGemv<T>::call32(
        handle, m, n, alpha_arg, d_a, m, stride_a, d_x, 1, stride_x, beta_arg,
        d_y, 1, stride_y, batch_count));
  }
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<T> out(y.size());
  CHECK_CUDA(cudaMemcpy(out.data(), d_y, out.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
  for (int batch = 0; batch < batch_count; ++batch) {
    for (int i = 0; i < m; ++i) {
      if (!Ops<T>::near(out[batch * stride_y + i], expected[batch][i])) {
        std::printf(
            "%s strided batched gemv mismatch batch %d index %d use_64=%d "
            "device_scalars=%d\n",
            name, batch, i, use_64 ? 1 : 0, device_scalars ? 1 : 0);
        std::exit(EXIT_FAILURE);
      }
    }
  }

  cudaFree(d_beta);
  cudaFree(d_alpha);
  cudaFree(d_y);
  cudaFree(d_x);
  cudaFree(d_a);
}

template <typename T>
static void run_type(cublasHandle_t handle, const char *name) {
  run_case<T>(handle, name, false, false);
  run_case<T>(handle, name, false, true);
  run_case<T>(handle, name, true, false);
  run_case<T>(handle, name, true, true);
  run_batched_case<T>(handle, name, false, false);
  run_batched_case<T>(handle, name, false, true);
  run_batched_case<T>(handle, name, true, false);
  run_batched_case<T>(handle, name, true, true);
  run_strided_batched_case<T>(handle, name, false, false);
  run_strided_batched_case<T>(handle, name, false, true);
  run_strided_batched_case<T>(handle, name, true, false);
  run_strided_batched_case<T>(handle, name, true, true);
}

int main() {
  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));
  run_type<float>(handle, "sgemv");
  run_type<double>(handle, "dgemv");
  run_type<cuComplex>(handle, "cgemv");
  run_type<cuDoubleComplex>(handle, "zgemv");
  CHECK_CUBLAS(cublasDestroy(handle));
  return 0;
}
