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
  static float mul(float a, float b) { return a * b; }
  static bool near(float a, float b) { return std::fabs(a - b) < 1.0e-4f; }
};

template <> struct Ops<double> {
  static double value(float real, float) { return static_cast<double>(real); }
  static double mul(double a, double b) { return a * b; }
  static bool near(double a, double b) { return std::fabs(a - b) < 1.0e-9; }
};

template <> struct Ops<cuComplex> {
  static cuComplex value(float real, float imag) {
    return make_cuComplex(real, imag);
  }
  static cuComplex mul(cuComplex a, cuComplex b) { return cuCmulf(a, b); }
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
  static cuDoubleComplex mul(cuDoubleComplex a, cuDoubleComplex b) {
    return cuCmul(a, b);
  }
  static bool near(cuDoubleComplex a, cuDoubleComplex b) {
    return std::fabs(a.x - b.x) < 1.0e-9 &&
           std::fabs(a.y - b.y) < 1.0e-9;
  }
};

template <typename T> struct Dgmm;

template <> struct Dgmm<float> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasSideMode_t mode,
                               int m, int n, const float *a, int lda,
                               const float *x, int incx, float *c, int ldc) {
    return cublasSdgmm(handle, mode, m, n, a, lda, x, incx, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasSideMode_t mode,
                               int64_t m, int64_t n, const float *a,
                               int64_t lda, const float *x, int64_t incx,
                               float *c, int64_t ldc) {
    return cublasSdgmm_64(handle, mode, m, n, a, lda, x, incx, c, ldc);
  }
};

template <> struct Dgmm<double> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasSideMode_t mode,
                               int m, int n, const double *a, int lda,
                               const double *x, int incx, double *c,
                               int ldc) {
    return cublasDdgmm(handle, mode, m, n, a, lda, x, incx, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasSideMode_t mode,
                               int64_t m, int64_t n, const double *a,
                               int64_t lda, const double *x, int64_t incx,
                               double *c, int64_t ldc) {
    return cublasDdgmm_64(handle, mode, m, n, a, lda, x, incx, c, ldc);
  }
};

template <> struct Dgmm<cuComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasSideMode_t mode,
                               int m, int n, const cuComplex *a, int lda,
                               const cuComplex *x, int incx, cuComplex *c,
                               int ldc) {
    return cublasCdgmm(handle, mode, m, n, a, lda, x, incx, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasSideMode_t mode,
                               int64_t m, int64_t n, const cuComplex *a,
                               int64_t lda, const cuComplex *x, int64_t incx,
                               cuComplex *c, int64_t ldc) {
    return cublasCdgmm_64(handle, mode, m, n, a, lda, x, incx, c, ldc);
  }
};

template <> struct Dgmm<cuDoubleComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasSideMode_t mode,
                               int m, int n, const cuDoubleComplex *a, int lda,
                               const cuDoubleComplex *x, int incx,
                               cuDoubleComplex *c, int ldc) {
    return cublasZdgmm(handle, mode, m, n, a, lda, x, incx, c, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasSideMode_t mode,
                               int64_t m, int64_t n, const cuDoubleComplex *a,
                               int64_t lda, const cuDoubleComplex *x,
                               int64_t incx, cuDoubleComplex *c, int64_t ldc) {
    return cublasZdgmm_64(handle, mode, m, n, a, lda, x, incx, c, ldc);
  }
};

template <typename T>
static std::vector<T> expected_dgmm(const std::vector<T> &a,
                                    const std::vector<T> &x,
                                    cublasSideMode_t mode, int m, int n,
                                    int incx) {
  std::vector<T> out(m * n);
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < m; ++row) {
      const int x_index = (mode == CUBLAS_SIDE_LEFT) ? row * incx : col * incx;
      out[row + col * m] = Ops<T>::mul(a[row + col * m], x[x_index]);
    }
  }
  return out;
}

template <typename T>
static void run_case(cublasHandle_t handle, const char *name, bool use_64,
                     cublasSideMode_t mode) {
  const int m = 3;
  const int n = 4;
  const int incx = 2;
  const int x_len = ((mode == CUBLAS_SIDE_LEFT) ? m : n) * incx;

  std::vector<T> a(m * n);
  std::vector<T> c(m * n);
  std::vector<T> x(x_len);
  for (int i = 0; i < m * n; ++i) {
    a[i] = Ops<T>::value(static_cast<float>(i + 1),
                         static_cast<float>((i % 3) - 1) * 0.25f);
    c[i] = Ops<T>::value(0.0f, 0.0f);
  }
  for (int i = 0; i < x_len; ++i) {
    x[i] = Ops<T>::value(static_cast<float>(i + 2),
                         static_cast<float>((i % 2) + 1) * -0.125f);
  }

  std::vector<T> expected = expected_dgmm(a, x, mode, m, n, incx);

  T *d_a = nullptr;
  T *d_c = nullptr;
  T *d_x = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, a.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_c, c.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(T)));
  CHECK_CUDA(
      cudaMemcpy(d_a, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_c, c.data(), c.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_x, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));

  if (use_64) {
    CHECK_CUBLAS(Dgmm<T>::call64(handle, mode, m, n, d_a, m, d_x, incx, d_c,
                                 m));
  } else {
    CHECK_CUBLAS(Dgmm<T>::call32(handle, mode, m, n, d_a, m, d_x, incx, d_c,
                                 m));
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<T> out(c.size());
  CHECK_CUDA(cudaMemcpy(out.data(), d_c, out.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < out.size(); ++i) {
    if (!Ops<T>::near(out[i], expected[i])) {
      std::printf("%s dgmm mismatch index %zu use_64=%d mode=%d\n", name, i,
                  use_64 ? 1 : 0, static_cast<int>(mode));
      std::exit(EXIT_FAILURE);
    }
  }

  cudaFree(d_x);
  cudaFree(d_c);
  cudaFree(d_a);
}

template <typename T>
static void run_type(cublasHandle_t handle, const char *name) {
  run_case<T>(handle, name, false, CUBLAS_SIDE_LEFT);
  run_case<T>(handle, name, false, CUBLAS_SIDE_RIGHT);
  run_case<T>(handle, name, true, CUBLAS_SIDE_LEFT);
  run_case<T>(handle, name, true, CUBLAS_SIDE_RIGHT);
}

int main() {
  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));

  run_type<float>(handle, "float");
  run_type<double>(handle, "double");
  run_type<cuComplex>(handle, "complex-float");
  run_type<cuDoubleComplex>(handle, "complex-double");

  CHECK_CUBLAS(cublasDestroy(handle));
  return EXIT_SUCCESS;
}
