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
  static float conj(float a) { return a; }
  static float alpha() { return -0.75f; }
  static bool near(float a, float b) { return std::fabs(a - b) < 1.0e-4f; }
};

template <> struct Ops<double> {
  static double value(float real, float) { return static_cast<double>(real); }
  static double add(double a, double b) { return a + b; }
  static double mul(double a, double b) { return a * b; }
  static double conj(double a) { return a; }
  static double alpha() { return -0.75; }
  static bool near(double a, double b) { return std::fabs(a - b) < 1.0e-9; }
};

template <> struct Ops<cuComplex> {
  static cuComplex value(float real, float imag) {
    return make_cuComplex(real, imag);
  }
  static cuComplex add(cuComplex a, cuComplex b) { return cuCaddf(a, b); }
  static cuComplex mul(cuComplex a, cuComplex b) { return cuCmulf(a, b); }
  static cuComplex conj(cuComplex a) { return cuConjf(a); }
  static cuComplex alpha() { return make_cuComplex(-0.625f, 0.25f); }
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
  static cuDoubleComplex conj(cuDoubleComplex a) { return cuConj(a); }
  static cuDoubleComplex alpha() { return make_cuDoubleComplex(-0.625, 0.25); }
  static bool near(cuDoubleComplex a, cuDoubleComplex b) {
    return std::fabs(a.x - b.x) < 1.0e-9 &&
           std::fabs(a.y - b.y) < 1.0e-9;
  }
};

template <typename T> struct Syr2;

template <> struct Syr2<float> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const float *alpha, const float *x,
                               int incx, const float *y, int incy, float *a,
                               int lda) {
    return cublasSsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, a, lda);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const float *alpha, const float *x,
                               int64_t incx, const float *y, int64_t incy,
                               float *a, int64_t lda) {
    return cublasSsyr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, a,
                             lda);
  }
};

template <> struct Syr2<double> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const double *alpha, const double *x,
                               int incx, const double *y, int incy, double *a,
                               int lda) {
    return cublasDsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, a, lda);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const double *alpha, const double *x,
                               int64_t incx, const double *y, int64_t incy,
                               double *a, int64_t lda) {
    return cublasDsyr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, a,
                             lda);
  }
};

template <> struct Syr2<cuComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuComplex *alpha,
                               const cuComplex *x, int incx,
                               const cuComplex *y, int incy, cuComplex *a,
                               int lda) {
    return cublasCsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, a, lda);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const cuComplex *alpha,
                               const cuComplex *x, int64_t incx,
                               const cuComplex *y, int64_t incy, cuComplex *a,
                               int64_t lda) {
    return cublasCsyr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, a,
                             lda);
  }
};

template <> struct Syr2<cuDoubleComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuDoubleComplex *alpha,
                               const cuDoubleComplex *x, int incx,
                               const cuDoubleComplex *y, int incy,
                               cuDoubleComplex *a, int lda) {
    return cublasZsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, a, lda);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const cuDoubleComplex *alpha,
                               const cuDoubleComplex *x, int64_t incx,
                               const cuDoubleComplex *y, int64_t incy,
                               cuDoubleComplex *a, int64_t lda) {
    return cublasZsyr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, a,
                             lda);
  }
};

template <typename T> struct Her2;

template <> struct Her2<cuComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuComplex *alpha,
                               const cuComplex *x, int incx,
                               const cuComplex *y, int incy, cuComplex *a,
                               int lda) {
    return cublasCher2_v2(handle, uplo, n, alpha, x, incx, y, incy, a, lda);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const cuComplex *alpha,
                               const cuComplex *x, int64_t incx,
                               const cuComplex *y, int64_t incy, cuComplex *a,
                               int64_t lda) {
    return cublasCher2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, a,
                             lda);
  }
};

template <> struct Her2<cuDoubleComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuDoubleComplex *alpha,
                               const cuDoubleComplex *x, int incx,
                               const cuDoubleComplex *y, int incy,
                               cuDoubleComplex *a, int lda) {
    return cublasZher2_v2(handle, uplo, n, alpha, x, incx, y, incy, a, lda);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const cuDoubleComplex *alpha,
                               const cuDoubleComplex *x, int64_t incx,
                               const cuDoubleComplex *y, int64_t incy,
                               cuDoubleComplex *a, int64_t lda) {
    return cublasZher2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, a,
                             lda);
  }
};

template <typename T> struct Spr2;

template <> struct Spr2<float> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const float *alpha, const float *x,
                               int incx, const float *y, int incy, float *ap) {
    return cublasSspr2_v2(handle, uplo, n, alpha, x, incx, y, incy, ap);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const float *alpha, const float *x,
                               int64_t incx, const float *y, int64_t incy,
                               float *ap) {
    return cublasSspr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, ap);
  }
};

template <> struct Spr2<double> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const double *alpha, const double *x,
                               int incx, const double *y, int incy,
                               double *ap) {
    return cublasDspr2_v2(handle, uplo, n, alpha, x, incx, y, incy, ap);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const double *alpha, const double *x,
                               int64_t incx, const double *y, int64_t incy,
                               double *ap) {
    return cublasDspr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, ap);
  }
};

template <typename T> struct Hpr2;

template <> struct Hpr2<cuComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuComplex *alpha,
                               const cuComplex *x, int incx,
                               const cuComplex *y, int incy, cuComplex *ap) {
    return cublasChpr2_v2(handle, uplo, n, alpha, x, incx, y, incy, ap);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const cuComplex *alpha,
                               const cuComplex *x, int64_t incx,
                               const cuComplex *y, int64_t incy,
                               cuComplex *ap) {
    return cublasChpr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, ap);
  }
};

template <> struct Hpr2<cuDoubleComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuDoubleComplex *alpha,
                               const cuDoubleComplex *x, int incx,
                               const cuDoubleComplex *y, int incy,
                               cuDoubleComplex *ap) {
    return cublasZhpr2_v2(handle, uplo, n, alpha, x, incx, y, incy, ap);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const cuDoubleComplex *alpha,
                               const cuDoubleComplex *x, int64_t incx,
                               const cuDoubleComplex *y, int64_t incy,
                               cuDoubleComplex *ap) {
    return cublasZhpr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, ap);
  }
};

static bool in_triangle(cublasFillMode_t uplo, int row, int col) {
  return uplo == CUBLAS_FILL_MODE_UPPER ? row <= col : row >= col;
}

static int packed_index(cublasFillMode_t uplo, int n, int row, int col) {
  if (uplo == CUBLAS_FILL_MODE_UPPER) {
    return col * (col + 1) / 2 + row;
  }
  return col * n - col * (col - 1) / 2 + (row - col);
}

template <typename T>
static T matrix_value(int row, int col, bool hermitian) {
  float imag = hermitian && row == col ? 0.0f
                                       : static_cast<float>(row - col) * 0.125f;
  return Ops<T>::value(static_cast<float>(4 * col + row + 1), imag);
}

template <typename T>
static T vector_value(int i, int seed) {
  return Ops<T>::value(static_cast<float>(seed + i),
                       static_cast<float>(seed - 2 * i) * 0.125f);
}

template <typename T>
static void fill_dense(std::vector<T> &a, int n, int lda, bool hermitian) {
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < lda; ++row) {
      a[row + col * lda] =
          row < n ? matrix_value<T>(row, col, hermitian)
                  : Ops<T>::value(static_cast<float>(-200 - row - col), 0.0f);
    }
  }
}

template <typename T>
static void fill_packed(std::vector<T> &ap, cublasFillMode_t uplo, int n,
                        bool hermitian) {
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      if (in_triangle(uplo, row, col)) {
        ap[packed_index(uplo, n, row, col)] =
            matrix_value<T>(row, col, hermitian);
      }
    }
  }
}

template <typename T>
static void fill_vector(std::vector<T> &x, int n, int incx, int seed) {
  for (int i = 0; i < n; ++i) {
    x[i * incx] = vector_value<T>(i, seed);
  }
}

template <typename T>
static void expected_syr2_dense(std::vector<T> &a, const std::vector<T> &x,
                                const std::vector<T> &y, cublasFillMode_t uplo,
                                int n, int lda, int incx, int incy, T alpha) {
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      if (in_triangle(uplo, row, col)) {
        T left = Ops<T>::mul(x[row * incx], y[col * incy]);
        T right = Ops<T>::mul(y[row * incy], x[col * incx]);
        T update = Ops<T>::mul(alpha, Ops<T>::add(left, right));
        a[row + col * lda] = Ops<T>::add(a[row + col * lda], update);
      }
    }
  }
}

template <typename T>
static void expected_her2_dense(std::vector<T> &a, const std::vector<T> &x,
                                const std::vector<T> &y, cublasFillMode_t uplo,
                                int n, int lda, int incx, int incy, T alpha) {
  T alpha_conj = Ops<T>::conj(alpha);
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      if (in_triangle(uplo, row, col)) {
        T left = Ops<T>::mul(Ops<T>::mul(alpha, x[row * incx]),
                             Ops<T>::conj(y[col * incy]));
        T right = Ops<T>::mul(Ops<T>::mul(alpha_conj, y[row * incy]),
                              Ops<T>::conj(x[col * incx]));
        a[row + col * lda] =
            Ops<T>::add(a[row + col * lda], Ops<T>::add(left, right));
      }
    }
  }
}

template <typename T>
static void expected_spr2(std::vector<T> &ap, const std::vector<T> &x,
                          const std::vector<T> &y, cublasFillMode_t uplo,
                          int n, int incx, int incy, T alpha) {
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      if (in_triangle(uplo, row, col)) {
        int idx = packed_index(uplo, n, row, col);
        T left = Ops<T>::mul(x[row * incx], y[col * incy]);
        T right = Ops<T>::mul(y[row * incy], x[col * incx]);
        T update = Ops<T>::mul(alpha, Ops<T>::add(left, right));
        ap[idx] = Ops<T>::add(ap[idx], update);
      }
    }
  }
}

template <typename T>
static void expected_hpr2(std::vector<T> &ap, const std::vector<T> &x,
                          const std::vector<T> &y, cublasFillMode_t uplo,
                          int n, int incx, int incy, T alpha) {
  T alpha_conj = Ops<T>::conj(alpha);
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      if (in_triangle(uplo, row, col)) {
        int idx = packed_index(uplo, n, row, col);
        T left = Ops<T>::mul(Ops<T>::mul(alpha, x[row * incx]),
                             Ops<T>::conj(y[col * incy]));
        T right = Ops<T>::mul(Ops<T>::mul(alpha_conj, y[row * incy]),
                              Ops<T>::conj(x[col * incx]));
        ap[idx] = Ops<T>::add(ap[idx], Ops<T>::add(left, right));
      }
    }
  }
}

template <typename T>
static void check_dense(const char *name, const std::vector<T> &out,
                        const std::vector<T> &expected,
                        cublasFillMode_t uplo, int n, int lda, bool use_64,
                        bool device_scalar) {
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      if (in_triangle(uplo, row, col) &&
          !Ops<T>::near(out[row + col * lda], expected[row + col * lda])) {
        std::printf("%s mismatch row=%d col=%d uplo=%d use64=%d device=%d\n",
                    name, row, col, static_cast<int>(uplo), use_64 ? 1 : 0,
                    device_scalar ? 1 : 0);
        std::exit(EXIT_FAILURE);
      }
    }
  }
}

template <typename T>
static void check_packed(const char *name, const std::vector<T> &out,
                         const std::vector<T> &expected,
                         cublasFillMode_t uplo, int n, bool use_64,
                         bool device_scalar) {
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      if (in_triangle(uplo, row, col)) {
        int idx = packed_index(uplo, n, row, col);
        if (!Ops<T>::near(out[idx], expected[idx])) {
          std::printf("%s mismatch row=%d col=%d uplo=%d use64=%d device=%d\n",
                      name, row, col, static_cast<int>(uplo), use_64 ? 1 : 0,
                      device_scalar ? 1 : 0);
          std::exit(EXIT_FAILURE);
        }
      }
    }
  }
}

template <typename T>
static void run_syr2_case(cublasHandle_t handle, const char *name,
                          cublasFillMode_t uplo, bool use_64,
                          bool device_scalar) {
  const int n = 5;
  const int lda = 7;
  const int incx = 2;
  const int incy = 3;
  std::vector<T> a(lda * n);
  std::vector<T> x(1 + (n - 1) * incx);
  std::vector<T> y(1 + (n - 1) * incy);
  fill_dense(a, n, lda, false);
  fill_vector(x, n, incx, 3);
  fill_vector(y, n, incy, 7);
  T alpha = Ops<T>::alpha();
  std::vector<T> expected = a;
  expected_syr2_dense(expected, x, y, uplo, n, lda, incx, incy, alpha);

  T *d_a = nullptr;
  T *d_x = nullptr;
  T *d_y = nullptr;
  T *d_alpha = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, a.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_y, y.size() * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_a, a.data(), a.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_x, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_y, y.data(), y.size() * sizeof(T), cudaMemcpyHostToDevice));

  const T *alpha_arg = &alpha;
  if (device_scalar) {
    CHECK_CUDA(cudaMalloc(&d_alpha, sizeof(T)));
    CHECK_CUDA(cudaMemcpy(d_alpha, &alpha, sizeof(T), cudaMemcpyHostToDevice));
    alpha_arg = d_alpha;
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  } else {
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  }

  if (use_64) {
    CHECK_CUBLAS(Syr2<T>::call64(handle, uplo, n, alpha_arg, d_x, incx, d_y,
                                 incy, d_a, lda));
  } else {
    CHECK_CUBLAS(Syr2<T>::call32(handle, uplo, n, alpha_arg, d_x, incx, d_y,
                                 incy, d_a, lda));
  }
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<T> out(a.size());
  CHECK_CUDA(cudaMemcpy(out.data(), d_a, out.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
  check_dense(name, out, expected, uplo, n, lda, use_64, device_scalar);

  if (d_alpha != nullptr) {
    cudaFree(d_alpha);
  }
  cudaFree(d_y);
  cudaFree(d_x);
  cudaFree(d_a);
}

template <typename T>
static void run_her2_case(cublasHandle_t handle, const char *name,
                          cublasFillMode_t uplo, bool use_64,
                          bool device_scalar) {
  const int n = 5;
  const int lda = 7;
  const int incx = 2;
  const int incy = 3;
  std::vector<T> a(lda * n);
  std::vector<T> x(1 + (n - 1) * incx);
  std::vector<T> y(1 + (n - 1) * incy);
  fill_dense(a, n, lda, true);
  fill_vector(x, n, incx, 3);
  fill_vector(y, n, incy, 7);
  T alpha = Ops<T>::alpha();
  std::vector<T> expected = a;
  expected_her2_dense(expected, x, y, uplo, n, lda, incx, incy, alpha);

  T *d_a = nullptr;
  T *d_x = nullptr;
  T *d_y = nullptr;
  T *d_alpha = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, a.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_y, y.size() * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_a, a.data(), a.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_x, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_y, y.data(), y.size() * sizeof(T), cudaMemcpyHostToDevice));

  const T *alpha_arg = &alpha;
  if (device_scalar) {
    CHECK_CUDA(cudaMalloc(&d_alpha, sizeof(T)));
    CHECK_CUDA(cudaMemcpy(d_alpha, &alpha, sizeof(T), cudaMemcpyHostToDevice));
    alpha_arg = d_alpha;
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  } else {
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  }

  if (use_64) {
    CHECK_CUBLAS(Her2<T>::call64(handle, uplo, n, alpha_arg, d_x, incx, d_y,
                                 incy, d_a, lda));
  } else {
    CHECK_CUBLAS(Her2<T>::call32(handle, uplo, n, alpha_arg, d_x, incx, d_y,
                                 incy, d_a, lda));
  }
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<T> out(a.size());
  CHECK_CUDA(cudaMemcpy(out.data(), d_a, out.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
  check_dense(name, out, expected, uplo, n, lda, use_64, device_scalar);

  if (d_alpha != nullptr) {
    cudaFree(d_alpha);
  }
  cudaFree(d_y);
  cudaFree(d_x);
  cudaFree(d_a);
}

template <typename T>
static void run_spr2_case(cublasHandle_t handle, const char *name,
                          cublasFillMode_t uplo, bool use_64,
                          bool device_scalar) {
  const int n = 5;
  const int incx = 2;
  const int incy = 3;
  const int packed_len = n * (n + 1) / 2;
  std::vector<T> ap(packed_len);
  std::vector<T> x(1 + (n - 1) * incx);
  std::vector<T> y(1 + (n - 1) * incy);
  fill_packed(ap, uplo, n, false);
  fill_vector(x, n, incx, 3);
  fill_vector(y, n, incy, 7);
  T alpha = Ops<T>::alpha();
  std::vector<T> expected = ap;
  expected_spr2(expected, x, y, uplo, n, incx, incy, alpha);

  T *d_ap = nullptr;
  T *d_x = nullptr;
  T *d_y = nullptr;
  T *d_alpha = nullptr;
  CHECK_CUDA(cudaMalloc(&d_ap, ap.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_y, y.size() * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_ap, ap.data(), ap.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_x, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_y, y.data(), y.size() * sizeof(T), cudaMemcpyHostToDevice));

  const T *alpha_arg = &alpha;
  if (device_scalar) {
    CHECK_CUDA(cudaMalloc(&d_alpha, sizeof(T)));
    CHECK_CUDA(cudaMemcpy(d_alpha, &alpha, sizeof(T), cudaMemcpyHostToDevice));
    alpha_arg = d_alpha;
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  } else {
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  }

  if (use_64) {
    CHECK_CUBLAS(Spr2<T>::call64(handle, uplo, n, alpha_arg, d_x, incx, d_y,
                                 incy, d_ap));
  } else {
    CHECK_CUBLAS(Spr2<T>::call32(handle, uplo, n, alpha_arg, d_x, incx, d_y,
                                 incy, d_ap));
  }
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<T> out(ap.size());
  CHECK_CUDA(cudaMemcpy(out.data(), d_ap, out.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
  check_packed(name, out, expected, uplo, n, use_64, device_scalar);

  if (d_alpha != nullptr) {
    cudaFree(d_alpha);
  }
  cudaFree(d_y);
  cudaFree(d_x);
  cudaFree(d_ap);
}

template <typename T>
static void run_hpr2_case(cublasHandle_t handle, const char *name,
                          cublasFillMode_t uplo, bool use_64,
                          bool device_scalar) {
  const int n = 5;
  const int incx = 2;
  const int incy = 3;
  const int packed_len = n * (n + 1) / 2;
  std::vector<T> ap(packed_len);
  std::vector<T> x(1 + (n - 1) * incx);
  std::vector<T> y(1 + (n - 1) * incy);
  fill_packed(ap, uplo, n, true);
  fill_vector(x, n, incx, 3);
  fill_vector(y, n, incy, 7);
  T alpha = Ops<T>::alpha();
  std::vector<T> expected = ap;
  expected_hpr2(expected, x, y, uplo, n, incx, incy, alpha);

  T *d_ap = nullptr;
  T *d_x = nullptr;
  T *d_y = nullptr;
  T *d_alpha = nullptr;
  CHECK_CUDA(cudaMalloc(&d_ap, ap.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_y, y.size() * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_ap, ap.data(), ap.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_x, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_y, y.data(), y.size() * sizeof(T), cudaMemcpyHostToDevice));

  const T *alpha_arg = &alpha;
  if (device_scalar) {
    CHECK_CUDA(cudaMalloc(&d_alpha, sizeof(T)));
    CHECK_CUDA(cudaMemcpy(d_alpha, &alpha, sizeof(T), cudaMemcpyHostToDevice));
    alpha_arg = d_alpha;
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  } else {
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  }

  if (use_64) {
    CHECK_CUBLAS(Hpr2<T>::call64(handle, uplo, n, alpha_arg, d_x, incx, d_y,
                                 incy, d_ap));
  } else {
    CHECK_CUBLAS(Hpr2<T>::call32(handle, uplo, n, alpha_arg, d_x, incx, d_y,
                                 incy, d_ap));
  }
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<T> out(ap.size());
  CHECK_CUDA(cudaMemcpy(out.data(), d_ap, out.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
  check_packed(name, out, expected, uplo, n, use_64, device_scalar);

  if (d_alpha != nullptr) {
    cudaFree(d_alpha);
  }
  cudaFree(d_y);
  cudaFree(d_x);
  cudaFree(d_ap);
}

template <typename T>
static void run_syr2_type(cublasHandle_t handle, const char *name) {
  cublasFillMode_t uplos[] = {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER};
  for (int u = 0; u < 2; ++u) {
    for (int use_64 = 0; use_64 < 2; ++use_64) {
      for (int device_scalar = 0; device_scalar < 2; ++device_scalar) {
        run_syr2_case<T>(handle, name, uplos[u], use_64 != 0,
                         device_scalar != 0);
      }
    }
  }
}

template <typename T>
static void run_her2_type(cublasHandle_t handle, const char *name) {
  cublasFillMode_t uplos[] = {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER};
  for (int u = 0; u < 2; ++u) {
    for (int use_64 = 0; use_64 < 2; ++use_64) {
      for (int device_scalar = 0; device_scalar < 2; ++device_scalar) {
        run_her2_case<T>(handle, name, uplos[u], use_64 != 0,
                         device_scalar != 0);
      }
    }
  }
}

template <typename T>
static void run_spr2_type(cublasHandle_t handle, const char *name) {
  cublasFillMode_t uplos[] = {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER};
  for (int u = 0; u < 2; ++u) {
    for (int use_64 = 0; use_64 < 2; ++use_64) {
      for (int device_scalar = 0; device_scalar < 2; ++device_scalar) {
        run_spr2_case<T>(handle, name, uplos[u], use_64 != 0,
                         device_scalar != 0);
      }
    }
  }
}

template <typename T>
static void run_hpr2_type(cublasHandle_t handle, const char *name) {
  cublasFillMode_t uplos[] = {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER};
  for (int u = 0; u < 2; ++u) {
    for (int use_64 = 0; use_64 < 2; ++use_64) {
      for (int device_scalar = 0; device_scalar < 2; ++device_scalar) {
        run_hpr2_case<T>(handle, name, uplos[u], use_64 != 0,
                         device_scalar != 0);
      }
    }
  }
}

int main() {
  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));

  run_syr2_type<float>(handle, "ssyr2");
  run_syr2_type<double>(handle, "dsyr2");
  run_syr2_type<cuComplex>(handle, "csyr2");
  run_syr2_type<cuDoubleComplex>(handle, "zsyr2");
  run_her2_type<cuComplex>(handle, "cher2");
  run_her2_type<cuDoubleComplex>(handle, "zher2");
  run_spr2_type<float>(handle, "sspr2");
  run_spr2_type<double>(handle, "dspr2");
  run_hpr2_type<cuComplex>(handle, "chpr2");
  run_hpr2_type<cuDoubleComplex>(handle, "zhpr2");

  CHECK_CUBLAS(cublasDestroy(handle));
  return EXIT_SUCCESS;
}
