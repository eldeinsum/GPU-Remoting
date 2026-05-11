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
  typedef float Real;
  static float value(float real, float) { return real; }
  static float add(float a, float b) { return a + b; }
  static float mul(float a, float b) { return a * b; }
  static float conj(float a) { return a; }
  static float scale_real(float a, float r) { return a * r; }
  static float alpha() { return 0.75f; }
  static float herm_alpha() { return 0.75f; }
  static bool near(float a, float b) { return std::fabs(a - b) < 1.0e-4f; }
};

template <> struct Ops<double> {
  typedef double Real;
  static double value(float real, float) { return static_cast<double>(real); }
  static double add(double a, double b) { return a + b; }
  static double mul(double a, double b) { return a * b; }
  static double conj(double a) { return a; }
  static double scale_real(double a, double r) { return a * r; }
  static double alpha() { return 0.75; }
  static double herm_alpha() { return 0.75; }
  static bool near(double a, double b) { return std::fabs(a - b) < 1.0e-9; }
};

template <> struct Ops<cuComplex> {
  typedef float Real;
  static cuComplex value(float real, float imag) {
    return make_cuComplex(real, imag);
  }
  static cuComplex add(cuComplex a, cuComplex b) { return cuCaddf(a, b); }
  static cuComplex mul(cuComplex a, cuComplex b) { return cuCmulf(a, b); }
  static cuComplex conj(cuComplex a) { return cuConjf(a); }
  static cuComplex scale_real(cuComplex a, float r) {
    return make_cuComplex(a.x * r, a.y * r);
  }
  static cuComplex alpha() { return make_cuComplex(0.625f, -0.25f); }
  static float herm_alpha() { return 0.625f; }
  static bool near(cuComplex a, cuComplex b) {
    return std::fabs(a.x - b.x) < 1.0e-3f &&
           std::fabs(a.y - b.y) < 1.0e-3f;
  }
};

template <> struct Ops<cuDoubleComplex> {
  typedef double Real;
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
  static cuDoubleComplex scale_real(cuDoubleComplex a, double r) {
    return make_cuDoubleComplex(a.x * r, a.y * r);
  }
  static cuDoubleComplex alpha() { return make_cuDoubleComplex(0.625, -0.25); }
  static double herm_alpha() { return 0.625; }
  static bool near(cuDoubleComplex a, cuDoubleComplex b) {
    return std::fabs(a.x - b.x) < 1.0e-9 &&
           std::fabs(a.y - b.y) < 1.0e-9;
  }
};

template <typename T> struct Syr;

template <> struct Syr<float> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const float *alpha, const float *x,
                               int incx, float *a, int lda) {
    return cublasSsyr_v2(handle, uplo, n, alpha, x, incx, a, lda);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const float *alpha, const float *x,
                               int64_t incx, float *a, int64_t lda) {
    return cublasSsyr_v2_64(handle, uplo, n, alpha, x, incx, a, lda);
  }
};

template <> struct Syr<double> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const double *alpha, const double *x,
                               int incx, double *a, int lda) {
    return cublasDsyr_v2(handle, uplo, n, alpha, x, incx, a, lda);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const double *alpha, const double *x,
                               int64_t incx, double *a, int64_t lda) {
    return cublasDsyr_v2_64(handle, uplo, n, alpha, x, incx, a, lda);
  }
};

template <> struct Syr<cuComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuComplex *alpha,
                               const cuComplex *x, int incx, cuComplex *a,
                               int lda) {
    return cublasCsyr_v2(handle, uplo, n, alpha, x, incx, a, lda);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const cuComplex *alpha,
                               const cuComplex *x, int64_t incx, cuComplex *a,
                               int64_t lda) {
    return cublasCsyr_v2_64(handle, uplo, n, alpha, x, incx, a, lda);
  }
};

template <> struct Syr<cuDoubleComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuDoubleComplex *alpha,
                               const cuDoubleComplex *x, int incx,
                               cuDoubleComplex *a, int lda) {
    return cublasZsyr_v2(handle, uplo, n, alpha, x, incx, a, lda);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const cuDoubleComplex *alpha,
                               const cuDoubleComplex *x, int64_t incx,
                               cuDoubleComplex *a, int64_t lda) {
    return cublasZsyr_v2_64(handle, uplo, n, alpha, x, incx, a, lda);
  }
};

template <typename T> struct Her;

template <> struct Her<cuComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const float *alpha, const cuComplex *x,
                               int incx, cuComplex *a, int lda) {
    return cublasCher_v2(handle, uplo, n, alpha, x, incx, a, lda);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const float *alpha,
                               const cuComplex *x, int64_t incx, cuComplex *a,
                               int64_t lda) {
    return cublasCher_v2_64(handle, uplo, n, alpha, x, incx, a, lda);
  }
};

template <> struct Her<cuDoubleComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const double *alpha,
                               const cuDoubleComplex *x, int incx,
                               cuDoubleComplex *a, int lda) {
    return cublasZher_v2(handle, uplo, n, alpha, x, incx, a, lda);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const double *alpha,
                               const cuDoubleComplex *x, int64_t incx,
                               cuDoubleComplex *a, int64_t lda) {
    return cublasZher_v2_64(handle, uplo, n, alpha, x, incx, a, lda);
  }
};

template <typename T> struct Spr;

template <> struct Spr<float> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const float *alpha, const float *x,
                               int incx, float *ap) {
    return cublasSspr_v2(handle, uplo, n, alpha, x, incx, ap);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const float *alpha, const float *x,
                               int64_t incx, float *ap) {
    return cublasSspr_v2_64(handle, uplo, n, alpha, x, incx, ap);
  }
};

template <> struct Spr<double> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const double *alpha, const double *x,
                               int incx, double *ap) {
    return cublasDspr_v2(handle, uplo, n, alpha, x, incx, ap);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const double *alpha, const double *x,
                               int64_t incx, double *ap) {
    return cublasDspr_v2_64(handle, uplo, n, alpha, x, incx, ap);
  }
};

template <typename T> struct Hpr;

template <> struct Hpr<cuComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const float *alpha, const cuComplex *x,
                               int incx, cuComplex *ap) {
    return cublasChpr_v2(handle, uplo, n, alpha, x, incx, ap);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const float *alpha,
                               const cuComplex *x, int64_t incx,
                               cuComplex *ap) {
    return cublasChpr_v2_64(handle, uplo, n, alpha, x, incx, ap);
  }
};

template <> struct Hpr<cuDoubleComplex> {
  static cublasStatus_t call32(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const double *alpha,
                               const cuDoubleComplex *x, int incx,
                               cuDoubleComplex *ap) {
    return cublasZhpr_v2(handle, uplo, n, alpha, x, incx, ap);
  }
  static cublasStatus_t call64(cublasHandle_t handle, cublasFillMode_t uplo,
                               int64_t n, const double *alpha,
                               const cuDoubleComplex *x, int64_t incx,
                               cuDoubleComplex *ap) {
    return cublasZhpr_v2_64(handle, uplo, n, alpha, x, incx, ap);
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
  return Ops<T>::value(static_cast<float>(3 * col + row + 1), imag);
}

template <typename T>
static T vector_value(int i) {
  return Ops<T>::value(static_cast<float>(i + 2),
                       static_cast<float>(i - 1) * 0.25f);
}

template <typename T>
static void fill_dense(std::vector<T> &a, int n, int lda, bool hermitian) {
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < lda; ++row) {
      a[row + col * lda] =
          row < n ? matrix_value<T>(row, col, hermitian)
                  : Ops<T>::value(static_cast<float>(-100 - row - col), 0.0f);
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
static void fill_vector(std::vector<T> &x, int n, int incx) {
  for (int i = 0; i < n; ++i) {
    x[i * incx] = vector_value<T>(i);
  }
}

template <typename T>
static void expected_syr_dense(std::vector<T> &a, const std::vector<T> &x,
                               cublasFillMode_t uplo, int n, int lda,
                               int incx, T alpha) {
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      if (in_triangle(uplo, row, col)) {
        T update = Ops<T>::mul(alpha, Ops<T>::mul(x[row * incx], x[col * incx]));
        a[row + col * lda] = Ops<T>::add(a[row + col * lda], update);
      }
    }
  }
}

template <typename T>
static void expected_her_dense(std::vector<T> &a, const std::vector<T> &x,
                               cublasFillMode_t uplo, int n, int lda,
                               int incx, typename Ops<T>::Real alpha) {
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      if (in_triangle(uplo, row, col)) {
        T product = Ops<T>::mul(x[row * incx], Ops<T>::conj(x[col * incx]));
        T update = Ops<T>::scale_real(product, alpha);
        a[row + col * lda] = Ops<T>::add(a[row + col * lda], update);
      }
    }
  }
}

template <typename T>
static void expected_spr(std::vector<T> &ap, const std::vector<T> &x,
                         cublasFillMode_t uplo, int n, int incx, T alpha) {
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      if (in_triangle(uplo, row, col)) {
        int idx = packed_index(uplo, n, row, col);
        T update = Ops<T>::mul(alpha, Ops<T>::mul(x[row * incx], x[col * incx]));
        ap[idx] = Ops<T>::add(ap[idx], update);
      }
    }
  }
}

template <typename T>
static void expected_hpr(std::vector<T> &ap, const std::vector<T> &x,
                         cublasFillMode_t uplo, int n, int incx,
                         typename Ops<T>::Real alpha) {
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      if (in_triangle(uplo, row, col)) {
        int idx = packed_index(uplo, n, row, col);
        T product = Ops<T>::mul(x[row * incx], Ops<T>::conj(x[col * incx]));
        T update = Ops<T>::scale_real(product, alpha);
        ap[idx] = Ops<T>::add(ap[idx], update);
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
static void run_syr_case(cublasHandle_t handle, const char *name,
                         cublasFillMode_t uplo, bool use_64,
                         bool device_scalar) {
  const int n = 5;
  const int lda = 7;
  const int incx = 2;
  std::vector<T> a(lda * n);
  std::vector<T> x(1 + (n - 1) * incx);
  fill_dense(a, n, lda, false);
  fill_vector(x, n, incx);
  T alpha = Ops<T>::alpha();
  std::vector<T> expected = a;
  expected_syr_dense(expected, x, uplo, n, lda, incx, alpha);

  T *d_a = nullptr;
  T *d_x = nullptr;
  T *d_alpha = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, a.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_a, a.data(), a.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_x, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));

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
    CHECK_CUBLAS(Syr<T>::call64(handle, uplo, n, alpha_arg, d_x, incx, d_a,
                                lda));
  } else {
    CHECK_CUBLAS(Syr<T>::call32(handle, uplo, n, alpha_arg, d_x, incx, d_a,
                                lda));
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
  cudaFree(d_x);
  cudaFree(d_a);
}

template <typename T>
static void run_her_case(cublasHandle_t handle, const char *name,
                         cublasFillMode_t uplo, bool use_64,
                         bool device_scalar) {
  typedef typename Ops<T>::Real Real;
  const int n = 5;
  const int lda = 7;
  const int incx = 2;
  std::vector<T> a(lda * n);
  std::vector<T> x(1 + (n - 1) * incx);
  fill_dense(a, n, lda, true);
  fill_vector(x, n, incx);
  Real alpha = Ops<T>::herm_alpha();
  std::vector<T> expected = a;
  expected_her_dense(expected, x, uplo, n, lda, incx, alpha);

  T *d_a = nullptr;
  T *d_x = nullptr;
  Real *d_alpha = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, a.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_a, a.data(), a.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_x, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));

  const Real *alpha_arg = &alpha;
  if (device_scalar) {
    CHECK_CUDA(cudaMalloc(&d_alpha, sizeof(Real)));
    CHECK_CUDA(
        cudaMemcpy(d_alpha, &alpha, sizeof(Real), cudaMemcpyHostToDevice));
    alpha_arg = d_alpha;
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  } else {
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  }

  if (use_64) {
    CHECK_CUBLAS(Her<T>::call64(handle, uplo, n, alpha_arg, d_x, incx, d_a,
                                lda));
  } else {
    CHECK_CUBLAS(Her<T>::call32(handle, uplo, n, alpha_arg, d_x, incx, d_a,
                                lda));
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
  cudaFree(d_x);
  cudaFree(d_a);
}

template <typename T>
static void run_spr_case(cublasHandle_t handle, const char *name,
                         cublasFillMode_t uplo, bool use_64,
                         bool device_scalar) {
  const int n = 5;
  const int incx = 2;
  const int packed_len = n * (n + 1) / 2;
  std::vector<T> ap(packed_len);
  std::vector<T> x(1 + (n - 1) * incx);
  fill_packed(ap, uplo, n, false);
  fill_vector(x, n, incx);
  T alpha = Ops<T>::alpha();
  std::vector<T> expected = ap;
  expected_spr(expected, x, uplo, n, incx, alpha);

  T *d_ap = nullptr;
  T *d_x = nullptr;
  T *d_alpha = nullptr;
  CHECK_CUDA(cudaMalloc(&d_ap, ap.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_ap, ap.data(), ap.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_x, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));

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
    CHECK_CUBLAS(Spr<T>::call64(handle, uplo, n, alpha_arg, d_x, incx, d_ap));
  } else {
    CHECK_CUBLAS(Spr<T>::call32(handle, uplo, n, alpha_arg, d_x, incx, d_ap));
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
  cudaFree(d_x);
  cudaFree(d_ap);
}

template <typename T>
static void run_hpr_case(cublasHandle_t handle, const char *name,
                         cublasFillMode_t uplo, bool use_64,
                         bool device_scalar) {
  typedef typename Ops<T>::Real Real;
  const int n = 5;
  const int incx = 2;
  const int packed_len = n * (n + 1) / 2;
  std::vector<T> ap(packed_len);
  std::vector<T> x(1 + (n - 1) * incx);
  fill_packed(ap, uplo, n, true);
  fill_vector(x, n, incx);
  Real alpha = Ops<T>::herm_alpha();
  std::vector<T> expected = ap;
  expected_hpr(expected, x, uplo, n, incx, alpha);

  T *d_ap = nullptr;
  T *d_x = nullptr;
  Real *d_alpha = nullptr;
  CHECK_CUDA(cudaMalloc(&d_ap, ap.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_ap, ap.data(), ap.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_x, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));

  const Real *alpha_arg = &alpha;
  if (device_scalar) {
    CHECK_CUDA(cudaMalloc(&d_alpha, sizeof(Real)));
    CHECK_CUDA(
        cudaMemcpy(d_alpha, &alpha, sizeof(Real), cudaMemcpyHostToDevice));
    alpha_arg = d_alpha;
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  } else {
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  }

  if (use_64) {
    CHECK_CUBLAS(Hpr<T>::call64(handle, uplo, n, alpha_arg, d_x, incx, d_ap));
  } else {
    CHECK_CUBLAS(Hpr<T>::call32(handle, uplo, n, alpha_arg, d_x, incx, d_ap));
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
  cudaFree(d_x);
  cudaFree(d_ap);
}

template <typename T>
static void run_syr_type(cublasHandle_t handle, const char *name) {
  cublasFillMode_t uplos[] = {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER};
  for (int u = 0; u < 2; ++u) {
    for (int use_64 = 0; use_64 < 2; ++use_64) {
      for (int device_scalar = 0; device_scalar < 2; ++device_scalar) {
        run_syr_case<T>(handle, name, uplos[u], use_64 != 0,
                        device_scalar != 0);
      }
    }
  }
}

template <typename T>
static void run_her_type(cublasHandle_t handle, const char *name) {
  cublasFillMode_t uplos[] = {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER};
  for (int u = 0; u < 2; ++u) {
    for (int use_64 = 0; use_64 < 2; ++use_64) {
      for (int device_scalar = 0; device_scalar < 2; ++device_scalar) {
        run_her_case<T>(handle, name, uplos[u], use_64 != 0,
                        device_scalar != 0);
      }
    }
  }
}

template <typename T>
static void run_spr_type(cublasHandle_t handle, const char *name) {
  cublasFillMode_t uplos[] = {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER};
  for (int u = 0; u < 2; ++u) {
    for (int use_64 = 0; use_64 < 2; ++use_64) {
      for (int device_scalar = 0; device_scalar < 2; ++device_scalar) {
        run_spr_case<T>(handle, name, uplos[u], use_64 != 0,
                        device_scalar != 0);
      }
    }
  }
}

template <typename T>
static void run_hpr_type(cublasHandle_t handle, const char *name) {
  cublasFillMode_t uplos[] = {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER};
  for (int u = 0; u < 2; ++u) {
    for (int use_64 = 0; use_64 < 2; ++use_64) {
      for (int device_scalar = 0; device_scalar < 2; ++device_scalar) {
        run_hpr_case<T>(handle, name, uplos[u], use_64 != 0,
                        device_scalar != 0);
      }
    }
  }
}

int main() {
  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));

  run_syr_type<float>(handle, "ssyr");
  run_syr_type<double>(handle, "dsyr");
  run_syr_type<cuComplex>(handle, "csyr");
  run_syr_type<cuDoubleComplex>(handle, "zsyr");
  run_her_type<cuComplex>(handle, "cher");
  run_her_type<cuDoubleComplex>(handle, "zher");
  run_spr_type<float>(handle, "sspr");
  run_spr_type<double>(handle, "dspr");
  run_hpr_type<cuComplex>(handle, "chpr");
  run_hpr_type<cuDoubleComplex>(handle, "zhpr");

  CHECK_CUBLAS(cublasDestroy(handle));
  return EXIT_SUCCESS;
}
