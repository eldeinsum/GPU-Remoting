#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cmath>
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
  static bool near(float a, float b) { return std::fabs(a - b) < 1.0e-4f; }
};

template <> struct Ops<double> {
  static double value(float real, float) { return static_cast<double>(real); }
  static bool near(double a, double b) { return std::fabs(a - b) < 1.0e-9; }
};

template <> struct Ops<cuComplex> {
  static cuComplex value(float real, float imag) {
    return make_cuComplex(real, imag);
  }
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
  static bool near(cuDoubleComplex a, cuDoubleComplex b) {
    return std::fabs(a.x - b.x) < 1.0e-9 &&
           std::fabs(a.y - b.y) < 1.0e-9;
  }
};

template <typename T> struct TriPack;

template <> struct TriPack<float> {
  static cublasStatus_t trttp(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const float *a, int lda, float *ap) {
    return cublasStrttp(handle, uplo, n, a, lda, ap);
  }
  static cublasStatus_t tpttr(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const float *ap, float *a, int lda) {
    return cublasStpttr(handle, uplo, n, ap, a, lda);
  }
};

template <> struct TriPack<double> {
  static cublasStatus_t trttp(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const double *a, int lda, double *ap) {
    return cublasDtrttp(handle, uplo, n, a, lda, ap);
  }
  static cublasStatus_t tpttr(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const double *ap, double *a, int lda) {
    return cublasDtpttr(handle, uplo, n, ap, a, lda);
  }
};

template <> struct TriPack<cuComplex> {
  static cublasStatus_t trttp(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const cuComplex *a, int lda,
                              cuComplex *ap) {
    return cublasCtrttp(handle, uplo, n, a, lda, ap);
  }
  static cublasStatus_t tpttr(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const cuComplex *ap, cuComplex *a,
                              int lda) {
    return cublasCtpttr(handle, uplo, n, ap, a, lda);
  }
};

template <> struct TriPack<cuDoubleComplex> {
  static cublasStatus_t trttp(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const cuDoubleComplex *a, int lda,
                              cuDoubleComplex *ap) {
    return cublasZtrttp(handle, uplo, n, a, lda, ap);
  }
  static cublasStatus_t tpttr(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const cuDoubleComplex *ap,
                              cuDoubleComplex *a, int lda) {
    return cublasZtpttr(handle, uplo, n, ap, a, lda);
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
static std::vector<T> expected_packed(const std::vector<T> &a,
                                      cublasFillMode_t uplo, int n, int lda) {
  std::vector<T> ap(n * (n + 1) / 2);
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      if (in_triangle(uplo, row, col)) {
        ap[packed_index(uplo, n, row, col)] = a[row + col * lda];
      }
    }
  }
  return ap;
}

template <typename T>
static void run_case(cublasHandle_t handle, const char *name,
                     cublasFillMode_t uplo) {
  const int n = 4;
  const int lda = 6;
  const int packed_len = n * (n + 1) / 2;

  std::vector<T> a(lda * n);
  std::vector<T> unpacked(lda * n);
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < lda; ++row) {
      a[row + col * lda] =
          Ops<T>::value(static_cast<float>(10 * col + row + 1),
                        static_cast<float>((row + col) % 3) * 0.125f);
      unpacked[row + col * lda] = Ops<T>::value(-77.0f, -3.0f);
    }
  }

  std::vector<T> expected = expected_packed(a, uplo, n, lda);

  T *d_a = nullptr;
  T *d_ap = nullptr;
  T *d_unpacked = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, a.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_ap, packed_len * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_unpacked, unpacked.size() * sizeof(T)));
  CHECK_CUDA(
      cudaMemcpy(d_a, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_unpacked, unpacked.data(),
                        unpacked.size() * sizeof(T),
                        cudaMemcpyHostToDevice));

  CHECK_CUBLAS(TriPack<T>::trttp(handle, uplo, n, d_a, lda, d_ap));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<T> packed(packed_len);
  CHECK_CUDA(cudaMemcpy(packed.data(), d_ap, packed.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
  for (int i = 0; i < packed_len; ++i) {
    if (!Ops<T>::near(packed[i], expected[i])) {
      std::printf("%s trttp mismatch index %d uplo=%d\n", name, i,
                  static_cast<int>(uplo));
      std::exit(EXIT_FAILURE);
    }
  }

  CHECK_CUBLAS(TriPack<T>::tpttr(handle, uplo, n, d_ap, d_unpacked, lda));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<T> out(unpacked.size());
  CHECK_CUDA(cudaMemcpy(out.data(), d_unpacked, out.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      if (!in_triangle(uplo, row, col)) {
        continue;
      }
      const int idx = row + col * lda;
      if (!Ops<T>::near(out[idx], a[idx])) {
        std::printf("%s tpttr mismatch row=%d col=%d uplo=%d\n", name, row,
                    col, static_cast<int>(uplo));
        std::exit(EXIT_FAILURE);
      }
    }
  }

  cudaFree(d_unpacked);
  cudaFree(d_ap);
  cudaFree(d_a);
}

template <typename T>
static void run_type(cublasHandle_t handle, const char *name) {
  run_case<T>(handle, name, CUBLAS_FILL_MODE_UPPER);
  run_case<T>(handle, name, CUBLAS_FILL_MODE_LOWER);
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
