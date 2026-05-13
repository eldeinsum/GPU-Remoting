#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::printf("CUDA failure %s:%d: %s\n", __FILE__, __LINE__,             \
                  cudaGetErrorString(err__));                                  \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CHECK_CUBLAS(call)                                                     \
  do {                                                                         \
    cublasStatus_t st__ = (call);                                              \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::printf("cuBLAS failure %s:%d: %d\n", __FILE__, __LINE__,           \
                  static_cast<int>(st__));                                     \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

static cuComplex add(cuComplex a, cuComplex b) { return cuCaddf(a, b); }
static cuComplex mul(cuComplex a, cuComplex b) { return cuCmulf(a, b); }
static cuComplex conjv(cuComplex a) { return cuConjf(a); }
static cuComplex scale(cuComplex a, cuComplex x) { return cuCmulf(a, x); }
static cuComplex scale(float a, cuComplex x) {
  return make_cuComplex(a * x.x, a * x.y);
}

static bool near(cuComplex a, cuComplex b, float eps) {
  return std::fabs(a.x - b.x) < eps && std::fabs(a.y - b.y) < eps;
}

static cuComplex a_value(int row, int col) {
  return make_cuComplex(static_cast<float>(2 * col + row + 1),
                        static_cast<float>(row - col) * 0.125f);
}

static cuComplex c_value(int row, int col, bool hermitian) {
  float imag = hermitian && row == col ? 0.0f
                                       : static_cast<float>(col - row) * 0.0625f;
  return make_cuComplex(static_cast<float>(3 * col - row + 2), imag);
}

template <typename Api>
static void expected_rankk(std::vector<cuComplex> &c,
                           const std::vector<cuComplex> &a, int n, int k,
                           int lda, int ldc, typename Api::Scalar alpha,
                           typename Api::Scalar beta) {
  std::vector<cuComplex> base = c;
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row <= col; ++row) {
      cuComplex sum = make_cuComplex(0.0f, 0.0f);
      for (int inner = 0; inner < k; ++inner) {
        cuComplex left = a[row + inner * lda];
        cuComplex right = a[col + inner * lda];
        if (Api::hermitian) {
          right = conjv(right);
        }
        sum = add(sum, mul(left, right));
      }
      c[row + col * ldc] =
          add(scale(alpha, sum), scale(beta, base[row + col * ldc]));
      if (Api::hermitian && row == col) {
        c[row + col * ldc].y = 0.0f;
      }
    }
  }
}

struct CsyrkExApi {
  typedef cuComplex Scalar;
  static const bool hermitian = false;
  static float eps() { return 1.0e-3f; }
  static Scalar alpha() { return make_cuComplex(-0.625f, 0.25f); }
  static Scalar beta() { return make_cuComplex(0.5f, -0.125f); }
  static cublasStatus_t call32(cublasHandle_t h, int n, int k,
                               const Scalar *alpha, const cuComplex *a,
                               int lda, const Scalar *beta, cuComplex *c,
                               int ldc) {
    return cublasCsyrkEx(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, alpha,
                         a, CUDA_C_32F, lda, beta, c, CUDA_C_32F, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t h, int64_t n, int64_t k,
                               const Scalar *alpha, const cuComplex *a,
                               int64_t lda, const Scalar *beta, cuComplex *c,
                               int64_t ldc) {
    return cublasCsyrkEx_64(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k,
                            alpha, a, CUDA_C_32F, lda, beta, c, CUDA_C_32F,
                            ldc);
  }
};

struct Csyrk3mExApi : CsyrkExApi {
  static float eps() { return 1.0e-2f; }
  static cublasStatus_t call32(cublasHandle_t h, int n, int k,
                               const Scalar *alpha, const cuComplex *a,
                               int lda, const Scalar *beta, cuComplex *c,
                               int ldc) {
    return cublasCsyrk3mEx(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k,
                           alpha, a, CUDA_C_32F, lda, beta, c, CUDA_C_32F,
                           ldc);
  }
  static cublasStatus_t call64(cublasHandle_t h, int64_t n, int64_t k,
                               const Scalar *alpha, const cuComplex *a,
                               int64_t lda, const Scalar *beta, cuComplex *c,
                               int64_t ldc) {
    return cublasCsyrk3mEx_64(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k,
                              alpha, a, CUDA_C_32F, lda, beta, c, CUDA_C_32F,
                              ldc);
  }
};

struct CherkExApi {
  typedef float Scalar;
  static const bool hermitian = true;
  static float eps() { return 1.0e-3f; }
  static Scalar alpha() { return -0.75f; }
  static Scalar beta() { return 0.5f; }
  static cublasStatus_t call32(cublasHandle_t h, int n, int k,
                               const Scalar *alpha, const cuComplex *a,
                               int lda, const Scalar *beta, cuComplex *c,
                               int ldc) {
    return cublasCherkEx(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, alpha,
                         a, CUDA_C_32F, lda, beta, c, CUDA_C_32F, ldc);
  }
  static cublasStatus_t call64(cublasHandle_t h, int64_t n, int64_t k,
                               const Scalar *alpha, const cuComplex *a,
                               int64_t lda, const Scalar *beta, cuComplex *c,
                               int64_t ldc) {
    return cublasCherkEx_64(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k,
                            alpha, a, CUDA_C_32F, lda, beta, c, CUDA_C_32F,
                            ldc);
  }
};

struct Cherk3mExApi : CherkExApi {
  static float eps() { return 1.0e-2f; }
  static cublasStatus_t call32(cublasHandle_t h, int n, int k,
                               const Scalar *alpha, const cuComplex *a,
                               int lda, const Scalar *beta, cuComplex *c,
                               int ldc) {
    return cublasCherk3mEx(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k,
                           alpha, a, CUDA_C_32F, lda, beta, c, CUDA_C_32F,
                           ldc);
  }
  static cublasStatus_t call64(cublasHandle_t h, int64_t n, int64_t k,
                               const Scalar *alpha, const cuComplex *a,
                               int64_t lda, const Scalar *beta, cuComplex *c,
                               int64_t ldc) {
    return cublasCherk3mEx_64(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k,
                              alpha, a, CUDA_C_32F, lda, beta, c, CUDA_C_32F,
                              ldc);
  }
};

template <typename Api>
static void run_case(cublasHandle_t handle, const char *name, bool use_64,
                     bool device_scalars) {
  const int n = 3;
  const int k = 2;
  const int lda = n + 1;
  const int ldc = n + 1;
  std::vector<cuComplex> a(lda * k);
  std::vector<cuComplex> c(ldc * n);
  for (int col = 0; col < k; ++col) {
    for (int row = 0; row < lda; ++row) {
      a[row + col * lda] =
          row < n ? a_value(row, col) : make_cuComplex(-100.0f, 0.0f);
    }
  }
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < ldc; ++row) {
      c[row + col * ldc] =
          row < n ? c_value(row, col, Api::hermitian)
                  : make_cuComplex(-200.0f, 0.0f);
    }
  }
  std::vector<cuComplex> expected = c;
  typename Api::Scalar alpha = Api::alpha();
  typename Api::Scalar beta = Api::beta();
  expected_rankk<Api>(expected, a, n, k, lda, ldc, alpha, beta);

  cuComplex *d_a = nullptr;
  cuComplex *d_c = nullptr;
  typename Api::Scalar *d_alpha = nullptr;
  typename Api::Scalar *d_beta = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, a.size() * sizeof(cuComplex)));
  CHECK_CUDA(cudaMalloc(&d_c, c.size() * sizeof(cuComplex)));
  CHECK_CUDA(
      cudaMemcpy(d_a, a.data(), a.size() * sizeof(cuComplex), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_c, c.data(), c.size() * sizeof(cuComplex), cudaMemcpyHostToDevice));

  const typename Api::Scalar *alpha_arg = &alpha;
  const typename Api::Scalar *beta_arg = &beta;
  if (device_scalars) {
    CHECK_CUDA(cudaMalloc(&d_alpha, sizeof(typename Api::Scalar)));
    CHECK_CUDA(cudaMalloc(&d_beta, sizeof(typename Api::Scalar)));
    CHECK_CUDA(cudaMemcpy(d_alpha, &alpha, sizeof(typename Api::Scalar),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_beta, &beta, sizeof(typename Api::Scalar),
                          cudaMemcpyHostToDevice));
    alpha_arg = d_alpha;
    beta_arg = d_beta;
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  } else {
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  }

  if (use_64) {
    CHECK_CUBLAS(Api::call64(handle, n, k, alpha_arg, d_a, lda, beta_arg, d_c, ldc));
  } else {
    CHECK_CUBLAS(Api::call32(handle, n, k, alpha_arg, d_a, lda, beta_arg, d_c, ldc));
  }
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<cuComplex> out(c.size());
  CHECK_CUDA(cudaMemcpy(out.data(), d_c, out.size() * sizeof(cuComplex),
                        cudaMemcpyDeviceToHost));
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row <= col; ++row) {
      if (!near(out[row + col * ldc], expected[row + col * ldc], Api::eps())) {
        std::printf("%s mismatch row=%d col=%d use64=%d device=%d\n", name,
                    row, col, use_64 ? 1 : 0, device_scalars ? 1 : 0);
        std::exit(EXIT_FAILURE);
      }
    }
  }

  if (d_beta != nullptr) cudaFree(d_beta);
  if (d_alpha != nullptr) cudaFree(d_alpha);
  cudaFree(d_c);
  cudaFree(d_a);
}

template <typename Api>
static void run_all(cublasHandle_t handle, const char *name) {
  run_case<Api>(handle, name, false, false);
  run_case<Api>(handle, name, false, true);
  run_case<Api>(handle, name, true, false);
  run_case<Api>(handle, name, true, true);
}

int main() {
  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));

  run_all<CsyrkExApi>(handle, "cublasCsyrkEx");
  run_all<Csyrk3mExApi>(handle, "cublasCsyrk3mEx");
  run_all<CherkExApi>(handle, "cublasCherkEx");
  run_all<Cherk3mExApi>(handle, "cublasCherk3mEx");

  CHECK_CUBLAS(cublasDestroy(handle));
  std::puts("cuBLAS rank-k Ex API test passed");
  return 0;
}
