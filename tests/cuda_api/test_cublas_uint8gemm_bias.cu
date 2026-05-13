#include <cublas_v2.h>
#include <cuda_runtime.h>

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

static void expect_bytes(const std::vector<unsigned char> &got,
                         const std::vector<unsigned char> &expected) {
  if (got.size() != expected.size()) {
    std::printf("size mismatch\n");
    std::exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != expected[i]) {
      std::printf("mismatch at %zu got=%u expected=%u\n", i,
                  static_cast<unsigned>(got[i]),
                  static_cast<unsigned>(expected[i]));
      std::exit(EXIT_FAILURE);
    }
  }
}

int main() {
  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));

  const int m = 2;
  const int n = 2;
  const int k = 2;
  const int lda = m;
  const int ldb = k;
  const int ldc = m;
  std::vector<unsigned char> a = {1, 3, 2, 4};
  std::vector<unsigned char> b = {5, 7, 6, 8};
  std::vector<unsigned char> c(4, 0);
  unsigned char *d_a = nullptr;
  unsigned char *d_b = nullptr;
  unsigned char *d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, a.size()));
  CHECK_CUDA(cudaMalloc(&d_b, b.size()));
  CHECK_CUDA(cudaMalloc(&d_c, c.size()));
  CHECK_CUDA(cudaMemcpy(d_a, a.data(), a.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b.data(), b.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_c, 0, c.size()));

  CHECK_CUBLAS(cublasUint8gemmBias(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, d_a, 0, lda,
      d_b, 0, ldb, d_c, 0, ldc, 1, 0));
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(c.data(), d_c, c.size(), cudaMemcpyDeviceToHost));
  expect_bytes(c, {19, 43, 22, 50});

  CHECK_CUDA(cudaFree(d_c));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUBLAS(cublasDestroy(handle));
  std::puts("cuBLAS uint8 GEMM bias test passed");
  return 0;
}
