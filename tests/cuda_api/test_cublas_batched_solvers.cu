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
  static float zero() { return 0.0f; }
  static float one() { return 1.0f; }
  static float value(double real, double) { return static_cast<float>(real); }
  static float add(float a, float b) { return a + b; }
  static float sub(float a, float b) { return a - b; }
  static float mul(float a, float b) { return a * b; }
  static double abs(float a) { return std::fabs(a); }
  static bool finite(float a) { return std::isfinite(a); }
  static double tol() { return 2.0e-3; }
};

template <> struct Ops<double> {
  static double zero() { return 0.0; }
  static double one() { return 1.0; }
  static double value(double real, double) { return real; }
  static double add(double a, double b) { return a + b; }
  static double sub(double a, double b) { return a - b; }
  static double mul(double a, double b) { return a * b; }
  static double abs(double a) { return std::fabs(a); }
  static bool finite(double a) { return std::isfinite(a); }
  static double tol() { return 1.0e-10; }
};

template <> struct Ops<cuComplex> {
  static cuComplex zero() { return make_cuComplex(0.0f, 0.0f); }
  static cuComplex one() { return make_cuComplex(1.0f, 0.0f); }
  static cuComplex value(double real, double imag) {
    return make_cuComplex(static_cast<float>(real), static_cast<float>(imag));
  }
  static cuComplex add(cuComplex a, cuComplex b) { return cuCaddf(a, b); }
  static cuComplex sub(cuComplex a, cuComplex b) { return cuCsubf(a, b); }
  static cuComplex mul(cuComplex a, cuComplex b) { return cuCmulf(a, b); }
  static double abs(cuComplex a) { return std::hypot(a.x, a.y); }
  static bool finite(cuComplex a) { return std::isfinite(a.x) && std::isfinite(a.y); }
  static double tol() { return 2.0e-3; }
};

template <> struct Ops<cuDoubleComplex> {
  static cuDoubleComplex zero() { return make_cuDoubleComplex(0.0, 0.0); }
  static cuDoubleComplex one() { return make_cuDoubleComplex(1.0, 0.0); }
  static cuDoubleComplex value(double real, double imag) {
    return make_cuDoubleComplex(real, imag);
  }
  static cuDoubleComplex add(cuDoubleComplex a, cuDoubleComplex b) { return cuCadd(a, b); }
  static cuDoubleComplex sub(cuDoubleComplex a, cuDoubleComplex b) { return cuCsub(a, b); }
  static cuDoubleComplex mul(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }
  static double abs(cuDoubleComplex a) { return std::hypot(a.x, a.y); }
  static bool finite(cuDoubleComplex a) { return std::isfinite(a.x) && std::isfinite(a.y); }
  static double tol() { return 1.0e-10; }
};

template <typename T> struct BatchedSolver;

template <> struct BatchedSolver<float> {
  static cublasStatus_t getrf(cublasHandle_t h, int n, float **a, int lda, int *p,
                              int *info, int batch) {
    return cublasSgetrfBatched(h, n, a, lda, p, info, batch);
  }
  static cublasStatus_t getri(cublasHandle_t h, int n, const float **a, int lda,
                              const int *p, float **c, int ldc, int *info,
                              int batch) {
    return cublasSgetriBatched(h, n, a, lda, p, c, ldc, info, batch);
  }
  static cublasStatus_t getrs(cublasHandle_t h, cublasOperation_t trans, int n,
                              int nrhs, const float **a, int lda,
                              const int *p, float **b, int ldb, int *info,
                              int batch) {
    return cublasSgetrsBatched(h, trans, n, nrhs, a, lda, p, b, ldb, info, batch);
  }
  static cublasStatus_t matinv(cublasHandle_t h, int n, const float **a, int lda,
                               float **ainv, int lda_inv, int *info, int batch) {
    return cublasSmatinvBatched(h, n, a, lda, ainv, lda_inv, info, batch);
  }
  static cublasStatus_t geqrf(cublasHandle_t h, int m, int n, float **a, int lda,
                              float **tau, int *info, int batch) {
    return cublasSgeqrfBatched(h, m, n, a, lda, tau, info, batch);
  }
  static cublasStatus_t gels(cublasHandle_t h, cublasOperation_t trans, int m,
                             int n, int nrhs, float **a, int lda, float **c,
                             int ldc, int *info, int *dev_info, int batch) {
    return cublasSgelsBatched(h, trans, m, n, nrhs, a, lda, c, ldc, info,
                              dev_info, batch);
  }
};

template <> struct BatchedSolver<double> {
  static cublasStatus_t getrf(cublasHandle_t h, int n, double **a, int lda,
                              int *p, int *info, int batch) {
    return cublasDgetrfBatched(h, n, a, lda, p, info, batch);
  }
  static cublasStatus_t getri(cublasHandle_t h, int n, const double **a, int lda,
                              const int *p, double **c, int ldc, int *info,
                              int batch) {
    return cublasDgetriBatched(h, n, a, lda, p, c, ldc, info, batch);
  }
  static cublasStatus_t getrs(cublasHandle_t h, cublasOperation_t trans, int n,
                              int nrhs, const double **a, int lda,
                              const int *p, double **b, int ldb, int *info,
                              int batch) {
    return cublasDgetrsBatched(h, trans, n, nrhs, a, lda, p, b, ldb, info, batch);
  }
  static cublasStatus_t matinv(cublasHandle_t h, int n, const double **a, int lda,
                               double **ainv, int lda_inv, int *info, int batch) {
    return cublasDmatinvBatched(h, n, a, lda, ainv, lda_inv, info, batch);
  }
  static cublasStatus_t geqrf(cublasHandle_t h, int m, int n, double **a, int lda,
                              double **tau, int *info, int batch) {
    return cublasDgeqrfBatched(h, m, n, a, lda, tau, info, batch);
  }
  static cublasStatus_t gels(cublasHandle_t h, cublasOperation_t trans, int m,
                             int n, int nrhs, double **a, int lda, double **c,
                             int ldc, int *info, int *dev_info, int batch) {
    return cublasDgelsBatched(h, trans, m, n, nrhs, a, lda, c, ldc, info,
                              dev_info, batch);
  }
};

template <> struct BatchedSolver<cuComplex> {
  static cublasStatus_t getrf(cublasHandle_t h, int n, cuComplex **a, int lda,
                              int *p, int *info, int batch) {
    return cublasCgetrfBatched(h, n, a, lda, p, info, batch);
  }
  static cublasStatus_t getri(cublasHandle_t h, int n, const cuComplex **a,
                              int lda, const int *p, cuComplex **c, int ldc,
                              int *info, int batch) {
    return cublasCgetriBatched(h, n, a, lda, p, c, ldc, info, batch);
  }
  static cublasStatus_t getrs(cublasHandle_t h, cublasOperation_t trans, int n,
                              int nrhs, const cuComplex **a, int lda,
                              const int *p, cuComplex **b, int ldb, int *info,
                              int batch) {
    return cublasCgetrsBatched(h, trans, n, nrhs, a, lda, p, b, ldb, info, batch);
  }
  static cublasStatus_t matinv(cublasHandle_t h, int n, const cuComplex **a,
                               int lda, cuComplex **ainv, int lda_inv,
                               int *info, int batch) {
    return cublasCmatinvBatched(h, n, a, lda, ainv, lda_inv, info, batch);
  }
  static cublasStatus_t geqrf(cublasHandle_t h, int m, int n, cuComplex **a,
                              int lda, cuComplex **tau, int *info, int batch) {
    return cublasCgeqrfBatched(h, m, n, a, lda, tau, info, batch);
  }
  static cublasStatus_t gels(cublasHandle_t h, cublasOperation_t trans, int m,
                             int n, int nrhs, cuComplex **a, int lda,
                             cuComplex **c, int ldc, int *info, int *dev_info,
                             int batch) {
    return cublasCgelsBatched(h, trans, m, n, nrhs, a, lda, c, ldc, info,
                              dev_info, batch);
  }
};

template <> struct BatchedSolver<cuDoubleComplex> {
  static cublasStatus_t getrf(cublasHandle_t h, int n, cuDoubleComplex **a,
                              int lda, int *p, int *info, int batch) {
    return cublasZgetrfBatched(h, n, a, lda, p, info, batch);
  }
  static cublasStatus_t getri(cublasHandle_t h, int n,
                              const cuDoubleComplex **a, int lda,
                              const int *p, cuDoubleComplex **c, int ldc,
                              int *info, int batch) {
    return cublasZgetriBatched(h, n, a, lda, p, c, ldc, info, batch);
  }
  static cublasStatus_t getrs(cublasHandle_t h, cublasOperation_t trans, int n,
                              int nrhs, const cuDoubleComplex **a, int lda,
                              const int *p, cuDoubleComplex **b, int ldb,
                              int *info, int batch) {
    return cublasZgetrsBatched(h, trans, n, nrhs, a, lda, p, b, ldb, info, batch);
  }
  static cublasStatus_t matinv(cublasHandle_t h, int n,
                               const cuDoubleComplex **a, int lda,
                               cuDoubleComplex **ainv, int lda_inv, int *info,
                               int batch) {
    return cublasZmatinvBatched(h, n, a, lda, ainv, lda_inv, info, batch);
  }
  static cublasStatus_t geqrf(cublasHandle_t h, int m, int n,
                              cuDoubleComplex **a, int lda,
                              cuDoubleComplex **tau, int *info, int batch) {
    return cublasZgeqrfBatched(h, m, n, a, lda, tau, info, batch);
  }
  static cublasStatus_t gels(cublasHandle_t h, cublasOperation_t trans, int m,
                             int n, int nrhs, cuDoubleComplex **a, int lda,
                             cuDoubleComplex **c, int ldc, int *info,
                             int *dev_info, int batch) {
    return cublasZgelsBatched(h, trans, m, n, nrhs, a, lda, c, ldc, info,
                              dev_info, batch);
  }
};

template <typename T> struct DeviceBatch {
  int rows;
  int cols;
  int lda;
  int batch;
  int stride;
  std::vector<T> host;
  T *data;
  T **ptrs;

  DeviceBatch(int rows_, int cols_, int lda_, int batch_)
      : rows(rows_), cols(cols_), lda(lda_), batch(batch_),
        stride(lda_ * cols_), host(batch_ * lda_ * cols_), data(NULL),
        ptrs(NULL) {}

  DeviceBatch(const DeviceBatch &) = delete;
  DeviceBatch &operator=(const DeviceBatch &) = delete;

  ~DeviceBatch() {
    if (ptrs != NULL) {
      cudaFree(ptrs);
    }
    if (data != NULL) {
      cudaFree(data);
    }
  }

  void upload() {
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&data),
                          host.size() * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(data, host.data(), host.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
    std::vector<T *> host_ptrs(batch);
    for (int b = 0; b < batch; ++b) {
      host_ptrs[b] = data + b * stride;
    }
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&ptrs),
                          batch * sizeof(T *)));
    CHECK_CUDA(cudaMemcpy(ptrs, host_ptrs.data(), batch * sizeof(T *),
                          cudaMemcpyHostToDevice));
  }

  void download() {
    CHECK_CUDA(cudaMemcpy(host.data(), data, host.size() * sizeof(T),
                          cudaMemcpyDeviceToHost));
  }

  T get(int b, int row, int col) const {
    return host[b * stride + row + col * lda];
  }

  void set(int b, int row, int col, T value) {
    host[b * stride + row + col * lda] = value;
  }
};

template <typename T>
static const T **const_ptrs(T **ptrs) {
  return reinterpret_cast<const T **>(static_cast<void *>(ptrs));
}

static void check_device_info(const char *name, const int *d_info, int count) {
  std::vector<int> info(count, -1);
  CHECK_CUDA(cudaMemcpy(info.data(), d_info, count * sizeof(int),
                        cudaMemcpyDeviceToHost));
  for (int i = 0; i < count; ++i) {
    if (info[i] != 0) {
      std::printf("%s info[%d] = %d\n", name, i, info[i]);
      std::exit(EXIT_FAILURE);
    }
  }
}

template <typename T>
static void fill_square(DeviceBatch<T> &a) {
  for (int b = 0; b < a.batch; ++b) {
    for (int col = 0; col < a.cols; ++col) {
      for (int row = 0; row < a.lda; ++row) {
        T value = Ops<T>::zero();
        if (row < a.rows) {
          const double diag = row == col ? 4.0 + 0.25 * b : 0.0;
          const double real = diag + 0.2 * (row + 1) - 0.15 * (col + 1);
          const double imag = 0.03 * (row - col) + 0.01 * b;
          value = Ops<T>::value(real, imag);
        }
        a.set(b, row, col, value);
      }
    }
  }
}

template <typename T>
static void fill_tall(DeviceBatch<T> &a) {
  for (int b = 0; b < a.batch; ++b) {
    for (int col = 0; col < a.cols; ++col) {
      for (int row = 0; row < a.lda; ++row) {
        T value = Ops<T>::zero();
        if (row < a.rows) {
          const double diag = row == col ? 2.0 + 0.2 * b : 0.0;
          const double real = diag + 0.25 * (row + 1) + 0.1 * (col + 1);
          const double imag = 0.02 * (row - col) + 0.01 * b;
          value = Ops<T>::value(real, imag);
        }
        a.set(b, row, col, value);
      }
    }
  }
}

template <typename T>
static T matmul_at(const DeviceBatch<T> &a, const DeviceBatch<T> &b, int batch,
                   int row, int col, int inner) {
  T sum = Ops<T>::zero();
  for (int k = 0; k < inner; ++k) {
    sum = Ops<T>::add(sum, Ops<T>::mul(a.get(batch, row, k), b.get(batch, k, col)));
  }
  return sum;
}

template <typename T>
static void check_identity(const char *name, const DeviceBatch<T> &a,
                           const DeviceBatch<T> &inv, int n) {
  for (int b = 0; b < a.batch; ++b) {
    for (int col = 0; col < n; ++col) {
      for (int row = 0; row < n; ++row) {
        const T got = matmul_at(a, inv, b, row, col, n);
        const T want = row == col ? Ops<T>::one() : Ops<T>::zero();
        if (Ops<T>::abs(Ops<T>::sub(got, want)) > Ops<T>::tol() * 16.0) {
          std::printf("%s identity mismatch batch=%d row=%d col=%d err=%g\n",
                      name, b, row, col, Ops<T>::abs(Ops<T>::sub(got, want)));
          std::exit(EXIT_FAILURE);
        }
      }
    }
  }
}

template <typename T>
static void fill_rhs_from_solution(const DeviceBatch<T> &a, DeviceBatch<T> &b,
                                   std::vector<T> &solution, int nrhs) {
  solution.assign(a.batch * a.cols * nrhs, Ops<T>::zero());
  for (int batch = 0; batch < a.batch; ++batch) {
    for (int col = 0; col < nrhs; ++col) {
      for (int row = 0; row < a.cols; ++row) {
        solution[batch * a.cols * nrhs + row + col * a.cols] =
            Ops<T>::value(0.5 + 0.1 * row + 0.2 * col + 0.05 * batch,
                          0.03 * (row - col) + 0.01 * batch);
      }
    }
    for (int col = 0; col < nrhs; ++col) {
      for (int row = 0; row < b.lda; ++row) {
        T value = Ops<T>::zero();
        if (row < a.rows) {
          for (int k = 0; k < a.cols; ++k) {
            const T x = solution[batch * a.cols * nrhs + k + col * a.cols];
            value = Ops<T>::add(value, Ops<T>::mul(a.get(batch, row, k), x));
          }
        }
        b.set(batch, row, col, value);
      }
    }
  }
}

template <typename T>
static void check_solution(const char *name, const DeviceBatch<T> &b,
                           const std::vector<T> &solution, int n, int nrhs) {
  for (int batch = 0; batch < b.batch; ++batch) {
    for (int col = 0; col < nrhs; ++col) {
      for (int row = 0; row < n; ++row) {
        const T want = solution[batch * n * nrhs + row + col * n];
        const T got = b.get(batch, row, col);
        if (Ops<T>::abs(Ops<T>::sub(got, want)) > Ops<T>::tol() * 16.0) {
          std::printf("%s solution mismatch batch=%d row=%d col=%d err=%g\n",
                      name, batch, row, col, Ops<T>::abs(Ops<T>::sub(got, want)));
          std::exit(EXIT_FAILURE);
        }
      }
    }
  }
}

template <typename T>
static void run_lu_inverse_and_solve(cublasHandle_t handle, const char *name) {
  const int n = 3;
  const int nrhs = 2;
  const int batch = 2;
  DeviceBatch<T> original(n, n, n + 1, batch);
  fill_square(original);

  DeviceBatch<T> lu(n, n, n + 1, batch);
  lu.host = original.host;
  DeviceBatch<T> rhs(n, nrhs, n + 1, batch);
  std::vector<T> solution;
  fill_rhs_from_solution(original, rhs, solution, nrhs);
  original.upload();
  lu.upload();
  rhs.upload();

  int *d_piv = NULL;
  int *d_info = NULL;
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_piv), n * batch * sizeof(int)));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_info), batch * sizeof(int)));

  CHECK_CUBLAS(BatchedSolver<T>::getrf(handle, n, lu.ptrs, lu.lda, d_piv,
                                       d_info, batch));
  CHECK_CUDA(cudaDeviceSynchronize());
  check_device_info(name, d_info, batch);

  int host_info = -1;
  CHECK_CUBLAS(BatchedSolver<T>::getrs(
      handle, CUBLAS_OP_N, n, nrhs, const_ptrs(lu.ptrs), lu.lda, d_piv,
      rhs.ptrs, rhs.lda, &host_info, batch));
  if (host_info != 0) {
    std::printf("%s getrs host info = %d\n", name, host_info);
    std::exit(EXIT_FAILURE);
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  rhs.download();
  check_solution(name, rhs, solution, n, nrhs);

  DeviceBatch<T> inv(n, n, n + 1, batch);
  inv.upload();
  CHECK_CUBLAS(BatchedSolver<T>::getri(
      handle, n, const_ptrs(lu.ptrs), lu.lda, d_piv, inv.ptrs, inv.lda,
      d_info, batch));
  CHECK_CUDA(cudaDeviceSynchronize());
  check_device_info(name, d_info, batch);
  inv.download();
  check_identity(name, original, inv, n);

  cudaFree(d_info);
  cudaFree(d_piv);
}

template <typename T>
static void run_matinv(cublasHandle_t handle, const char *name) {
  const int n = 3;
  const int batch = 2;
  DeviceBatch<T> a(n, n, n + 1, batch);
  DeviceBatch<T> inv(n, n, n + 1, batch);
  fill_square(a);
  a.upload();
  inv.upload();

  int *d_info = NULL;
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_info), batch * sizeof(int)));
  CHECK_CUBLAS(BatchedSolver<T>::matinv(
      handle, n, const_ptrs(a.ptrs), a.lda, inv.ptrs, inv.lda, d_info, batch));
  CHECK_CUDA(cudaDeviceSynchronize());
  check_device_info(name, d_info, batch);
  inv.download();
  check_identity(name, a, inv, n);
  cudaFree(d_info);
}

template <typename T>
static void run_qr(cublasHandle_t handle, const char *name) {
  const int m = 4;
  const int n = 3;
  const int batch = 2;
  const int k = 3;
  DeviceBatch<T> a(m, n, m + 1, batch);
  DeviceBatch<T> tau(k, 1, k, batch);
  fill_tall(a);
  a.upload();
  tau.upload();

  int host_info = -1;
  CHECK_CUBLAS(BatchedSolver<T>::geqrf(handle, m, n, a.ptrs, a.lda, tau.ptrs,
                                       &host_info, batch));
  if (host_info != 0) {
    std::printf("%s geqrf host info = %d\n", name, host_info);
    std::exit(EXIT_FAILURE);
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  tau.download();
  for (size_t i = 0; i < tau.host.size(); ++i) {
    if (!Ops<T>::finite(tau.host[i])) {
      std::printf("%s geqrf produced non-finite tau at %zu\n", name, i);
      std::exit(EXIT_FAILURE);
    }
  }
}

template <typename T>
static void run_gels(cublasHandle_t handle, const char *name) {
  const int m = 4;
  const int n = 3;
  const int nrhs = 1;
  const int batch = 2;
  DeviceBatch<T> a(m, n, m, batch);
  DeviceBatch<T> c(m, nrhs, m, batch);
  std::vector<T> solution;
  fill_tall(a);
  fill_rhs_from_solution(a, c, solution, nrhs);
  a.upload();
  c.upload();

  int *d_info = NULL;
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_info), batch * sizeof(int)));
  int host_info = -1;
  CHECK_CUBLAS(BatchedSolver<T>::gels(handle, CUBLAS_OP_N, m, n, nrhs, a.ptrs,
                                      a.lda, c.ptrs, c.lda, &host_info, d_info,
                                      batch));
  if (host_info != 0) {
    std::printf("%s gels host info = %d\n", name, host_info);
    std::exit(EXIT_FAILURE);
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  check_device_info(name, d_info, batch);
  c.download();
  check_solution(name, c, solution, n, nrhs);
  cudaFree(d_info);
}

template <typename T>
static void run_all(cublasHandle_t handle, const char *name) {
  run_lu_inverse_and_solve<T>(handle, name);
  run_matinv<T>(handle, name);
  run_qr<T>(handle, name);
  run_gels<T>(handle, name);
}

int main() {
  cublasHandle_t handle = NULL;
  CHECK_CUBLAS(cublasCreate(&handle));

  run_all<float>(handle, "float");
  run_all<double>(handle, "double");
  run_all<cuComplex>(handle, "complex-float");
  run_all<cuDoubleComplex>(handle, "complex-double");

  CHECK_CUBLAS(cublasDestroy(handle));
  std::printf("cublas batched solver APIs ok\n");
  return 0;
}
