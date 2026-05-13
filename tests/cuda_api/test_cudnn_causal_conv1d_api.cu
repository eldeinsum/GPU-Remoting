#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_subquadratic_ops.h>

#define CUDA_CALL(f)                                                           \
  do {                                                                         \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CUDA error " << err      \
                << " (" << cudaGetErrorString(err) << ")" << std::endl;       \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define CUDNN_CALL(f)                                                          \
  do {                                                                         \
    cudnnStatus_t err = (f);                                                   \
    if (err != CUDNN_STATUS_SUCCESS) {                                         \
      std::cerr << __FILE__ << ":" << __LINE__ << ": cuDNN error " << err     \
                << " (" << cudnnGetErrorString(err) << ")" << std::endl;      \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

template <typename T>
T *device_from_host(const std::vector<T> &host) {
  T *device = nullptr;
  CUDA_CALL(cudaMalloc(&device, host.size() * sizeof(T)));
  CUDA_CALL(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                       cudaMemcpyHostToDevice));
  return device;
}

std::vector<float> host_from_device(const float *device, size_t count) {
  std::vector<float> host(count);
  CUDA_CALL(cudaMemcpy(host.data(), device, count * sizeof(float),
                       cudaMemcpyDeviceToHost));
  return host;
}

void expect_finite(const std::vector<float> &actual, const char *label) {
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i])) {
      std::cerr << label << "[" << i << "] is not finite" << std::endl;
      std::exit(1);
    }
  }
}

int main() {
  constexpr int batch = 1;
  constexpr int dim = 8;
  constexpr int seq_len = 16;
  constexpr int kernel_size = 2;
  constexpr int x_count = batch * dim * seq_len;
  constexpr int weight_count = dim * kernel_size;

  std::vector<float> x_host(x_count);
  for (int i = 0; i < x_count; ++i) {
    x_host[i] = static_cast<float>((i % 7) + 1) / 7.0f;
  }

  float *x = device_from_host(x_host);
  float *weight = device_from_host(std::vector<float>(weight_count, 0.25f));
  float *bias = device_from_host(std::vector<float>(dim, 0.1f));
  float *y = device_from_host(std::vector<float>(x_count, 0.0f));
  float *dy = device_from_host(std::vector<float>(x_count, 1.0f));
  float *dx = device_from_host(std::vector<float>(x_count, 0.0f));
  float *dweight = device_from_host(std::vector<float>(weight_count, 0.0f));
  float *dbias = device_from_host(std::vector<float>(dim, 0.0f));

  CUDNN_CALL(cudnnCausalConv1dForward(
      nullptr, x, weight, bias, y, batch, dim, seq_len, kernel_size,
      CUDNN_DATA_FLOAT, CUDNN_CAUSAL_CONV1D_ACTIVATION_IDENTITY));
  CUDA_CALL(cudaDeviceSynchronize());
  expect_finite(host_from_device(y, x_count), "causal conv1d forward");

  CUDNN_CALL(cudnnCausalConv1dBackward(
      nullptr, x, weight, bias, dy, dx, dweight, dbias, batch, dim, seq_len,
      kernel_size, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT,
      CUDNN_CAUSAL_CONV1D_ACTIVATION_IDENTITY));
  CUDA_CALL(cudaDeviceSynchronize());
  expect_finite(host_from_device(dx, x_count), "causal conv1d dx");
  expect_finite(host_from_device(dweight, weight_count), "causal conv1d dweight");
  expect_finite(host_from_device(dbias, dim), "causal conv1d dbias");

  CUDA_CALL(cudaFree(dbias));
  CUDA_CALL(cudaFree(dweight));
  CUDA_CALL(cudaFree(dx));
  CUDA_CALL(cudaFree(dy));
  CUDA_CALL(cudaFree(y));
  CUDA_CALL(cudaFree(bias));
  CUDA_CALL(cudaFree(weight));
  CUDA_CALL(cudaFree(x));

  std::cout << "cuDNN causal conv1d coverage ok" << std::endl;
  return 0;
}
