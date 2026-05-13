#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cudnn.h>

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

void upload(float *device, const std::vector<float> &host) {
  CUDA_CALL(cudaMemcpy(device, host.data(), host.size() * sizeof(float),
                       cudaMemcpyHostToDevice));
}

void expect_close(const std::vector<float> &actual,
                  const std::vector<float> &expected, float tolerance,
                  const char *label) {
  if (actual.size() != expected.size()) {
    std::cerr << label << ": size mismatch" << std::endl;
    std::exit(1);
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (std::fabs(actual[i] - expected[i]) > tolerance) {
      std::cerr << label << "[" << i << "] expected " << expected[i] << " got "
                << actual[i] << std::endl;
      std::exit(1);
    }
  }
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
  cudnnHandle_t handle = nullptr;
  CUDNN_CALL(cudnnCreate(&handle));

  cudnnTensorDescriptor_t desc = nullptr;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, 1, 1, 2, 4));

  const std::vector<float> values = {-2.0f, -1.0f, 0.0f, 1.0f,
                                     2.0f,  3.0f,  4.0f, 5.0f};
  float *x = device_from_host(values);
  float *y = device_from_host(std::vector<float>(8, 0.0f));
  float *dy = device_from_host(std::vector<float>(8, 1.0f));
  float *dx = device_from_host(std::vector<float>(8, 0.0f));

  cudnnDropoutDescriptor_t dropout_desc = nullptr;
  CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
  size_t states_size = 0;
  CUDNN_CALL(cudnnDropoutGetStatesSize(handle, &states_size));
  void *states = nullptr;
  if (states_size != 0) {
    CUDA_CALL(cudaMalloc(&states, states_size));
  }
  CUDNN_CALL(cudnnSetDropoutDescriptor(dropout_desc, handle, 0.0f, states,
                                       states_size, 1234ULL));
  float dropout = -1.0f;
  void *returned_states = nullptr;
  unsigned long long seed = 0;
  CUDNN_CALL(
      cudnnGetDropoutDescriptor(dropout_desc, handle, &dropout, &returned_states,
                                &seed));
  expect_close({dropout}, {0.0f}, 1e-6f, "dropout probability");
  if (returned_states != states) {
    std::cerr << "dropout states pointer mismatch" << std::endl;
    std::exit(1);
  }
  size_t reserve_size = 0;
  CUDNN_CALL(cudnnDropoutGetReserveSpaceSize(desc, &reserve_size));
  void *reserve = nullptr;
  if (reserve_size != 0) {
    CUDA_CALL(cudaMalloc(&reserve, reserve_size));
  }
  CUDNN_CALL(cudnnDropoutForward(handle, dropout_desc, desc, x, desc, y,
                                 reserve, reserve_size));
  expect_close(host_from_device(y, 8), values, 1e-5f, "dropout forward");
  CUDNN_CALL(cudnnDropoutBackward(handle, dropout_desc, desc, dy, desc, dx,
                                  reserve, reserve_size));
  expect_close(host_from_device(dx, 8), std::vector<float>(8, 1.0f), 1e-5f,
               "dropout backward");
  CUDNN_CALL(cudnnRestoreDropoutDescriptor(dropout_desc, handle, 0.0f, states,
                                           states_size, 1234ULL));

  cudnnTensorDescriptor_t bn_x_desc = nullptr;
  cudnnTensorDescriptor_t bn_desc = nullptr;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&bn_x_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&bn_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(bn_x_desc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, 1, 2, 1, 4));
  CUDNN_CALL(
      cudnnDeriveBNTensorDescriptor(bn_desc, bn_x_desc, CUDNN_BATCHNORM_SPATIAL));

  float *bn_x = device_from_host(
      std::vector<float>{1.0f, 3.0f, 5.0f, 7.0f, 2.0f, 4.0f, 6.0f, 8.0f});
  float *bn_y = device_from_host(std::vector<float>(8, 0.0f));
  float *bn_scale = device_from_host(std::vector<float>{1.0f, 1.0f});
  float *bn_bias = device_from_host(std::vector<float>{0.0f, 0.0f});
  float *running_mean = device_from_host(std::vector<float>{0.0f, 0.0f});
  float *running_var = device_from_host(std::vector<float>{0.0f, 0.0f});
  float *save_mean = device_from_host(std::vector<float>{0.0f, 0.0f});
  float *save_inv_var = device_from_host(std::vector<float>{0.0f, 0.0f});

  float one = 1.0f;
  float zero = 0.0f;
  CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
      handle, CUDNN_BATCHNORM_SPATIAL, &one, &zero, bn_x_desc, bn_x, bn_x_desc,
      bn_y, bn_desc, bn_scale, bn_bias, 1.0, running_mean, running_var,
      CUDNN_BN_MIN_EPSILON, save_mean, save_inv_var));
  const float inv_std = 1.0f / std::sqrt(5.0f + CUDNN_BN_MIN_EPSILON);
  expect_close(host_from_device(bn_y, 8),
               {-3.0f * inv_std, -1.0f * inv_std, 1.0f * inv_std,
                3.0f * inv_std, -3.0f * inv_std, -1.0f * inv_std,
                1.0f * inv_std, 3.0f * inv_std},
               1e-4f, "batchnorm forward");
  expect_close(host_from_device(save_mean, 2), {4.0f, 5.0f}, 1e-5f,
               "batchnorm saved mean");

  float *bn_dy = device_from_host(std::vector<float>(8, 1.0f));
  float *bn_dx = device_from_host(std::vector<float>(8, 0.0f));
  float *bn_dscale = device_from_host(std::vector<float>(2, 0.0f));
  float *bn_dbias = device_from_host(std::vector<float>(2, 0.0f));
  CUDNN_CALL(cudnnBatchNormalizationBackward(
      handle, CUDNN_BATCHNORM_SPATIAL, &one, &zero, &one, &zero, bn_x_desc,
      bn_x, bn_x_desc, bn_dy, bn_x_desc, bn_dx, bn_desc, bn_scale, bn_dscale,
      bn_dbias, CUDNN_BN_MIN_EPSILON, save_mean, save_inv_var));
  expect_finite(host_from_device(bn_dx, 8), "batchnorm backward data");
  expect_close(host_from_device(bn_dscale, 2), {0.0f, 0.0f}, 1e-4f,
               "batchnorm scale diff");
  expect_close(host_from_device(bn_dbias, 2), {4.0f, 4.0f}, 1e-5f,
               "batchnorm bias diff");

  CUDA_CALL(cudaFree(bn_dbias));
  CUDA_CALL(cudaFree(bn_dscale));
  CUDA_CALL(cudaFree(bn_dx));
  CUDA_CALL(cudaFree(bn_dy));
  CUDA_CALL(cudaFree(save_inv_var));
  CUDA_CALL(cudaFree(save_mean));
  CUDA_CALL(cudaFree(running_var));
  CUDA_CALL(cudaFree(running_mean));
  CUDA_CALL(cudaFree(bn_bias));
  CUDA_CALL(cudaFree(bn_scale));
  CUDA_CALL(cudaFree(bn_y));
  CUDA_CALL(cudaFree(bn_x));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(bn_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(bn_x_desc));

  if (reserve) CUDA_CALL(cudaFree(reserve));
  if (states) CUDA_CALL(cudaFree(states));
  CUDNN_CALL(cudnnDestroyDropoutDescriptor(dropout_desc));
  CUDA_CALL(cudaFree(dx));
  CUDA_CALL(cudaFree(dy));
  CUDA_CALL(cudaFree(y));
  CUDA_CALL(cudaFree(x));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
  CUDNN_CALL(cudnnDestroy(handle));

  std::cout << "cuDNN dropout and batchnorm coverage ok" << std::endl;
  return 0;
}
