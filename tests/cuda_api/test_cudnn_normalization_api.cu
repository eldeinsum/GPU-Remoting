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

void *cuda_alloc_bytes(size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  void *ptr = nullptr;
  CUDA_CALL(cudaMalloc(&ptr, bytes));
  return ptr;
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
  constexpr int n = 1;
  constexpr int c = 2;
  constexpr int h = 1;
  constexpr int w = 4;
  constexpr int element_count = n * c * h * w;
  constexpr int param_count = c;

  cudnnHandle_t handle = nullptr;
  CUDNN_CALL(cudnnCreate(&handle));

  cudnnTensorDescriptor_t x_desc = nullptr;
  cudnnTensorDescriptor_t scale_bias_desc = nullptr;
  cudnnTensorDescriptor_t mean_var_desc = nullptr;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&scale_bias_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&mean_var_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, n, c, h, w));
  CUDNN_CALL(cudnnDeriveNormTensorDescriptor(
      scale_bias_desc, mean_var_desc, x_desc, CUDNN_NORM_PER_CHANNEL, 1));

  cudnnActivationDescriptor_t activation_desc = nullptr;
  CUDNN_CALL(cudnnCreateActivationDescriptor(&activation_desc));
  CUDNN_CALL(cudnnSetActivationDescriptor(
      activation_desc, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0));

  float *x = device_from_host(
      std::vector<float>{1.0f, 3.0f, 5.0f, 7.0f, 2.0f, 4.0f, 6.0f, 8.0f});
  float *z = device_from_host(std::vector<float>(element_count, 0.0f));
  float *y = device_from_host(std::vector<float>(element_count, 0.0f));
  float *dy = device_from_host(std::vector<float>(element_count, 1.0f));
  float *dx = device_from_host(std::vector<float>(element_count, 0.0f));
  float *dz = device_from_host(std::vector<float>(element_count, 0.0f));
  float *scale = device_from_host(std::vector<float>(param_count, 1.0f));
  float *bias = device_from_host(std::vector<float>(param_count, 0.0f));
  float *running_mean = device_from_host(std::vector<float>(param_count, 0.0f));
  float *running_var = device_from_host(std::vector<float>(param_count, 0.0f));
  float *save_mean = device_from_host(std::vector<float>(param_count, 0.0f));
  float *save_inv_var = device_from_host(std::vector<float>(param_count, 0.0f));
  float *dscale = device_from_host(std::vector<float>(param_count, 0.0f));
  float *dbias = device_from_host(std::vector<float>(param_count, 0.0f));

  const cudnnNormMode_t mode = CUDNN_NORM_PER_CHANNEL;
  const cudnnNormOps_t ops = CUDNN_NORM_OPS_NORM;
  const cudnnNormAlgo_t algo = CUDNN_NORM_ALGO_STANDARD;
  const float one = 1.0f;
  const float zero = 0.0f;

  size_t forward_workspace_size = 0;
  CUDNN_CALL(cudnnGetNormalizationForwardTrainingWorkspaceSize(
      handle, mode, ops, algo, x_desc, x_desc, x_desc, scale_bias_desc,
      activation_desc, mean_var_desc, &forward_workspace_size, 1));
  size_t reserve_size = 0;
  CUDNN_CALL(cudnnGetNormalizationTrainingReserveSpaceSize(
      handle, mode, ops, algo, activation_desc, x_desc, &reserve_size, 1));
  void *forward_workspace = cuda_alloc_bytes(forward_workspace_size);
  void *reserve = cuda_alloc_bytes(reserve_size);

  CUDNN_CALL(cudnnNormalizationForwardTraining(
      handle, mode, ops, algo, &one, &zero, x_desc, x, scale_bias_desc, scale,
      bias, 1.0, mean_var_desc, running_mean, running_var, CUDNN_BN_MIN_EPSILON,
      save_mean, save_inv_var, activation_desc, x_desc, z, x_desc, y,
      forward_workspace, forward_workspace_size, reserve, reserve_size, 1));
  CUDA_CALL(cudaDeviceSynchronize());
  expect_finite(host_from_device(y, element_count), "normalization training y");
  expect_finite(host_from_device(save_mean, param_count),
                "normalization saved mean");
  expect_finite(host_from_device(save_inv_var, param_count),
                "normalization saved inv var");

  size_t backward_workspace_size = 0;
  CUDNN_CALL(cudnnGetNormalizationBackwardWorkspaceSize(
      handle, mode, ops, algo, x_desc, x_desc, x_desc, x_desc, x_desc,
      scale_bias_desc, activation_desc, mean_var_desc, &backward_workspace_size,
      1));
  void *backward_workspace = cuda_alloc_bytes(backward_workspace_size);
  CUDNN_CALL(cudnnNormalizationBackward(
      handle, mode, ops, algo, &one, &zero, &one, &zero, x_desc, x, x_desc, y,
      x_desc, dy, x_desc, dz, x_desc, dx, scale_bias_desc, scale, bias, dscale,
      dbias, CUDNN_BN_MIN_EPSILON, mean_var_desc, save_mean, save_inv_var,
      activation_desc, backward_workspace, backward_workspace_size, reserve,
      reserve_size, 1));
  CUDA_CALL(cudaDeviceSynchronize());
  expect_finite(host_from_device(dx, element_count), "normalization dx");
  expect_finite(host_from_device(dscale, param_count), "normalization dscale");
  expect_finite(host_from_device(dbias, param_count), "normalization dbias");

  CUDNN_CALL(cudnnNormalizationForwardInference(
      handle, mode, ops, algo, &one, &zero, x_desc, x, scale_bias_desc, scale,
      bias, mean_var_desc, running_mean, running_var, x_desc, z, activation_desc,
      x_desc, y, CUDNN_BN_MIN_EPSILON, 1));
  CUDA_CALL(cudaDeviceSynchronize());
  expect_finite(host_from_device(y, element_count), "normalization inference y");

  if (backward_workspace) CUDA_CALL(cudaFree(backward_workspace));
  if (reserve) CUDA_CALL(cudaFree(reserve));
  if (forward_workspace) CUDA_CALL(cudaFree(forward_workspace));
  CUDA_CALL(cudaFree(dbias));
  CUDA_CALL(cudaFree(dscale));
  CUDA_CALL(cudaFree(save_inv_var));
  CUDA_CALL(cudaFree(save_mean));
  CUDA_CALL(cudaFree(running_var));
  CUDA_CALL(cudaFree(running_mean));
  CUDA_CALL(cudaFree(bias));
  CUDA_CALL(cudaFree(scale));
  CUDA_CALL(cudaFree(dz));
  CUDA_CALL(cudaFree(dx));
  CUDA_CALL(cudaFree(dy));
  CUDA_CALL(cudaFree(y));
  CUDA_CALL(cudaFree(z));
  CUDA_CALL(cudaFree(x));
  CUDNN_CALL(cudnnDestroyActivationDescriptor(activation_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(mean_var_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(scale_bias_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroy(handle));

  std::cout << "cuDNN normalization coverage ok" << std::endl;
  return 0;
}
