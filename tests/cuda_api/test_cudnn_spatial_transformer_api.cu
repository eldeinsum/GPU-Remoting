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
  constexpr int c = 1;
  constexpr int h = 2;
  constexpr int w = 2;
  constexpr int element_count = n * c * h * w;
  constexpr int grid_count = n * h * w * 2;
  constexpr int theta_count = n * 2 * 3;

  cudnnHandle_t handle = nullptr;
  CUDNN_CALL(cudnnCreate(&handle));

  cudnnTensorDescriptor_t tensor_desc = nullptr;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&tensor_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(tensor_desc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, n, c, h, w));

  cudnnSpatialTransformerDescriptor_t st_desc = nullptr;
  CUDNN_CALL(cudnnCreateSpatialTransformerDescriptor(&st_desc));
  const int st_dims[4] = {n, c, h, w};
  CUDNN_CALL(cudnnSetSpatialTransformerNdDescriptor(
      st_desc, CUDNN_SAMPLER_BILINEAR, CUDNN_DATA_FLOAT, 4, st_dims));

  float *theta =
      device_from_host(std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});
  float *grid = device_from_host(std::vector<float>(grid_count, 0.0f));
  float *x = device_from_host(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  float *y = device_from_host(std::vector<float>(element_count, 0.0f));
  float *dy = device_from_host(std::vector<float>(element_count, 1.0f));
  float *dx = device_from_host(std::vector<float>(element_count, 0.0f));
  float *dgrid = device_from_host(std::vector<float>(grid_count, 0.0f));
  float *dtheta = device_from_host(std::vector<float>(theta_count, 0.0f));

  const float one = 1.0f;
  const float zero = 0.0f;
  CUDNN_CALL(
      cudnnSpatialTfGridGeneratorForward(handle, st_desc, theta, grid));
  CUDA_CALL(cudaDeviceSynchronize());
  expect_finite(host_from_device(grid, grid_count), "spatial grid");

  CUDNN_CALL(cudnnSpatialTfSamplerForward(handle, st_desc, &one, tensor_desc, x,
                                          grid, &zero, tensor_desc, y));
  CUDA_CALL(cudaDeviceSynchronize());
  expect_finite(host_from_device(y, element_count), "spatial sampler y");

  CUDNN_CALL(cudnnSpatialTfSamplerBackward(
      handle, st_desc, &one, tensor_desc, x, &zero, tensor_desc, dx, &one,
      tensor_desc, dy, grid, &zero, dgrid));
  CUDA_CALL(cudaDeviceSynchronize());
  expect_finite(host_from_device(dx, element_count), "spatial sampler dx");
  expect_finite(host_from_device(dgrid, grid_count), "spatial sampler dgrid");

  CUDNN_CALL(
      cudnnSpatialTfGridGeneratorBackward(handle, st_desc, dgrid, dtheta));
  CUDA_CALL(cudaDeviceSynchronize());
  expect_finite(host_from_device(dtheta, theta_count), "spatial dtheta");

  CUDNN_CALL(cudnnDestroySpatialTransformerDescriptor(st_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(tensor_desc));
  CUDNN_CALL(cudnnDestroy(handle));

  CUDA_CALL(cudaFree(dtheta));
  CUDA_CALL(cudaFree(dgrid));
  CUDA_CALL(cudaFree(dx));
  CUDA_CALL(cudaFree(dy));
  CUDA_CALL(cudaFree(y));
  CUDA_CALL(cudaFree(x));
  CUDA_CALL(cudaFree(grid));
  CUDA_CALL(cudaFree(theta));

  std::cout << "cuDNN spatial transformer coverage ok" << std::endl;
  return 0;
}
