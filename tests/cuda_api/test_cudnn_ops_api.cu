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

  const std::vector<float> x_host = {-2.0f, -1.0f, 0.0f, 1.0f,
                                     2.0f,  3.0f,  4.0f, 5.0f};
  float *x = device_from_host(x_host);
  float *y = device_from_host(std::vector<float>(8, 0.0f));
  float *dy = device_from_host(std::vector<float>(8, 1.0f));
  float *dx = device_from_host(std::vector<float>(8, 0.0f));

  float one = 1.0f;
  float zero = 0.0f;
  float half = 0.5f;
  float two = 2.0f;
  float three = 3.0f;
  float four = 4.0f;

  CUDNN_CALL(cudnnSetTensor(handle, desc, y, &three));
  expect_close(host_from_device(y, 8), std::vector<float>(8, 3.0f), 1e-5f,
               "set tensor");

  CUDNN_CALL(cudnnScaleTensor(handle, desc, y, &two));
  expect_close(host_from_device(y, 8), std::vector<float>(8, 6.0f), 1e-5f,
               "scale tensor");

  upload(y, std::vector<float>(8, 0.0f));
  CUDNN_CALL(cudnnTransformTensor(handle, &half, desc, x, &zero, desc, y));
  expect_close(host_from_device(y, 8),
               {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f},
               1e-5f, "transform tensor");

  cudnnTensorTransformDescriptor_t transform_desc = nullptr;
  CUDNN_CALL(cudnnCreateTensorTransformDescriptor(&transform_desc));
  int pad_before[4] = {0, 0, 0, 0};
  int pad_after[4] = {0, 0, 0, 0};
  unsigned fold[4] = {1, 1, 1, 1};
  CUDNN_CALL(cudnnSetTensorTransformDescriptor(
      transform_desc, 4, CUDNN_TENSOR_NCHW, pad_before, pad_after, fold,
      CUDNN_TRANSFORM_FOLD));
  upload(y, std::vector<float>(8, 0.0f));
  CUDNN_CALL(
      cudnnTransformTensorEx(handle, transform_desc, &one, desc, x, &zero, desc, y));
  expect_close(host_from_device(y, 8), x_host, 1e-5f, "transform tensor ex");

  cudnnTensorDescriptor_t scalar_desc = nullptr;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&scalar_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(scalar_desc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, 1, 1, 1, 1));
  float *scalar = device_from_host(std::vector<float>{2.0f});
  upload(y, std::vector<float>(8, 1.0f));
  CUDNN_CALL(cudnnAddTensor(handle, &three, scalar_desc, scalar, &four, desc, y));
  expect_close(host_from_device(y, 8), std::vector<float>(8, 10.0f), 1e-5f,
               "add tensor");

  cudnnActivationDescriptor_t activation = nullptr;
  CUDNN_CALL(cudnnCreateActivationDescriptor(&activation));
  CUDNN_CALL(cudnnSetActivationDescriptor(
      activation, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));
  upload(y, std::vector<float>(8, 0.0f));
  CUDNN_CALL(cudnnActivationForward(handle, activation, &one, desc, x, &zero,
                                    desc, y));
  expect_close(host_from_device(y, 8),
               {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, 1e-5f,
               "activation forward");

  upload(dx, std::vector<float>(8, 0.0f));
  CUDNN_CALL(cudnnActivationBackward(handle, activation, &one, desc, y, desc, dy,
                                     desc, x, &zero, desc, dx));
  expect_close(host_from_device(dx, 8),
               {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, 1e-5f,
               "activation backward");

  upload(y, std::vector<float>(8, 0.0f));
  CUDNN_CALL(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE,
                                 CUDNN_SOFTMAX_MODE_INSTANCE, &one, desc, x,
                                 &zero, desc, y));
  std::vector<float> softmax = host_from_device(y, 8);
  float softmax_sum = 0.0f;
  for (float value : softmax) {
    softmax_sum += value;
  }
  expect_close({softmax_sum}, {1.0f}, 1e-5f, "softmax sum");

  upload(dx, std::vector<float>(8, 0.0f));
  CUDNN_CALL(cudnnSoftmaxBackward(handle, CUDNN_SOFTMAX_ACCURATE,
                                  CUDNN_SOFTMAX_MODE_INSTANCE, &one, desc, y,
                                  desc, dy, &zero, desc, dx));
  expect_close(host_from_device(dx, 8), std::vector<float>(8, 0.0f), 1e-4f,
               "softmax backward");

  cudnnOpTensorDescriptor_t op_desc = nullptr;
  CUDNN_CALL(cudnnCreateOpTensorDescriptor(&op_desc));
  CUDNN_CALL(cudnnSetOpTensorDescriptor(op_desc, CUDNN_OP_TENSOR_ADD,
                                        CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
  upload(y, std::vector<float>(8, 0.0f));
  CUDNN_CALL(cudnnOpTensor(handle, op_desc, &one, desc, x, &two, desc, dy, &zero,
                           desc, y));
  expect_close(host_from_device(y, 8),
               {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}, 1e-5f,
               "op tensor");

  cudnnReduceTensorDescriptor_t reduce_desc = nullptr;
  CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&reduce_desc));
  CUDNN_CALL(cudnnSetReduceTensorDescriptor(
      reduce_desc, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT,
      CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES,
      CUDNN_32BIT_INDICES));
  size_t reduce_indices_size = 0;
  size_t reduce_workspace_size = 0;
  CUDNN_CALL(cudnnGetReductionIndicesSize(handle, reduce_desc, desc, scalar_desc,
                                          &reduce_indices_size));
  CUDNN_CALL(cudnnGetReductionWorkspaceSize(handle, reduce_desc, desc, scalar_desc,
                                            &reduce_workspace_size));
  void *reduce_indices = nullptr;
  void *reduce_workspace = nullptr;
  if (reduce_indices_size != 0) {
    CUDA_CALL(cudaMalloc(&reduce_indices, reduce_indices_size));
  }
  if (reduce_workspace_size != 0) {
    CUDA_CALL(cudaMalloc(&reduce_workspace, reduce_workspace_size));
  }
  upload(scalar, std::vector<float>{0.0f});
  CUDNN_CALL(cudnnReduceTensor(handle, reduce_desc, reduce_indices,
                               reduce_indices_size, reduce_workspace,
                               reduce_workspace_size, &one, desc, x, &zero,
                               scalar_desc, scalar));
  expect_close(host_from_device(scalar, 1), {12.0f}, 1e-5f, "reduce tensor");

  cudnnPoolingDescriptor_t pool_desc = nullptr;
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_desc));
  CUDNN_CALL(cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX,
                                         CUDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0,
                                         2, 2));
  cudnnTensorDescriptor_t pool_out_desc = nullptr;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&pool_out_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(pool_out_desc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, 1, 1, 1, 2));
  float *pool_y = device_from_host(std::vector<float>(2, 0.0f));
  float *pool_dy = device_from_host(std::vector<float>{1.0f, 2.0f});
  CUDNN_CALL(cudnnPoolingForward(handle, pool_desc, &one, desc, x, &zero,
                                 pool_out_desc, pool_y));
  expect_close(host_from_device(pool_y, 2), {3.0f, 5.0f}, 1e-5f,
               "pooling forward");
  upload(dx, std::vector<float>(8, 0.0f));
  CUDNN_CALL(cudnnPoolingBackward(handle, pool_desc, &one, pool_out_desc, pool_y,
                                  pool_out_desc, pool_dy, desc, x, &zero, desc,
                                  dx));
  expect_close(host_from_device(dx, 8),
               {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 2.0f}, 1e-5f,
               "pooling backward");

  cudnnLRNDescriptor_t lrn_desc = nullptr;
  CUDNN_CALL(cudnnCreateLRNDescriptor(&lrn_desc));
  CUDNN_CALL(cudnnSetLRNDescriptor(lrn_desc, 3, 1.0, 0.75, 2.0));
  cudnnTensorDescriptor_t lrn_tensor_desc = nullptr;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&lrn_tensor_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(lrn_tensor_desc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, 1, 3, 1, 1));
  float *lrn_x = device_from_host(std::vector<float>{1.0f, 2.0f, 3.0f});
  float *lrn_y = device_from_host(std::vector<float>(3, 0.0f));
  float *lrn_dy = device_from_host(std::vector<float>(3, 1.0f));
  float *lrn_dx = device_from_host(std::vector<float>(3, 0.0f));
  CUDNN_CALL(cudnnLRNCrossChannelForward(handle, lrn_desc,
                                         CUDNN_LRN_CROSS_CHANNEL_DIM1, &one,
                                         lrn_tensor_desc, lrn_x, &zero,
                                         lrn_tensor_desc, lrn_y));
  expect_finite(host_from_device(lrn_y, 3), "lrn forward");
  CUDNN_CALL(cudnnLRNCrossChannelBackward(
      handle, lrn_desc, CUDNN_LRN_CROSS_CHANNEL_DIM1, &one, lrn_tensor_desc,
      lrn_y, lrn_tensor_desc, lrn_dy, lrn_tensor_desc, lrn_x, &zero,
      lrn_tensor_desc, lrn_dx));
  expect_finite(host_from_device(lrn_dx, 3), "lrn backward");

  cudnnTensorDescriptor_t bias_desc = nullptr;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, 1, 1, 1, 1));
  float *bias_grad = device_from_host(std::vector<float>{0.0f});
  CUDNN_CALL(cudnnConvolutionBackwardBias(handle, &one, desc, x, &zero,
                                          bias_desc, bias_grad));
  expect_close(host_from_device(bias_grad, 1), {12.0f}, 1e-5f,
               "convolution backward bias");

  cudnnTensorDescriptor_t conv_x_desc = nullptr;
  cudnnTensorDescriptor_t conv_y_desc = nullptr;
  cudnnFilterDescriptor_t filter_desc = nullptr;
  cudnnConvolutionDescriptor_t conv_desc = nullptr;
  cudnnActivationDescriptor_t identity = nullptr;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&conv_x_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&conv_y_desc));
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnCreateActivationDescriptor(&identity));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(conv_x_desc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, 1, 1, 2, 2));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(conv_y_desc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, 1, 1, 2, 2));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW, 1, 1, 1, 1));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
      conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
  CUDNN_CALL(cudnnSetActivationDescriptor(
      identity, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0));
  float *conv_x = device_from_host(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  float *conv_w = device_from_host(std::vector<float>{2.0f});
  float *conv_z = device_from_host(std::vector<float>(4, 10.0f));
  float *conv_bias = device_from_host(std::vector<float>{1.0f});
  float *conv_y = device_from_host(std::vector<float>(4, 0.0f));
  cudnnConvolutionFwdAlgoPerf_t perf = {};
  int returned_algo_count = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(
      handle, conv_x_desc, filter_desc, conv_desc, conv_y_desc, 1,
      &returned_algo_count, &perf));
  size_t conv_workspace_size = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
      handle, conv_x_desc, filter_desc, conv_desc, conv_y_desc, perf.algo,
      &conv_workspace_size));
  void *conv_workspace = nullptr;
  if (conv_workspace_size != 0) {
    CUDA_CALL(cudaMalloc(&conv_workspace, conv_workspace_size));
  }
  CUDNN_CALL(cudnnConvolutionBiasActivationForward(
      handle, &one, conv_x_desc, conv_x, filter_desc, conv_w, conv_desc,
      perf.algo, conv_workspace, conv_workspace_size, &half, conv_y_desc, conv_z,
      bias_desc, conv_bias, identity, conv_y_desc, conv_y));
  expect_close(host_from_device(conv_y, 4), {8.0f, 10.0f, 12.0f, 14.0f},
               1e-5f, "convolution bias activation forward");

  if (conv_workspace) CUDA_CALL(cudaFree(conv_workspace));
  CUDA_CALL(cudaFree(conv_y));
  CUDA_CALL(cudaFree(conv_bias));
  CUDA_CALL(cudaFree(conv_z));
  CUDA_CALL(cudaFree(conv_w));
  CUDA_CALL(cudaFree(conv_x));
  CUDNN_CALL(cudnnDestroyActivationDescriptor(identity));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(conv_y_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(conv_x_desc));

  CUDNN_CALL(cudnnDestroyTensorDescriptor(bias_desc));
  CUDA_CALL(cudaFree(bias_grad));
  CUDA_CALL(cudaFree(lrn_dx));
  CUDA_CALL(cudaFree(lrn_dy));
  CUDA_CALL(cudaFree(lrn_y));
  CUDA_CALL(cudaFree(lrn_x));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(lrn_tensor_desc));
  CUDNN_CALL(cudnnDestroyLRNDescriptor(lrn_desc));
  CUDA_CALL(cudaFree(pool_dy));
  CUDA_CALL(cudaFree(pool_y));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(pool_out_desc));
  CUDNN_CALL(cudnnDestroyPoolingDescriptor(pool_desc));
  if (reduce_workspace) CUDA_CALL(cudaFree(reduce_workspace));
  if (reduce_indices) CUDA_CALL(cudaFree(reduce_indices));
  CUDNN_CALL(cudnnDestroyReduceTensorDescriptor(reduce_desc));
  CUDNN_CALL(cudnnDestroyOpTensorDescriptor(op_desc));
  CUDNN_CALL(cudnnDestroyActivationDescriptor(activation));
  CUDA_CALL(cudaFree(scalar));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(scalar_desc));
  CUDNN_CALL(cudnnDestroyTensorTransformDescriptor(transform_desc));
  CUDA_CALL(cudaFree(dx));
  CUDA_CALL(cudaFree(dy));
  CUDA_CALL(cudaFree(y));
  CUDA_CALL(cudaFree(x));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
  CUDNN_CALL(cudnnDestroy(handle));

  std::cout << "cuDNN operation coverage ok" << std::endl;
  return 0;
}
