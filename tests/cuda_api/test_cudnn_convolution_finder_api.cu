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

void expect_close(const std::vector<float> &actual,
                  const std::vector<float> &expected, float tolerance,
                  const char *label) {
  if (actual.size() != expected.size()) {
    std::cerr << label << " size mismatch" << std::endl;
    std::exit(1);
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]) ||
        std::fabs(actual[i] - expected[i]) > tolerance) {
      std::cerr << label << "[" << i << "] = " << actual[i]
                << ", expected " << expected[i] << std::endl;
      std::exit(1);
    }
  }
}

void expect_find_result(int returned_count, const char *label) {
  if (returned_count <= 0) {
    std::cerr << label << " returned no algorithms" << std::endl;
    std::exit(1);
  }
}

bool allow_deprecated_status(cudnnStatus_t status, const char *label) {
  if (status == CUDNN_STATUS_SUCCESS) {
    return true;
  }
  if (status == CUDNN_STATUS_NOT_SUPPORTED) {
    return false;
  }
  std::cerr << label << " failed with cuDNN error " << status << " ("
            << cudnnGetErrorString(status) << ")" << std::endl;
  std::exit(1);
}

int main() {
  constexpr int n = 1;
  constexpr int c = 1;
  constexpr int h = 3;
  constexpr int w = 3;
  constexpr int k = 1;
  constexpr int r = 1;
  constexpr int s = 1;
  constexpr int element_count = n * c * h * w;
  constexpr int filter_count = k * c * r * s;

  cudnnHandle_t handle = nullptr;
  CUDNN_CALL(cudnnCreate(&handle));

  cudnnTensorDescriptor_t x_desc = nullptr;
  cudnnTensorDescriptor_t y_desc = nullptr;
  cudnnFilterDescriptor_t filter_desc = nullptr;
  cudnnConvolutionDescriptor_t conv_desc = nullptr;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, n, c, h, w));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW, k, c, r, s));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
      conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

  int out_n = 0;
  int out_c = 0;
  int out_h = 0;
  int out_w = 0;
  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
      conv_desc, x_desc, filter_desc, &out_n, &out_c, &out_h, &out_w));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, out_n, out_c, out_h,
                                        out_w));
  const int output_count = out_n * out_c * out_h * out_w;

  float *x = device_from_host(
      std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                         9.0f});
  float *filter = device_from_host(std::vector<float>{1.0f});
  float *y = device_from_host(std::vector<float>(output_count, 0.0f));
  float *dy = device_from_host(std::vector<float>(output_count, 1.0f));
  float *dx = device_from_host(std::vector<float>(element_count, 0.0f));
  float *dw = device_from_host(std::vector<float>(filter_count, 0.0f));
  void *workspace = cuda_alloc_bytes(1 << 20);

  cudnnConvolutionFwdAlgoPerf_t fwd_perf[2] = {};
  int returned_count = 0;
  CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(
      handle, x_desc, filter_desc, conv_desc, y_desc, 2, &returned_count,
      fwd_perf));
  expect_find_result(returned_count, "forward finder");

  returned_count = 0;
  CUDNN_CALL(cudnnFindConvolutionForwardAlgorithmEx(
      handle, x_desc, x, filter_desc, filter, conv_desc, y_desc, y, 2,
      &returned_count, fwd_perf, workspace, 1 << 20));
  expect_find_result(returned_count, "forward finder ex");

  cudnnConvolutionBwdDataAlgoPerf_t bwd_data_perf[2] = {};
  returned_count = 0;
  CUDNN_CALL(cudnnFindConvolutionBackwardDataAlgorithm(
      handle, filter_desc, y_desc, conv_desc, x_desc, 2, &returned_count,
      bwd_data_perf));
  expect_find_result(returned_count, "backward data finder");

  returned_count = 0;
  CUDNN_CALL(cudnnFindConvolutionBackwardDataAlgorithmEx(
      handle, filter_desc, filter, y_desc, dy, conv_desc, x_desc, dx, 2,
      &returned_count, bwd_data_perf, workspace, 1 << 20));
  expect_find_result(returned_count, "backward data finder ex");

  cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_perf[2] = {};
  returned_count = 0;
  CUDNN_CALL(cudnnFindConvolutionBackwardFilterAlgorithm(
      handle, x_desc, y_desc, conv_desc, filter_desc, 2, &returned_count,
      bwd_filter_perf));
  expect_find_result(returned_count, "backward filter finder");

  returned_count = 0;
  CUDNN_CALL(cudnnFindConvolutionBackwardFilterAlgorithmEx(
      handle, x_desc, x, y_desc, dy, conv_desc, filter_desc, dw, 2,
      &returned_count, bwd_filter_perf, workspace, 1 << 20));
  expect_find_result(returned_count, "backward filter finder ex");

  float *col = device_from_host(std::vector<float>(output_count, 0.0f));
  CUDNN_CALL(cudnnIm2Col(handle, x_desc, x, filter_desc, conv_desc, col));
  CUDA_CALL(cudaDeviceSynchronize());
  expect_close(host_from_device(col, output_count), host_from_device(x, element_count),
               1e-5f, "im2col");

  cudnnTensorTransformDescriptor_t transform_desc = nullptr;
  CUDNN_CALL(cudnnCreateTensorTransformDescriptor(&transform_desc));
  int pad_before[4] = {0, 0, 0, 0};
  int pad_after[4] = {0, 0, 0, 0};
  unsigned fold[4] = {1, 1, 1, 1};
  CUDNN_CALL(cudnnSetTensorTransformDescriptor(
      transform_desc, 4, CUDNN_TENSOR_NCHW, pad_before, pad_after, fold,
      CUDNN_TRANSFORM_FOLD));

  cudnnFilterDescriptor_t folded_filter_desc = nullptr;
  cudnnTensorDescriptor_t padded_diff_desc = nullptr;
  cudnnConvolutionDescriptor_t folded_conv_desc = nullptr;
  cudnnTensorDescriptor_t folded_grad_desc = nullptr;
  cudnnTensorTransformDescriptor_t filter_fold_trans_desc = nullptr;
  cudnnTensorTransformDescriptor_t diff_pad_trans_desc = nullptr;
  cudnnTensorTransformDescriptor_t grad_fold_trans_desc = nullptr;
  cudnnTensorTransformDescriptor_t grad_unfold_trans_desc = nullptr;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&folded_filter_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&padded_diff_desc));
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&folded_conv_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&folded_grad_desc));
  CUDNN_CALL(cudnnCreateTensorTransformDescriptor(&filter_fold_trans_desc));
  CUDNN_CALL(cudnnCreateTensorTransformDescriptor(&diff_pad_trans_desc));
  CUDNN_CALL(cudnnCreateTensorTransformDescriptor(&grad_fold_trans_desc));
  CUDNN_CALL(cudnnCreateTensorTransformDescriptor(&grad_unfold_trans_desc));

  CUDNN_CALL(cudnnGetFoldedConvBackwardDataDescriptors(
      handle, filter_desc, y_desc, conv_desc, x_desc, CUDNN_TENSOR_NCHW,
      folded_filter_desc, padded_diff_desc, folded_conv_desc, folded_grad_desc,
      filter_fold_trans_desc, diff_pad_trans_desc, grad_fold_trans_desc,
      grad_unfold_trans_desc));
  cudnnDataType_t folded_type = CUDNN_DATA_DOUBLE;
  cudnnTensorFormat_t folded_format = CUDNN_TENSOR_NHWC;
  int folded_k = 0;
  int folded_c = 0;
  int folded_h = 0;
  int folded_w = 0;
  CUDNN_CALL(cudnnGetFilter4dDescriptor(folded_filter_desc, &folded_type,
                                        &folded_format, &folded_k, &folded_c,
                                        &folded_h, &folded_w));
  if (folded_type != CUDNN_DATA_FLOAT || folded_k <= 0 || folded_c <= 0 ||
      folded_h <= 0 || folded_w <= 0) {
    std::cerr << "invalid folded convolution filter descriptor" << std::endl;
    std::exit(1);
  }

  const float one = 1.0f;
  const float zero = 0.0f;
  float *transformed_filter =
      device_from_host(std::vector<float>(filter_count, 0.0f));
  if (allow_deprecated_status(
          cudnnTransformFilter(handle, transform_desc, &one, filter_desc,
                               filter, &zero, filter_desc, transformed_filter),
          "transform filter")) {
    CUDA_CALL(cudaDeviceSynchronize());
    expect_close(host_from_device(transformed_filter, filter_count), {1.0f},
                 1e-5f, "transform filter");
  }

  float *bias = device_from_host(std::vector<float>{2.0f});
  float *reordered_filter =
      device_from_host(std::vector<float>(filter_count, 0.0f));
  float *reordered_bias = device_from_host(std::vector<float>{0.0f});
  if (allow_deprecated_status(
          cudnnReorderFilterAndBias(handle, filter_desc, CUDNN_NO_REORDER,
                                    filter, reordered_filter, 1, bias,
                                    reordered_bias),
          "reorder filter and bias")) {
    CUDA_CALL(cudaDeviceSynchronize());
    expect_close(host_from_device(reordered_filter, filter_count), {1.0f},
                 1e-5f, "reordered filter");
    expect_close(host_from_device(reordered_bias, 1), {2.0f}, 1e-5f,
                 "reordered bias");
  }

  CUDA_CALL(cudaFree(reordered_bias));
  CUDA_CALL(cudaFree(reordered_filter));
  CUDA_CALL(cudaFree(bias));
  CUDA_CALL(cudaFree(transformed_filter));
  CUDNN_CALL(cudnnDestroyTensorTransformDescriptor(grad_unfold_trans_desc));
  CUDNN_CALL(cudnnDestroyTensorTransformDescriptor(grad_fold_trans_desc));
  CUDNN_CALL(cudnnDestroyTensorTransformDescriptor(diff_pad_trans_desc));
  CUDNN_CALL(cudnnDestroyTensorTransformDescriptor(filter_fold_trans_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(folded_grad_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(folded_conv_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(padded_diff_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(folded_filter_desc));
  CUDNN_CALL(cudnnDestroyTensorTransformDescriptor(transform_desc));
  CUDA_CALL(cudaFree(col));
  if (workspace) CUDA_CALL(cudaFree(workspace));
  CUDA_CALL(cudaFree(dw));
  CUDA_CALL(cudaFree(dx));
  CUDA_CALL(cudaFree(dy));
  CUDA_CALL(cudaFree(y));
  CUDA_CALL(cudaFree(filter));
  CUDA_CALL(cudaFree(x));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroy(handle));

  std::cout << "cuDNN convolution finder coverage ok" << std::endl;
  return 0;
}
