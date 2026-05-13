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

template <typename T, typename U>
void expect_eq(T actual, U expected, const char *label) {
  if (!(actual == expected)) {
    std::cerr << label << ": expected " << expected << " got " << actual
              << std::endl;
    std::exit(1);
  }
}

void expect_finite(const float *device, size_t count, const char *label) {
  std::vector<float> host(count);
  CUDA_CALL(cudaMemcpy(host.data(), device, count * sizeof(float),
                       cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < host.size(); ++i) {
    if (!std::isfinite(host[i])) {
      std::cerr << label << "[" << i << "] is not finite" << std::endl;
      std::exit(1);
    }
  }
}

int main() {
  cudnnCTCLossDescriptor_t ctc_desc = nullptr;
  CUDNN_CALL(cudnnCreateCTCLossDescriptor(&ctc_desc));

  cudnnDataType_t comp_type = CUDNN_DATA_DOUBLE;
  CUDNN_CALL(cudnnSetCTCLossDescriptor(ctc_desc, CUDNN_DATA_FLOAT));
  CUDNN_CALL(cudnnGetCTCLossDescriptor(ctc_desc, &comp_type));
  expect_eq(comp_type, CUDNN_DATA_FLOAT, "ctc comp type");

  cudnnLossNormalizationMode_t norm_mode = CUDNN_LOSS_NORMALIZATION_NONE;
  cudnnNanPropagation_t grad_mode = CUDNN_PROPAGATE_NAN;
  CUDNN_CALL(cudnnSetCTCLossDescriptorEx(
      ctc_desc, CUDNN_DATA_DOUBLE, CUDNN_LOSS_NORMALIZATION_SOFTMAX,
      CUDNN_NOT_PROPAGATE_NAN));
  CUDNN_CALL(
      cudnnGetCTCLossDescriptorEx(ctc_desc, &comp_type, &norm_mode, &grad_mode));
  expect_eq(comp_type, CUDNN_DATA_DOUBLE, "ctc ex comp type");
  expect_eq(norm_mode, CUDNN_LOSS_NORMALIZATION_SOFTMAX, "ctc ex norm");
  expect_eq(grad_mode, CUDNN_NOT_PROPAGATE_NAN, "ctc ex grad mode");

  int max_label_length = 0;
  CUDNN_CALL(cudnnSetCTCLossDescriptor_v8(
      ctc_desc, CUDNN_DATA_FLOAT, CUDNN_LOSS_NORMALIZATION_SOFTMAX,
      CUDNN_NOT_PROPAGATE_NAN, 2));
  CUDNN_CALL(cudnnGetCTCLossDescriptor_v8(
      ctc_desc, &comp_type, &norm_mode, &grad_mode, &max_label_length));
  expect_eq(comp_type, CUDNN_DATA_FLOAT, "ctc v8 comp type");
  expect_eq(norm_mode, CUDNN_LOSS_NORMALIZATION_SOFTMAX, "ctc v8 norm");
  expect_eq(grad_mode, CUDNN_NOT_PROPAGATE_NAN, "ctc v8 grad mode");
  expect_eq(max_label_length, 2, "ctc v8 max label");

  cudnnCTCGradMode_t ctc_grad_mode = CUDNN_CTC_SKIP_OOB_GRADIENTS;
  CUDNN_CALL(cudnnSetCTCLossDescriptor_v9(
      ctc_desc, CUDNN_DATA_FLOAT, CUDNN_LOSS_NORMALIZATION_NONE,
      CUDNN_CTC_ZERO_OOB_GRADIENTS, 2));
  CUDNN_CALL(cudnnGetCTCLossDescriptor_v9(
      ctc_desc, &comp_type, &norm_mode, &ctc_grad_mode, &max_label_length));
  expect_eq(comp_type, CUDNN_DATA_FLOAT, "ctc v9 comp type");
  expect_eq(norm_mode, CUDNN_LOSS_NORMALIZATION_NONE, "ctc v9 norm");
  expect_eq(ctc_grad_mode, CUDNN_CTC_ZERO_OOB_GRADIENTS, "ctc v9 grad mode");
  expect_eq(max_label_length, 2, "ctc v9 max label");

  cudnnHandle_t handle = nullptr;
  CUDNN_CALL(cudnnCreate(&handle));
  cudnnTensorDescriptor_t probs_desc = nullptr;
  cudnnTensorDescriptor_t gradients_desc = nullptr;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&probs_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&gradients_desc));
  int dims[3] = {3, 1, 4};
  int strides[3] = {4, 4, 1};
  CUDNN_CALL(
      cudnnSetTensorNdDescriptor(probs_desc, CUDNN_DATA_FLOAT, 3, dims, strides));
  CUDNN_CALL(cudnnSetTensorNdDescriptor(gradients_desc, CUDNN_DATA_FLOAT, 3,
                                        dims, strides));
  CUDNN_CALL(cudnnSetCTCLossDescriptor_v8(
      ctc_desc, CUDNN_DATA_FLOAT, CUDNN_LOSS_NORMALIZATION_SOFTMAX,
      CUDNN_NOT_PROPAGATE_NAN, 2));
  size_t workspace_size = 0;
  CUDNN_CALL(cudnnGetCTCLossWorkspaceSize_v8(
      handle, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, probs_desc,
      gradients_desc, &workspace_size));

  float *probs = device_from_host(std::vector<float>{
      0.1f, 0.6f, 0.2f, 0.1f, 0.1f, 0.2f, 0.6f, 0.1f, 0.6f, 0.1f, 0.2f,
      0.1f});
  int *labels = device_from_host(std::vector<int>{1, 2});
  int *label_lengths = device_from_host(std::vector<int>{2});
  int *input_lengths = device_from_host(std::vector<int>{3});
  float *costs = device_from_host(std::vector<float>{0.0f});
  float *gradients = device_from_host(std::vector<float>(12, 0.0f));
  void *workspace = cuda_alloc_bytes(workspace_size);
  CUDNN_CALL(cudnnCTCLoss_v8(
      handle, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, probs_desc, probs,
      labels, label_lengths, input_lengths, costs, gradients_desc, gradients,
      workspace_size, workspace));
  CUDA_CALL(cudaDeviceSynchronize());
  expect_finite(costs, 1, "ctc cost");
  expect_finite(gradients, 12, "ctc gradients");

  if (workspace) CUDA_CALL(cudaFree(workspace));
  CUDA_CALL(cudaFree(gradients));
  CUDA_CALL(cudaFree(costs));
  CUDA_CALL(cudaFree(input_lengths));
  CUDA_CALL(cudaFree(label_lengths));
  CUDA_CALL(cudaFree(labels));
  CUDA_CALL(cudaFree(probs));

  CUDNN_CALL(cudnnDestroyTensorDescriptor(gradients_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(probs_desc));
  CUDNN_CALL(cudnnDestroy(handle));
  CUDNN_CALL(cudnnDestroyCTCLossDescriptor(ctc_desc));

  std::cout << "cuDNN CTC descriptor coverage ok" << std::endl;
  return 0;
}
