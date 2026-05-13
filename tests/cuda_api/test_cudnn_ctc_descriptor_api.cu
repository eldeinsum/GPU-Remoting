#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>
#include <cudnn.h>

#define CUDNN_CALL(f)                                                          \
  do {                                                                         \
    cudnnStatus_t err = (f);                                                   \
    if (err != CUDNN_STATUS_SUCCESS) {                                         \
      std::cerr << __FILE__ << ":" << __LINE__ << ": cuDNN error " << err     \
                << " (" << cudnnGetErrorString(err) << ")" << std::endl;      \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

template <typename T, typename U>
void expect_eq(T actual, U expected, const char *label) {
  if (!(actual == expected)) {
    std::cerr << label << ": expected " << expected << " got " << actual
              << std::endl;
    std::exit(1);
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
  size_t workspace_size = 0;
  CUDNN_CALL(cudnnGetCTCLossWorkspaceSize_v8(
      handle, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, probs_desc,
      gradients_desc, &workspace_size));

  CUDNN_CALL(cudnnDestroyTensorDescriptor(gradients_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(probs_desc));
  CUDNN_CALL(cudnnDestroy(handle));
  CUDNN_CALL(cudnnDestroyCTCLossDescriptor(ctc_desc));

  std::cout << "cuDNN CTC descriptor coverage ok" << std::endl;
  return 0;
}
