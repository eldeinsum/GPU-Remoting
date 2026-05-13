#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>
#include <cudnn.h>

#define CUDA_CALL(f)                                                           \
  do {                                                                         \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CUDA error "             \
                << cudaGetErrorString(err) << std::endl;                      \
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

int main() {
  cudnnFusedOpsConstParamPack_t const_pack = nullptr;
  cudnnFusedOpsVariantParamPack_t var_pack = nullptr;
  CUDNN_CALL(cudnnCreateFusedOpsConstParamPack(
      &const_pack, CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING));
  CUDNN_CALL(cudnnCreateFusedOpsVariantParamPack(
      &var_pack, CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING));

  cudnnBatchNormMode_t bn_mode = CUDNN_BATCHNORM_SPATIAL;
  CUDNN_CALL(cudnnSetFusedOpsConstParamPackAttribute(
      const_pack, CUDNN_PARAM_BN_MODE, &bn_mode));

  cudnnBatchNormMode_t got_bn_mode =
      static_cast<cudnnBatchNormMode_t>(-1);
  int is_null = -1;
  CUDNN_CALL(cudnnGetFusedOpsConstParamPackAttribute(
      const_pack, CUDNN_PARAM_BN_MODE, &got_bn_mode, &is_null));
  if (got_bn_mode != bn_mode || is_null != 0) {
    std::cerr << "cuDNN fused-op BN mode did not round trip" << std::endl;
    std::exit(1);
  }

  cudnnFusedOpsPointerPlaceHolder_t placeholder = CUDNN_PTR_16B_ALIGNED;
  CUDNN_CALL(cudnnSetFusedOpsConstParamPackAttribute(
      const_pack, CUDNN_PARAM_BN_SCALE_PLACEHOLDER, &placeholder));

  cudnnFusedOpsPointerPlaceHolder_t got_placeholder = CUDNN_PTR_NULL;
  is_null = -1;
  CUDNN_CALL(cudnnGetFusedOpsConstParamPackAttribute(
      const_pack, CUDNN_PARAM_BN_SCALE_PLACEHOLDER, &got_placeholder,
      &is_null));
  if (got_placeholder != placeholder || is_null != 0) {
    std::cerr << "cuDNN fused-op placeholder did not round trip" << std::endl;
    std::exit(1);
  }

  cudnnTensorDescriptor_t desc = nullptr;
  cudnnTensorDescriptor_t got_desc = nullptr;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&got_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, 1, 2, 3, 4));
  CUDNN_CALL(cudnnSetFusedOpsConstParamPackAttribute(
      const_pack, CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC, desc));

  is_null = -1;
  CUDNN_CALL(cudnnGetFusedOpsConstParamPackAttribute(
      const_pack, CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC, got_desc, &is_null));

  cudnnDataType_t data_type;
  int n = 0;
  int c = 0;
  int h = 0;
  int w = 0;
  int n_stride = 0;
  int c_stride = 0;
  int h_stride = 0;
  int w_stride = 0;
  CUDNN_CALL(cudnnGetTensor4dDescriptor(got_desc, &data_type, &n, &c, &h, &w,
                                        &n_stride, &c_stride, &h_stride,
                                        &w_stride));
  if (is_null != 0 || data_type != CUDNN_DATA_FLOAT || n != 1 || c != 2 ||
      h != 3 || w != 4) {
    std::cerr << "cuDNN fused-op descriptor attribute did not round trip"
              << std::endl;
    std::exit(1);
  }

  size_t workspace_size = 1234;
  CUDNN_CALL(cudnnSetFusedOpsVariantParamPackAttribute(
      var_pack, CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES, &workspace_size));
  size_t got_workspace_size = 0;
  CUDNN_CALL(cudnnGetFusedOpsVariantParamPackAttribute(
      var_pack, CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES,
      &got_workspace_size));
  if (got_workspace_size != workspace_size) {
    std::cerr << "cuDNN fused-op workspace scalar did not round trip"
              << std::endl;
    std::exit(1);
  }

  double epsilon = 1e-5;
  CUDNN_CALL(cudnnSetFusedOpsVariantParamPackAttribute(
      var_pack, CUDNN_SCALAR_DOUBLE_BN_EPSILON, &epsilon));
  double got_epsilon = 0.0;
  CUDNN_CALL(cudnnGetFusedOpsVariantParamPackAttribute(
      var_pack, CUDNN_SCALAR_DOUBLE_BN_EPSILON, &got_epsilon));
  if (got_epsilon != epsilon) {
    std::cerr << "cuDNN fused-op double scalar did not round trip"
              << std::endl;
    std::exit(1);
  }

  void *workspace = nullptr;
  CUDA_CALL(cudaMalloc(&workspace, 16));
  CUDNN_CALL(cudnnSetFusedOpsVariantParamPackAttribute(
      var_pack, CUDNN_PTR_WORKSPACE, workspace));
  void *got_workspace = nullptr;
  CUDNN_CALL(cudnnGetFusedOpsVariantParamPackAttribute(
      var_pack, CUDNN_PTR_WORKSPACE, &got_workspace));
  if (got_workspace != workspace) {
    std::cerr << "cuDNN fused-op device pointer did not round trip"
              << std::endl;
    std::exit(1);
  }

  CUDA_CALL(cudaFree(workspace));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(got_desc));
  CUDNN_CALL(cudnnDestroyFusedOpsVariantParamPack(var_pack));
  CUDNN_CALL(cudnnDestroyFusedOpsConstParamPack(const_pack));

  std::cout << "cuDNN fused-op attribute coverage ok" << std::endl;
  return 0;
}
