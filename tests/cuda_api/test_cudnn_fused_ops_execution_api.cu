#include <cmath>
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

static void expect_close(float got, float want, const char *name) {
  if (std::fabs(got - want) > 1e-5f) {
    std::cerr << name << " mismatch: got " << got << ", want " << want
              << std::endl;
    std::exit(1);
  }
}

int main() {
  constexpr int channels = 2;
  constexpr double epsilon = 1e-5;

  cudnnHandle_t handle = nullptr;
  cudnnTensorDescriptor_t desc = nullptr;
  cudnnFusedOpsConstParamPack_t const_pack = nullptr;
  cudnnFusedOpsVariantParamPack_t var_pack = nullptr;
  cudnnFusedOpsPlan_t plan = nullptr;

  CUDNN_CALL(cudnnCreate(&handle));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, 1, channels, 1, 1));

  CUDNN_CALL(cudnnCreateFusedOpsConstParamPack(
      &const_pack, CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE));
  CUDNN_CALL(cudnnCreateFusedOpsVariantParamPack(
      &var_pack, CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE));
  CUDNN_CALL(cudnnCreateFusedOpsPlan(
      &plan, CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE));

  cudnnBatchNormMode_t bn_mode = CUDNN_BATCHNORM_SPATIAL;
  cudnnFusedOpsPointerPlaceHolder_t aligned = CUDNN_PTR_16B_ALIGNED;

  CUDNN_CALL(cudnnSetFusedOpsConstParamPackAttribute(
      const_pack, CUDNN_PARAM_BN_MODE, &bn_mode));
  CUDNN_CALL(cudnnSetFusedOpsConstParamPackAttribute(
      const_pack, CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC, desc));
  CUDNN_CALL(cudnnSetFusedOpsConstParamPackAttribute(
      const_pack, CUDNN_PARAM_BN_EQSCALEBIAS_DESC, desc));
  CUDNN_CALL(cudnnSetFusedOpsConstParamPackAttribute(
      const_pack, CUDNN_PARAM_BN_SCALE_PLACEHOLDER, &aligned));
  CUDNN_CALL(cudnnSetFusedOpsConstParamPackAttribute(
      const_pack, CUDNN_PARAM_BN_BIAS_PLACEHOLDER, &aligned));
  CUDNN_CALL(cudnnSetFusedOpsConstParamPackAttribute(
      const_pack, CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER, &aligned));
  CUDNN_CALL(cudnnSetFusedOpsConstParamPackAttribute(
      const_pack, CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER, &aligned));
  CUDNN_CALL(cudnnSetFusedOpsConstParamPackAttribute(
      const_pack, CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER, &aligned));
  CUDNN_CALL(cudnnSetFusedOpsConstParamPackAttribute(
      const_pack, CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER, &aligned));

  size_t workspace_size = 0;
  CUDNN_CALL(
      cudnnMakeFusedOpsPlan(handle, plan, const_pack, &workspace_size));

  const float scale_h[channels] = {1.5f, -2.0f};
  const float bias_h[channels] = {0.25f, 0.5f};
  const float mean_h[channels] = {2.0f, -1.0f};
  const float var_h[channels] = {4.0f, 9.0f};
  float eqscale_h[channels] = {};
  float eqbias_h[channels] = {};

  float *scale = nullptr;
  float *bias = nullptr;
  float *mean = nullptr;
  float *var = nullptr;
  float *eqscale = nullptr;
  float *eqbias = nullptr;
  void *workspace = nullptr;

  CUDA_CALL(cudaMalloc(&scale, sizeof(scale_h)));
  CUDA_CALL(cudaMalloc(&bias, sizeof(bias_h)));
  CUDA_CALL(cudaMalloc(&mean, sizeof(mean_h)));
  CUDA_CALL(cudaMalloc(&var, sizeof(var_h)));
  CUDA_CALL(cudaMalloc(&eqscale, sizeof(eqscale_h)));
  CUDA_CALL(cudaMalloc(&eqbias, sizeof(eqbias_h)));
  if (workspace_size > 0) {
    CUDA_CALL(cudaMalloc(&workspace, workspace_size));
  }

  CUDA_CALL(cudaMemcpy(scale, scale_h, sizeof(scale_h),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(bias, bias_h, sizeof(bias_h), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(mean, mean_h, sizeof(mean_h), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(var, var_h, sizeof(var_h), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemset(eqscale, 0, sizeof(eqscale_h)));
  CUDA_CALL(cudaMemset(eqbias, 0, sizeof(eqbias_h)));

  CUDNN_CALL(cudnnSetFusedOpsVariantParamPackAttribute(
      var_pack, CUDNN_PTR_BN_SCALE, scale));
  CUDNN_CALL(cudnnSetFusedOpsVariantParamPackAttribute(
      var_pack, CUDNN_PTR_BN_BIAS, bias));
  CUDNN_CALL(cudnnSetFusedOpsVariantParamPackAttribute(
      var_pack, CUDNN_PTR_BN_RUNNING_MEAN, mean));
  CUDNN_CALL(cudnnSetFusedOpsVariantParamPackAttribute(
      var_pack, CUDNN_PTR_BN_RUNNING_VAR, var));
  CUDNN_CALL(cudnnSetFusedOpsVariantParamPackAttribute(
      var_pack, CUDNN_PTR_BN_EQSCALE, eqscale));
  CUDNN_CALL(cudnnSetFusedOpsVariantParamPackAttribute(
      var_pack, CUDNN_PTR_BN_EQBIAS, eqbias));
  CUDNN_CALL(cudnnSetFusedOpsVariantParamPackAttribute(
      var_pack, CUDNN_PTR_WORKSPACE, workspace));
  CUDNN_CALL(cudnnSetFusedOpsVariantParamPackAttribute(
      var_pack, CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES,
      &workspace_size));
  CUDNN_CALL(cudnnSetFusedOpsVariantParamPackAttribute(
      var_pack, CUDNN_SCALAR_DOUBLE_BN_EPSILON,
      const_cast<double *>(&epsilon)));

  CUDNN_CALL(cudnnFusedOpsExecute(handle, plan, var_pack));
  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaMemcpy(eqscale_h, eqscale, sizeof(eqscale_h),
                       cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(eqbias_h, eqbias, sizeof(eqbias_h),
                       cudaMemcpyDeviceToHost));

  for (int i = 0; i < channels; ++i) {
    const float inv_std =
        1.0f / std::sqrt(var_h[i] + static_cast<float>(epsilon));
    const float expected_scale = scale_h[i] * inv_std;
    const float expected_bias = bias_h[i] - mean_h[i] * expected_scale;
    expect_close(eqscale_h[i], expected_scale, "eqscale");
    expect_close(eqbias_h[i], expected_bias, "eqbias");
  }

  if (workspace != nullptr) {
    CUDA_CALL(cudaFree(workspace));
  }
  CUDA_CALL(cudaFree(eqbias));
  CUDA_CALL(cudaFree(eqscale));
  CUDA_CALL(cudaFree(var));
  CUDA_CALL(cudaFree(mean));
  CUDA_CALL(cudaFree(bias));
  CUDA_CALL(cudaFree(scale));
  CUDNN_CALL(cudnnDestroyFusedOpsPlan(plan));
  CUDNN_CALL(cudnnDestroyFusedOpsVariantParamPack(var_pack));
  CUDNN_CALL(cudnnDestroyFusedOpsConstParamPack(const_pack));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
  CUDNN_CALL(cudnnDestroy(handle));

  std::cout << "cuDNN fused-op execution coverage ok" << std::endl;
  return 0;
}
