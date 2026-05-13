#include <cstdlib>
#include <iostream>

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

int main() {
  const cudnnFusedOps_t ops[] = {
      CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS,
      CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD,
      CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING,
      CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE,
      CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION,
      CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK,
      CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM,
  };

  for (size_t i = 0; i < sizeof(ops) / sizeof(ops[0]); ++i) {
    cudnnFusedOpsConstParamPack_t const_pack = nullptr;
    cudnnFusedOpsVariantParamPack_t var_pack = nullptr;
    cudnnFusedOpsPlan_t plan = nullptr;

    CUDNN_CALL(cudnnCreateFusedOpsConstParamPack(&const_pack, ops[i]));
    CUDNN_CALL(cudnnCreateFusedOpsVariantParamPack(&var_pack, ops[i]));
    CUDNN_CALL(cudnnCreateFusedOpsPlan(&plan, ops[i]));

    if (const_pack == nullptr || var_pack == nullptr || plan == nullptr) {
      std::cerr << "cuDNN fused-op lifecycle returned a null handle"
                << std::endl;
      std::exit(1);
    }

    CUDNN_CALL(cudnnDestroyFusedOpsPlan(plan));
    CUDNN_CALL(cudnnDestroyFusedOpsVariantParamPack(var_pack));
    CUDNN_CALL(cudnnDestroyFusedOpsConstParamPack(const_pack));
  }

  std::cout << "cuDNN fused-op lifecycle coverage ok" << std::endl;
  return 0;
}
