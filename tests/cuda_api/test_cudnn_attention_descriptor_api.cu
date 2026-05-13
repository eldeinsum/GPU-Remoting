#include <cstdint>
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

static void check_weight_desc(cudnnTensorDescriptor_t desc, int kind) {
  cudnnDataType_t data_type;
  int nb_dims = 0;
  int dims[8] = {0};
  int strides[8] = {0};
  CUDNN_CALL(cudnnGetTensorNdDescriptor(desc, 8, &data_type, &nb_dims, dims,
                                        strides));
  if (data_type != CUDNN_DATA_FLOAT || nb_dims != 3) {
    std::cerr << "unexpected attention weight descriptor metadata" << std::endl;
    std::exit(1);
  }

  const int expected0 = kind == CUDNN_MH_ATTN_O_BIASES ? 1 : 2;
  const int expected2 = kind >= CUDNN_MH_ATTN_Q_BIASES ? 1 : 8;
  if (dims[0] != expected0 || dims[1] != 8 || dims[2] != expected2) {
    std::cerr << "unexpected attention weight descriptor shape" << std::endl;
    std::exit(1);
  }
}

int main() {
  cudnnHandle_t handle = nullptr;
  cudnnAttnDescriptor_t attn = nullptr;
  cudnnDropoutDescriptor_t attn_dropout = nullptr;
  cudnnDropoutDescriptor_t post_dropout = nullptr;

  CUDNN_CALL(cudnnCreate(&handle));
  CUDNN_CALL(cudnnCreateAttnDescriptor(&attn));
  CUDNN_CALL(cudnnCreateDropoutDescriptor(&attn_dropout));
  CUDNN_CALL(cudnnCreateDropoutDescriptor(&post_dropout));
  CUDNN_CALL(cudnnSetDropoutDescriptor(attn_dropout, handle, 0.0f, nullptr, 0,
                                       0));
  CUDNN_CALL(cudnnSetDropoutDescriptor(post_dropout, handle, 0.0f, nullptr, 0,
                                       0));

  const unsigned mode =
      CUDNN_ATTN_QUERYMAP_ONE_TO_ONE | CUDNN_ATTN_ENABLE_PROJ_BIASES;
  CUDNN_CALL(cudnnSetAttnDescriptor(
      attn, mode, 2, 0.125, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT,
      CUDNN_DEFAULT_MATH, attn_dropout, post_dropout, 8, 8, 8, 8, 8, 8, 8, 4,
      4, 2, 1));

  unsigned got_mode = 0;
  int heads = 0;
  double scale = 0.0;
  cudnnDataType_t data_type;
  cudnnDataType_t compute_type;
  cudnnMathType_t math_type;
  cudnnDropoutDescriptor_t got_attn_dropout = nullptr;
  cudnnDropoutDescriptor_t got_post_dropout = nullptr;
  int q_size = 0;
  int k_size = 0;
  int v_size = 0;
  int q_proj = 0;
  int k_proj = 0;
  int v_proj = 0;
  int o_proj = 0;
  int qo_max = 0;
  int kv_max = 0;
  int batch = 0;
  int beam = 0;

  CUDNN_CALL(cudnnGetAttnDescriptor(
      attn, &got_mode, &heads, &scale, &data_type, &compute_type, &math_type,
      &got_attn_dropout, &got_post_dropout, &q_size, &k_size, &v_size, &q_proj,
      &k_proj, &v_proj, &o_proj, &qo_max, &kv_max, &batch, &beam));

  if (got_mode != mode || heads != 2 || scale != 0.125 ||
      data_type != CUDNN_DATA_FLOAT || compute_type != CUDNN_DATA_FLOAT ||
      math_type != CUDNN_DEFAULT_MATH || got_attn_dropout != attn_dropout ||
      got_post_dropout != post_dropout || q_size != 8 || k_size != 8 ||
      v_size != 8 || q_proj != 8 || k_proj != 8 || v_proj != 8 ||
      o_proj != 8 || qo_max != 4 || kv_max != 4 || batch != 2 ||
      beam != 1) {
    std::cerr << "cuDNN attention descriptor did not round trip" << std::endl;
    std::exit(1);
  }

  size_t weight_size = 0;
  size_t workspace_size = 0;
  size_t reserve_size = 0;
  CUDNN_CALL(cudnnGetMultiHeadAttnBuffers(handle, attn, &weight_size,
                                          &workspace_size, &reserve_size));
  if (weight_size == 0 || reserve_size == 0) {
    std::cerr << "cuDNN attention buffers were not sized" << std::endl;
    std::exit(1);
  }

  void *weights = nullptr;
  CUDA_CALL(cudaMalloc(&weights, weight_size));
  const std::uintptr_t weight_begin = reinterpret_cast<std::uintptr_t>(weights);
  const std::uintptr_t weight_end = weight_begin + weight_size;

  for (int kind = 0; kind < CUDNN_ATTN_WKIND_COUNT; ++kind) {
    cudnnTensorDescriptor_t weight_desc = nullptr;
    void *weight_addr = nullptr;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&weight_desc));
    CUDNN_CALL(cudnnGetMultiHeadAttnWeights(
        handle, attn, static_cast<cudnnMultiHeadAttnWeightKind_t>(kind),
        weight_size, weights, weight_desc, &weight_addr));

    const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(weight_addr);
    if (addr < weight_begin || addr >= weight_end) {
      std::cerr << "attention weight address outside weight buffer"
                << std::endl;
      std::exit(1);
    }
    check_weight_desc(weight_desc, kind);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(weight_desc));
  }

  CUDA_CALL(cudaFree(weights));
  CUDNN_CALL(cudnnDestroyAttnDescriptor(attn));
  CUDNN_CALL(cudnnDestroyDropoutDescriptor(attn_dropout));
  CUDNN_CALL(cudnnDestroyDropoutDescriptor(post_dropout));
  CUDNN_CALL(cudnnDestroy(handle));

  std::cout << "cuDNN attention descriptor coverage ok" << std::endl;
  return 0;
}
