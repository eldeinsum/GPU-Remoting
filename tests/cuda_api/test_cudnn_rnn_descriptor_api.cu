#include <cmath>
#include <cstdlib>
#include <iostream>

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

template <typename T, typename U>
void expect_eq(T actual, U expected, const char *label) {
  if (!(actual == expected)) {
    std::cerr << label << ": expected " << expected << " got " << actual
              << std::endl;
    std::exit(1);
  }
}

void expect_close(double actual, double expected, double tolerance,
                  const char *label) {
  if (std::fabs(actual - expected) > tolerance) {
    std::cerr << label << ": expected " << expected << " got " << actual
              << std::endl;
    std::exit(1);
  }
}

int main() {
  cudnnHandle_t handle = nullptr;
  CUDNN_CALL(cudnnCreate(&handle));

  cudnnDropoutDescriptor_t dropout_desc = nullptr;
  CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
  size_t states_size = 0;
  CUDNN_CALL(cudnnDropoutGetStatesSize(handle, &states_size));
  void *states = nullptr;
  if (states_size != 0) {
    CUDA_CALL(cudaMalloc(&states, states_size));
  }
  CUDNN_CALL(cudnnSetDropoutDescriptor(dropout_desc, handle, 0.0f, states,
                                       states_size, 7ULL));

  cudnnRNNDescriptor_t rnn_desc = nullptr;
  CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc));
  CUDNN_CALL(cudnnSetRNNDescriptor_v8(
      rnn_desc, CUDNN_RNN_ALGO_STANDARD, CUDNN_RNN_RELU,
      CUDNN_RNN_DOUBLE_BIAS, CUDNN_UNIDIRECTIONAL, CUDNN_LINEAR_INPUT,
      CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH, 4, 3, 3, 1,
      dropout_desc, 0));

  cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_COUNT;
  cudnnRNNMode_t cell_mode = CUDNN_GRU;
  cudnnRNNBiasMode_t bias_mode = CUDNN_RNN_NO_BIAS;
  cudnnDirectionMode_t dir_mode = CUDNN_BIDIRECTIONAL;
  cudnnRNNInputMode_t input_mode = CUDNN_SKIP_INPUT;
  cudnnDataType_t data_type = CUDNN_DATA_DOUBLE;
  cudnnDataType_t math_prec = CUDNN_DATA_DOUBLE;
  cudnnMathType_t math_type = CUDNN_TENSOR_OP_MATH;
  int input_size = 0;
  int hidden_size = 0;
  int proj_size = -1;
  int num_layers = 0;
  cudnnDropoutDescriptor_t returned_dropout = nullptr;
  unsigned aux_flags = 1;
  CUDNN_CALL(cudnnGetRNNDescriptor_v8(
      rnn_desc, &algo, &cell_mode, &bias_mode, &dir_mode, &input_mode,
      &data_type, &math_prec, &math_type, &input_size, &hidden_size, &proj_size,
      &num_layers, &returned_dropout, &aux_flags));
  expect_eq(algo, CUDNN_RNN_ALGO_STANDARD, "rnn algo");
  expect_eq(cell_mode, CUDNN_RNN_RELU, "rnn cell");
  expect_eq(bias_mode, CUDNN_RNN_DOUBLE_BIAS, "rnn bias");
  expect_eq(dir_mode, CUDNN_UNIDIRECTIONAL, "rnn dir");
  expect_eq(input_mode, CUDNN_LINEAR_INPUT, "rnn input mode");
  expect_eq(data_type, CUDNN_DATA_FLOAT, "rnn data type");
  expect_eq(math_prec, CUDNN_DATA_FLOAT, "rnn math precision");
  expect_eq(math_type, CUDNN_DEFAULT_MATH, "rnn math type");
  expect_eq(input_size, 4, "rnn input size");
  expect_eq(hidden_size, 3, "rnn hidden size");
  expect_eq(proj_size, 3, "rnn projection size");
  expect_eq(num_layers, 1, "rnn layers");
  expect_eq(returned_dropout, dropout_desc, "rnn dropout descriptor");
  expect_eq(aux_flags, 0U, "rnn aux flags");

  CUDNN_CALL(cudnnRNNSetClip_v8(rnn_desc, CUDNN_RNN_CLIP_MINMAX,
                                CUDNN_NOT_PROPAGATE_NAN, -1.5, 1.5));
  cudnnRNNClipMode_t clip_mode = CUDNN_RNN_CLIP_NONE;
  cudnnNanPropagation_t clip_nan = CUDNN_PROPAGATE_NAN;
  double lclip = 0.0;
  double rclip = 0.0;
  CUDNN_CALL(cudnnRNNGetClip_v8(rnn_desc, &clip_mode, &clip_nan, &lclip, &rclip));
  expect_eq(clip_mode, CUDNN_RNN_CLIP_MINMAX, "rnn clip mode v8");
  expect_eq(clip_nan, CUDNN_NOT_PROPAGATE_NAN, "rnn clip nan v8");
  expect_close(lclip, -1.5, 1e-12, "rnn left clip v8");
  expect_close(rclip, 1.5, 1e-12, "rnn right clip v8");

  CUDNN_CALL(cudnnRNNSetClip_v9(rnn_desc, CUDNN_RNN_CLIP_MINMAX, -2.0, 2.0));
  clip_mode = CUDNN_RNN_CLIP_NONE;
  lclip = 0.0;
  rclip = 0.0;
  CUDNN_CALL(cudnnRNNGetClip_v9(rnn_desc, &clip_mode, &lclip, &rclip));
  expect_eq(clip_mode, CUDNN_RNN_CLIP_MINMAX, "rnn clip mode v9");
  expect_close(lclip, -2.0, 1e-12, "rnn left clip v9");
  expect_close(rclip, 2.0, 1e-12, "rnn right clip v9");

  size_t weight_space_size = 0;
  CUDNN_CALL(cudnnGetRNNWeightSpaceSize(handle, rnn_desc, &weight_space_size));
  if (weight_space_size == 0) {
    std::cerr << "rnn weight space is empty" << std::endl;
    std::exit(1);
  }

  CUDNN_CALL(cudnnDestroyRNNDescriptor(rnn_desc));
  CUDNN_CALL(cudnnDestroyDropoutDescriptor(dropout_desc));
  if (states) CUDA_CALL(cudaFree(states));
  CUDNN_CALL(cudnnDestroy(handle));

  std::cout << "cuDNN RNN descriptor coverage ok" << std::endl;
  return 0;
}
