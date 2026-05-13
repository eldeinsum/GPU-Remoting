#include <cmath>
#include <cstdint>
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

template <typename T, typename U>
void expect_eq(T actual, U expected, const char *label) {
  if (!(actual == expected)) {
    std::cerr << label << ": expected " << expected << " got " << actual
              << std::endl;
    std::exit(1);
  }
}

void expect_close(float actual, float expected, float tolerance,
                  const char *label) {
  if (std::fabs(actual - expected) > tolerance) {
    std::cerr << label << ": expected " << expected << " got " << actual
              << std::endl;
    std::exit(1);
  }
}

void *cuda_alloc_bytes(size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  void *ptr = nullptr;
  CUDA_CALL(cudaMalloc(&ptr, bytes));
  return ptr;
}

void check_zero_buffer(const std::vector<float> &values, const char *label) {
  for (size_t i = 0; i < values.size(); ++i) {
    if (std::fabs(values[i]) > 1e-6f) {
      std::cerr << label << "[" << i << "] expected 0 got " << values[i]
                << std::endl;
      std::exit(1);
    }
  }
}

int main() {
  const int max_seq_length = 2;
  const int batch_size = 2;
  const int input_size = 4;
  const int hidden_size = 3;
  const int num_layers = 1;
  const int seq_lengths[batch_size] = {max_seq_length, max_seq_length};
  const float padding_fill = 0.0f;

  cudnnHandle_t handle = nullptr;
  CUDNN_CALL(cudnnCreate(&handle));
  cudnnStatus_t runtime_status = CUDNN_STATUS_NOT_INITIALIZED;
  CUDNN_CALL(cudnnQueryRuntimeError(handle, &runtime_status,
                                    CUDNN_ERRQUERY_RAWCODE, nullptr));
  if (runtime_status != CUDNN_STATUS_SUCCESS) {
    std::cerr << "unexpected cuDNN runtime status " << runtime_status
              << std::endl;
    std::exit(1);
  }

  cudnnDropoutDescriptor_t dropout_desc = nullptr;
  CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
  size_t states_size = 0;
  CUDNN_CALL(cudnnDropoutGetStatesSize(handle, &states_size));
  void *states = cuda_alloc_bytes(states_size);
  CUDNN_CALL(cudnnSetDropoutDescriptor(dropout_desc, handle, 0.0f, states,
                                       states_size, 11ULL));

  cudnnRNNDescriptor_t rnn_desc = nullptr;
  CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc));
  CUDNN_CALL(cudnnSetRNNDescriptor_v8(
      rnn_desc, CUDNN_RNN_ALGO_STANDARD, CUDNN_RNN_RELU,
      CUDNN_RNN_DOUBLE_BIAS, CUDNN_UNIDIRECTIONAL, CUDNN_LINEAR_INPUT,
      CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH, input_size,
      hidden_size, hidden_size, num_layers, dropout_desc, 0));
  CUDNN_CALL(cudnnBuildRNNDynamic(handle, rnn_desc, batch_size));

  cudnnRNNDataDescriptor_t x_desc = nullptr;
  cudnnRNNDataDescriptor_t y_desc = nullptr;
  CUDNN_CALL(cudnnCreateRNNDataDescriptor(&x_desc));
  CUDNN_CALL(cudnnCreateRNNDataDescriptor(&y_desc));
  CUDNN_CALL(cudnnSetRNNDataDescriptor(
      x_desc, CUDNN_DATA_FLOAT, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
      max_seq_length, batch_size, input_size, seq_lengths,
      const_cast<float *>(&padding_fill)));
  CUDNN_CALL(cudnnSetRNNDataDescriptor(
      y_desc, CUDNN_DATA_FLOAT, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
      max_seq_length, batch_size, hidden_size, seq_lengths,
      const_cast<float *>(&padding_fill)));

  cudnnDataType_t returned_type = CUDNN_DATA_DOUBLE;
  cudnnRNNDataLayout_t returned_layout =
      CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED;
  int returned_max_seq = 0;
  int returned_batch = 0;
  int returned_vector = 0;
  int returned_lengths[batch_size] = {0, 0};
  float returned_padding = 1.0f;
  CUDNN_CALL(cudnnGetRNNDataDescriptor(
      x_desc, &returned_type, &returned_layout, &returned_max_seq,
      &returned_batch, &returned_vector, batch_size, returned_lengths,
      &returned_padding));
  expect_eq(returned_type, CUDNN_DATA_FLOAT, "rnn data type");
  expect_eq(returned_layout, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
            "rnn data layout");
  expect_eq(returned_max_seq, max_seq_length, "rnn max seq");
  expect_eq(returned_batch, batch_size, "rnn batch");
  expect_eq(returned_vector, input_size, "rnn vector");
  expect_eq(returned_lengths[0], seq_lengths[0], "rnn seq length 0");
  expect_eq(returned_lengths[1], seq_lengths[1], "rnn seq length 1");
  expect_close(returned_padding, 0.0f, 1e-6f, "rnn padding fill");

  cudnnSeqDataDescriptor_t seq_desc = nullptr;
  CUDNN_CALL(cudnnCreateSeqDataDescriptor(&seq_desc));
  const int seq_dims[4] = {max_seq_length, batch_size, 1, input_size};
  const cudnnSeqDataAxis_t seq_axes[4] = {
      CUDNN_SEQDATA_TIME_DIM, CUDNN_SEQDATA_BATCH_DIM,
      CUDNN_SEQDATA_BEAM_DIM, CUDNN_SEQDATA_VECT_DIM};
  CUDNN_CALL(cudnnSetSeqDataDescriptor(
      seq_desc, CUDNN_DATA_FLOAT, 4, seq_dims, seq_axes, batch_size,
      seq_lengths, const_cast<float *>(&padding_fill)));
  cudnnDataType_t seq_type = CUDNN_DATA_DOUBLE;
  int seq_nb_dims = 0;
  int seq_dims_out[4] = {0, 0, 0, 0};
  cudnnSeqDataAxis_t seq_axes_out[4] = {};
  size_t seq_length_size = 0;
  int seq_lengths_out[batch_size] = {0, 0};
  float seq_padding = 1.0f;
  CUDNN_CALL(cudnnGetSeqDataDescriptor(
      seq_desc, &seq_type, &seq_nb_dims, 4, seq_dims_out, seq_axes_out,
      &seq_length_size, batch_size, seq_lengths_out, &seq_padding));
  expect_eq(seq_type, CUDNN_DATA_FLOAT, "seq data type");
  expect_eq(seq_nb_dims, 4, "seq dims");
  expect_eq(seq_length_size, static_cast<size_t>(batch_size),
            "seq length size");
  for (int i = 0; i < 4; ++i) {
    expect_eq(seq_dims_out[i], seq_dims[i], "seq dim");
    expect_eq(seq_axes_out[i], seq_axes[i], "seq axis");
  }
  expect_eq(seq_lengths_out[0], seq_lengths[0], "seq length 0");
  expect_eq(seq_lengths_out[1], seq_lengths[1], "seq length 1");
  expect_close(seq_padding, 0.0f, 1e-6f, "seq padding fill");

  size_t weight_space_size = 0;
  CUDNN_CALL(cudnnGetRNNWeightSpaceSize(handle, rnn_desc, &weight_space_size));
  if (weight_space_size == 0) {
    std::cerr << "RNN weight space is empty" << std::endl;
    std::exit(1);
  }
  void *weights = cuda_alloc_bytes(weight_space_size);
  void *dweights = cuda_alloc_bytes(weight_space_size);
  CUDA_CALL(cudaMemset(weights, 0, weight_space_size));
  CUDA_CALL(cudaMemset(dweights, 0, weight_space_size));

  cudnnTensorDescriptor_t matrix_desc = nullptr;
  cudnnTensorDescriptor_t bias_desc = nullptr;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&matrix_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
  void *matrix_addr = nullptr;
  void *bias_addr = nullptr;
  CUDNN_CALL(cudnnGetRNNWeightParams(
      handle, rnn_desc, 0, weight_space_size, weights, 0, matrix_desc,
      &matrix_addr, bias_desc, &bias_addr));
  uintptr_t weight_base = reinterpret_cast<uintptr_t>(weights);
  uintptr_t weight_end = weight_base + weight_space_size;
  uintptr_t matrix_value = reinterpret_cast<uintptr_t>(matrix_addr);
  if (matrix_addr == nullptr || matrix_value < weight_base ||
      matrix_value >= weight_end) {
    std::cerr << "RNN matrix weight pointer outside weight space" << std::endl;
    std::exit(1);
  }

  size_t workspace_size = 0;
  size_t reserve_size = 0;
  CUDNN_CALL(cudnnGetRNNTempSpaceSizes(handle, rnn_desc,
                                       CUDNN_FWD_MODE_TRAINING, x_desc,
                                       &workspace_size, &reserve_size));
  void *workspace = cuda_alloc_bytes(workspace_size);
  void *reserve = cuda_alloc_bytes(reserve_size);

  const size_t x_count =
      static_cast<size_t>(max_seq_length * batch_size * input_size);
  const size_t y_count =
      static_cast<size_t>(max_seq_length * batch_size * hidden_size);
  const size_t h_count =
      static_cast<size_t>(num_layers * batch_size * hidden_size);
  std::vector<float> host_x(x_count);
  std::vector<float> host_dy(y_count, 1.0f);
  for (size_t i = 0; i < host_x.size(); ++i) {
    host_x[i] = 0.01f * static_cast<float>(i + 1);
  }

  float *x = nullptr;
  float *y = nullptr;
  float *dy = nullptr;
  float *dx = nullptr;
  float *hx = nullptr;
  float *hy = nullptr;
  float *dhy = nullptr;
  float *dhx = nullptr;
  int *dev_seq_lengths = nullptr;
  CUDA_CALL(cudaMalloc(&x, x_count * sizeof(float)));
  CUDA_CALL(cudaMalloc(&y, y_count * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dy, y_count * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dx, x_count * sizeof(float)));
  CUDA_CALL(cudaMalloc(&hx, h_count * sizeof(float)));
  CUDA_CALL(cudaMalloc(&hy, h_count * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dhy, h_count * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dhx, h_count * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dev_seq_lengths, batch_size * sizeof(int)));
  CUDA_CALL(cudaMemcpy(x, host_x.data(), x_count * sizeof(float),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dy, host_dy.data(), y_count * sizeof(float),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_seq_lengths, seq_lengths, batch_size * sizeof(int),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemset(y, 0xff, y_count * sizeof(float)));
  CUDA_CALL(cudaMemset(dx, 0xff, x_count * sizeof(float)));
  CUDA_CALL(cudaMemset(hx, 0, h_count * sizeof(float)));
  CUDA_CALL(cudaMemset(hy, 0, h_count * sizeof(float)));
  CUDA_CALL(cudaMemset(dhy, 0, h_count * sizeof(float)));
  CUDA_CALL(cudaMemset(dhx, 0, h_count * sizeof(float)));

  cudnnTensorDescriptor_t h_desc = nullptr;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&h_desc));
  const int h_dims[3] = {num_layers, batch_size, hidden_size};
  const int h_strides[3] = {batch_size * hidden_size, hidden_size, 1};
  CUDNN_CALL(
      cudnnSetTensorNdDescriptor(h_desc, CUDNN_DATA_FLOAT, 3, h_dims, h_strides));

  CUDNN_CALL(cudnnRNNForward(
      handle, rnn_desc, CUDNN_FWD_MODE_TRAINING, dev_seq_lengths, x_desc, x,
      y_desc, y, h_desc, hx, hy, nullptr, nullptr, nullptr, weight_space_size,
      weights, workspace_size, workspace, reserve_size, reserve));
  CUDA_CALL(cudaDeviceSynchronize());

  std::vector<float> host_y(y_count);
  CUDA_CALL(cudaMemcpy(host_y.data(), y, y_count * sizeof(float),
                       cudaMemcpyDeviceToHost));
  check_zero_buffer(host_y, "rnn y");

  CUDNN_CALL(cudnnRNNBackwardData_v8(
      handle, rnn_desc, dev_seq_lengths, y_desc, y, dy, x_desc, dx, h_desc, hx,
      dhy, dhx, nullptr, nullptr, nullptr, nullptr, weight_space_size, weights,
      workspace_size, workspace, reserve_size, reserve));
  CUDA_CALL(cudaDeviceSynchronize());

  std::vector<float> host_dx(x_count);
  CUDA_CALL(cudaMemcpy(host_dx.data(), dx, x_count * sizeof(float),
                       cudaMemcpyDeviceToHost));
  check_zero_buffer(host_dx, "rnn dx");

  CUDNN_CALL(cudnnRNNBackwardWeights_v8(
      handle, rnn_desc, CUDNN_WGRAD_MODE_ADD, dev_seq_lengths, x_desc, x,
      h_desc, hx, y_desc, y, weight_space_size, dweights, workspace_size,
      workspace, reserve_size, reserve));
  CUDA_CALL(cudaDeviceSynchronize());

  CUDNN_CALL(cudnnDestroyTensorDescriptor(h_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(matrix_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(bias_desc));
  CUDNN_CALL(cudnnDestroySeqDataDescriptor(seq_desc));
  CUDNN_CALL(cudnnDestroyRNNDataDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyRNNDataDescriptor(y_desc));
  CUDNN_CALL(cudnnDestroyRNNDescriptor(rnn_desc));
  CUDNN_CALL(cudnnDestroyDropoutDescriptor(dropout_desc));
  CUDNN_CALL(cudnnDestroy(handle));

  CUDA_CALL(cudaFree(dev_seq_lengths));
  CUDA_CALL(cudaFree(dhx));
  CUDA_CALL(cudaFree(dhy));
  CUDA_CALL(cudaFree(hy));
  CUDA_CALL(cudaFree(hx));
  CUDA_CALL(cudaFree(dx));
  CUDA_CALL(cudaFree(dy));
  CUDA_CALL(cudaFree(y));
  CUDA_CALL(cudaFree(x));
  if (reserve) CUDA_CALL(cudaFree(reserve));
  if (workspace) CUDA_CALL(cudaFree(workspace));
  CUDA_CALL(cudaFree(dweights));
  CUDA_CALL(cudaFree(weights));
  if (states) CUDA_CALL(cudaFree(states));

  std::cout << "cuDNN RNN execution coverage ok" << std::endl;
  return 0;
}
