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

void *cuda_alloc_bytes(size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  void *ptr = nullptr;
  CUDA_CALL(cudaMalloc(&ptr, bytes));
  return ptr;
}

void check_finite_buffer(const std::vector<float> &values, const char *label) {
  for (size_t i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values[i])) {
      std::cerr << label << "[" << i << "] is not finite" << std::endl;
      std::exit(1);
    }
  }
}

int main() {
  const int seq_length = 2;
  const int batch_size = 1;
  const int beam_size = 1;
  const int vector_size = 8;
  const int num_heads = 2;
  const int element_count = seq_length * batch_size * beam_size * vector_size;
  const int seq_lengths[batch_size * beam_size] = {seq_length};
  const int lo_win[seq_length] = {0, 0};
  const int hi_win[seq_length] = {seq_length, seq_length};
  const float padding_fill = 0.0f;

  cudnnHandle_t handle = nullptr;
  cudnnAttnDescriptor_t attn = nullptr;
  cudnnDropoutDescriptor_t attn_dropout = nullptr;
  cudnnDropoutDescriptor_t post_dropout = nullptr;
  cudnnSeqDataDescriptor_t q_desc = nullptr;
  cudnnSeqDataDescriptor_t k_desc = nullptr;
  cudnnSeqDataDescriptor_t v_desc = nullptr;
  cudnnSeqDataDescriptor_t o_desc = nullptr;

  CUDNN_CALL(cudnnCreate(&handle));
  CUDNN_CALL(cudnnCreateAttnDescriptor(&attn));
  CUDNN_CALL(cudnnCreateDropoutDescriptor(&attn_dropout));
  CUDNN_CALL(cudnnCreateDropoutDescriptor(&post_dropout));
  size_t dropout_states_size = 0;
  CUDNN_CALL(cudnnDropoutGetStatesSize(handle, &dropout_states_size));
  void *attn_dropout_states = cuda_alloc_bytes(dropout_states_size);
  void *post_dropout_states = cuda_alloc_bytes(dropout_states_size);
  CUDNN_CALL(cudnnSetDropoutDescriptor(attn_dropout, handle, 0.0f,
                                       attn_dropout_states, dropout_states_size,
                                       13));
  CUDNN_CALL(cudnnSetDropoutDescriptor(post_dropout, handle, 0.0f,
                                       post_dropout_states, dropout_states_size,
                                       17));

  CUDNN_CALL(cudnnSetAttnDescriptor(
      attn, CUDNN_ATTN_QUERYMAP_ONE_TO_ONE, num_heads, 1.0,
      CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH, attn_dropout,
      post_dropout, vector_size, vector_size, vector_size, vector_size,
      vector_size, vector_size, vector_size, seq_length, seq_length, batch_size,
      beam_size));

  CUDNN_CALL(cudnnCreateSeqDataDescriptor(&q_desc));
  CUDNN_CALL(cudnnCreateSeqDataDescriptor(&k_desc));
  CUDNN_CALL(cudnnCreateSeqDataDescriptor(&v_desc));
  CUDNN_CALL(cudnnCreateSeqDataDescriptor(&o_desc));

  const int seq_dims[CUDNN_SEQDATA_DIM_COUNT] = {seq_length, batch_size,
                                                 beam_size, vector_size};
  const cudnnSeqDataAxis_t seq_axes[CUDNN_SEQDATA_DIM_COUNT] = {
      CUDNN_SEQDATA_TIME_DIM, CUDNN_SEQDATA_BATCH_DIM,
      CUDNN_SEQDATA_BEAM_DIM, CUDNN_SEQDATA_VECT_DIM};
  CUDNN_CALL(cudnnSetSeqDataDescriptor(
      q_desc, CUDNN_DATA_FLOAT, CUDNN_SEQDATA_DIM_COUNT, seq_dims, seq_axes,
      batch_size * beam_size, seq_lengths, const_cast<float *>(&padding_fill)));
  CUDNN_CALL(cudnnSetSeqDataDescriptor(
      k_desc, CUDNN_DATA_FLOAT, CUDNN_SEQDATA_DIM_COUNT, seq_dims, seq_axes,
      batch_size * beam_size, seq_lengths, const_cast<float *>(&padding_fill)));
  CUDNN_CALL(cudnnSetSeqDataDescriptor(
      v_desc, CUDNN_DATA_FLOAT, CUDNN_SEQDATA_DIM_COUNT, seq_dims, seq_axes,
      batch_size * beam_size, seq_lengths, const_cast<float *>(&padding_fill)));
  CUDNN_CALL(cudnnSetSeqDataDescriptor(
      o_desc, CUDNN_DATA_FLOAT, CUDNN_SEQDATA_DIM_COUNT, seq_dims, seq_axes,
      batch_size * beam_size, seq_lengths, const_cast<float *>(&padding_fill)));

  size_t weight_size = 0;
  size_t workspace_size = 0;
  size_t reserve_size = 0;
  CUDNN_CALL(cudnnGetMultiHeadAttnBuffers(handle, attn, &weight_size,
                                          &workspace_size, &reserve_size));
  if (weight_size == 0 || reserve_size == 0) {
    std::cerr << "multi-head attention buffers were not sized" << std::endl;
    std::exit(1);
  }

  float *queries = nullptr;
  float *keys = nullptr;
  float *values = nullptr;
  float *out = nullptr;
  float *dout = nullptr;
  float *dqueries = nullptr;
  float *dkeys = nullptr;
  float *dvalues = nullptr;
  int *dev_seq_qo = nullptr;
  int *dev_seq_kv = nullptr;
  void *weights = cuda_alloc_bytes(weight_size);
  void *dweights = cuda_alloc_bytes(weight_size);
  void *workspace = cuda_alloc_bytes(workspace_size);
  void *reserve = cuda_alloc_bytes(reserve_size);

  CUDA_CALL(cudaMalloc(&queries, element_count * sizeof(float)));
  CUDA_CALL(cudaMalloc(&keys, element_count * sizeof(float)));
  CUDA_CALL(cudaMalloc(&values, element_count * sizeof(float)));
  CUDA_CALL(cudaMalloc(&out, element_count * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dout, element_count * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dqueries, element_count * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dkeys, element_count * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dvalues, element_count * sizeof(float)));
  CUDA_CALL(cudaMalloc(&dev_seq_qo, sizeof(seq_lengths)));
  CUDA_CALL(cudaMalloc(&dev_seq_kv, sizeof(seq_lengths)));

  std::vector<float> host_input(element_count);
  std::vector<float> host_dout(element_count, 1.0f);
  for (int i = 0; i < element_count; ++i) {
    host_input[i] = 0.01f * static_cast<float>(i + 1);
  }
  CUDA_CALL(cudaMemcpy(queries, host_input.data(),
                       element_count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(keys, host_input.data(), element_count * sizeof(float),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(values, host_input.data(),
                       element_count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dout, host_dout.data(), element_count * sizeof(float),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_seq_qo, seq_lengths, sizeof(seq_lengths),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_seq_kv, seq_lengths, sizeof(seq_lengths),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemset(weights, 0, weight_size));
  CUDA_CALL(cudaMemset(dweights, 0, weight_size));
  CUDA_CALL(cudaMemset(out, 0xff, element_count * sizeof(float)));
  CUDA_CALL(cudaMemset(dqueries, 0xff, element_count * sizeof(float)));
  CUDA_CALL(cudaMemset(dkeys, 0xff, element_count * sizeof(float)));
  CUDA_CALL(cudaMemset(dvalues, 0xff, element_count * sizeof(float)));

  CUDNN_CALL(cudnnMultiHeadAttnForward(
      handle, attn, -1, lo_win, hi_win, dev_seq_qo, dev_seq_kv, q_desc,
      queries, nullptr, k_desc, keys, v_desc, values, o_desc, out, weight_size,
      weights, workspace_size, workspace, reserve_size, reserve));
  CUDA_CALL(cudaDeviceSynchronize());

  std::vector<float> host_out(element_count);
  CUDA_CALL(cudaMemcpy(host_out.data(), out, element_count * sizeof(float),
                       cudaMemcpyDeviceToHost));
  check_finite_buffer(host_out, "attention out");

  CUDNN_CALL(cudnnMultiHeadAttnBackwardData(
      handle, attn, lo_win, hi_win, dev_seq_qo, dev_seq_kv, o_desc, dout,
      q_desc, dqueries, queries, k_desc, dkeys, keys, v_desc, dvalues, values,
      weight_size, weights, workspace_size, workspace, reserve_size, reserve));
  CUDA_CALL(cudaDeviceSynchronize());

  std::vector<float> host_dqueries(element_count);
  CUDA_CALL(cudaMemcpy(host_dqueries.data(), dqueries,
                       element_count * sizeof(float), cudaMemcpyDeviceToHost));
  check_finite_buffer(host_dqueries, "attention dqueries");

  CUDNN_CALL(cudnnMultiHeadAttnBackwardWeights(
      handle, attn, CUDNN_WGRAD_MODE_SET, q_desc, queries, k_desc, keys, v_desc,
      values, o_desc, dout, weight_size, weights, dweights, workspace_size,
      workspace, reserve_size, reserve));
  CUDA_CALL(cudaDeviceSynchronize());

  CUDNN_CALL(cudnnDestroySeqDataDescriptor(o_desc));
  CUDNN_CALL(cudnnDestroySeqDataDescriptor(v_desc));
  CUDNN_CALL(cudnnDestroySeqDataDescriptor(k_desc));
  CUDNN_CALL(cudnnDestroySeqDataDescriptor(q_desc));
  CUDNN_CALL(cudnnDestroyAttnDescriptor(attn));
  CUDNN_CALL(cudnnDestroyDropoutDescriptor(attn_dropout));
  CUDNN_CALL(cudnnDestroyDropoutDescriptor(post_dropout));
  CUDNN_CALL(cudnnDestroy(handle));

  CUDA_CALL(cudaFree(dev_seq_kv));
  CUDA_CALL(cudaFree(dev_seq_qo));
  CUDA_CALL(cudaFree(dvalues));
  CUDA_CALL(cudaFree(dkeys));
  CUDA_CALL(cudaFree(dqueries));
  CUDA_CALL(cudaFree(dout));
  CUDA_CALL(cudaFree(out));
  CUDA_CALL(cudaFree(values));
  CUDA_CALL(cudaFree(keys));
  CUDA_CALL(cudaFree(queries));
  if (reserve) CUDA_CALL(cudaFree(reserve));
  if (workspace) CUDA_CALL(cudaFree(workspace));
  CUDA_CALL(cudaFree(dweights));
  CUDA_CALL(cudaFree(weights));
  if (post_dropout_states) CUDA_CALL(cudaFree(post_dropout_states));
  if (attn_dropout_states) CUDA_CALL(cudaFree(attn_dropout_states));

  std::cout << "cuDNN attention execution coverage ok" << std::endl;
  return 0;
}
