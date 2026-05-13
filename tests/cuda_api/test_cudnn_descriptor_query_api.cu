#include <cuda_runtime.h>
#include <cudnn.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#define CUDA_CALL(expr) do { \
  cudaError_t status = (expr); \
  if (status != cudaSuccess) { \
    std::cerr << __FILE__ << ":" << __LINE__ << ": CUDA error " << status << std::endl; \
    std::exit(1); \
  } \
} while (0)

#define CUDNN_CALL(expr) do { \
  cudnnStatus_t status = (expr); \
  if (status != CUDNN_STATUS_SUCCESS) { \
    std::cerr << __FILE__ << ":" << __LINE__ << ": cuDNN error " << status << std::endl; \
    std::exit(1); \
  } \
} while (0)

#define CHECK_TRUE(expr) do { \
  if (!(expr)) { \
    std::cerr << __FILE__ << ":" << __LINE__ << ": check failed: " #expr << std::endl; \
    std::exit(1); \
  } \
} while (0)

template <typename T, typename U>
void check_eq(T actual, U expected, const char *actual_expr, const char *expected_expr, int line) {
  if (!(actual == expected)) {
    std::cerr << __FILE__ << ":" << line << ": expected " << actual_expr << " == "
              << expected_expr << ", got " << actual << " vs " << expected << std::endl;
    std::exit(1);
  }
}

#define CHECK_EQ(actual, expected) check_eq((actual), (expected), #actual, #expected, __LINE__)

int main() {
  int major = 0;
  int minor = 0;
  int patch = 0;
  CUDNN_CALL(cudnnGetProperty(MAJOR_VERSION, &major));
  CUDNN_CALL(cudnnGetProperty(MINOR_VERSION, &minor));
  CUDNN_CALL(cudnnGetProperty(PATCH_LEVEL, &patch));
  CHECK_EQ(cudnnGetVersion(), static_cast<size_t>(major * 10000 + minor * 100 + patch));
  CHECK_TRUE(cudnnGetMaxDeviceVersion() >= 700);
  CHECK_TRUE(cudnnGetCudartVersion() >= 11000);
  CHECK_TRUE(cudnnGetErrorString(CUDNN_STATUS_SUCCESS) != nullptr);
  char last_error[64] = {};
  cudnnGetLastErrorString(last_error, sizeof(last_error));
  CHECK_EQ(last_error[sizeof(last_error) - 1], '\0');
  CUDNN_CALL(cudnnGraphVersionCheck());
  CUDNN_CALL(cudnnAdvVersionCheck());
  CUDNN_CALL(cudnnCnnVersionCheck());
  CUDNN_CALL(cudnnOpsVersionCheck());
  CUDNN_CALL(cudnnSubquadraticOpsVersionCheck());

  cudnnHandle_t handle;
  CUDNN_CALL(cudnnCreate(&handle));

  cudaStream_t stream;
  CUDA_CALL(cudaStreamCreate(&stream));
  CUDNN_CALL(cudnnSetStream(handle, stream));
  cudaStream_t returned_stream = nullptr;
  CUDNN_CALL(cudnnGetStream(handle, &returned_stream));
  CHECK_EQ(returned_stream, stream);

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
      x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 3, 8, 10));

  cudnnDataType_t data_type;
  int n = 0, c = 0, h = 0, w = 0;
  int ns = 0, cs = 0, hs = 0, ws = 0;
  CUDNN_CALL(cudnnGetTensor4dDescriptor(
      x_desc, &data_type, &n, &c, &h, &w, &ns, &cs, &hs, &ws));
  CHECK_EQ(data_type, CUDNN_DATA_FLOAT);
  CHECK_EQ(n, 2);
  CHECK_EQ(c, 3);
  CHECK_EQ(h, 8);
  CHECK_EQ(w, 10);
  CHECK_EQ(ns, 240);
  CHECK_EQ(cs, 80);
  CHECK_EQ(hs, 10);
  CHECK_EQ(ws, 1);
  size_t tensor_size = 0;
  CUDNN_CALL(cudnnGetTensorSizeInBytes(x_desc, &tensor_size));
  CHECK_EQ(tensor_size, static_cast<size_t>(2 * 3 * 8 * 10 * sizeof(float)));

  CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
      x_desc, CUDNN_DATA_FLOAT, 2, 3, 8, 10, 240, 80, 10, 1));
  CUDNN_CALL(cudnnGetTensor4dDescriptor(
      x_desc, &data_type, &n, &c, &h, &w, &ns, &cs, &hs, &ws));
  CHECK_EQ(ns, 240);
  CHECK_EQ(cs, 80);
  CHECK_EQ(hs, 10);
  CHECK_EQ(ws, 1);

  int dims[4] = {2, 3, 8, 10};
  CUDNN_CALL(cudnnSetTensorNdDescriptorEx(
      x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 4, dims));
  int returned_dims = 0;
  int dims_out[4] = {};
  int strides_out[4] = {};
  CUDNN_CALL(cudnnGetTensorNdDescriptor(
      x_desc, 4, &data_type, &returned_dims, dims_out, strides_out));
  CHECK_EQ(returned_dims, 4);
  CHECK_EQ(dims_out[0], 2);
  CHECK_EQ(dims_out[1], 3);
  CHECK_EQ(dims_out[2], 8);
  CHECK_EQ(dims_out[3], 10);
  CHECK_EQ(strides_out[0], 240);
  CHECK_EQ(strides_out[1], 80);
  CHECK_EQ(strides_out[2], 10);
  CHECK_EQ(strides_out[3], 1);

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
      y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 4, 4, 5));

  cudnnTensorDescriptor_t bn_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&bn_desc));
  CUDNN_CALL(cudnnDeriveBNTensorDescriptor(bn_desc, x_desc, CUDNN_BATCHNORM_SPATIAL));
  CUDNN_CALL(cudnnGetTensor4dDescriptor(
      bn_desc, &data_type, &n, &c, &h, &w, &ns, &cs, &hs, &ws));
  CHECK_EQ(n, 1);
  CHECK_EQ(c, 3);
  CHECK_EQ(h, 1);
  CHECK_EQ(w, 1);

  cudnnTensorDescriptor_t norm_scale_bias_desc;
  cudnnTensorDescriptor_t norm_mean_var_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&norm_scale_bias_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&norm_mean_var_desc));
  CUDNN_CALL(cudnnDeriveNormTensorDescriptor(
      norm_scale_bias_desc, norm_mean_var_desc, x_desc, CUDNN_NORM_PER_CHANNEL, 1));

  cudnnFilterDescriptor_t filter_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(
      filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, 3, 3, 3));
  cudnnTensorFormat_t tensor_format;
  int k = 0;
  CUDNN_CALL(cudnnGetFilter4dDescriptor(
      filter_desc, &data_type, &tensor_format, &k, &c, &h, &w));
  CHECK_EQ(data_type, CUDNN_DATA_FLOAT);
  CHECK_EQ(tensor_format, CUDNN_TENSOR_NCHW);
  CHECK_EQ(k, 4);
  CHECK_EQ(c, 3);
  CHECK_EQ(h, 3);
  CHECK_EQ(w, 3);
  int filter_dims = 0;
  int filter_dims_out[4] = {};
  CUDNN_CALL(cudnnGetFilterNdDescriptor(
      filter_desc, 4, &data_type, &tensor_format, &filter_dims, filter_dims_out));
  CHECK_EQ(filter_dims, 4);
  CHECK_EQ(filter_dims_out[0], 4);
  CHECK_EQ(filter_dims_out[1], 3);
  CHECK_EQ(filter_dims_out[2], 3);
  CHECK_EQ(filter_dims_out[3], 3);
  size_t filter_size = 0;
  CUDNN_CALL(cudnnGetFilterSizeInBytes(filter_desc, &filter_size));
  CHECK_EQ(filter_size, static_cast<size_t>(4 * 3 * 3 * 3 * sizeof(float)));

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
      conv_desc, 1, 1, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  CUDNN_CALL(cudnnSetConvolutionGroupCount(conv_desc, 1));
  CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH));
  CUDNN_CALL(cudnnSetConvolutionReorderType(conv_desc, CUDNN_DEFAULT_REORDER));

  int pad_h = 0, pad_w = 0, stride_h = 0, stride_w = 0, dilation_h = 0, dilation_w = 0;
  cudnnConvolutionMode_t conv_mode;
  CUDNN_CALL(cudnnGetConvolution2dDescriptor(
      conv_desc, &pad_h, &pad_w, &stride_h, &stride_w, &dilation_h, &dilation_w,
      &conv_mode, &data_type));
  CHECK_EQ(pad_h, 1);
  CHECK_EQ(pad_w, 1);
  CHECK_EQ(stride_h, 2);
  CHECK_EQ(stride_w, 2);
  CHECK_EQ(dilation_h, 1);
  CHECK_EQ(dilation_w, 1);
  CHECK_EQ(conv_mode, CUDNN_CROSS_CORRELATION);
  CHECK_EQ(data_type, CUDNN_DATA_FLOAT);
  int group_count = 0;
  CUDNN_CALL(cudnnGetConvolutionGroupCount(conv_desc, &group_count));
  CHECK_EQ(group_count, 1);
  cudnnMathType_t math_type;
  CUDNN_CALL(cudnnGetConvolutionMathType(conv_desc, &math_type));
  CHECK_EQ(math_type, CUDNN_DEFAULT_MATH);
  cudnnReorderType_t reorder_type;
  CUDNN_CALL(cudnnGetConvolutionReorderType(conv_desc, &reorder_type));
  CHECK_EQ(reorder_type, CUDNN_DEFAULT_REORDER);

  int out_n = 0, out_c = 0, out_h = 0, out_w = 0;
  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
      conv_desc, x_desc, filter_desc, &out_n, &out_c, &out_h, &out_w));
  CHECK_EQ(out_n, 2);
  CHECK_EQ(out_c, 4);
  CHECK_EQ(out_h, 4);
  CHECK_EQ(out_w, 5);
  int conv_dims = 0;
  int pad_out[2] = {};
  int stride_out[2] = {};
  int dilation_out[2] = {};
  CUDNN_CALL(cudnnGetConvolutionNdDescriptor(
      conv_desc, 2, &conv_dims, pad_out, stride_out, dilation_out, &conv_mode, &data_type));
  CHECK_EQ(conv_dims, 2);
  CHECK_EQ(pad_out[0], 1);
  CHECK_EQ(pad_out[1], 1);
  CHECK_EQ(stride_out[0], 2);
  CHECK_EQ(stride_out[1], 2);
  CHECK_EQ(dilation_out[0], 1);
  CHECK_EQ(dilation_out[1], 1);

  int algo_count = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &algo_count));
  CHECK_TRUE(algo_count > 0);
  size_t workspace_size = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
      handle, x_desc, filter_desc, conv_desc, y_desc,
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &workspace_size));
  CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle, &algo_count));
  CHECK_TRUE(algo_count > 0);
  CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
      handle, filter_desc, y_desc, conv_desc, x_desc,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, &workspace_size));
  CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle, &algo_count));
  CHECK_TRUE(algo_count > 0);
  CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      handle, x_desc, y_desc, conv_desc, filter_desc,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, &workspace_size));

  cudnnActivationDescriptor_t activation_desc;
  CUDNN_CALL(cudnnCreateActivationDescriptor(&activation_desc));
  CUDNN_CALL(cudnnSetActivationDescriptor(
      activation_desc, CUDNN_ACTIVATION_SWISH, CUDNN_NOT_PROPAGATE_NAN, 0.0));
  CUDNN_CALL(cudnnSetActivationDescriptorSwishBeta(activation_desc, 1.5));
  cudnnActivationMode_t activation_mode;
  cudnnNanPropagation_t nan_opt;
  double coef = 0.0;
  CUDNN_CALL(cudnnGetActivationDescriptor(
      activation_desc, &activation_mode, &nan_opt, &coef));
  CHECK_EQ(activation_mode, CUDNN_ACTIVATION_SWISH);
  CHECK_EQ(nan_opt, CUDNN_NOT_PROPAGATE_NAN);
  double swish_beta = 0.0;
  CUDNN_CALL(cudnnGetActivationDescriptorSwishBeta(activation_desc, &swish_beta));
  CHECK_TRUE(std::abs(swish_beta - 1.5) < 1e-12);

  cudnnOpTensorDescriptor_t op_desc;
  CUDNN_CALL(cudnnCreateOpTensorDescriptor(&op_desc));
  CUDNN_CALL(cudnnSetOpTensorDescriptor(
      op_desc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
  cudnnOpTensorOp_t op_kind;
  CUDNN_CALL(cudnnGetOpTensorDescriptor(op_desc, &op_kind, &data_type, &nan_opt));
  CHECK_EQ(op_kind, CUDNN_OP_TENSOR_ADD);
  CHECK_EQ(data_type, CUDNN_DATA_FLOAT);
  CHECK_EQ(nan_opt, CUDNN_NOT_PROPAGATE_NAN);

  cudnnReduceTensorDescriptor_t reduce_desc;
  CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&reduce_desc));
  CUDNN_CALL(cudnnSetReduceTensorDescriptor(
      reduce_desc, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN,
      CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));
  cudnnReduceTensorOp_t reduce_op;
  cudnnReduceTensorIndices_t reduce_indices;
  cudnnIndicesType_t indices_type;
  CUDNN_CALL(cudnnGetReduceTensorDescriptor(
      reduce_desc, &reduce_op, &data_type, &nan_opt, &reduce_indices, &indices_type));
  CHECK_EQ(reduce_op, CUDNN_REDUCE_TENSOR_ADD);
  CHECK_EQ(data_type, CUDNN_DATA_FLOAT);
  CHECK_EQ(nan_opt, CUDNN_NOT_PROPAGATE_NAN);
  CHECK_EQ(reduce_indices, CUDNN_REDUCE_TENSOR_NO_INDICES);
  CHECK_EQ(indices_type, CUDNN_32BIT_INDICES);

  cudnnPoolingDescriptor_t pooling_desc;
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&pooling_desc));
  CUDNN_CALL(cudnnSetPooling2dDescriptor(
      pooling_desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2));
  cudnnPoolingMode_t pooling_mode;
  int win_h = 0, win_w = 0, pool_pad_h = 0, pool_pad_w = 0, pool_stride_h = 0, pool_stride_w = 0;
  CUDNN_CALL(cudnnGetPooling2dDescriptor(
      pooling_desc, &pooling_mode, &nan_opt, &win_h, &win_w,
      &pool_pad_h, &pool_pad_w, &pool_stride_h, &pool_stride_w));
  CHECK_EQ(pooling_mode, CUDNN_POOLING_MAX);
  CHECK_EQ(nan_opt, CUDNN_NOT_PROPAGATE_NAN);
  CHECK_EQ(win_h, 2);
  CHECK_EQ(win_w, 2);
  int pool_out[4] = {};
  CUDNN_CALL(cudnnGetPoolingNdForwardOutputDim(pooling_desc, x_desc, 4, pool_out));
  CHECK_EQ(pool_out[0], 2);
  CHECK_EQ(pool_out[1], 3);
  CHECK_EQ(pool_out[2], 4);
  CHECK_EQ(pool_out[3], 5);
  CUDNN_CALL(cudnnGetPooling2dForwardOutputDim(
      pooling_desc, x_desc, &out_n, &out_c, &out_h, &out_w));
  CHECK_EQ(out_h, 4);
  CHECK_EQ(out_w, 5);
  int window_dims[2] = {2, 2};
  int padding_dims[2] = {0, 0};
  int stride_dims[2] = {2, 2};
  CUDNN_CALL(cudnnSetPoolingNdDescriptor(
      pooling_desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2,
      window_dims, padding_dims, stride_dims));
  int pool_dims = 0;
  int window_out[2] = {};
  int padding_out[2] = {};
  int pool_stride_out[2] = {};
  CUDNN_CALL(cudnnGetPoolingNdDescriptor(
      pooling_desc, 2, &pooling_mode, &nan_opt, &pool_dims,
      window_out, padding_out, pool_stride_out));
  CHECK_EQ(pool_dims, 2);
  CHECK_EQ(window_out[0], 2);
  CHECK_EQ(window_out[1], 2);

  cudnnLRNDescriptor_t lrn_desc;
  CUDNN_CALL(cudnnCreateLRNDescriptor(&lrn_desc));
  CUDNN_CALL(cudnnSetLRNDescriptor(lrn_desc, 5, 1.0, 0.75, 2.0));
  unsigned lrn_n = 0;
  double lrn_alpha = 0.0, lrn_beta = 0.0, lrn_k = 0.0;
  CUDNN_CALL(cudnnGetLRNDescriptor(lrn_desc, &lrn_n, &lrn_alpha, &lrn_beta, &lrn_k));
  CHECK_EQ(lrn_n, 5u);
  CHECK_TRUE(std::abs(lrn_alpha - 1.0) < 1e-12);
  CHECK_TRUE(std::abs(lrn_beta - 0.75) < 1e-12);
  CHECK_TRUE(std::abs(lrn_k - 2.0) < 1e-12);

  cudnnTensorTransformDescriptor_t transform_desc;
  CUDNN_CALL(cudnnCreateTensorTransformDescriptor(&transform_desc));
  int pad_before[4] = {0, 0, 0, 0};
  int pad_after[4] = {0, 0, 0, 0};
  unsigned fold[4] = {1, 1, 1, 1};
  CUDNN_CALL(cudnnSetTensorTransformDescriptor(
      transform_desc, 4, CUDNN_TENSOR_NCHW, pad_before, pad_after, fold, CUDNN_TRANSFORM_FOLD));
  int pad_before_out[4] = {};
  int pad_after_out[4] = {};
  unsigned fold_out[4] = {};
  cudnnFoldingDirection_t fold_direction;
  CUDNN_CALL(cudnnGetTensorTransformDescriptor(
      transform_desc, 4, &tensor_format, pad_before_out, pad_after_out, fold_out, &fold_direction));
  CHECK_EQ(tensor_format, CUDNN_TENSOR_NCHW);
  CHECK_EQ(fold_direction, CUDNN_TRANSFORM_FOLD);
  CHECK_EQ(fold_out[0], 1u);

  CUDNN_CALL(cudnnDestroyTensorTransformDescriptor(transform_desc));
  CUDNN_CALL(cudnnDestroyLRNDescriptor(lrn_desc));
  CUDNN_CALL(cudnnDestroyPoolingDescriptor(pooling_desc));
  CUDNN_CALL(cudnnDestroyReduceTensorDescriptor(reduce_desc));
  CUDNN_CALL(cudnnDestroyOpTensorDescriptor(op_desc));
  CUDNN_CALL(cudnnDestroyActivationDescriptor(activation_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(norm_mean_var_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(norm_scale_bias_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(bn_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDA_CALL(cudaStreamDestroy(stream));
  CUDNN_CALL(cudnnDestroy(handle));

  std::cout << "cuDNN descriptor/query coverage ok" << std::endl;
  return 0;
}
