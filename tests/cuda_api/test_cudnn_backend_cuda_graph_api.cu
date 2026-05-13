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

static void set_attr(cudnnBackendDescriptor_t desc,
                     cudnnBackendAttributeName_t name,
                     cudnnBackendAttributeType_t type, int64_t count,
                     const void *value) {
  CUDNN_CALL(cudnnBackendSetAttribute(desc, name, type, count, value));
}

static cudnnBackendDescriptor_t make_tensor(int64_t uid) {
  cudnnBackendDescriptor_t desc = nullptr;
  CUDNN_CALL(
      cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &desc));

  const cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
  const int64_t dims[] = {1, 4, 1, 1};
  const int64_t strides[] = {4, 1, 1, 1};
  const int64_t alignment = 16;

  set_attr(desc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1,
           &data_type);
  set_attr(desc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dims);
  set_attr(desc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, strides);
  set_attr(desc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
  set_attr(desc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1,
           &alignment);
  CUDNN_CALL(cudnnBackendFinalize(desc));
  return desc;
}

struct BackendPlan {
  cudnnBackendDescriptor_t x_desc = nullptr;
  cudnnBackendDescriptor_t y_desc = nullptr;
  cudnnBackendDescriptor_t pointwise_desc = nullptr;
  cudnnBackendDescriptor_t operation_desc = nullptr;
  cudnnBackendDescriptor_t graph_desc = nullptr;
  cudnnBackendDescriptor_t heur_desc = nullptr;
  std::vector<cudnnBackendDescriptor_t> engine_configs;
  cudnnBackendDescriptor_t plan_desc = nullptr;
  int64_t workspace_size = 0;
};

static BackendPlan make_identity_plan(cudnnHandle_t handle) {
  BackendPlan plan;
  plan.x_desc = make_tensor(1);
  plan.y_desc = make_tensor(2);

  CUDNN_CALL(cudnnBackendCreateDescriptor(CUDNN_BACKEND_POINTWISE_DESCRIPTOR,
                                          &plan.pointwise_desc));
  const cudnnPointwiseMode_t pointwise_mode = CUDNN_POINTWISE_IDENTITY;
  const cudnnDataType_t math_prec = CUDNN_DATA_FLOAT;
  set_attr(plan.pointwise_desc, CUDNN_ATTR_POINTWISE_MODE,
           CUDNN_TYPE_POINTWISE_MODE, 1, &pointwise_mode);
  set_attr(plan.pointwise_desc, CUDNN_ATTR_POINTWISE_MATH_PREC,
           CUDNN_TYPE_DATA_TYPE, 1, &math_prec);
  CUDNN_CALL(cudnnBackendFinalize(plan.pointwise_desc));

  CUDNN_CALL(cudnnBackendCreateDescriptor(
      CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, &plan.operation_desc));
  set_attr(plan.operation_desc, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
           CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &plan.pointwise_desc);
  set_attr(plan.operation_desc, CUDNN_ATTR_OPERATION_POINTWISE_XDESC,
           CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &plan.x_desc);
  set_attr(plan.operation_desc, CUDNN_ATTR_OPERATION_POINTWISE_YDESC,
           CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &plan.y_desc);
  CUDNN_CALL(cudnnBackendFinalize(plan.operation_desc));

  CUDNN_CALL(cudnnBackendCreateDescriptor(
      CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &plan.graph_desc));
  cudnnBackendDescriptor_t ops[] = {plan.operation_desc};
  set_attr(plan.graph_desc, CUDNN_ATTR_OPERATIONGRAPH_OPS,
           CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops);
  set_attr(plan.graph_desc, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE,
           1, &handle);
  CUDNN_CALL(cudnnBackendFinalize(plan.graph_desc));

  CUDNN_CALL(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR,
                                          &plan.heur_desc));
  const cudnnBackendHeurMode_t heur_mode = CUDNN_HEUR_MODE_INSTANT;
  set_attr(plan.heur_desc, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
           CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &plan.graph_desc);
  set_attr(plan.heur_desc, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1,
           &heur_mode);
  CUDNN_CALL(cudnnBackendFinalize(plan.heur_desc));

  int64_t engine_count = 0;
  CUDNN_CALL(cudnnBackendGetAttribute(
      plan.heur_desc, CUDNN_ATTR_ENGINEHEUR_RESULTS,
      CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &engine_count, nullptr));
  if (engine_count <= 0) {
    std::cerr << "cuDNN backend heuristics returned no engine configs"
              << std::endl;
    std::exit(1);
  }

  const int64_t requested = engine_count < 8 ? engine_count : 8;
  plan.engine_configs.resize(static_cast<size_t>(requested));
  for (int64_t i = 0; i < requested; ++i) {
    CUDNN_CALL(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR,
                                            &plan.engine_configs[i]));
  }

  int64_t returned = 0;
  CUDNN_CALL(cudnnBackendGetAttribute(
      plan.heur_desc, CUDNN_ATTR_ENGINEHEUR_RESULTS,
      CUDNN_TYPE_BACKEND_DESCRIPTOR, requested, &returned,
      plan.engine_configs.data()));

  for (int64_t i = 0; i < returned; ++i) {
    cudnnBackendDescriptor_t candidate = nullptr;
    CUDNN_CALL(cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &candidate));
    cudnnStatus_t status = cudnnBackendSetAttribute(
        candidate, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &plan.engine_configs[i]);
    if (status == CUDNN_STATUS_SUCCESS) {
      status = cudnnBackendSetAttribute(candidate, CUDNN_ATTR_EXECUTION_PLAN_HANDLE,
                                        CUDNN_TYPE_HANDLE, 1, &handle);
    }
    if (status == CUDNN_STATUS_SUCCESS) {
      status = cudnnBackendFinalize(candidate);
    }
    if (status == CUDNN_STATUS_SUCCESS) {
      plan.plan_desc = candidate;
      break;
    }
    CUDNN_CALL(cudnnBackendDestroyDescriptor(candidate));
  }

  if (plan.plan_desc == nullptr) {
    std::cerr << "cuDNN backend could not finalize an execution plan"
              << std::endl;
    std::exit(1);
  }

  int64_t count = 0;
  CUDNN_CALL(cudnnBackendGetAttribute(
      plan.plan_desc, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
      CUDNN_TYPE_INT64, 1, &count, &plan.workspace_size));
  return plan;
}

static cudnnBackendDescriptor_t make_variant_pack(void *x, void *y,
                                                  void *workspace) {
  cudnnBackendDescriptor_t pack = nullptr;
  CUDNN_CALL(
      cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
                                   &pack));
  int64_t uids[] = {1, 2};
  void *ptrs[] = {x, y};
  set_attr(pack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 2,
           ptrs);
  set_attr(pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 2,
           uids);
  set_attr(pack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1,
           &workspace);
  CUDNN_CALL(cudnnBackendFinalize(pack));
  return pack;
}

static void destroy_plan(BackendPlan &plan) {
  if (plan.plan_desc != nullptr) {
    CUDNN_CALL(cudnnBackendDestroyDescriptor(plan.plan_desc));
  }
  for (size_t i = 0; i < plan.engine_configs.size(); ++i) {
    if (plan.engine_configs[i] != nullptr) {
      CUDNN_CALL(cudnnBackendDestroyDescriptor(plan.engine_configs[i]));
    }
  }
  if (plan.heur_desc != nullptr) {
    CUDNN_CALL(cudnnBackendDestroyDescriptor(plan.heur_desc));
  }
  if (plan.graph_desc != nullptr) {
    CUDNN_CALL(cudnnBackendDestroyDescriptor(plan.graph_desc));
  }
  if (plan.operation_desc != nullptr) {
    CUDNN_CALL(cudnnBackendDestroyDescriptor(plan.operation_desc));
  }
  if (plan.pointwise_desc != nullptr) {
    CUDNN_CALL(cudnnBackendDestroyDescriptor(plan.pointwise_desc));
  }
  if (plan.y_desc != nullptr) {
    CUDNN_CALL(cudnnBackendDestroyDescriptor(plan.y_desc));
  }
  if (plan.x_desc != nullptr) {
    CUDNN_CALL(cudnnBackendDestroyDescriptor(plan.x_desc));
  }
}

static void launch_graph_and_check(cudaGraph_t graph, const float *expected,
                                   float *device_output) {
  cudaGraphExec_t exec = nullptr;
  CUDA_CALL(cudaGraphInstantiate(&exec, graph, 0));
  CUDA_CALL(cudaGraphLaunch(exec, 0));
  CUDA_CALL(cudaDeviceSynchronize());

  float got[4] = {};
  CUDA_CALL(cudaMemcpy(got, device_output, sizeof(got), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 4; ++i) {
    expect_close(got[i], expected[i], "backend graph output");
  }
  CUDA_CALL(cudaGraphExecDestroy(exec));
}

int main() {
  cudnnHandle_t handle = nullptr;
  CUDNN_CALL(cudnnCreate(&handle));
  BackendPlan plan = make_identity_plan(handle);

  const float input_a_h[4] = {-2.0f, -1.0f, 0.5f, 3.0f};
  const float input_b_h[4] = {7.0f, 8.0f, 9.0f, 10.0f};
  float *input_a = nullptr;
  float *output_a = nullptr;
  float *input_b = nullptr;
  float *output_b = nullptr;
  void *workspace = nullptr;

  CUDA_CALL(cudaMalloc(&input_a, sizeof(input_a_h)));
  CUDA_CALL(cudaMalloc(&output_a, sizeof(input_a_h)));
  CUDA_CALL(cudaMalloc(&input_b, sizeof(input_b_h)));
  CUDA_CALL(cudaMalloc(&output_b, sizeof(input_b_h)));
  if (plan.workspace_size > 0) {
    CUDA_CALL(cudaMalloc(&workspace, static_cast<size_t>(plan.workspace_size)));
  }
  CUDA_CALL(cudaMemcpy(input_a, input_a_h, sizeof(input_a_h),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(input_b, input_b_h, sizeof(input_b_h),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemset(output_a, 0, sizeof(input_a_h)));
  CUDA_CALL(cudaMemset(output_b, 0, sizeof(input_b_h)));

  cudnnBackendDescriptor_t variant_a =
      make_variant_pack(input_a, output_a, workspace);
  cudnnBackendDescriptor_t variant_b =
      make_variant_pack(input_b, output_b, workspace);

  CUDNN_CALL(cudnnBackendExecute(handle, plan.plan_desc, variant_a));
  CUDA_CALL(cudaDeviceSynchronize());
  float execute_got[4] = {};
  CUDA_CALL(cudaMemcpy(execute_got, output_a, sizeof(execute_got),
                       cudaMemcpyDeviceToHost));
  for (int i = 0; i < 4; ++i) {
    expect_close(execute_got[i], input_a_h[i], "backend execute output");
  }
  CUDA_CALL(cudaMemset(output_a, 0, sizeof(input_a_h)));

  cudaGraph_t graph = nullptr;
  CUDA_CALL(cudaGraphCreate(&graph, 0));
  CUDNN_CALL(
      cudnnBackendPopulateCudaGraph(handle, plan.plan_desc, variant_a, graph));
  launch_graph_and_check(graph, input_a_h, output_a);
  CUDA_CALL(cudaGraphDestroy(graph));

  cudaGraph_t updated_graph = nullptr;
  CUDA_CALL(cudaGraphCreate(&updated_graph, 0));
  CUDNN_CALL(cudnnBackendPopulateCudaGraph(handle, plan.plan_desc, variant_a,
                                           updated_graph));
  CUDNN_CALL(cudnnBackendUpdateCudaGraph(handle, plan.plan_desc, variant_b,
                                         updated_graph));
  launch_graph_and_check(updated_graph, input_b_h, output_b);
  CUDA_CALL(cudaGraphDestroy(updated_graph));

  CUDNN_CALL(cudnnBackendDestroyDescriptor(variant_b));
  CUDNN_CALL(cudnnBackendDestroyDescriptor(variant_a));
  if (workspace != nullptr) {
    CUDA_CALL(cudaFree(workspace));
  }
  CUDA_CALL(cudaFree(output_b));
  CUDA_CALL(cudaFree(input_b));
  CUDA_CALL(cudaFree(output_a));
  CUDA_CALL(cudaFree(input_a));
  destroy_plan(plan);
  CUDNN_CALL(cudnnDestroy(handle));

  std::cout << "cuDNN backend CUDA graph coverage ok" << std::endl;
  return 0;
}
