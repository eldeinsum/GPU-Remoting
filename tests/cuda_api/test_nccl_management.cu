#include <cuda_runtime.h>
#include <nccl.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool cuda_ok(cudaError_t status, const char *call) {
  if (status == cudaSuccess) {
    return true;
  }
  fprintf(stderr, "%s failed: %s\n", call, cudaGetErrorString(status));
  return false;
}

static bool nccl_ok(ncclResult_t status, const char *call) {
  if (status == ncclSuccess) {
    return true;
  }
  fprintf(stderr, "%s failed with NCCL status %d\n", call, (int)status);
  return false;
}

static bool check_values(const char *label, int rank, const float *actual,
                         const float *expected, int count) {
  for (int i = 0; i < count; ++i) {
    if (fabsf(actual[i] - expected[i]) > 1e-4f) {
      fprintf(stderr, "%s rank %d mismatch at %d: got %.3f expected %.3f\n",
              label, rank, i, actual[i], expected[i]);
      return false;
    }
  }
  return true;
}

static bool sync_all(const int *devs, cudaStream_t *streams, int nranks) {
  for (int rank = 0; rank < nranks; ++rank) {
    if (!cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") ||
        !cuda_ok(cudaStreamSynchronize(streams[rank]),
                 "cudaStreamSynchronize")) {
      return false;
    }
  }
  return true;
}

static bool allreduce_sum(const int *devs, ncclComm_t *comms,
                          cudaStream_t *streams, float **send, float **recv,
                          ncclRedOp_t *ops, int count) {
  if (!nccl_ok(ncclGroupStart(), "ncclGroupStart")) {
    return false;
  }
  for (int rank = 0; rank < 2; ++rank) {
    if (!cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") ||
        !nccl_ok(ncclAllReduce(send[rank], recv[rank], count, ncclFloat,
                               ops[rank], comms[rank], streams[rank]),
                 "ncclAllReduce")) {
      return false;
    }
  }
  return nccl_ok(ncclGroupEnd(), "ncclGroupEnd") &&
         sync_all(devs, streams, 2);
}

int main(void) {
  int version = 0;
  if (!nccl_ok(ncclGetVersion(&version), "ncclGetVersion") || version <= 0) {
    return EXIT_FAILURE;
  }
  const char *invalid_text = ncclGetErrorString(ncclInvalidArgument);
  if (invalid_text == NULL || strlen(invalid_text) == 0) {
    fprintf(stderr, "ncclGetErrorString returned empty text\n");
    return EXIT_FAILURE;
  }

  int device_count = 0;
  if (!cuda_ok(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) {
    return EXIT_FAILURE;
  }
  if (device_count < 2) {
    printf("NCCL management test skipped; need at least two CUDA devices\n");
    return EXIT_SUCCESS;
  }

  void *nccl_alloc = NULL;
  if (!cuda_ok(cudaSetDevice(0), "cudaSetDevice") ||
      !nccl_ok(ncclMemAlloc(&nccl_alloc, 256), "ncclMemAlloc") ||
      nccl_alloc == NULL ||
      !cuda_ok(cudaMemset(nccl_alloc, 0, 256), "cudaMemset(ncclMemAlloc)") ||
      !nccl_ok(ncclMemFree(nccl_alloc), "ncclMemFree")) {
    return EXIT_FAILURE;
  }

  const int nranks = 2;
  const int devs[nranks] = {0, 1};
  const int elem_count = 4;
  ncclComm_t revoke_comms[nranks];
  if (!nccl_ok(ncclCommInitAll(revoke_comms, nranks, devs),
               "ncclCommInitAll(revoke)")) {
    return EXIT_FAILURE;
  }
  if (!nccl_ok(ncclGroupStart(), "ncclGroupStart(revoke)")) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    if (!nccl_ok(ncclCommRevoke(revoke_comms[rank], NCCL_REVOKE_DEFAULT),
                 "ncclCommRevoke")) {
      return EXIT_FAILURE;
    }
  }
  if (!nccl_ok(ncclGroupEnd(), "ncclGroupEnd(revoke)")) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    ncclResult_t revoke_state = ncclInProgress;
    for (int attempt = 0; attempt < 100 && revoke_state == ncclInProgress;
         ++attempt) {
      if (!nccl_ok(ncclCommGetAsyncError(revoke_comms[rank], &revoke_state),
                   "ncclCommGetAsyncError(revoke)")) {
        return EXIT_FAILURE;
      }
    }
    if (revoke_state != ncclSuccess) {
      fprintf(stderr, "NCCL revoke did not quiesce for rank %d\n", rank);
      return EXIT_FAILURE;
    }
    if (!nccl_ok(ncclCommDestroy(revoke_comms[rank]),
                 "ncclCommDestroy(revoke)")) {
      return EXIT_FAILURE;
    }
  }

  ncclComm_t comms[nranks];
  if (!nccl_ok(ncclCommInitAll(comms, nranks, devs), "ncclCommInitAll")) {
    return EXIT_FAILURE;
  }
  if (!nccl_ok(ncclGroupStart(), "ncclGroupStart(suspend)")) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    if (!nccl_ok(ncclCommSuspend(comms[rank], NCCL_SUSPEND_MEM),
                 "ncclCommSuspend")) {
      return EXIT_FAILURE;
    }
  }
  if (!nccl_ok(ncclGroupEnd(), "ncclGroupEnd(suspend)")) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    uint64_t suspended = 0;
    if (!nccl_ok(ncclCommMemStats(comms[rank], ncclStatGpuMemSuspended,
                                  &suspended),
                 "ncclCommMemStats(suspended)")) {
      return EXIT_FAILURE;
    }
  }
  if (!nccl_ok(ncclGroupStart(), "ncclGroupStart(resume)")) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    if (!nccl_ok(ncclCommResume(comms[rank]), "ncclCommResume")) {
      return EXIT_FAILURE;
    }
  }
  if (!nccl_ok(ncclGroupEnd(), "ncclGroupEnd(resume)")) {
    return EXIT_FAILURE;
  }

  cudaStream_t streams[nranks];
  float *send[nranks];
  float *recv[nranks];
  void *registration_handles[nranks] = {NULL, NULL};
  for (int rank = 0; rank < nranks; ++rank) {
    if (!cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") ||
        !cuda_ok(cudaStreamCreate(&streams[rank]), "cudaStreamCreate") ||
        !cuda_ok(cudaMalloc(&send[rank], elem_count * sizeof(float)),
                 "cudaMalloc(send)") ||
        !cuda_ok(cudaMalloc(&recv[rank], elem_count * sizeof(float)),
                 "cudaMalloc(recv)") ||
        !nccl_ok(ncclCommRegister(comms[rank], send[rank],
                                  elem_count * sizeof(float),
                                  &registration_handles[rank]),
                 "ncclCommRegister")) {
      return EXIT_FAILURE;
    }
    if (registration_handles[rank] == NULL) {
      fprintf(stderr, "ncclCommRegister returned a null handle\n");
      return EXIT_FAILURE;
    }
  }
  for (int rank = 0; rank < nranks; ++rank) {
    if (!nccl_ok(ncclCommDeregister(comms[rank], registration_handles[rank]),
                 "ncclCommDeregister")) {
      return EXIT_FAILURE;
    }
  }

  ncclUniqueId scalable_id;
  ncclComm_t scalable_comms[nranks];
  ncclConfig_t scalable_config = NCCL_CONFIG_INITIALIZER;
  if (!nccl_ok(ncclGetUniqueId(&scalable_id), "ncclGetUniqueId") ||
      !nccl_ok(ncclGroupStart(), "ncclGroupStart")) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    if (!cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") ||
        !nccl_ok(ncclCommInitRankScalable(&scalable_comms[rank], nranks, rank,
                                          1, &scalable_id, &scalable_config),
                 "ncclCommInitRankScalable")) {
      return EXIT_FAILURE;
    }
  }
  if (!nccl_ok(ncclGroupEnd(), "ncclGroupEnd")) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    if (!nccl_ok(ncclCommDestroy(scalable_comms[rank]),
                 "ncclCommDestroy(scalable)")) {
      return EXIT_FAILURE;
    }
  }

  const float rank0_values[elem_count] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float rank1_values[elem_count] = {10.0f, 20.0f, 30.0f, 40.0f};
  const float sum_expected[elem_count] = {11.0f, 22.0f, 33.0f, 44.0f};
  for (int rank = 0; rank < nranks; ++rank) {
    const float *values = rank == 0 ? rank0_values : rank1_values;
    if (!cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") ||
        !cuda_ok(cudaMemcpy(send[rank], values, elem_count * sizeof(float),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy H2D") ||
        !cuda_ok(cudaMemset(recv[rank], 0, elem_count * sizeof(float)),
                 "cudaMemset")) {
      return EXIT_FAILURE;
    }
  }

  ncclSimInfo_t sim_info = NCCL_SIM_INFO_INITIALIZER;
  if (!nccl_ok(ncclGroupStart(), "ncclGroupStart(simulate)")) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    if (!cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") ||
        !nccl_ok(ncclAllReduce(send[rank], recv[rank], elem_count, ncclFloat,
                               ncclSum, comms[rank], streams[rank]),
                 "ncclAllReduce(simulate)")) {
      return EXIT_FAILURE;
    }
  }
  if (!nccl_ok(ncclGroupSimulateEnd(&sim_info), "ncclGroupSimulateEnd") ||
      sim_info.size != sizeof(ncclSimInfo_t)) {
    fprintf(stderr, "invalid NCCL simulation info\n");
    return EXIT_FAILURE;
  }

  ncclRedOp_t sum_ops[nranks] = {ncclSum, ncclSum};
  if (!allreduce_sum(devs, comms, streams, send, recv, sum_ops,
                     elem_count)) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    float actual[elem_count] = {0.0f};
    if (!cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") ||
        !cuda_ok(cudaMemcpy(actual, recv[rank], elem_count * sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy D2H") ||
        !check_values("allreduce", rank, actual, sum_expected,
                      elem_count) ||
        !cuda_ok(cudaMemset(recv[rank], 0, elem_count * sizeof(float)),
                 "cudaMemset")) {
      return EXIT_FAILURE;
    }
  }

  ncclRedOp_t premul_ops[nranks];
  float scale = 2.0f;
  for (int rank = 0; rank < nranks; ++rank) {
    if (!cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") ||
        !nccl_ok(ncclRedOpCreatePreMulSum(&premul_ops[rank], &scale,
                                          ncclFloat,
                                          ncclScalarHostImmediate,
                                          comms[rank]),
                 "ncclRedOpCreatePreMulSum")) {
      return EXIT_FAILURE;
    }
  }
  if (!allreduce_sum(devs, comms, streams, send, recv, premul_ops,
                     elem_count)) {
    return EXIT_FAILURE;
  }
  const float premul_expected[elem_count] = {22.0f, 44.0f, 66.0f, 88.0f};
  for (int rank = 0; rank < nranks; ++rank) {
    float actual[elem_count] = {0.0f};
    if (!cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") ||
        !cuda_ok(cudaMemcpy(actual, recv[rank], elem_count * sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy D2H") ||
        !check_values("premul allreduce", rank, actual, premul_expected,
                      elem_count) ||
        !nccl_ok(ncclRedOpDestroy(premul_ops[rank], comms[rank]),
                 "ncclRedOpDestroy")) {
      return EXIT_FAILURE;
    }
  }

  for (int rank = 0; rank < nranks; ++rank) {
    uint64_t mem_total = 0;
    if (!nccl_ok(ncclCommMemStats(comms[rank], ncclStatGpuMemTotal,
                                  &mem_total),
                 "ncclCommMemStats")) {
      return EXIT_FAILURE;
    }
  }

  for (int rank = 0; rank < nranks; ++rank) {
    if (!cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice")) {
      return EXIT_FAILURE;
    }
    cudaFree(send[rank]);
    cudaFree(recv[rank]);
    cudaStreamDestroy(streams[rank]);
    ncclCommDestroy(comms[rank]);
  }

  printf("NCCL management coverage test passed\n");
  return EXIT_SUCCESS;
}
