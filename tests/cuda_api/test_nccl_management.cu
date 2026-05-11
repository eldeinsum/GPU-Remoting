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
  for (int rank = 0; rank < nranks; ++rank) {
    const char *last_error = ncclGetLastError(comms[rank]);
    if (last_error == NULL) {
      fprintf(stderr, "ncclGetLastError returned null text\n");
      return EXIT_FAILURE;
    }
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
  void *rma_source_buffers[nranks] = {NULL, NULL};
  void *window_buffers[nranks] = {NULL, NULL};
  void *registration_handles[nranks] = {NULL, NULL};
  ncclWindow_t rma_source_windows[nranks] = {NULL, NULL};
  ncclWindow_t windows[nranks] = {NULL, NULL};
  for (int rank = 0; rank < nranks; ++rank) {
    if (!cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") ||
        !cuda_ok(cudaStreamCreate(&streams[rank]), "cudaStreamCreate") ||
        !cuda_ok(cudaMalloc(&send[rank], elem_count * sizeof(float)),
                 "cudaMalloc(send)") ||
        !cuda_ok(cudaMalloc(&recv[rank], elem_count * sizeof(float)),
                 "cudaMalloc(recv)") ||
        !nccl_ok(ncclMemAlloc(&rma_source_buffers[rank],
                              NCCL_WIN_REQUIRED_ALIGNMENT),
                 "ncclMemAlloc(rma source)") ||
        rma_source_buffers[rank] == NULL ||
        !cuda_ok(cudaMemset(rma_source_buffers[rank], 0,
                            NCCL_WIN_REQUIRED_ALIGNMENT),
                 "cudaMemset(rma source)") ||
        !nccl_ok(ncclMemAlloc(&window_buffers[rank],
                              NCCL_WIN_REQUIRED_ALIGNMENT),
                 "ncclMemAlloc(window)") ||
        window_buffers[rank] == NULL ||
        !cuda_ok(cudaMemset(window_buffers[rank], 0,
                            NCCL_WIN_REQUIRED_ALIGNMENT),
                 "cudaMemset(window)") ||
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
  if (!nccl_ok(ncclGroupStart(), "ncclGroupStart(window register)")) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    if (!cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") ||
        !nccl_ok(ncclCommWindowRegister(
                     comms[rank], rma_source_buffers[rank],
                     NCCL_WIN_REQUIRED_ALIGNMENT, &rma_source_windows[rank],
                     NCCL_WIN_COLL_SYMMETRIC),
                 "ncclCommWindowRegister(source)") ||
        !nccl_ok(ncclCommWindowRegister(
                     comms[rank], window_buffers[rank],
                     NCCL_WIN_REQUIRED_ALIGNMENT, &windows[rank],
                     NCCL_WIN_COLL_SYMMETRIC),
                 "ncclCommWindowRegister")) {
      return EXIT_FAILURE;
    }
  }
  if (!nccl_ok(ncclGroupEnd(), "ncclGroupEnd(window register)")) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    void *user_ptr = NULL;
    if (rma_source_windows[rank] == NULL ||
        !nccl_ok(ncclWinGetUserPtr(comms[rank], rma_source_windows[rank],
                                   &user_ptr),
                 "ncclWinGetUserPtr(source)") ||
        user_ptr != rma_source_buffers[rank]) {
      fprintf(stderr, "NCCL source window user pointer mismatch for rank %d\n",
              rank);
      return EXIT_FAILURE;
    }
    user_ptr = NULL;
    if (windows[rank] == NULL ||
        !nccl_ok(ncclWinGetUserPtr(comms[rank], windows[rank], &user_ptr),
                 "ncclWinGetUserPtr") ||
        user_ptr != window_buffers[rank]) {
      fprintf(stderr, "NCCL window user pointer mismatch for rank %d\n", rank);
      return EXIT_FAILURE;
    }
  }

  ncclWaitSignalDesc_t signal_wait_from_rank1 = {1, 1, 0, 0};
  ncclWaitSignalDesc_t signal_wait_from_rank0 = {1, 0, 0, 0};
  if (!nccl_ok(ncclGroupStart(), "ncclGroupStart(signal)")) {
    return EXIT_FAILURE;
  }
  if (!cuda_ok(cudaSetDevice(devs[0]), "cudaSetDevice") ||
      !nccl_ok(ncclSignal(1, 0, 0, 0, comms[0], streams[0]),
               "ncclSignal(rank0)") ||
      !nccl_ok(ncclWaitSignal(1, &signal_wait_from_rank1, comms[0],
                              streams[0]),
               "ncclWaitSignal(signal rank0)") ||
      !cuda_ok(cudaSetDevice(devs[1]), "cudaSetDevice") ||
      !nccl_ok(ncclSignal(0, 0, 0, 0, comms[1], streams[1]),
               "ncclSignal(rank1)") ||
      !nccl_ok(ncclWaitSignal(1, &signal_wait_from_rank0, comms[1],
                              streams[1]),
               "ncclWaitSignal(signal rank1)")) {
    return EXIT_FAILURE;
  }
  if (!nccl_ok(ncclGroupEnd(), "ncclGroupEnd(signal)") ||
      !sync_all(devs, streams, nranks)) {
    return EXIT_FAILURE;
  }

  const float signal_rank0_values[elem_count] = {5.0f, 6.0f, 7.0f, 8.0f};
  const float signal_rank1_values[elem_count] = {50.0f, 60.0f, 70.0f, 80.0f};
  if (!cuda_ok(cudaSetDevice(devs[0]), "cudaSetDevice") ||
      !cuda_ok(cudaMemcpy(rma_source_buffers[0], signal_rank0_values,
                          elem_count * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy H2D(signal rank0)") ||
      !cuda_ok(cudaSetDevice(devs[1]), "cudaSetDevice") ||
      !cuda_ok(cudaMemcpy(rma_source_buffers[1], signal_rank1_values,
                          elem_count * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy H2D(signal rank1)") ||
      !cuda_ok(cudaSetDevice(devs[0]), "cudaSetDevice") ||
      !cuda_ok(cudaMemset(window_buffers[0], 0, elem_count * sizeof(float)),
               "cudaMemset(window signal rank0)") ||
      !cuda_ok(cudaSetDevice(devs[1]), "cudaSetDevice") ||
      !cuda_ok(cudaMemset(window_buffers[1], 0, elem_count * sizeof(float)),
               "cudaMemset(window signal rank1)")) {
    return EXIT_FAILURE;
  }

  ncclWaitSignalDesc_t wait_from_rank1 = {1, 1, 0, 0};
  ncclWaitSignalDesc_t wait_from_rank0 = {1, 0, 0, 0};
  if (!nccl_ok(ncclGroupStart(), "ncclGroupStart(put signal)")) {
    return EXIT_FAILURE;
  }
  if (!cuda_ok(cudaSetDevice(devs[0]), "cudaSetDevice") ||
      !nccl_ok(ncclWaitSignal(1, &wait_from_rank1, comms[0], streams[0]),
               "ncclWaitSignal(put rank0)") ||
      !nccl_ok(ncclPutSignal(rma_source_buffers[0], elem_count, ncclFloat, 1,
                             windows[0], 0, 0, 0, 0, comms[0], streams[0]),
               "ncclPutSignal(rank0)") ||
      !cuda_ok(cudaSetDevice(devs[1]), "cudaSetDevice") ||
      !nccl_ok(ncclPutSignal(rma_source_buffers[1], elem_count, ncclFloat, 0,
                             windows[1], 0, 0, 0, 0, comms[1], streams[1]),
               "ncclPutSignal(rank1)") ||
      !nccl_ok(ncclWaitSignal(1, &wait_from_rank0, comms[1], streams[1]),
               "ncclWaitSignal(put rank1)")) {
    return EXIT_FAILURE;
  }
  if (!nccl_ok(ncclGroupEnd(), "ncclGroupEnd(put signal)") ||
      !sync_all(devs, streams, nranks)) {
    return EXIT_FAILURE;
  }

  float signal_rank0_actual[elem_count] = {0.0f};
  float signal_rank1_actual[elem_count] = {0.0f};
  if (!cuda_ok(cudaSetDevice(devs[0]), "cudaSetDevice") ||
      !cuda_ok(cudaMemcpy(signal_rank0_actual, window_buffers[0],
                          elem_count * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H(signal rank0)") ||
      !check_values("put signal", 0, signal_rank0_actual,
                    signal_rank1_values, elem_count) ||
      !cuda_ok(cudaSetDevice(devs[1]), "cudaSetDevice") ||
      !cuda_ok(cudaMemcpy(signal_rank1_actual, window_buffers[1],
                          elem_count * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H(signal rank1)") ||
      !check_values("put signal", 1, signal_rank1_actual,
                    signal_rank0_values, elem_count)) {
    return EXIT_FAILURE;
  }

  if (!nccl_ok(ncclGroupStart(), "ncclGroupStart(window deregister)")) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    if (!nccl_ok(ncclCommWindowDeregister(comms[rank],
                                          rma_source_windows[rank]),
                 "ncclCommWindowDeregister(source)") ||
        !nccl_ok(ncclCommWindowDeregister(comms[rank], windows[rank]),
                 "ncclCommWindowDeregister")) {
      return EXIT_FAILURE;
    }
  }
  if (!nccl_ok(ncclGroupEnd(), "ncclGroupEnd(window deregister)")) {
    return EXIT_FAILURE;
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

  ncclRedOp_t sum_ops[nranks] = {ncclSum, ncclSum};
  ncclComm_t split_comms[nranks] = {NULL, NULL};
  if (!nccl_ok(ncclGroupStart(), "ncclGroupStart(split)")) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    if (!cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") ||
        !nccl_ok(ncclCommSplit(comms[rank], 0, rank, &split_comms[rank],
                               NULL),
                 "ncclCommSplit")) {
      return EXIT_FAILURE;
    }
  }
  if (!nccl_ok(ncclGroupEnd(), "ncclGroupEnd(split)")) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    int split_count = 0;
    int split_rank = -1;
    if (!nccl_ok(ncclCommCount(split_comms[rank], &split_count),
                 "ncclCommCount(split)") ||
        !nccl_ok(ncclCommUserRank(split_comms[rank], &split_rank),
                 "ncclCommUserRank(split)") ||
        split_count != nranks || split_rank != rank) {
      fprintf(stderr, "NCCL split communicator mismatch for rank %d\n", rank);
      return EXIT_FAILURE;
    }
  }
  if (!allreduce_sum(devs, split_comms, streams, send, recv, sum_ops,
                     elem_count)) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    float actual[elem_count] = {0.0f};
    if (!cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") ||
        !cuda_ok(cudaMemcpy(actual, recv[rank], elem_count * sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy D2H(split)") ||
        !check_values("split allreduce", rank, actual, sum_expected,
                      elem_count) ||
        !cuda_ok(cudaMemset(recv[rank], 0, elem_count * sizeof(float)),
                 "cudaMemset(split)")) {
      return EXIT_FAILURE;
    }
  }
  for (int rank = 0; rank < nranks; ++rank) {
    if (!nccl_ok(ncclCommDestroy(split_comms[rank]),
                 "ncclCommDestroy(split)")) {
      return EXIT_FAILURE;
    }
  }

  int shrink_exclude[1] = {1};
  ncclComm_t shrink_comm = NULL;
  if (!cuda_ok(cudaSetDevice(devs[0]), "cudaSetDevice") ||
      !nccl_ok(ncclCommShrink(comms[0], shrink_exclude, 1, &shrink_comm,
                              NULL, NCCL_SHRINK_DEFAULT),
               "ncclCommShrink")) {
    return EXIT_FAILURE;
  }
  int shrink_count = 0;
  int shrink_rank = -1;
  if (!nccl_ok(ncclCommCount(shrink_comm, &shrink_count),
               "ncclCommCount(shrink)") ||
      !nccl_ok(ncclCommUserRank(shrink_comm, &shrink_rank),
               "ncclCommUserRank(shrink)") ||
      shrink_count != 1 || shrink_rank != 0) {
    fprintf(stderr, "NCCL shrink communicator mismatch\n");
    return EXIT_FAILURE;
  }

  ncclUniqueId grow_id;
  ncclComm_t grow_comms[nranks] = {NULL, NULL};
  if (!nccl_ok(ncclCommGetUniqueId(shrink_comm, &grow_id),
               "ncclCommGetUniqueId(grow)") ||
      !nccl_ok(ncclGroupStart(), "ncclGroupStart(grow)")) {
    return EXIT_FAILURE;
  }
  if (!cuda_ok(cudaSetDevice(devs[0]), "cudaSetDevice") ||
      !nccl_ok(ncclCommGrow(shrink_comm, nranks, &grow_id, -1,
                            &grow_comms[0], NULL),
               "ncclCommGrow(existing)") ||
      !cuda_ok(cudaSetDevice(devs[1]), "cudaSetDevice") ||
      !nccl_ok(ncclCommGrow(NULL, nranks, &grow_id, 1, &grow_comms[1],
                            NULL),
               "ncclCommGrow(new)")) {
    return EXIT_FAILURE;
  }
  if (!nccl_ok(ncclGroupEnd(), "ncclGroupEnd(grow)")) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    int grow_count = 0;
    int grow_rank = -1;
    if (!nccl_ok(ncclCommCount(grow_comms[rank], &grow_count),
                 "ncclCommCount(grow)") ||
        !nccl_ok(ncclCommUserRank(grow_comms[rank], &grow_rank),
                 "ncclCommUserRank(grow)") ||
        grow_count != nranks || grow_rank != rank) {
      fprintf(stderr, "NCCL grow communicator mismatch for rank %d\n", rank);
      return EXIT_FAILURE;
    }
  }
  if (!allreduce_sum(devs, grow_comms, streams, send, recv, sum_ops,
                     elem_count)) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    float actual[elem_count] = {0.0f};
    if (!cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") ||
        !cuda_ok(cudaMemcpy(actual, recv[rank], elem_count * sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy D2H(grow)") ||
        !check_values("grow allreduce", rank, actual, sum_expected,
                      elem_count) ||
        !cuda_ok(cudaMemset(recv[rank], 0, elem_count * sizeof(float)),
                 "cudaMemset(grow)")) {
      return EXIT_FAILURE;
    }
  }
  for (int rank = 0; rank < nranks; ++rank) {
    if (!nccl_ok(ncclCommDestroy(grow_comms[rank]),
                 "ncclCommDestroy(grow)")) {
      return EXIT_FAILURE;
    }
  }
  if (!nccl_ok(ncclCommDestroy(shrink_comm), "ncclCommDestroy(shrink)")) {
    return EXIT_FAILURE;
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
    ncclMemFree(rma_source_buffers[rank]);
    ncclMemFree(window_buffers[rank]);
    cudaStreamDestroy(streams[rank]);
    ncclCommDestroy(comms[rank]);
  }

  printf("NCCL management coverage test passed\n");
  return EXIT_SUCCESS;
}
