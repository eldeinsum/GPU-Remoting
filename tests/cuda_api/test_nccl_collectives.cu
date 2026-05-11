#include <cuda_runtime.h>
#include <nccl.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

int main(void) {
  int device_count = 0;
  if (!cuda_ok(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) {
    return EXIT_FAILURE;
  }
  if (device_count < 2) {
    printf("NCCL collective test skipped; need at least two CUDA devices\n");
    return EXIT_SUCCESS;
  }

  const int nranks = 2;
  const int devs[nranks] = {0, 1};
  const int max_elems = 8;

  ncclComm_t comms[nranks];
  if (!nccl_ok(ncclCommInitAll(comms, nranks, devs), "ncclCommInitAll")) {
    return EXIT_FAILURE;
  }

  ncclUniqueId unique_id;
  if (!nccl_ok(ncclCommGetUniqueId(comms[0], &unique_id),
               "ncclCommGetUniqueId")) {
    return EXIT_FAILURE;
  }

  for (int rank = 0; rank < nranks; ++rank) {
    int count = 0;
    int user_rank = -1;
    int cu_device = -1;
    if (!nccl_ok(ncclCommCount(comms[rank], &count), "ncclCommCount") ||
        !nccl_ok(ncclCommUserRank(comms[rank], &user_rank),
                 "ncclCommUserRank") ||
        !nccl_ok(ncclCommCuDevice(comms[rank], &cu_device),
                 "ncclCommCuDevice")) {
      return EXIT_FAILURE;
    }
    if (count != nranks || user_rank != rank || cu_device != devs[rank]) {
      fprintf(stderr, "unexpected communicator metadata for rank %d\n", rank);
      return EXIT_FAILURE;
    }
  }

  cudaStream_t streams[nranks];
  float *send[nranks];
  float *recv[nranks];
  for (int rank = 0; rank < nranks; ++rank) {
    if (!cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") ||
        !cuda_ok(cudaStreamCreate(&streams[rank]), "cudaStreamCreate") ||
        !cuda_ok(cudaMalloc(&send[rank], max_elems * sizeof(float)),
                 "cudaMalloc(send)") ||
        !cuda_ok(cudaMalloc(&recv[rank], max_elems * sizeof(float)),
                 "cudaMalloc(recv)")) {
      return EXIT_FAILURE;
    }
  }

  auto copy_to_device = [&](int rank, const float *values, int count) -> bool {
    return cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") &&
           cuda_ok(cudaMemcpy(send[rank], values, count * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D");
  };
  auto clear_recv = [&](int rank) -> bool {
    return cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") &&
           cuda_ok(cudaMemset(recv[rank], 0, max_elems * sizeof(float)),
                   "cudaMemset");
  };
  auto sync_all = [&]() -> bool {
    for (int rank = 0; rank < nranks; ++rank) {
      if (!cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") ||
          !cuda_ok(cudaStreamSynchronize(streams[rank]),
                   "cudaStreamSynchronize")) {
        return false;
      }
    }
    return true;
  };
  auto read_recv = [&](int rank, float *values, int count) -> bool {
    return cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") &&
           cuda_ok(cudaMemcpy(values, recv[rank], count * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H");
  };

  const float broadcast_root[max_elems] = {7.0f, 8.0f, 9.0f, 10.0f};
  const float zeros[max_elems] = {0.0f};
  for (int rank = 0; rank < nranks; ++rank) {
    if (!copy_to_device(rank, rank == 0 ? broadcast_root : zeros, 4) ||
        !clear_recv(rank)) {
      return EXIT_FAILURE;
    }
  }
  if (!nccl_ok(ncclGroupStart(), "ncclGroupStart")) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    if (!cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") ||
        !nccl_ok(ncclBroadcast(send[rank], recv[rank], 4, ncclFloat, 0,
                               comms[rank], streams[rank]),
                 "ncclBroadcast")) {
      return EXIT_FAILURE;
    }
  }
  if (!nccl_ok(ncclGroupEnd(), "ncclGroupEnd") || !sync_all()) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    float actual[max_elems] = {0.0f};
    if (!read_recv(rank, actual, 4) ||
        !check_values("broadcast", rank, actual, broadcast_root, 4)) {
      return EXIT_FAILURE;
    }
  }

  const float reduce_rank0[max_elems] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float reduce_rank1[max_elems] = {10.0f, 20.0f, 30.0f, 40.0f};
  const float reduce_expected[max_elems] = {11.0f, 22.0f, 33.0f, 44.0f};
  if (!copy_to_device(0, reduce_rank0, 4) ||
      !copy_to_device(1, reduce_rank1, 4) || !clear_recv(0) ||
      !clear_recv(1)) {
    return EXIT_FAILURE;
  }
  if (!nccl_ok(ncclGroupStart(), "ncclGroupStart")) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    if (!cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") ||
        !nccl_ok(ncclReduce(send[rank], recv[rank], 4, ncclFloat, ncclSum, 1,
                            comms[rank], streams[rank]),
                 "ncclReduce")) {
      return EXIT_FAILURE;
    }
  }
  if (!nccl_ok(ncclGroupEnd(), "ncclGroupEnd") || !sync_all()) {
    return EXIT_FAILURE;
  }
  float reduce_actual[max_elems] = {0.0f};
  if (!read_recv(1, reduce_actual, 4) ||
      !check_values("reduce", 1, reduce_actual, reduce_expected, 4)) {
    return EXIT_FAILURE;
  }

  if (!copy_to_device(0, reduce_rank0, 4) ||
      !copy_to_device(1, reduce_rank1, 4) || !clear_recv(0) ||
      !clear_recv(1)) {
    return EXIT_FAILURE;
  }
  if (!nccl_ok(ncclGroupStart(), "ncclGroupStart")) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    if (!cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") ||
        !nccl_ok(ncclReduceScatter(send[rank], recv[rank], 2, ncclFloat,
                                   ncclSum, comms[rank], streams[rank]),
                 "ncclReduceScatter")) {
      return EXIT_FAILURE;
    }
  }
  if (!nccl_ok(ncclGroupEnd(), "ncclGroupEnd") || !sync_all()) {
    return EXIT_FAILURE;
  }
  const float reduce_scatter_rank0[2] = {11.0f, 22.0f};
  const float reduce_scatter_rank1[2] = {33.0f, 44.0f};
  for (int rank = 0; rank < nranks; ++rank) {
    float actual[2] = {0.0f};
    const float *expected =
        rank == 0 ? reduce_scatter_rank0 : reduce_scatter_rank1;
    if (!read_recv(rank, actual, 2) ||
        !check_values("reduce_scatter", rank, actual, expected, 2)) {
      return EXIT_FAILURE;
    }
  }

  const float alltoall_rank0[max_elems] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float alltoall_rank1[max_elems] = {10.0f, 20.0f, 30.0f, 40.0f};
  const float alltoall_expected0[max_elems] = {1.0f, 2.0f, 10.0f, 20.0f};
  const float alltoall_expected1[max_elems] = {3.0f, 4.0f, 30.0f, 40.0f};
  if (!copy_to_device(0, alltoall_rank0, 4) ||
      !copy_to_device(1, alltoall_rank1, 4) || !clear_recv(0) ||
      !clear_recv(1)) {
    return EXIT_FAILURE;
  }
  if (!nccl_ok(ncclGroupStart(), "ncclGroupStart")) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    if (!cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") ||
        !nccl_ok(ncclAlltoAll(send[rank], recv[rank], 2, ncclFloat,
                              comms[rank], streams[rank]),
                 "ncclAlltoAll")) {
      return EXIT_FAILURE;
    }
  }
  if (!nccl_ok(ncclGroupEnd(), "ncclGroupEnd") || !sync_all()) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    float actual[max_elems] = {0.0f};
    const float *expected =
        rank == 0 ? alltoall_expected0 : alltoall_expected1;
    if (!read_recv(rank, actual, 4) ||
        !check_values("alltoall", rank, actual, expected, 4)) {
      return EXIT_FAILURE;
    }
  }

  const float gather_rank0[max_elems] = {5.0f, 6.0f};
  const float gather_rank1[max_elems] = {7.0f, 8.0f};
  const float gather_expected[max_elems] = {5.0f, 6.0f, 7.0f, 8.0f};
  if (!copy_to_device(0, gather_rank0, 2) ||
      !copy_to_device(1, gather_rank1, 2) || !clear_recv(0) ||
      !clear_recv(1)) {
    return EXIT_FAILURE;
  }
  if (!nccl_ok(ncclGroupStart(), "ncclGroupStart")) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    if (!cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") ||
        !nccl_ok(ncclGather(send[rank], recv[rank], 2, ncclFloat, 0,
                            comms[rank], streams[rank]),
                 "ncclGather")) {
      return EXIT_FAILURE;
    }
  }
  if (!nccl_ok(ncclGroupEnd(), "ncclGroupEnd") || !sync_all()) {
    return EXIT_FAILURE;
  }
  float gather_actual[max_elems] = {0.0f};
  if (!read_recv(0, gather_actual, 4) ||
      !check_values("gather", 0, gather_actual, gather_expected, 4)) {
    return EXIT_FAILURE;
  }

  const float scatter_root[max_elems] = {11.0f, 12.0f, 13.0f, 14.0f};
  const float scatter_expected0[2] = {11.0f, 12.0f};
  const float scatter_expected1[2] = {13.0f, 14.0f};
  if (!copy_to_device(0, zeros, 4) || !copy_to_device(1, scatter_root, 4) ||
      !clear_recv(0) || !clear_recv(1)) {
    return EXIT_FAILURE;
  }
  if (!nccl_ok(ncclGroupStart(), "ncclGroupStart")) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    if (!cuda_ok(cudaSetDevice(devs[rank]), "cudaSetDevice") ||
        !nccl_ok(ncclScatter(send[rank], recv[rank], 2, ncclFloat, 1,
                             comms[rank], streams[rank]),
                 "ncclScatter")) {
      return EXIT_FAILURE;
    }
  }
  if (!nccl_ok(ncclGroupEnd(), "ncclGroupEnd") || !sync_all()) {
    return EXIT_FAILURE;
  }
  for (int rank = 0; rank < nranks; ++rank) {
    float actual[2] = {0.0f};
    const float *expected = rank == 0 ? scatter_expected0 : scatter_expected1;
    if (!read_recv(rank, actual, 2) ||
        !check_values("scatter", rank, actual, expected, 2)) {
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

  printf("NCCL collective coverage test passed\n");
  return EXIT_SUCCESS;
}
