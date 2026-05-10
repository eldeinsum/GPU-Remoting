#include <cuda_runtime_api.h>
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int check_cuda(cudaError_t result, const char *label) {
  if (result != cudaSuccess) {
    fprintf(stderr, "%s failed: %s\n", label, cudaGetErrorString(result));
    return 1;
  }
  return 0;
}

static int check_nvml(nvmlReturn_t result, const char *label) {
  if (result != NVML_SUCCESS) {
    fprintf(stderr, "%s failed: %s\n", label, nvmlErrorString(result));
    return 1;
  }
  return 0;
}

static int child_main() {
  if (check_nvml(nvmlInit_v2(), "child nvmlInit_v2")) {
    return 10;
  }

  unsigned int count = 0;
  if (check_nvml(nvmlDeviceGetCount_v2(&count), "child nvmlDeviceGetCount_v2")) {
    return 11;
  }
  if (count == 0) {
    fprintf(stderr, "child saw no NVML devices\n");
    return 12;
  }

  if (check_cuda(cudaSetDevice(0), "child cudaSetDevice")) {
    return 13;
  }
  if (check_cuda(cudaFree(nullptr), "child cudaFree")) {
    return 14;
  }

  if (check_nvml(nvmlShutdown(), "child nvmlShutdown")) {
    return 15;
  }
  return 0;
}

int main() {
  const char *preload = getenv("LD_PRELOAD");
  if (preload == NULL || strstr(preload, "libclient.so") == NULL) {
    printf("fork after CUDA init test skipped without GPU-Remoting preload\n");
    return 0;
  }

  if (check_cuda(cudaSetDevice(0), "parent cudaSetDevice")) {
    return 1;
  }
  if (check_cuda(cudaFree(nullptr), "parent cudaFree before fork")) {
    return 2;
  }

  pid_t pid = fork();
  if (pid < 0) {
    perror("fork");
    return 3;
  }
  if (pid == 0) {
    exit(child_main());
  }

  int status = 0;
  if (waitpid(pid, &status, 0) != pid) {
    perror("waitpid");
    return 4;
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    fprintf(stderr, "child failed with status %d\n", status);
    return 5;
  }

  if (check_cuda(cudaFree(nullptr), "parent cudaFree after fork")) {
    return 6;
  }

  printf("fork after CUDA init test passed\n");
  return 0;
}
