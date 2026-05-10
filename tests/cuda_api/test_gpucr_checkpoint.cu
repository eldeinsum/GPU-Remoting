#include <cuda_runtime.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

static int check(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    std::cerr << what << ": " << cudaGetErrorString(err) << std::endl;
    return 1;
  }
  return 0;
}

int main() {
  constexpr int n = 1 << 20;
  std::vector<unsigned char> input(n);
  std::vector<unsigned char> output(n, 0);
  for (int i = 0; i < n; ++i) {
    input[i] = static_cast<unsigned char>((i * 17 + 3) & 0xff);
  }

  unsigned char *device = nullptr;
  if (check(cudaMalloc(reinterpret_cast<void **>(&device), input.size()),
            "cudaMalloc")) {
    return 1;
  }
  if (check(cudaMemcpy(device, input.data(), input.size(), cudaMemcpyHostToDevice),
            "cudaMemcpyHtoD")) {
    return 1;
  }
  if (check(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
    return 1;
  }

  std::cout << "READY pid=" << getpid() << " ptr=" << static_cast<void *>(device)
            << std::endl;

  const char *hold_seconds = std::getenv("GPU_REMOTING_CHECKPOINT_HOLD");
  if (hold_seconds != nullptr) {
    std::this_thread::sleep_for(std::chrono::seconds(std::atoi(hold_seconds)));
  }

  if (check(cudaMemcpy(output.data(), device, output.size(), cudaMemcpyDeviceToHost),
            "cudaMemcpyDtoH")) {
    return 1;
  }
  for (int i = 0; i < n; ++i) {
    if (input[i] != output[i]) {
      std::cerr << "mismatch at " << i << ": " << static_cast<int>(input[i])
                << " != " << static_cast<int>(output[i]) << std::endl;
      return 1;
    }
  }

  if (check(cudaFree(device), "cudaFree")) {
    return 1;
  }
  std::cout << "OK" << std::endl;
  return 0;
}
