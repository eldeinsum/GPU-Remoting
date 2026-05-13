#include <cstdlib>
#include <iostream>

#include <cudnn.h>

#define CUDNN_CALL(f)                                                          \
  do {                                                                         \
    cudnnStatus_t err = (f);                                                   \
    if (err != CUDNN_STATUS_SUCCESS) {                                         \
      std::cerr << __FILE__ << ":" << __LINE__ << ": cuDNN error " << err     \
                << " (" << cudnnGetErrorString(err) << ")" << std::endl;      \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

void debug_callback(cudnnSeverity_t, void *, const cudnnDebug_t *, const char *) {
}

int main() {
  unsigned mask = 0;
  void *udata = nullptr;
  cudnnCallback_t callback = nullptr;
  CUDNN_CALL(cudnnGetCallback(&mask, &udata, &callback));
  if (mask != 0 || udata != nullptr || callback != nullptr) {
    std::cerr << "unexpected initial cuDNN callback state" << std::endl;
    std::exit(1);
  }

  int cookie = 42;
  const unsigned enabled_mask = CUDNN_SEV_ERROR_EN | CUDNN_SEV_WARNING_EN;
  CUDNN_CALL(cudnnSetCallback(enabled_mask, &cookie, debug_callback));
  CUDNN_CALL(cudnnGetCallback(&mask, &udata, &callback));
  if (mask != enabled_mask || udata != &cookie || callback != debug_callback) {
    std::cerr << "cuDNN callback state did not round trip" << std::endl;
    std::exit(1);
  }

  CUDNN_CALL(cudnnSetCallback(0, nullptr, nullptr));
  CUDNN_CALL(cudnnGetCallback(&mask, &udata, &callback));
  if (mask != CUDNN_SEV_ERROR_EN || udata != nullptr || callback != nullptr) {
    std::cerr << "cuDNN callback clear state did not match cuDNN" << std::endl;
    std::exit(1);
  }

  std::cout << "cuDNN callback coverage ok" << std::endl;
  return 0;
}
