#include <cublasLt.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static void check_cublas(cublasStatus_t status, const char *expr, const char *file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLASLt error %d for %s at %s:%d\n", status, expr, file, line);
        exit(1);
    }
}

#define CHECK_CUBLAS(expr) check_cublas((expr), #expr, __FILE__, __LINE__)

int main() {
    char log_path[256];
    snprintf(log_path, sizeof(log_path), "/tmp/gpu_remoting_cublaslt_logger_%ld.log", (long)getpid());
    unlink(log_path);

    CHECK_CUBLAS(cublasLtLoggerSetLevel(1));
    CHECK_CUBLAS(cublasLtLoggerSetMask(0));
    CHECK_CUBLAS(cublasLtLoggerOpenFile(log_path));
    CHECK_CUBLAS(cublasLtLoggerForceDisable());

    unlink(log_path);
    printf("cuBLASLt logger API test passed\n");
    return 0;
}
