#include <cuda.h>

#include <cstdio>
#include <unistd.h>

#define CHECK_DRV(call)                                                        \
    do {                                                                       \
        CUresult result = (call);                                              \
        if (result != CUDA_SUCCESS) {                                          \
            const char *name = nullptr;                                        \
            cuGetErrorName(result, &name);                                     \
            std::fprintf(stderr, "%s failed: %s (%d)\n", #call,               \
                         name == nullptr ? "unknown" : name,                  \
                         static_cast<int>(result));                            \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static bool skip_if_unsupported(CUresult result, const char *call)
{
    if (result == CUDA_ERROR_NOT_SUPPORTED) {
        std::printf("%s unsupported on this driver\n", call);
        return true;
    }
    if (result == CUDA_ERROR_NOT_INITIALIZED) {
        std::printf("%s unavailable before CUDA process initialization\n",
                    call);
        return true;
    }
    return false;
}

int main()
{
    CHECK_DRV(cuInit(0));

    const int pid = getpid();
    CUprocessState state = CU_PROCESS_STATE_FAILED;
    CUresult result = cuCheckpointProcessGetState(pid, &state);
    if (skip_if_unsupported(result, "cuCheckpointProcessGetState")) {
        return 0;
    }
    if (result != CUDA_SUCCESS || state != CU_PROCESS_STATE_RUNNING) {
        std::fprintf(stderr, "unexpected initial checkpoint state: %d (%d)\n",
                     static_cast<int>(state), static_cast<int>(result));
        return 1;
    }

    int restore_tid = 0;
    CHECK_DRV(cuCheckpointProcessGetRestoreThreadId(pid, &restore_tid));
    if (restore_tid <= 0) {
        std::fprintf(stderr, "invalid restore thread id: %d\n", restore_tid);
        return 1;
    }

    CUcheckpointLockArgs lock_args = {};
    lock_args.timeoutMs = 5000;
    CHECK_DRV(cuCheckpointProcessLock(pid, &lock_args));
    CHECK_DRV(cuCheckpointProcessGetState(pid, &state));
    if (state != CU_PROCESS_STATE_LOCKED) {
        std::fprintf(stderr, "expected locked state, got %d\n",
                     static_cast<int>(state));
        return 1;
    }

    CUcheckpointCheckpointArgs checkpoint_args = {};
    CHECK_DRV(cuCheckpointProcessCheckpoint(pid, &checkpoint_args));
    CHECK_DRV(cuCheckpointProcessGetState(pid, &state));
    if (state != CU_PROCESS_STATE_CHECKPOINTED) {
        std::fprintf(stderr, "expected checkpointed state, got %d\n",
                     static_cast<int>(state));
        return 1;
    }

    CUcheckpointRestoreArgs restore_args = {};
    CHECK_DRV(cuCheckpointProcessRestore(pid, &restore_args));
    CHECK_DRV(cuCheckpointProcessGetState(pid, &state));
    if (state != CU_PROCESS_STATE_LOCKED) {
        std::fprintf(stderr, "expected locked state after restore, got %d\n",
                     static_cast<int>(state));
        return 1;
    }

    CUcheckpointUnlockArgs unlock_args = {};
    CHECK_DRV(cuCheckpointProcessUnlock(pid, &unlock_args));
    CHECK_DRV(cuCheckpointProcessGetState(pid, &state));
    if (state != CU_PROCESS_STATE_RUNNING) {
        std::fprintf(stderr, "expected running state after unlock, got %d\n",
                     static_cast<int>(state));
        return 1;
    }

    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));

    std::puts("checkpoint process API test passed");
    return 0;
}
