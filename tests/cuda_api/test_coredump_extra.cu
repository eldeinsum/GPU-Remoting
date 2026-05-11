#include <cuda.h>

#include <cstdio>
#include <cstring>

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

int main()
{
    CHECK_DRV(cuInit(0));

    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));

    CUcontext context = nullptr;
    CHECK_DRV(cuCtxCreate(&context, nullptr, 0, device));

    size_t local_required = 0;
    CHECK_DRV(cuCoredumpGetAttribute(CU_COREDUMP_FILE, nullptr,
                                     &local_required));
    if (local_required == 0 || local_required > 1024) {
        std::fprintf(stderr, "unexpected local coredump file size: %zu\n",
                     local_required);
        return 1;
    }

    const char local_file[] = "/tmp/gpu_remoting_local_core";
    size_t local_file_size = sizeof(local_file);
    CHECK_DRV(cuCoredumpSetAttribute(CU_COREDUMP_FILE,
                                     const_cast<char *>(local_file),
                                     &local_file_size));

    char local_file_out[1024] = {};
    local_file_size = sizeof(local_file_out);
    CHECK_DRV(cuCoredumpGetAttribute(CU_COREDUMP_FILE, local_file_out,
                                     &local_file_size));
    if (std::strcmp(local_file_out, local_file) != 0) {
        std::fprintf(stderr, "local coredump file mismatch: %s\n",
                     local_file_out);
        return 1;
    }

    unsigned int local_flags = CU_COREDUMP_SKIP_GLOBAL_MEMORY |
                               CU_COREDUMP_SKIP_SHARED_MEMORY;
    size_t local_flags_size = sizeof(local_flags);
    CHECK_DRV(cuCoredumpSetAttribute(CU_COREDUMP_GENERATION_FLAGS,
                                     &local_flags, &local_flags_size));

    unsigned int local_flags_out = 0;
    local_flags_size = sizeof(local_flags_out);
    CHECK_DRV(cuCoredumpGetAttribute(CU_COREDUMP_GENERATION_FLAGS,
                                     &local_flags_out, &local_flags_size));
    if (local_flags_out != local_flags) {
        std::fprintf(stderr, "local coredump flags mismatch: %u != %u\n",
                     local_flags_out, local_flags);
        return 1;
    }

    size_t global_required = 0;
    CHECK_DRV(cuCoredumpGetAttributeGlobal(CU_COREDUMP_FILE, nullptr,
                                           &global_required));
    if (global_required == 0 || global_required > 1024) {
        std::fprintf(stderr, "unexpected global coredump file size: %zu\n",
                     global_required);
        return 1;
    }

    const char global_file[] = "/tmp/gpu_remoting_global_core";
    size_t global_file_size = sizeof(global_file);
    CHECK_DRV(cuCoredumpSetAttributeGlobal(CU_COREDUMP_FILE,
                                           const_cast<char *>(global_file),
                                           &global_file_size));

    char global_file_out[1024] = {};
    global_file_size = sizeof(global_file_out);
    CHECK_DRV(cuCoredumpGetAttributeGlobal(CU_COREDUMP_FILE, global_file_out,
                                           &global_file_size));
    if (std::strcmp(global_file_out, global_file) != 0) {
        std::fprintf(stderr, "global coredump file mismatch: %s\n",
                     global_file_out);
        return 1;
    }

    unsigned int global_flags = CU_COREDUMP_SKIP_LOCAL_MEMORY |
                                CU_COREDUMP_SKIP_CONSTBANK_MEMORY;
    size_t global_flags_size = sizeof(global_flags);
    CHECK_DRV(cuCoredumpSetAttributeGlobal(CU_COREDUMP_GENERATION_FLAGS,
                                           &global_flags, &global_flags_size));

    unsigned int global_flags_out = 0;
    global_flags_size = sizeof(global_flags_out);
    CHECK_DRV(cuCoredumpGetAttributeGlobal(CU_COREDUMP_GENERATION_FLAGS,
                                           &global_flags_out,
                                           &global_flags_size));
    if (global_flags_out != global_flags) {
        std::fprintf(stderr, "global coredump flags mismatch: %u != %u\n",
                     global_flags_out, global_flags);
        return 1;
    }

    CHECK_DRV(cuCtxDestroy(context));

    std::puts("coredump API test passed");
    return 0;
}
