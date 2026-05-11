#include <cuda.h>

#include <cstdio>
#include <cstring>
#include <vector>

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

static const char kPtx[] = R"ptx(
.version 7.8
.target sm_52
.address_size 64

.visible .entry load_data_ex_kernel()
{
    ret;
}
)ptx";

int main()
{
    CHECK_DRV(cuInit(0));

    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));
    CUcontext context = nullptr;
    CHECK_DRV(cuDevicePrimaryCtxRetain(&context, device));
    CHECK_DRV(cuCtxSetCurrent(context));

    CUmodule module = nullptr;
    CHECK_DRV(cuModuleLoadDataEx(&module, kPtx, 0, nullptr, nullptr));

    CUfunction function = nullptr;
    CHECK_DRV(cuModuleGetFunction(&function, module, "load_data_ex_kernel"));
    if (function == nullptr) {
        std::fprintf(stderr, "cuModuleGetFunction returned null\n");
        return 1;
    }

    const char *name = nullptr;
    CHECK_DRV(cuFuncGetName(&name, function));
    if (name == nullptr || std::strcmp(name, "load_data_ex_kernel") != 0) {
        std::fprintf(stderr, "unexpected function name: %s\n",
                     name == nullptr ? "(null)" : name);
        return 1;
    }

    unsigned int function_count = 0;
    CHECK_DRV(cuModuleGetFunctionCount(&function_count, module));
    if (function_count != 1) {
        std::fprintf(stderr, "unexpected function count: %u\n", function_count);
        return 1;
    }

    std::vector<CUfunction> functions(function_count);
    CHECK_DRV(
        cuModuleEnumerateFunctions(functions.data(), function_count, module));
    if (functions[0] == nullptr) {
        std::fprintf(stderr, "cuModuleEnumerateFunctions returned null\n");
        return 1;
    }

    const char *enumerated_name = nullptr;
    CHECK_DRV(cuFuncGetName(&enumerated_name, functions[0]));
    if (enumerated_name == nullptr ||
        std::strcmp(enumerated_name, "load_data_ex_kernel") != 0) {
        std::fprintf(stderr, "unexpected enumerated function name: %s\n",
                     enumerated_name == nullptr ? "(null)" : enumerated_name);
        return 1;
    }

    CHECK_DRV(cuModuleUnload(module));
    CHECK_DRV(cuDevicePrimaryCtxRelease(device));

    std::puts("module load data ex API test passed");
    return 0;
}
