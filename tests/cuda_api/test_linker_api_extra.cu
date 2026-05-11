#include <cuda.h>

#include <cstdio>
#include <cstring>
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

static const char kLinkPtx[] = R"ptx(
.version 7.8
.target sm_52
.address_size 64

.visible .entry linked_kernel(
    .param .u64 out,
    .param .u32 value
)
{
    .reg .u64 %rd<2>;
    .reg .u32 %r<3>;

    ld.param.u64 %rd1, [out];
    ld.param.u32 %r1, [value];
    add.u32 %r2, %r1, 9;
    st.global.u32 [%rd1], %r2;
    ret;
}
)ptx";

static int run_linked_kernel(const void *image, int value, int expected)
{
    CUmodule module = nullptr;
    CHECK_DRV(cuModuleLoadData(&module, image));

    CUfunction function = nullptr;
    CHECK_DRV(cuModuleGetFunction(&function, module, "linked_kernel"));

    CUdeviceptr output = 0;
    CHECK_DRV(cuMemAlloc(&output, sizeof(unsigned int)));
    CHECK_DRV(cuMemsetD32(output, 0, 1));

    void *args[] = {&output, &value};
    CHECK_DRV(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, args,
                             nullptr));
    CHECK_DRV(cuCtxSynchronize());

    unsigned int actual = 0;
    CHECK_DRV(cuMemcpyDtoH(&actual, output, sizeof(actual)));
    if (actual != static_cast<unsigned int>(expected)) {
        std::fprintf(stderr, "linked kernel output mismatch: %u\n", actual);
        return 1;
    }

    CHECK_DRV(cuMemFree(output));
    CHECK_DRV(cuModuleUnload(module));
    return 0;
}

static int complete_link(CUlinkState state, const void **image_out,
                         size_t *size_out)
{
    void *image = nullptr;
    size_t image_size = 0;
    CHECK_DRV(cuLinkComplete(state, &image, &image_size));
    if (image == nullptr || image_size == 0) {
        std::fprintf(stderr, "cuLinkComplete returned an empty image\n");
        return 1;
    }
    *image_out = image;
    *size_out = image_size;
    return 0;
}

static int run_link_add_data()
{
    CUlinkState state = nullptr;
    CHECK_DRV(cuLinkCreate(0, nullptr, nullptr, &state));

    CHECK_DRV(cuLinkAddData(state, CU_JIT_INPUT_PTX, (void *)kLinkPtx,
                            std::strlen(kLinkPtx) + 1, "linked_data.ptx", 0,
                            nullptr, nullptr));

    const void *image = nullptr;
    size_t image_size = 0;
    if (complete_link(state, &image, &image_size) != 0) {
        return 1;
    }
    if (run_linked_kernel(image, 33, 42) != 0) {
        return 1;
    }

    CHECK_DRV(cuLinkDestroy(state));
    return 0;
}

static int write_temp_ptx(char *path, size_t path_size)
{
    std::snprintf(path, path_size, "/tmp/gpu_remoting_link_%ld_XXXXXX",
                  static_cast<long>(getpid()));
    int fd = mkstemp(path);
    if (fd < 0) {
        std::perror("mkstemp");
        return 1;
    }

    const size_t len = std::strlen(kLinkPtx);
    ssize_t written = write(fd, kLinkPtx, len);
    int close_result = close(fd);
    if (written < 0 || static_cast<size_t>(written) != len ||
        close_result != 0) {
        std::perror("write ptx");
        unlink(path);
        return 1;
    }
    return 0;
}

static int run_link_add_file()
{
    char path[128] = {};
    if (write_temp_ptx(path, sizeof(path)) != 0) {
        return 1;
    }

    CUlinkState state = nullptr;
    CHECK_DRV(cuLinkCreate(0, nullptr, nullptr, &state));
    CHECK_DRV(cuLinkAddFile(state, CU_JIT_INPUT_PTX, path, 0, nullptr,
                            nullptr));

    const void *image = nullptr;
    size_t image_size = 0;
    if (complete_link(state, &image, &image_size) != 0) {
        unlink(path);
        return 1;
    }
    if (run_linked_kernel(image, 51, 60) != 0) {
        unlink(path);
        return 1;
    }

    CHECK_DRV(cuLinkDestroy(state));
    unlink(path);
    return 0;
}

int main()
{
    CHECK_DRV(cuInit(0));

    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));
    CUcontext context = nullptr;
    CHECK_DRV(cuDevicePrimaryCtxRetain(&context, device));
    CHECK_DRV(cuCtxSetCurrent(context));

    if (run_link_add_data() != 0) {
        return 1;
    }
    if (run_link_add_file() != 0) {
        return 1;
    }

    CHECK_DRV(cuDevicePrimaryCtxRelease(device));
    std::puts("linker API test passed");
    return 0;
}
