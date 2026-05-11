#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "%s failed: %s (%d)\n", #call,                \
                         cudaGetErrorString(err__), static_cast<int>(err__));   \
            return 1;                                                           \
        }                                                                       \
    } while (0)

#define CHECK_DRV(call)                                                         \
    do {                                                                        \
        CUresult err__ = (call);                                                \
        if (err__ != CUDA_SUCCESS) {                                            \
            const char *name__ = nullptr;                                       \
            cuGetErrorName(err__, &name__);                                     \
            std::fprintf(stderr, "%s failed: %s (%d)\n", #call,                \
                         name__ ? name__ : "unknown", static_cast<int>(err__));\
            return 1;                                                           \
        }                                                                       \
    } while (0)

struct HostState {
    int calls;
    int value;
};

static void CUDART_CB runtime_host_callback(void *data)
{
    HostState *state = static_cast<HostState *>(data);
    state->calls += 1;
    state->value += 11;
}

static void CUDART_CB runtime_host_callback_alt(void *data)
{
    HostState *state = static_cast<HostState *>(data);
    state->calls += 1;
    state->value += 17;
}

static void CUDA_CB driver_host_callback(void *data)
{
    HostState *state = static_cast<HostState *>(data);
    state->calls += 1;
    state->value += 23;
}

static void CUDA_CB driver_host_callback_alt(void *data)
{
    HostState *state = static_cast<HostState *>(data);
    state->calls += 1;
    state->value += 29;
}

static int validate_runtime_host_node()
{
    cudaGraph_t graph = nullptr;
    cudaGraphNode_t node = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaStream_t stream = nullptr;
    HostState initial = {};
    HostState updated = {};
    HostState exec_updated = {};

    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaGraphCreate(&graph, 0));

    cudaHostNodeParams params = {};
    params.fn = runtime_host_callback;
    params.userData = &initial;
    CHECK_CUDA(cudaGraphAddHostNode(&node, graph, nullptr, 0, &params));

    cudaHostNodeParams readback = {};
    CHECK_CUDA(cudaGraphHostNodeGetParams(node, &readback));
    if (readback.fn != runtime_host_callback || readback.userData != &initial) {
        std::fprintf(stderr, "runtime host node get returned unexpected params\n");
        return 1;
    }

    params.fn = runtime_host_callback_alt;
    params.userData = &updated;
    CHECK_CUDA(cudaGraphHostNodeSetParams(node, &params));
    CHECK_CUDA(cudaGraphHostNodeGetParams(node, &readback));
    if (readback.fn != runtime_host_callback_alt || readback.userData != &updated) {
        std::fprintf(stderr, "runtime host node get after set returned unexpected params\n");
        return 1;
    }

    CHECK_CUDA(cudaGraphInstantiate(&exec, graph, 0));

    params.fn = runtime_host_callback;
    params.userData = &exec_updated;
    CHECK_CUDA(cudaGraphExecHostNodeSetParams(exec, node, &params));
    CHECK_CUDA(cudaGraphLaunch(exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    if (initial.calls != 0 || updated.calls != 0 ||
        exec_updated.calls != 1 || exec_updated.value != 11) {
        std::fprintf(stderr,
                     "runtime callback states unexpected: initial=%d/%d updated=%d/%d exec=%d/%d\n",
                     initial.calls, initial.value, updated.calls, updated.value,
                     exec_updated.calls, exec_updated.value);
        return 1;
    }

    CHECK_CUDA(cudaGraphExecDestroy(exec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
}

static int validate_driver_host_node()
{
    CUgraph graph = nullptr;
    CUgraphNode node = nullptr;
    CUgraphExec exec = nullptr;
    CUstream stream = nullptr;
    HostState initial = {};
    HostState updated = {};
    HostState exec_updated = {};

    CHECK_DRV(cuInit(0));
    CUdevice dev = 0;
    CHECK_DRV(cuDeviceGet(&dev, 0));
    CUcontext ctx = nullptr;
    CHECK_DRV(cuCtxCreate(&ctx, nullptr, 0, dev));
    CHECK_DRV(cuStreamCreate(&stream, 0));
    CHECK_DRV(cuGraphCreate(&graph, 0));

    CUDA_HOST_NODE_PARAMS params = {};
    params.fn = driver_host_callback;
    params.userData = &initial;
    CHECK_DRV(cuGraphAddHostNode(&node, graph, nullptr, 0, &params));

    CUDA_HOST_NODE_PARAMS readback = {};
    CHECK_DRV(cuGraphHostNodeGetParams(node, &readback));
    if (readback.fn != driver_host_callback || readback.userData != &initial) {
        std::fprintf(stderr, "driver host node get returned unexpected params\n");
        return 1;
    }

    params.fn = driver_host_callback_alt;
    params.userData = &updated;
    CHECK_DRV(cuGraphHostNodeSetParams(node, &params));
    CHECK_DRV(cuGraphHostNodeGetParams(node, &readback));
    if (readback.fn != driver_host_callback_alt || readback.userData != &updated) {
        std::fprintf(stderr, "driver host node get after set returned unexpected params\n");
        return 1;
    }

    CHECK_DRV(cuGraphInstantiateWithFlags(&exec, graph, 0));

    params.fn = driver_host_callback;
    params.userData = &exec_updated;
    CHECK_DRV(cuGraphExecHostNodeSetParams(exec, node, &params));
    CHECK_DRV(cuGraphLaunch(exec, stream));
    CHECK_DRV(cuStreamSynchronize(stream));

    if (initial.calls != 0 || updated.calls != 0 ||
        exec_updated.calls != 1 || exec_updated.value != 23) {
        std::fprintf(stderr,
                     "driver callback states unexpected: initial=%d/%d updated=%d/%d exec=%d/%d\n",
                     initial.calls, initial.value, updated.calls, updated.value,
                     exec_updated.calls, exec_updated.value);
        return 1;
    }

    CHECK_DRV(cuGraphExecDestroy(exec));
    CHECK_DRV(cuGraphDestroy(graph));
    CHECK_DRV(cuStreamDestroy(stream));
    CHECK_DRV(cuCtxDestroy(ctx));
    return 0;
}

int main()
{
    if (validate_runtime_host_node() != 0) {
        return 1;
    }
    if (validate_driver_host_node() != 0) {
        return 1;
    }
    std::puts("graph host node API test passed");
    return 0;
}
