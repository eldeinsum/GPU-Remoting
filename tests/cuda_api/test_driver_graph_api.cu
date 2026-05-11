#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t result = (call);                                           \
        if (result != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s failed: %s (%d)\n", #call,               \
                         cudaGetErrorString(result), static_cast<int>(result)); \
            return 1;                                                          \
        }                                                                      \
    } while (0)

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

#define CHECK_NVRTC(call)                                                      \
    do {                                                                       \
        nvrtcResult result = (call);                                           \
        if (result != NVRTC_SUCCESS) {                                         \
            std::fprintf(stderr, "%s failed: %s (%d)\n", #call,               \
                         nvrtcGetErrorString(result), static_cast<int>(result)); \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static int compile_kernel_cubin(std::vector<char> *cubin, int major, int minor)
{
    static const char source[] = R"(
extern "C" __global__ void graph_kernel(int *out, int value, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        out[i] = value + i;
    }
}
)";

    nvrtcProgram program = nullptr;
    CHECK_NVRTC(nvrtcCreateProgram(&program, source,
                                   "test_driver_graph_api.cu", 0, nullptr,
                                   nullptr));
    std::string arch = "--gpu-architecture=sm_" + std::to_string(major) +
                       std::to_string(minor);
    const char *options[] = {arch.c_str(), "--std=c++11"};
    nvrtcResult compile_result = nvrtcCompileProgram(program, 2, options);
    size_t log_size = 0;
    nvrtcGetProgramLogSize(program, &log_size);
    if (log_size > 1) {
        std::vector<char> log(log_size);
        nvrtcGetProgramLog(program, log.data());
        std::fprintf(stderr, "%s\n", log.data());
    }
    if (compile_result != NVRTC_SUCCESS) {
        std::fprintf(stderr, "nvrtcCompileProgram failed: %s\n",
                     nvrtcGetErrorString(compile_result));
        nvrtcDestroyProgram(&program);
        return 1;
    }

    size_t cubin_size = 0;
    CHECK_NVRTC(nvrtcGetCUBINSize(program, &cubin_size));
    cubin->resize(cubin_size);
    CHECK_NVRTC(nvrtcGetCUBIN(program, cubin->data()));
    CHECK_NVRTC(nvrtcDestroyProgram(&program));
    return 0;
}

static bool contains_node(const CUgraphNode *nodes, size_t count, CUgraphNode node)
{
    for (size_t i = 0; i < count; ++i) {
        if (nodes[i] == node) {
            return true;
        }
    }
    return false;
}

static int check_empty_graph(CUgraph graph, CUgraphNode node_a, CUgraphNode node_b)
{
    size_t count = 0;
    CHECK_DRV(cuGraphGetNodes(graph, nullptr, &count));
    if (count != 2) {
        std::fprintf(stderr, "expected 2 graph nodes, got %zu\n", count);
        return 1;
    }

    CUgraphNode nodes[2] = {};
    CHECK_DRV(cuGraphGetNodes(graph, nodes, &count));
    if (count != 2 || !contains_node(nodes, count, node_a) ||
        !contains_node(nodes, count, node_b)) {
        std::fprintf(stderr, "graph node list did not contain expected nodes\n");
        return 1;
    }

    size_t roots = 0;
    CHECK_DRV(cuGraphGetRootNodes(graph, nullptr, &roots));
    if (roots != 1) {
        std::fprintf(stderr, "expected 1 root node, got %zu\n", roots);
        return 1;
    }
    CUgraphNode root_nodes[1] = {};
    CHECK_DRV(cuGraphGetRootNodes(graph, root_nodes, &roots));
    if (roots != 1 || root_nodes[0] != node_a) {
        std::fprintf(stderr, "unexpected root node\n");
        return 1;
    }

    size_t edges = 0;
    CHECK_DRV(cuGraphGetEdges(graph, nullptr, nullptr, nullptr, &edges));
    if (edges != 1) {
        std::fprintf(stderr, "expected 1 graph edge, got %zu\n", edges);
        return 1;
    }
    CUgraphNode edge_from[1] = {};
    CUgraphNode edge_to[1] = {};
    CUgraphEdgeData edge_data[1] = {};
    CHECK_DRV(cuGraphGetEdges(graph, edge_from, edge_to, edge_data, &edges));
    if (edges != 1 || edge_from[0] != node_a || edge_to[0] != node_b) {
        std::fprintf(stderr, "unexpected graph edge endpoints\n");
        return 1;
    }

    size_t dependencies = 0;
    CHECK_DRV(cuGraphNodeGetDependencies(node_b, nullptr, nullptr, &dependencies));
    if (dependencies != 1) {
        std::fprintf(stderr, "expected 1 dependency, got %zu\n", dependencies);
        return 1;
    }
    CUgraphNode dependency_nodes[1] = {};
    CUgraphEdgeData dependency_edges[1] = {};
    CHECK_DRV(cuGraphNodeGetDependencies(node_b, dependency_nodes, dependency_edges,
                                         &dependencies));
    if (dependencies != 1 || dependency_nodes[0] != node_a) {
        std::fprintf(stderr, "unexpected dependency node\n");
        return 1;
    }

    size_t dependents = 0;
    CHECK_DRV(cuGraphNodeGetDependentNodes(node_a, nullptr, nullptr, &dependents));
    if (dependents != 1) {
        std::fprintf(stderr, "expected 1 dependent, got %zu\n", dependents);
        return 1;
    }
    CUgraphNode dependent_nodes[1] = {};
    CUgraphEdgeData dependent_edges[1] = {};
    CHECK_DRV(cuGraphNodeGetDependentNodes(node_a, dependent_nodes, dependent_edges,
                                           &dependents));
    if (dependents != 1 || dependent_nodes[0] != node_b) {
        std::fprintf(stderr, "unexpected dependent node\n");
        return 1;
    }

    CUgraphNodeType node_type = CU_GRAPH_NODE_TYPE_KERNEL;
    CHECK_DRV(cuGraphNodeGetType(node_a, &node_type));
    if (node_type != CU_GRAPH_NODE_TYPE_EMPTY) {
        std::fprintf(stderr, "expected empty node type\n");
        return 1;
    }

    CUgraph containing_graph = nullptr;
    CHECK_DRV(cuGraphNodeGetContainingGraph(node_a, &containing_graph));
    if (containing_graph == nullptr) {
        std::fprintf(stderr, "cuGraphNodeGetContainingGraph returned null\n");
        return 1;
    }

    unsigned int graph_id = 0;
    CHECK_DRV(cuGraphGetId(graph, &graph_id));
    unsigned int local_id = 0;
    CHECK_DRV(cuGraphNodeGetLocalId(node_a, &local_id));
    unsigned long long tools_id = 0;
    CHECK_DRV(cuGraphNodeGetToolsId(node_a, &tools_id));

    return 0;
}

static int check_graph_memory_attributes(CUdevice device)
{
    unsigned long long value = 0;
    CHECK_DRV(cuDeviceGetGraphMemAttribute(
        device, CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT, &value));
    CHECK_DRV(cuDeviceGetGraphMemAttribute(
        device, CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT, &value));
    value = 0;
    CHECK_DRV(cuDeviceSetGraphMemAttribute(
        device, CU_GRAPH_MEM_ATTR_USED_MEM_HIGH, &value));
    CHECK_DRV(cuDeviceSetGraphMemAttribute(
        device, CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH, &value));
    CHECK_DRV(cuDeviceGraphMemTrim(device));

    return 0;
}

static int check_child_and_event_graph(CUstream stream)
{
    CUgraph graph = nullptr;
    CHECK_DRV(cuGraphCreate(&graph, 0));

    CUgraph child_graph = nullptr;
    CHECK_DRV(cuGraphCreate(&child_graph, 0));
    CUgraphNode child_inner_node = nullptr;
    CHECK_DRV(cuGraphAddEmptyNode(&child_inner_node, child_graph, nullptr, 0));

    CUgraphNode child_node = nullptr;
    CHECK_DRV(cuGraphAddChildGraphNode(&child_node, graph, nullptr, 0,
                                       child_graph));
    CUgraph child_clone = nullptr;
    CHECK_DRV(cuGraphChildGraphNodeGetGraph(child_node, &child_clone));
    size_t child_nodes = 0;
    CHECK_DRV(cuGraphGetNodes(child_clone, nullptr, &child_nodes));
    if (child_nodes != 1) {
        std::fprintf(stderr, "expected 1 child graph node, got %zu\n",
                     child_nodes);
        return 1;
    }
    CHECK_DRV(cuGraphDestroy(child_graph));

    CUevent event_initial = nullptr;
    CUevent event_replacement = nullptr;
    CHECK_DRV(cuEventCreate(&event_initial, CU_EVENT_DEFAULT));
    CHECK_DRV(cuEventCreate(&event_replacement, CU_EVENT_DEFAULT));

    CUgraphNode record_node = nullptr;
    CUgraphNode wait_node = nullptr;
    CHECK_DRV(cuGraphAddEventRecordNode(&record_node, graph, nullptr, 0,
                                        event_initial));
    CUevent queried_event = nullptr;
    CHECK_DRV(cuGraphEventRecordNodeGetEvent(record_node, &queried_event));
    if (queried_event != event_initial) {
        std::fprintf(stderr, "unexpected event record node event\n");
        return 1;
    }
    CHECK_DRV(cuGraphEventRecordNodeSetEvent(record_node, event_replacement));
    CHECK_DRV(cuGraphEventRecordNodeGetEvent(record_node, &queried_event));
    if (queried_event != event_replacement) {
        std::fprintf(stderr, "event record node set did not stick\n");
        return 1;
    }

    CHECK_DRV(cuGraphAddEventWaitNode(&wait_node, graph, nullptr, 0,
                                      event_replacement));
    CHECK_DRV(cuGraphEventWaitNodeGetEvent(wait_node, &queried_event));
    if (queried_event != event_replacement) {
        std::fprintf(stderr, "unexpected event wait node event\n");
        return 1;
    }
    CHECK_DRV(cuGraphEventWaitNodeSetEvent(wait_node, event_initial));
    CHECK_DRV(cuGraphEventWaitNodeGetEvent(wait_node, &queried_event));
    if (queried_event != event_initial) {
        std::fprintf(stderr, "event wait node set did not stick\n");
        return 1;
    }
    CHECK_DRV(cuGraphEventWaitNodeSetEvent(wait_node, event_replacement));

    CUgraphNode event_from[1] = {record_node};
    CUgraphNode event_to[1] = {wait_node};
    CHECK_DRV(cuGraphAddDependencies(graph, event_from, event_to, nullptr, 1));

    CUgraph replacement_child_graph = nullptr;
    CHECK_DRV(cuGraphCreate(&replacement_child_graph, 0));
    CUgraphNode replacement_child_node = nullptr;
    CHECK_DRV(cuGraphAddEmptyNode(&replacement_child_node,
                                  replacement_child_graph, nullptr, 0));

    CUgraphExec exec = nullptr;
    CHECK_DRV(cuGraphInstantiateWithFlags(&exec, graph, 0));
    CHECK_DRV(cuGraphExecChildGraphNodeSetParams(exec, child_node,
                                                 replacement_child_graph));
    CHECK_DRV(cuGraphExecEventRecordNodeSetEvent(exec, record_node,
                                                 event_replacement));
    CHECK_DRV(cuGraphExecEventWaitNodeSetEvent(exec, wait_node,
                                               event_replacement));
    CHECK_DRV(cuGraphUpload(exec, stream));
    CHECK_DRV(cuGraphLaunch(exec, stream));
    CHECK_DRV(cuStreamSynchronize(stream));
    CHECK_DRV(cuGraphExecDestroy(exec));

    CHECK_DRV(cuGraphDestroy(replacement_child_graph));
    CHECK_DRV(cuEventDestroy(event_replacement));
    CHECK_DRV(cuEventDestroy(event_initial));
    CHECK_DRV(cuGraphDestroy(graph));

    return 0;
}

static int check_memory_graph(CUstream stream, CUcontext context)
{
    const size_t bytes = 64;
    CUdeviceptr device_a = 0;
    CUdeviceptr device_b = 0;
    CHECK_DRV(cuMemAlloc(&device_a, bytes));
    CHECK_DRV(cuMemAlloc(&device_b, bytes));
    CHECK_DRV(cuMemsetD8(device_a, 0, bytes));
    CHECK_DRV(cuMemsetD8(device_b, 0, bytes));

    CUgraph graph = nullptr;
    CHECK_DRV(cuGraphCreate(&graph, 0));

    CUDA_MEMSET_NODE_PARAMS memset_params = {};
    memset_params.dst = device_a;
    memset_params.value = 0x5a;
    memset_params.elementSize = 1;
    memset_params.width = bytes;
    memset_params.height = 1;

    CUgraphNode memset_node = nullptr;
    CHECK_DRV(cuGraphAddMemsetNode(&memset_node, graph, nullptr, 0,
                                   &memset_params, context));
    CUDA_MEMSET_NODE_PARAMS queried_memset = {};
    CHECK_DRV(cuGraphMemsetNodeGetParams(memset_node, &queried_memset));
    if (queried_memset.dst != device_a || queried_memset.value != 0x5a ||
        queried_memset.width != bytes || queried_memset.height != 1) {
        std::fprintf(stderr, "unexpected graph memset params\n");
        return 1;
    }

    memset_params.value = 0x6b;
    CHECK_DRV(cuGraphMemsetNodeSetParams(memset_node, &memset_params));
    CHECK_DRV(cuGraphMemsetNodeGetParams(memset_node, &queried_memset));
    if (queried_memset.value != 0x6b) {
        std::fprintf(stderr, "graph memset params update did not stick\n");
        return 1;
    }

    CUDA_MEMCPY3D copy_params = {};
    copy_params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy_params.srcDevice = device_a;
    copy_params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy_params.dstDevice = device_b;
    copy_params.WidthInBytes = bytes;
    copy_params.Height = 1;
    copy_params.Depth = 1;

    CUgraphNode copy_node = nullptr;
    CHECK_DRV(cuGraphAddMemcpyNode(&copy_node, graph, nullptr, 0,
                                   &copy_params, context));
    CUDA_MEMCPY3D queried_copy = {};
    CHECK_DRV(cuGraphMemcpyNodeGetParams(copy_node, &queried_copy));
    if (queried_copy.srcDevice != device_a || queried_copy.dstDevice != device_b ||
        queried_copy.WidthInBytes != bytes || queried_copy.Height != 1 ||
        queried_copy.Depth != 1) {
        std::fprintf(stderr, "unexpected graph memcpy params\n");
        return 1;
    }
    CHECK_DRV(cuGraphMemcpyNodeSetParams(copy_node, &copy_params));

    CUgraphNode from[1] = {memset_node};
    CUgraphNode to[1] = {copy_node};
    CHECK_DRV(cuGraphAddDependencies(graph, from, to, nullptr, 1));

    CUgraphExec exec = nullptr;
    CHECK_DRV(cuGraphInstantiateWithFlags(&exec, graph, 0));
    memset_params.value = 0x7c;
    CHECK_DRV(cuGraphExecMemsetNodeSetParams(exec, memset_node, &memset_params,
                                             context));
    CHECK_DRV(cuGraphExecMemcpyNodeSetParams(exec, copy_node, &copy_params,
                                             context));

    unsigned int node_enabled = 0;
    CHECK_DRV(cuGraphNodeGetEnabled(exec, copy_node, &node_enabled));
    if (node_enabled != 1) {
        std::fprintf(stderr, "unexpected graph node enabled state: %u\n",
                     node_enabled);
        return 1;
    }
    CHECK_DRV(cuGraphNodeSetEnabled(exec, copy_node, 0));
    CHECK_DRV(cuGraphNodeGetEnabled(exec, copy_node, &node_enabled));
    if (node_enabled != 0) {
        std::fprintf(stderr, "graph node disable did not persist\n");
        return 1;
    }
    CHECK_DRV(cuMemsetD8(device_b, 0, bytes));
    CHECK_DRV(cuGraphLaunch(exec, stream));
    CHECK_DRV(cuStreamSynchronize(stream));
    unsigned char disabled_output[bytes];
    CHECK_DRV(cuMemcpyDtoH(disabled_output, device_b, bytes));
    for (size_t i = 0; i < bytes; ++i) {
        if (disabled_output[i] != 0) {
            std::fprintf(stderr, "disabled graph memcpy wrote byte %zu\n", i);
            return 1;
        }
    }
    CHECK_DRV(cuGraphNodeSetEnabled(exec, copy_node, 1));
    CHECK_DRV(cuGraphNodeGetEnabled(exec, copy_node, &node_enabled));
    if (node_enabled != 1) {
        std::fprintf(stderr, "graph node enable did not persist\n");
        return 1;
    }
    CHECK_DRV(cuGraphLaunch(exec, stream));
    CHECK_DRV(cuStreamSynchronize(stream));

    unsigned char output[bytes];
    CHECK_DRV(cuMemcpyDtoH(output, device_b, bytes));
    for (size_t i = 0; i < bytes; ++i) {
        if (output[i] != 0x7c) {
            std::fprintf(stderr, "graph memory output mismatch at %zu\n", i);
            return 1;
        }
    }

    CHECK_DRV(cuGraphExecDestroy(exec));
    CHECK_DRV(cuGraphDestroy(graph));
    CHECK_DRV(cuMemFree(device_b));
    CHECK_DRV(cuMemFree(device_a));

    return 0;
}

static int check_mem_alloc_graph(CUstream stream, CUcontext context)
{
    const size_t bytes = 64;
    CUdeviceptr copy_dst = 0;
    CHECK_DRV(cuMemAlloc(&copy_dst, bytes));
    CHECK_DRV(cuMemsetD8(copy_dst, 0, bytes));

    CUgraph graph = nullptr;
    CHECK_DRV(cuGraphCreate(&graph, 0));

    CUDA_MEM_ALLOC_NODE_PARAMS alloc_params = {};
    alloc_params.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    alloc_params.poolProps.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
    alloc_params.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    alloc_params.poolProps.location.id = 0;
    alloc_params.bytesize = bytes;

    CUgraphNode alloc_node = nullptr;
    CHECK_DRV(cuGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0,
                                     &alloc_params));
    if (alloc_params.dptr == 0) {
        std::fprintf(stderr, "graph alloc node returned null allocation\n");
        return 1;
    }

    CUDA_MEM_ALLOC_NODE_PARAMS queried_alloc = {};
    CHECK_DRV(cuGraphMemAllocNodeGetParams(alloc_node, &queried_alloc));
    if (queried_alloc.dptr != alloc_params.dptr ||
        queried_alloc.bytesize != bytes) {
        std::fprintf(stderr, "unexpected graph alloc node params\n");
        return 1;
    }

    CUDA_MEMSET_NODE_PARAMS memset_params = {};
    memset_params.dst = alloc_params.dptr;
    memset_params.value = 0x4d;
    memset_params.elementSize = 1;
    memset_params.width = bytes;
    memset_params.height = 1;
    CUgraphNode memset_node = nullptr;
    CHECK_DRV(cuGraphAddMemsetNode(&memset_node, graph, nullptr, 0,
                                   &memset_params, context));

    CUDA_MEMCPY3D copy_params = {};
    copy_params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy_params.srcDevice = alloc_params.dptr;
    copy_params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy_params.dstDevice = copy_dst;
    copy_params.WidthInBytes = bytes;
    copy_params.Height = 1;
    copy_params.Depth = 1;
    CUgraphNode copy_node = nullptr;
    CHECK_DRV(cuGraphAddMemcpyNode(&copy_node, graph, nullptr, 0,
                                   &copy_params, context));

    CUgraphNode free_node = nullptr;
    CHECK_DRV(cuGraphAddMemFreeNode(&free_node, graph, nullptr, 0,
                                    alloc_params.dptr));
    CUdeviceptr queried_free = 0;
    CHECK_DRV(cuGraphMemFreeNodeGetParams(free_node, &queried_free));
    if (queried_free != alloc_params.dptr) {
        std::fprintf(stderr, "unexpected graph free node pointer\n");
        return 1;
    }

    CUgraphNode from[3] = {alloc_node, memset_node, copy_node};
    CUgraphNode to[3] = {memset_node, copy_node, free_node};
    CHECK_DRV(cuGraphAddDependencies(graph, from, to, nullptr, 3));

    CUgraphExec exec = nullptr;
    CHECK_DRV(cuGraphInstantiateWithFlags(&exec, graph, 0));
    CHECK_DRV(cuGraphLaunch(exec, stream));
    CHECK_DRV(cuStreamSynchronize(stream));

    unsigned char output[bytes];
    CHECK_DRV(cuMemcpyDtoH(output, copy_dst, bytes));
    for (size_t i = 0; i < bytes; ++i) {
        if (output[i] != 0x4d) {
            std::fprintf(stderr, "graph alloc output mismatch at %zu\n", i);
            return 1;
        }
    }

    CHECK_DRV(cuGraphExecDestroy(exec));
    CHECK_DRV(cuGraphDestroy(graph));
    CHECK_DRV(cuMemFree(copy_dst));

    return 0;
}

static int check_generic_node_graph(CUstream stream, CUcontext context)
{
    const size_t bytes = 64;
    CUdeviceptr device = 0;
    CHECK_DRV(cuMemAlloc(&device, bytes));
    CHECK_DRV(cuMemsetD8(device, 0, bytes));

    CUgraph graph = nullptr;
    CHECK_DRV(cuGraphCreate(&graph, 0));

    CUgraphNodeParams params = {};
    params.type = CU_GRAPH_NODE_TYPE_MEMSET;
    params.memset.dst = device;
    params.memset.value = 0x21;
    params.memset.elementSize = 1;
    params.memset.width = bytes;
    params.memset.height = 1;
    params.memset.ctx = context;

    CUgraphNode node = nullptr;
    CHECK_DRV(cuGraphAddNode(&node, graph, nullptr, nullptr, 0, &params));

    CUgraphNodeParams queried = {};
    CHECK_DRV(cuGraphNodeGetParams(node, &queried));
    if (queried.type != CU_GRAPH_NODE_TYPE_MEMSET ||
        queried.memset.dst != device || queried.memset.value != 0x21 ||
        queried.memset.width != bytes || queried.memset.height != 1) {
        std::fprintf(stderr, "unexpected generic graph node params\n");
        return 1;
    }

    params.memset.value = 0x32;
    CHECK_DRV(cuGraphNodeSetParams(node, &params));
    queried = {};
    CHECK_DRV(cuGraphNodeGetParams(node, &queried));
    if (queried.memset.value != 0x32) {
        std::fprintf(stderr, "generic graph node update did not stick\n");
        return 1;
    }

    CUgraphExec exec = nullptr;
    CHECK_DRV(cuGraphInstantiateWithFlags(&exec, graph, 0));

    params.memset.value = 0x43;
    CHECK_DRV(cuGraphExecNodeSetParams(exec, node, &params));
    CHECK_DRV(cuGraphLaunch(exec, stream));
    CHECK_DRV(cuStreamSynchronize(stream));

    unsigned char output[bytes];
    CHECK_DRV(cuMemcpyDtoH(output, device, bytes));
    for (size_t i = 0; i < bytes; ++i) {
        if (output[i] != 0x43) {
            std::fprintf(stderr,
                         "generic graph node output mismatch at %zu: got %u\n",
                         i, static_cast<unsigned int>(output[i]));
            return 1;
        }
    }

    CHECK_DRV(cuGraphExecDestroy(exec));
    CHECK_DRV(cuGraphDestroy(graph));
    CHECK_DRV(cuMemFree(device));

    return 0;
}

static int check_stream_capture_graph(CUstream stream)
{
    const size_t bytes = 64;
    CUdeviceptr device = 0;
    CHECK_DRV(cuMemAlloc(&device, bytes));
    CHECK_DRV(cuMemsetD8(device, 0, bytes));

    CHECK_DRV(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));

    CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
    CHECK_DRV(cuStreamIsCapturing(stream, &status));
    if (status != CU_STREAM_CAPTURE_STATUS_ACTIVE) {
        std::fprintf(stderr, "unexpected stream capture status: %d\n",
                     static_cast<int>(status));
        return 1;
    }
    cuuint64_t capture_id = 0;
    CHECK_DRV(cuStreamGetCaptureInfo(stream, &status, &capture_id, nullptr,
                                     nullptr, nullptr, nullptr));
    if (status != CU_STREAM_CAPTURE_STATUS_ACTIVE || capture_id == 0) {
        std::fprintf(stderr, "unexpected stream capture info\n");
        return 1;
    }

    CHECK_DRV(cuMemsetD8Async(device, 0x54, bytes, stream));
    CUgraph captured = nullptr;
    CHECK_DRV(cuStreamEndCapture(stream, &captured));
    if (captured == nullptr) {
        std::fprintf(stderr, "stream capture returned a null graph\n");
        return 1;
    }

    CUgraphExec exec = nullptr;
    CHECK_DRV(cuGraphInstantiateWithFlags(&exec, captured, 0));
    CHECK_DRV(cuMemsetD8(device, 0, bytes));
    CHECK_DRV(cuGraphLaunch(exec, stream));
    CHECK_DRV(cuStreamSynchronize(stream));

    unsigned char output[bytes];
    CHECK_DRV(cuMemcpyDtoH(output, device, bytes));
    for (size_t i = 0; i < bytes; ++i) {
        if (output[i] != 0x54) {
            std::fprintf(stderr,
                         "captured graph output mismatch at %zu: got %u\n",
                         i, static_cast<unsigned int>(output[i]));
            return 1;
        }
    }

    CHECK_DRV(cuGraphExecDestroy(exec));
    CHECK_DRV(cuGraphDestroy(captured));

    CUgraph capture_graph = nullptr;
    CHECK_DRV(cuGraphCreate(&capture_graph, 0));
    CUgraphNode capture_dependency = nullptr;
    CHECK_DRV(cuGraphAddEmptyNode(&capture_dependency, capture_graph,
                                  nullptr, 0));
    CUgraphNode capture_updated_dependency = nullptr;
    CHECK_DRV(cuGraphAddEmptyNode(&capture_updated_dependency, capture_graph,
                                  nullptr, 0));

    CUgraphNode capture_dependencies[] = {capture_dependency};
    CHECK_DRV(cuStreamBeginCaptureToGraph(
        stream, capture_graph, capture_dependencies, nullptr, 1,
        CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
    CUgraphNode updated_dependencies[] = {capture_updated_dependency};
    CHECK_DRV(cuStreamUpdateCaptureDependencies(
        stream, updated_dependencies, nullptr, 1,
        CU_STREAM_SET_CAPTURE_DEPENDENCIES));
    CHECK_DRV(cuMemsetD8Async(device, 0x65, bytes, stream));
    CUgraph captured_to_graph = nullptr;
    CHECK_DRV(cuStreamEndCapture(stream, &captured_to_graph));
    if (captured_to_graph == nullptr) {
        std::fprintf(stderr, "capture-to-graph returned a null graph\n");
        return 1;
    }

    size_t nodes = 0;
    CHECK_DRV(cuGraphGetNodes(captured_to_graph, nullptr, &nodes));
    if (nodes != 3) {
        std::fprintf(stderr, "unexpected capture-to-graph node count: %zu\n",
                     nodes);
        return 1;
    }
    size_t edges = 0;
    CHECK_DRV(cuGraphGetEdges(captured_to_graph, nullptr, nullptr, nullptr,
                              &edges));
    if (edges != 1) {
        std::fprintf(stderr, "unexpected capture-to-graph edge count: %zu\n",
                     edges);
        return 1;
    }

    exec = nullptr;
    CHECK_DRV(cuGraphInstantiateWithFlags(&exec, captured_to_graph, 0));
    CHECK_DRV(cuMemsetD8(device, 0, bytes));
    CHECK_DRV(cuGraphLaunch(exec, stream));
    CHECK_DRV(cuStreamSynchronize(stream));

    CHECK_DRV(cuMemcpyDtoH(output, device, bytes));
    for (size_t i = 0; i < bytes; ++i) {
        if (output[i] != 0x65) {
            std::fprintf(stderr,
                         "capture-to-graph output mismatch at %zu: got %u\n",
                         i, static_cast<unsigned int>(output[i]));
            return 1;
        }
    }

    CHECK_DRV(cuGraphExecDestroy(exec));
    CHECK_DRV(cuGraphDestroy(captured_to_graph));
    CHECK_DRV(cuMemFree(device));

    return 0;
}

static int check_batch_mem_op_graph(CUstream stream, CUcontext context)
{
    CUdeviceptr device = 0;
    CHECK_DRV(cuMemAlloc(&device, sizeof(cuuint32_t)));
    CHECK_DRV(cuMemsetD32(device, 0, 1));

    CUgraph graph = nullptr;
    CHECK_DRV(cuGraphCreate(&graph, 0));

    CUstreamBatchMemOpParams op = {};
    op.writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    op.writeValue.address = device;
    op.writeValue.value = 0x12345678u;
    op.writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
    op.writeValue.alias = 0;

    CUDA_BATCH_MEM_OP_NODE_PARAMS params = {};
    params.ctx = context;
    params.count = 1;
    params.paramArray = &op;
    params.flags = 0;

    CUgraphNode node = nullptr;
    CHECK_DRV(cuGraphAddBatchMemOpNode(&node, graph, nullptr, 0, &params));

    CUDA_BATCH_MEM_OP_NODE_PARAMS queried = {};
    CHECK_DRV(cuGraphBatchMemOpNodeGetParams(node, &queried));
    if (queried.count != 1 || queried.paramArray == nullptr ||
        queried.paramArray[0].operation != CU_STREAM_MEM_OP_WRITE_VALUE_32 ||
        queried.paramArray[0].writeValue.address != device ||
        queried.paramArray[0].writeValue.value != 0x12345678u) {
        std::fprintf(stderr, "unexpected graph batch mem-op params\n");
        return 1;
    }

    op.writeValue.value = 0x23456789u;
    CHECK_DRV(cuGraphBatchMemOpNodeSetParams(node, &params));
    queried = {};
    CHECK_DRV(cuGraphBatchMemOpNodeGetParams(node, &queried));
    if (queried.paramArray == nullptr ||
        queried.paramArray[0].writeValue.value != 0x23456789u) {
        std::fprintf(stderr, "graph batch mem-op update did not stick\n");
        return 1;
    }

    CUgraphExec exec = nullptr;
    CHECK_DRV(cuGraphInstantiateWithFlags(&exec, graph, 0));

    op.writeValue.value = 0x3456789au;
    CHECK_DRV(cuGraphExecBatchMemOpNodeSetParams(exec, node, &params));
    CHECK_DRV(cuGraphLaunch(exec, stream));
    CHECK_DRV(cuStreamSynchronize(stream));

    cuuint32_t output = 0;
    CHECK_DRV(cuMemcpyDtoH(&output, device, sizeof(output)));
    if (output != 0x3456789au) {
        std::fprintf(stderr, "batch mem-op graph output mismatch: got 0x%x\n",
                     output);
        return 1;
    }

    CHECK_DRV(cuGraphExecDestroy(exec));
    CHECK_DRV(cuGraphDestroy(graph));
    CHECK_DRV(cuMemFree(device));

    return 0;
}

static int check_kernel_graph(CUstream stream, CUdevice device)
{
    int major = 0;
    int minor = 0;
    CHECK_DRV(cuDeviceGetAttribute(&major,
                                   CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                   device));
    CHECK_DRV(cuDeviceGetAttribute(&minor,
                                   CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                   device));

    std::vector<char> cubin;
    if (compile_kernel_cubin(&cubin, major, minor) != 0) {
        return 1;
    }

    CUmodule module = nullptr;
    CHECK_DRV(cuModuleLoadData(&module, cubin.data()));
    CUfunction function = nullptr;
    CHECK_DRV(cuModuleGetFunction(&function, module, "graph_kernel"));

    const int count = 64;
    int kernel_count = count;
    int value = 3;
    CUdeviceptr output_device = 0;
    CHECK_DRV(cuMemAlloc(&output_device, count * sizeof(int)));
    CHECK_DRV(cuMemsetD8(output_device, 0, count * sizeof(int)));

    CUgraph graph = nullptr;
    CHECK_DRV(cuGraphCreate(&graph, 0));

    void *args[] = {&output_device, &value, &kernel_count};
    CUDA_KERNEL_NODE_PARAMS kernel_params = {};
    kernel_params.func = function;
    kernel_params.gridDimX = 1;
    kernel_params.gridDimY = 1;
    kernel_params.gridDimZ = 1;
    kernel_params.blockDimX = count;
    kernel_params.blockDimY = 1;
    kernel_params.blockDimZ = 1;
    kernel_params.sharedMemBytes = 0;
    kernel_params.kernelParams = args;

    CUgraphNode kernel_node = nullptr;
    CHECK_DRV(cuGraphAddKernelNode(&kernel_node, graph, nullptr, 0,
                                   &kernel_params));

    CUgraphNode attr_copy_node = nullptr;
    CHECK_DRV(cuGraphAddKernelNode(&attr_copy_node, graph, nullptr, 0,
                                   &kernel_params));
    CUkernelNodeAttrValue attr_value = {};
    attr_value.priority = 0;
    CHECK_DRV(cuGraphKernelNodeSetAttribute(kernel_node,
                                            CU_KERNEL_NODE_ATTRIBUTE_PRIORITY,
                                            &attr_value));
    CHECK_DRV(cuGraphKernelNodeCopyAttributes(attr_copy_node, kernel_node));
    CUkernelNodeAttrValue queried_attr = {};
    CHECK_DRV(cuGraphKernelNodeGetAttribute(attr_copy_node,
                                            CU_KERNEL_NODE_ATTRIBUTE_PRIORITY,
                                            &queried_attr));
    if (queried_attr.priority != 0) {
        std::fprintf(stderr, "unexpected graph kernel priority attribute\n");
        return 1;
    }
    CHECK_DRV(cuGraphDestroyNode(attr_copy_node));

    CUDA_KERNEL_NODE_PARAMS queried_params = {};
    CHECK_DRV(cuGraphKernelNodeGetParams(kernel_node, &queried_params));
    if (queried_params.func != function || queried_params.gridDimX != 1 ||
        queried_params.blockDimX != static_cast<unsigned int>(count) ||
        queried_params.kernelParams == nullptr) {
        std::fprintf(stderr, "unexpected graph kernel node params\n");
        return 1;
    }
    int queried_value =
        *reinterpret_cast<int *>(queried_params.kernelParams[1]);
    if (queried_value != value) {
        std::fprintf(stderr, "unexpected graph kernel argument value\n");
        return 1;
    }

    value = 5;
    CHECK_DRV(cuGraphKernelNodeSetParams(kernel_node, &kernel_params));
    queried_params = {};
    CHECK_DRV(cuGraphKernelNodeGetParams(kernel_node, &queried_params));
    queried_value = *reinterpret_cast<int *>(queried_params.kernelParams[1]);
    if (queried_value != value) {
        std::fprintf(stderr, "graph kernel param update did not persist\n");
        return 1;
    }

    CUgraphExec exec = nullptr;
    CHECK_DRV(cuGraphInstantiateWithFlags(&exec, graph, 0));

    value = 7;
    CHECK_DRV(cuGraphExecKernelNodeSetParams(exec, kernel_node,
                                             &kernel_params));
    CHECK_DRV(cuGraphLaunch(exec, stream));
    CHECK_DRV(cuStreamSynchronize(stream));

    std::vector<int> output(count, 0);
    CHECK_DRV(cuMemcpyDtoH(output.data(), output_device,
                           output.size() * sizeof(output[0])));
    for (int i = 0; i < count; ++i) {
        int expected = value + i;
        if (output[i] != expected) {
            std::fprintf(stderr,
                         "graph kernel output mismatch at %d: got %d expected %d\n",
                         i, output[i], expected);
            return 1;
        }
    }

    CHECK_DRV(cuGraphExecDestroy(exec));
    CHECK_DRV(cuGraphDestroy(graph));
    CHECK_DRV(cuMemFree(output_device));
    CHECK_DRV(cuModuleUnload(module));

    return 0;
}

int main()
{
    CHECK_CUDA(cudaSetDevice(0));
    void *runtime_context_init = nullptr;
    CHECK_CUDA(cudaMalloc(&runtime_context_init, 1));
    CHECK_CUDA(cudaFree(runtime_context_init));

    CHECK_DRV(cuInit(0));
    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));
    if (check_graph_memory_attributes(device) != 0) {
        return 1;
    }

    CUstream stream = nullptr;
    CHECK_DRV(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
    CUcontext context = nullptr;
    CHECK_DRV(cuCtxGetCurrent(&context));

    CUgraph conditional_graph = nullptr;
    CHECK_DRV(cuGraphCreate(&conditional_graph, 0));
    CUgraphConditionalHandle conditional = 0;
    CHECK_DRV(cuGraphConditionalHandleCreate(&conditional, conditional_graph,
                                             context, 17, 0));
    if (conditional == 0) {
        std::fprintf(stderr,
                     "cuGraphConditionalHandleCreate returned zero handle\n");
        return 1;
    }
    CHECK_DRV(cuGraphDestroy(conditional_graph));

    CUgraph graph = nullptr;
    CHECK_DRV(cuGraphCreate(&graph, 0));
    CUgraphNode node_a = nullptr;
    CUgraphNode node_b = nullptr;
    CHECK_DRV(cuGraphAddEmptyNode(&node_a, graph, nullptr, 0));
    CHECK_DRV(cuGraphAddEmptyNode(&node_b, graph, nullptr, 0));
    CUgraphNode from[1] = {node_a};
    CUgraphNode to[1] = {node_b};
    CHECK_DRV(cuGraphAddDependencies(graph, from, to, nullptr, 1));

    if (check_empty_graph(graph, node_a, node_b) != 0) {
        return 1;
    }
    if (check_child_and_event_graph(stream) != 0) {
        return 1;
    }
    if (check_memory_graph(stream, context) != 0) {
        return 1;
    }
    if (check_mem_alloc_graph(stream, context) != 0) {
        return 1;
    }
    if (check_generic_node_graph(stream, context) != 0) {
        return 1;
    }
    if (check_stream_capture_graph(stream) != 0) {
        return 1;
    }
    if (check_batch_mem_op_graph(stream, context) != 0) {
        return 1;
    }
    if (check_kernel_graph(stream, device) != 0) {
        return 1;
    }

    CUgraph cloned_graph = nullptr;
    CHECK_DRV(cuGraphClone(&cloned_graph, graph));
    CUgraphNode cloned_node = nullptr;
    CHECK_DRV(cuGraphNodeFindInClone(&cloned_node, node_a, cloned_graph));
    CUgraphNodeType cloned_type = CU_GRAPH_NODE_TYPE_KERNEL;
    CHECK_DRV(cuGraphNodeGetType(cloned_node, &cloned_type));
    if (cloned_type != CU_GRAPH_NODE_TYPE_EMPTY) {
        std::fprintf(stderr, "expected cloned empty node type\n");
        return 1;
    }
    CHECK_DRV(cuGraphDestroy(cloned_graph));

    CUgraphExec exec = nullptr;
    CHECK_DRV(cuGraphInstantiateWithFlags(&exec, graph, 0));
    cuuint64_t exec_flags = 1;
    CHECK_DRV(cuGraphExecGetFlags(exec, &exec_flags));
    if (exec_flags != 0) {
        std::fprintf(stderr, "unexpected graph exec flags: %llu\n",
                     static_cast<unsigned long long>(exec_flags));
        return 1;
    }
    unsigned int exec_id = 0;
    CHECK_DRV(cuGraphExecGetId(exec, &exec_id));
    CUgraphExecUpdateResultInfo update_info = {};
    CHECK_DRV(cuGraphExecUpdate(exec, graph, &update_info));
    if (update_info.result != CU_GRAPH_EXEC_UPDATE_SUCCESS) {
        std::fprintf(stderr, "graph exec update result: %d\n",
                     static_cast<int>(update_info.result));
        return 1;
    }
    CHECK_DRV(cuGraphUpload(exec, stream));
    CHECK_DRV(cuGraphLaunch(exec, stream));
    CHECK_DRV(cuStreamSynchronize(stream));
    CHECK_DRV(cuGraphExecDestroy(exec));

    CUDA_GRAPH_INSTANTIATE_PARAMS instantiate_params = {};
    CHECK_DRV(cuGraphInstantiateWithParams(&exec, graph, &instantiate_params));
    if (instantiate_params.result_out != CUDA_GRAPH_INSTANTIATE_SUCCESS) {
        std::fprintf(stderr, "graph instantiate result: %d\n",
                     static_cast<int>(instantiate_params.result_out));
        return 1;
    }
    CHECK_DRV(cuGraphLaunch(exec, stream));
    CHECK_DRV(cuStreamSynchronize(stream));
    CHECK_DRV(cuGraphExecDestroy(exec));

    const char *dot_path = "/tmp/gpu_remoting_driver_graph.dot";
    std::remove(dot_path);
    CHECK_DRV(cuGraphDebugDotPrint(graph, dot_path, 0));
    FILE *dot = std::fopen(dot_path, "r");
    if (dot == nullptr) {
        std::fprintf(stderr, "cuGraphDebugDotPrint did not create output\n");
        return 1;
    }
    std::fclose(dot);
    std::remove(dot_path);

    CHECK_DRV(cuGraphRemoveDependencies(graph, from, to, nullptr, 1));
    size_t edges = 1;
    CHECK_DRV(cuGraphGetEdges(graph, nullptr, nullptr, nullptr, &edges));
    if (edges != 0) {
        std::fprintf(stderr, "expected no graph edges after removal, got %zu\n", edges);
        return 1;
    }

    CHECK_DRV(cuGraphDestroyNode(node_b));
    CHECK_DRV(cuGraphDestroyNode(node_a));
    CHECK_DRV(cuGraphDestroy(graph));
    CHECK_DRV(cuStreamDestroy(stream));

    std::puts("driver graph API test passed");
    return 0;
}
