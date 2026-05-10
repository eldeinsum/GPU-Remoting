#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
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

static int verify_value(const std::vector<unsigned char> &data,
                        unsigned char expected)
{
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] != expected) {
            std::fprintf(stderr, "mismatch at byte %zu: got %u expected %u\n",
                         i, static_cast<unsigned>(data[i]),
                         static_cast<unsigned>(expected));
            return 1;
        }
    }
    return 0;
}

int main()
{
    constexpr size_t kBytes = 256;

    CHECK_CUDA(cudaSetDevice(0));

    unsigned char *device = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&device), kBytes));
    CHECK_CUDA(cudaMemset(device, 0, kBytes));

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    unsigned long long capture_id = 0;
    CHECK_CUDA(cudaStreamGetCaptureInfo(stream, &status, &capture_id, nullptr,
                                        nullptr, nullptr, nullptr));
    if (status != cudaStreamCaptureStatusNone) {
        std::fprintf(stderr, "unexpected initial capture status %d\n",
                     static_cast<int>(status));
        return 1;
    }

    cudaStreamCaptureMode mode = cudaStreamCaptureModeThreadLocal;
    CHECK_CUDA(cudaThreadExchangeStreamCaptureMode(&mode));
    CHECK_CUDA(cudaThreadExchangeStreamCaptureMode(&mode));

    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    CHECK_CUDA(cudaStreamGetCaptureInfo(stream, &status, &capture_id, nullptr,
                                        nullptr, nullptr, nullptr));
    if (status != cudaStreamCaptureStatusActive || capture_id == 0) {
        std::fprintf(stderr, "bad capture info: status=%d id=%llu\n",
                     static_cast<int>(status), capture_id);
        return 1;
    }

    CHECK_CUDA(cudaMemsetAsync(device, 0x5a, kBytes, stream));

    cudaGraph_t graph = nullptr;
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    if (graph == nullptr) {
        std::fprintf(stderr, "cudaStreamEndCapture returned a null graph\n");
        return 1;
    }

    size_t nodes = 0;
    CHECK_CUDA(cudaGraphGetNodes(graph, nullptr, &nodes));
    if (nodes == 0) {
        std::fprintf(stderr, "captured graph has no nodes\n");
        return 1;
    }

    cudaGraphExec_t exec = nullptr;
    CHECK_CUDA(cudaGraphInstantiate(&exec, graph, 0));
    CHECK_CUDA(cudaMemset(device, 0, kBytes));
    CHECK_CUDA(cudaGraphLaunch(exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<unsigned char> output(kBytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), device, kBytes, cudaMemcpyDeviceToHost));
    if (verify_value(output, 0x5a) != 0) {
        return 1;
    }
    CHECK_CUDA(cudaGraphExecDestroy(exec));

    exec = nullptr;
    CHECK_CUDA(cudaGraphInstantiateWithFlags(&exec, graph, 0));
    CHECK_CUDA(cudaMemset(device, 0, kBytes));
    CHECK_CUDA(cudaGraphLaunch(exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    output.assign(kBytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), device, kBytes, cudaMemcpyDeviceToHost));
    if (verify_value(output, 0x5a) != 0) {
        return 1;
    }

    CHECK_CUDA(cudaGraphExecDestroy(exec));
    CHECK_CUDA(cudaGraphDestroy(graph));

    cudaGraph_t memset_graph = nullptr;
    CHECK_CUDA(cudaGraphCreate(&memset_graph, 0));
    cudaMemsetParams memset_params = {};
    memset_params.dst = device;
    memset_params.value = 0x3c;
    memset_params.elementSize = 1;
    memset_params.width = kBytes;
    memset_params.height = 1;

    cudaGraphNode_t memset_node = nullptr;
    CHECK_CUDA(cudaGraphAddMemsetNode(&memset_node, memset_graph, nullptr,
                                      0, &memset_params));
    cudaMemsetParams queried_memset = {};
    CHECK_CUDA(cudaGraphMemsetNodeGetParams(memset_node, &queried_memset));
    if (queried_memset.dst != device || queried_memset.value != 0x3c ||
        queried_memset.elementSize != 1 || queried_memset.width != kBytes ||
        queried_memset.height != 1) {
        std::fprintf(stderr, "unexpected graph memset node params\n");
        return 1;
    }

    memset_params.value = 0x2a;
    CHECK_CUDA(cudaGraphMemsetNodeSetParams(memset_node, &memset_params));
    CHECK_CUDA(cudaGraphMemsetNodeGetParams(memset_node, &queried_memset));
    if (queried_memset.value != 0x2a) {
        std::fprintf(stderr, "graph memset node set unexpected value\n");
        return 1;
    }

    exec = nullptr;
    CHECK_CUDA(cudaGraphInstantiate(&exec, memset_graph, 0));
    memset_params.value = 0x7f;
    CHECK_CUDA(cudaGraphExecMemsetNodeSetParams(exec, memset_node,
                                                &memset_params));
    CHECK_CUDA(cudaMemset(device, 0, kBytes));
    CHECK_CUDA(cudaGraphLaunch(exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    output.assign(kBytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), device, kBytes, cudaMemcpyDeviceToHost));
    if (verify_value(output, 0x7f) != 0) {
        return 1;
    }
    CHECK_CUDA(cudaGraphExecDestroy(exec));
    CHECK_CUDA(cudaGraphDestroy(memset_graph));

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(device));

    CHECK_CUDA(cudaStreamCreate(&stream));
    cudaGraph_t manual_graph = nullptr;
    CHECK_CUDA(cudaGraphCreate(&manual_graph, 0));
    if (manual_graph == nullptr) {
        std::fprintf(stderr, "cudaGraphCreate returned a null graph\n");
        return 1;
    }

    cudaGraphNode_t node_a = nullptr;
    cudaGraphNode_t node_b = nullptr;
    CHECK_CUDA(cudaGraphAddEmptyNode(&node_a, manual_graph, nullptr, 0));
    CHECK_CUDA(cudaGraphAddEmptyNode(&node_b, manual_graph, nullptr, 0));

    cudaGraphNode_t from[] = {node_a};
    cudaGraphNode_t to[] = {node_b};
    CHECK_CUDA(cudaGraphAddDependencies(manual_graph, from, to, nullptr, 1));

    size_t root_nodes = 0;
    CHECK_CUDA(cudaGraphGetRootNodes(manual_graph, nullptr, &root_nodes));
    if (root_nodes != 1) {
        std::fprintf(stderr, "unexpected root node count: %zu\n", root_nodes);
        return 1;
    }

    size_t edges = 0;
    CHECK_CUDA(cudaGraphGetEdges(manual_graph, nullptr, nullptr, nullptr, &edges));
    if (edges != 1) {
        std::fprintf(stderr, "unexpected edge count: %zu\n", edges);
        return 1;
    }

    unsigned int graph_id = 0;
    CHECK_CUDA(cudaGraphGetId(manual_graph, &graph_id));
    cudaGraphNodeType node_type = cudaGraphNodeTypeCount;
    CHECK_CUDA(cudaGraphNodeGetType(node_a, &node_type));
    if (node_type != cudaGraphNodeTypeEmpty) {
        std::fprintf(stderr, "unexpected node type: %d\n", static_cast<int>(node_type));
        return 1;
    }

    cudaGraph_t containing_graph = nullptr;
    CHECK_CUDA(cudaGraphNodeGetContainingGraph(node_a, &containing_graph));
    if (containing_graph == nullptr) {
        std::fprintf(stderr, "cudaGraphNodeGetContainingGraph returned null\n");
        return 1;
    }

    unsigned int local_id = 0;
    CHECK_CUDA(cudaGraphNodeGetLocalId(node_a, &local_id));
    unsigned long long tools_id = 0;
    CHECK_CUDA(cudaGraphNodeGetToolsId(node_a, &tools_id));

    size_t dependency_count = 0;
    CHECK_CUDA(cudaGraphNodeGetDependencies(node_b, nullptr, nullptr, &dependency_count));
    if (dependency_count != 1) {
        std::fprintf(stderr, "unexpected dependency count: %zu\n", dependency_count);
        return 1;
    }

    size_t dependent_count = 0;
    CHECK_CUDA(cudaGraphNodeGetDependentNodes(node_a, nullptr, nullptr, &dependent_count));
    if (dependent_count != 1) {
        std::fprintf(stderr, "unexpected dependent count: %zu\n", dependent_count);
        return 1;
    }

    cudaGraph_t cloned_graph = nullptr;
    CHECK_CUDA(cudaGraphClone(&cloned_graph, manual_graph));
    cudaGraphNode_t cloned_node_a = nullptr;
    CHECK_CUDA(cudaGraphNodeFindInClone(&cloned_node_a, node_a, cloned_graph));
    CHECK_CUDA(cudaGraphNodeGetType(cloned_node_a, &node_type));
    if (node_type != cudaGraphNodeTypeEmpty) {
        std::fprintf(stderr, "unexpected cloned node type: %d\n", static_cast<int>(node_type));
        return 1;
    }
    CHECK_CUDA(cudaGraphDestroy(cloned_graph));

    cudaGraph_t child_graph = nullptr;
    CHECK_CUDA(cudaGraphCreate(&child_graph, 0));
    cudaGraphNode_t child_inner_node = nullptr;
    CHECK_CUDA(cudaGraphAddEmptyNode(&child_inner_node, child_graph, nullptr, 0));
    cudaGraphNode_t child_graph_node = nullptr;
    CHECK_CUDA(cudaGraphAddChildGraphNode(&child_graph_node, manual_graph, nullptr, 0, child_graph));
    CHECK_CUDA(cudaGraphChildGraphNodeGetGraph(child_graph_node, &containing_graph));
    size_t child_nodes = 0;
    CHECK_CUDA(cudaGraphGetNodes(containing_graph, nullptr, &child_nodes));
    if (child_nodes != 1) {
        std::fprintf(stderr, "unexpected child graph node count: %zu\n", child_nodes);
        return 1;
    }
    CHECK_CUDA(cudaGraphDestroy(child_graph));

    cudaEvent_t event_initial = nullptr;
    cudaEvent_t event_replacement = nullptr;
    cudaEvent_t event = nullptr;
    CHECK_CUDA(cudaEventCreate(&event_initial));
    CHECK_CUDA(cudaEventCreate(&event_replacement));

    cudaGraphNode_t event_record_node = nullptr;
    cudaGraphNode_t event_wait_node = nullptr;
    CHECK_CUDA(cudaGraphAddEventRecordNode(&event_record_node, manual_graph,
                                           nullptr, 0, event_initial));
    CHECK_CUDA(cudaGraphEventRecordNodeGetEvent(event_record_node, &event));
    if (event != event_initial) {
        std::fprintf(stderr, "event record node returned unexpected event\n");
        return 1;
    }
    CHECK_CUDA(cudaGraphEventRecordNodeSetEvent(event_record_node,
                                                event_replacement));
    CHECK_CUDA(cudaGraphEventRecordNodeGetEvent(event_record_node, &event));
    if (event != event_replacement) {
        std::fprintf(stderr, "event record node set unexpected event\n");
        return 1;
    }

    CHECK_CUDA(cudaGraphAddEventWaitNode(&event_wait_node, manual_graph,
                                         nullptr, 0, event_replacement));
    CHECK_CUDA(cudaGraphEventWaitNodeGetEvent(event_wait_node, &event));
    if (event != event_replacement) {
        std::fprintf(stderr, "event wait node returned unexpected event\n");
        return 1;
    }
    CHECK_CUDA(cudaGraphEventWaitNodeSetEvent(event_wait_node, event_initial));
    CHECK_CUDA(cudaGraphEventWaitNodeGetEvent(event_wait_node, &event));
    if (event != event_initial) {
        std::fprintf(stderr, "event wait node set unexpected event\n");
        return 1;
    }
    CHECK_CUDA(cudaGraphEventWaitNodeSetEvent(event_wait_node,
                                              event_replacement));

    cudaGraphNode_t event_from[] = {event_record_node};
    cudaGraphNode_t event_to[] = {event_wait_node};
    CHECK_CUDA(cudaGraphAddDependencies(manual_graph, event_from, event_to,
                                        nullptr, 1));

    exec = nullptr;
    CHECK_CUDA(cudaGraphInstantiate(&exec, manual_graph, 0));
    cudaGraphExecUpdateResultInfo update_info = {};
    CHECK_CUDA(cudaGraphExecUpdate(exec, manual_graph, &update_info));
    if (update_info.result != cudaGraphExecUpdateSuccess) {
        std::fprintf(stderr, "graph exec update failed: %d\n", static_cast<int>(update_info.result));
        return 1;
    }
    unsigned long long exec_flags = 1;
    CHECK_CUDA(cudaGraphExecGetFlags(exec, &exec_flags));
    if (exec_flags != 0) {
        std::fprintf(stderr, "unexpected graph exec flags: %llu\n", exec_flags);
        return 1;
    }
    unsigned int exec_id = 0;
    CHECK_CUDA(cudaGraphExecGetId(exec, &exec_id));
    CHECK_CUDA(cudaGraphExecEventRecordNodeSetEvent(exec, event_record_node,
                                                    event_initial));
    CHECK_CUDA(cudaGraphExecEventWaitNodeSetEvent(exec, event_wait_node,
                                                  event_initial));
    CHECK_CUDA(cudaGraphUpload(exec, stream));
    CHECK_CUDA(cudaGraphLaunch(exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaGraphExecDestroy(exec));

    CHECK_CUDA(cudaGraphRemoveDependencies(manual_graph, event_from, event_to,
                                           nullptr, 1));
    CHECK_CUDA(cudaGraphRemoveDependencies(manual_graph, from, to, nullptr, 1));
    edges = 1;
    CHECK_CUDA(cudaGraphGetEdges(manual_graph, nullptr, nullptr, nullptr, &edges));
    if (edges != 0) {
        std::fprintf(stderr, "dependency removal left %zu edges\n", edges);
        return 1;
    }
    CHECK_CUDA(cudaGraphDestroyNode(node_b));
    CHECK_CUDA(cudaGraphDestroyNode(child_graph_node));
    CHECK_CUDA(cudaGraphDestroyNode(event_wait_node));
    CHECK_CUDA(cudaGraphDestroyNode(event_record_node));
    CHECK_CUDA(cudaEventDestroy(event_replacement));
    CHECK_CUDA(cudaEventDestroy(event_initial));
    nodes = 0;
    CHECK_CUDA(cudaGraphGetNodes(manual_graph, nullptr, &nodes));
    if (nodes != 1) {
        std::fprintf(stderr, "unexpected node count after destroy: %zu\n", nodes);
        return 1;
    }
    CHECK_CUDA(cudaGraphDestroy(manual_graph));
    CHECK_CUDA(cudaStreamDestroy(stream));

    std::puts("graph API test passed");
    return 0;
}
