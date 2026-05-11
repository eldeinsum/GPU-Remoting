#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

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
