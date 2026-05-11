#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
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

    cudaGraph_t capture_graph = nullptr;
    CHECK_CUDA(cudaGraphCreate(&capture_graph, 0));
    cudaGraphNode_t capture_dependency = nullptr;
    CHECK_CUDA(cudaGraphAddEmptyNode(&capture_dependency, capture_graph,
                                     nullptr, 0));
    cudaGraphNode_t capture_updated_dependency = nullptr;
    CHECK_CUDA(cudaGraphAddEmptyNode(&capture_updated_dependency,
                                     capture_graph, nullptr, 0));
    cudaGraphNode_t capture_dependencies[] = {capture_dependency};
    CHECK_CUDA(cudaStreamBeginCaptureToGraph(
        stream, capture_graph, capture_dependencies, nullptr, 1,
        cudaStreamCaptureModeThreadLocal));
    cudaGraphNode_t updated_dependencies[] = {capture_updated_dependency};
    CHECK_CUDA(cudaStreamUpdateCaptureDependencies(
        stream, updated_dependencies, nullptr, 1,
        cudaStreamSetCaptureDependencies));
    CHECK_CUDA(cudaMemsetAsync(device, 0x6b, kBytes, stream));
    cudaGraph_t captured_to_graph = nullptr;
    CHECK_CUDA(cudaStreamEndCapture(stream, &captured_to_graph));
    if (captured_to_graph == nullptr) {
        std::fprintf(stderr, "capture-to-graph returned a null graph\n");
        return 1;
    }
    nodes = 0;
    CHECK_CUDA(cudaGraphGetNodes(captured_to_graph, nullptr, &nodes));
    if (nodes != 3) {
        std::fprintf(stderr, "unexpected capture-to-graph node count: %zu\n",
                     nodes);
        return 1;
    }
    size_t capture_edges = 0;
    CHECK_CUDA(cudaGraphGetEdges(captured_to_graph, nullptr, nullptr,
                                 nullptr, &capture_edges));
    if (capture_edges != 1) {
        std::fprintf(stderr, "unexpected capture-to-graph edge count: %zu\n",
                     capture_edges);
        return 1;
    }
    cudaGraphInstantiateParams instantiate_params = {};
    CHECK_CUDA(cudaGraphInstantiateWithParams(&exec, captured_to_graph,
                                              &instantiate_params));
    if (instantiate_params.result_out != cudaGraphInstantiateSuccess ||
        instantiate_params.errNode_out != nullptr) {
        std::fprintf(stderr, "unexpected instantiate-with-params result\n");
        return 1;
    }
    CHECK_CUDA(cudaMemset(device, 0, kBytes));
    CHECK_CUDA(cudaGraphLaunch(exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    output.assign(kBytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), device, kBytes,
                          cudaMemcpyDeviceToHost));
    if (verify_value(output, 0x6b) != 0) {
        return 1;
    }
    CHECK_CUDA(cudaGraphExecDestroy(exec));
    CHECK_CUDA(cudaGraphDestroy(captured_to_graph));

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

    CHECK_CUDA(cudaMemset(device, 0, kBytes));
    cudaGraph_t generic_graph = nullptr;
    CHECK_CUDA(cudaGraphCreate(&generic_graph, 0));
    cudaGraphNodeParams generic_params = {};
    generic_params.type = cudaGraphNodeTypeMemset;
    generic_params.memset.dst = device;
    generic_params.memset.value = 0x34;
    generic_params.memset.elementSize = 1;
    generic_params.memset.width = kBytes;
    generic_params.memset.height = 1;

    cudaGraphNode_t generic_node = nullptr;
    CHECK_CUDA(cudaGraphAddNode(&generic_node, generic_graph, nullptr, nullptr,
                                0, &generic_params));
    cudaGraphNodeParams queried_generic = {};
    CHECK_CUDA(cudaGraphNodeGetParams(generic_node, &queried_generic));
    if (queried_generic.type != cudaGraphNodeTypeMemset ||
        queried_generic.memset.dst != device ||
        queried_generic.memset.value != 0x34 ||
        queried_generic.memset.width != kBytes ||
        queried_generic.memset.height != 1) {
        std::fprintf(stderr, "unexpected runtime generic graph node params\n");
        return 1;
    }
    generic_params.memset.value = 0x45;
    CHECK_CUDA(cudaGraphNodeSetParams(generic_node, &generic_params));
    queried_generic = {};
    CHECK_CUDA(cudaGraphNodeGetParams(generic_node, &queried_generic));
    if (queried_generic.memset.value != 0x45) {
        std::fprintf(stderr, "runtime generic graph update did not stick\n");
        return 1;
    }
    exec = nullptr;
    CHECK_CUDA(cudaGraphInstantiate(&exec, generic_graph, 0));
    generic_params.memset.value = 0x56;
    CHECK_CUDA(cudaGraphExecNodeSetParams(exec, generic_node,
                                          &generic_params));
    CHECK_CUDA(cudaGraphLaunch(exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    output.assign(kBytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), device, kBytes,
                          cudaMemcpyDeviceToHost));
    if (verify_value(output, 0x56) != 0) {
        return 1;
    }
    CHECK_CUDA(cudaGraphExecDestroy(exec));
    CHECK_CUDA(cudaGraphDestroy(generic_graph));
    CHECK_CUDA(cudaMemset(device, 0x7f, kBytes));

    unsigned char *copy_src = nullptr;
    unsigned char *copy_dst = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&copy_src), kBytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&copy_dst), kBytes));
    CHECK_CUDA(cudaMemset(copy_src, 0x11, kBytes));
    CHECK_CUDA(cudaMemset(copy_dst, 0, kBytes));

    cudaGraph_t memcpy_graph = nullptr;
    CHECK_CUDA(cudaGraphCreate(&memcpy_graph, 0));
    cudaGraphNode_t memcpy_node = nullptr;
    CHECK_CUDA(cudaGraphAddMemcpyNode1D(&memcpy_node, memcpy_graph, nullptr,
                                        0, copy_dst, copy_src, kBytes,
                                        cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaGraphMemcpyNodeSetParams1D(memcpy_node, copy_dst,
                                              copy_src, kBytes,
                                              cudaMemcpyDeviceToDevice));
    exec = nullptr;
    CHECK_CUDA(cudaGraphInstantiate(&exec, memcpy_graph, 0));
    CHECK_CUDA(cudaGraphLaunch(exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    output.assign(kBytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), copy_dst, kBytes,
                          cudaMemcpyDeviceToHost));
    if (verify_value(output, 0x11) != 0) {
        return 1;
    }

    CHECK_CUDA(cudaMemset(copy_dst, 0, kBytes));
    CHECK_CUDA(cudaGraphExecMemcpyNodeSetParams1D(exec, memcpy_node,
                                                  copy_dst, device, kBytes,
                                                  cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaGraphLaunch(exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    output.assign(kBytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), copy_dst, kBytes,
                          cudaMemcpyDeviceToHost));
    if (verify_value(output, 0x7f) != 0) {
        return 1;
    }
    CHECK_CUDA(cudaGraphExecDestroy(exec));
    CHECK_CUDA(cudaGraphDestroy(memcpy_graph));

    CHECK_CUDA(cudaMemset(copy_src, 0x22, kBytes));
    CHECK_CUDA(cudaMemset(copy_dst, 0, kBytes));

    cudaMemcpy3DParms copy_params = {};
    copy_params.srcPtr = make_cudaPitchedPtr(copy_src, kBytes, kBytes, 1);
    copy_params.dstPtr = make_cudaPitchedPtr(copy_dst, kBytes, kBytes, 1);
    copy_params.extent = make_cudaExtent(kBytes, 1, 1);
    copy_params.kind = cudaMemcpyDeviceToDevice;

    CHECK_CUDA(cudaGraphCreate(&memcpy_graph, 0));
    memcpy_node = nullptr;
    CHECK_CUDA(cudaGraphAddMemcpyNode(&memcpy_node, memcpy_graph, nullptr,
                                      0, &copy_params));
    cudaMemcpy3DParms queried_copy = {};
    CHECK_CUDA(cudaGraphMemcpyNodeGetParams(memcpy_node, &queried_copy));
    if (queried_copy.srcPtr.ptr != copy_src ||
        queried_copy.dstPtr.ptr != copy_dst ||
        queried_copy.extent.width != kBytes ||
        queried_copy.extent.height != 1 ||
        queried_copy.extent.depth != 1 ||
        queried_copy.kind != cudaMemcpyDeviceToDevice) {
        std::fprintf(stderr, "unexpected graph memcpy node params\n");
        return 1;
    }

    CHECK_CUDA(cudaMemset(device, 0x33, kBytes));
    copy_params.srcPtr = make_cudaPitchedPtr(device, kBytes, kBytes, 1);
    CHECK_CUDA(cudaGraphMemcpyNodeSetParams(memcpy_node, &copy_params));
    CHECK_CUDA(cudaGraphInstantiate(&exec, memcpy_graph, 0));
    CHECK_CUDA(cudaGraphLaunch(exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    output.assign(kBytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), copy_dst, kBytes,
                          cudaMemcpyDeviceToHost));
    if (verify_value(output, 0x33) != 0) {
        return 1;
    }

    CHECK_CUDA(cudaMemset(copy_src, 0x44, kBytes));
    CHECK_CUDA(cudaMemset(copy_dst, 0, kBytes));
    copy_params.srcPtr = make_cudaPitchedPtr(copy_src, kBytes, kBytes, 1);
    CHECK_CUDA(cudaGraphExecMemcpyNodeSetParams(exec, memcpy_node,
                                                &copy_params));
    CHECK_CUDA(cudaGraphLaunch(exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    output.assign(kBytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), copy_dst, kBytes,
                          cudaMemcpyDeviceToHost));
    if (verify_value(output, 0x44) != 0) {
        return 1;
    }

    unsigned int node_enabled = 0;
    CHECK_CUDA(cudaGraphNodeGetEnabled(exec, memcpy_node, &node_enabled));
    if (node_enabled != 1) {
        std::fprintf(stderr, "unexpected graph node enabled state: %u\n",
                     node_enabled);
        return 1;
    }
    CHECK_CUDA(cudaGraphNodeSetEnabled(exec, memcpy_node, 0));
    CHECK_CUDA(cudaGraphNodeGetEnabled(exec, memcpy_node, &node_enabled));
    if (node_enabled != 0) {
        std::fprintf(stderr, "graph node disable did not persist\n");
        return 1;
    }
    CHECK_CUDA(cudaMemset(copy_dst, 0, kBytes));
    CHECK_CUDA(cudaGraphLaunch(exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    output.assign(kBytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), copy_dst, kBytes,
                          cudaMemcpyDeviceToHost));
    if (verify_value(output, 0) != 0) {
        return 1;
    }
    CHECK_CUDA(cudaGraphNodeSetEnabled(exec, memcpy_node, 1));
    CHECK_CUDA(cudaGraphNodeGetEnabled(exec, memcpy_node, &node_enabled));
    if (node_enabled != 1) {
        std::fprintf(stderr, "graph node enable did not persist\n");
        return 1;
    }
    CHECK_CUDA(cudaGraphLaunch(exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    output.assign(kBytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), copy_dst, kBytes,
                          cudaMemcpyDeviceToHost));
    if (verify_value(output, 0x44) != 0) {
        return 1;
    }
    CHECK_CUDA(cudaGraphExecDestroy(exec));
    CHECK_CUDA(cudaGraphDestroy(memcpy_graph));

    cudaGraph_t alloc_graph = nullptr;
    CHECK_CUDA(cudaGraphCreate(&alloc_graph, 0));
    cudaMemAllocNodeParams alloc_params = {};
    alloc_params.poolProps.allocType = cudaMemAllocationTypePinned;
    alloc_params.poolProps.handleTypes = cudaMemHandleTypeNone;
    alloc_params.poolProps.location.type = cudaMemLocationTypeDevice;
    alloc_params.poolProps.location.id = 0;
    alloc_params.bytesize = kBytes;

    cudaGraphNode_t alloc_node = nullptr;
    CHECK_CUDA(cudaGraphAddMemAllocNode(&alloc_node, alloc_graph, nullptr,
                                        0, &alloc_params));
    if (alloc_params.dptr == nullptr) {
        std::fprintf(stderr, "graph alloc node returned null allocation\n");
        return 1;
    }
    cudaMemAllocNodeParams queried_alloc = {};
    CHECK_CUDA(cudaGraphMemAllocNodeGetParams(alloc_node, &queried_alloc));
    if (queried_alloc.dptr != alloc_params.dptr ||
        queried_alloc.bytesize != kBytes) {
        std::fprintf(stderr, "unexpected graph alloc node params\n");
        return 1;
    }

    cudaMemsetParams alloc_memset_params = {};
    alloc_memset_params.dst = alloc_params.dptr;
    alloc_memset_params.value = 0x55;
    alloc_memset_params.elementSize = 1;
    alloc_memset_params.width = kBytes;
    alloc_memset_params.height = 1;
    cudaGraphNode_t alloc_memset_node = nullptr;
    CHECK_CUDA(cudaGraphAddMemsetNode(&alloc_memset_node, alloc_graph,
                                      nullptr, 0, &alloc_memset_params));

    cudaMemcpy3DParms alloc_copy_params = {};
    alloc_copy_params.srcPtr = make_cudaPitchedPtr(alloc_params.dptr, kBytes,
                                                   kBytes, 1);
    alloc_copy_params.dstPtr = make_cudaPitchedPtr(copy_dst, kBytes, kBytes, 1);
    alloc_copy_params.extent = make_cudaExtent(kBytes, 1, 1);
    alloc_copy_params.kind = cudaMemcpyDeviceToDevice;
    cudaGraphNode_t alloc_copy_node = nullptr;
    CHECK_CUDA(cudaGraphAddMemcpyNode(&alloc_copy_node, alloc_graph, nullptr,
                                      0, &alloc_copy_params));

    cudaGraphNode_t free_node = nullptr;
    CHECK_CUDA(cudaGraphAddMemFreeNode(&free_node, alloc_graph, nullptr, 0,
                                       alloc_params.dptr));
    void *queried_free = nullptr;
    CHECK_CUDA(cudaGraphMemFreeNodeGetParams(free_node, &queried_free));
    if (queried_free != alloc_params.dptr) {
        std::fprintf(stderr, "unexpected graph free node pointer\n");
        return 1;
    }

    cudaGraphNode_t alloc_from[] = {alloc_node, alloc_memset_node,
                                    alloc_copy_node};
    cudaGraphNode_t alloc_to[] = {alloc_memset_node, alloc_copy_node,
                                  free_node};
    CHECK_CUDA(cudaGraphAddDependencies(alloc_graph, alloc_from, alloc_to,
                                        nullptr, 3));
    CHECK_CUDA(cudaMemset(copy_dst, 0, kBytes));
    exec = nullptr;
    CHECK_CUDA(cudaGraphInstantiate(&exec, alloc_graph, 0));
    CHECK_CUDA(cudaGraphLaunch(exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    output.assign(kBytes, 0);
    CHECK_CUDA(cudaMemcpy(output.data(), copy_dst, kBytes,
                          cudaMemcpyDeviceToHost));
    if (verify_value(output, 0x55) != 0) {
        return 1;
    }
    CHECK_CUDA(cudaGraphExecDestroy(exec));
    CHECK_CUDA(cudaGraphDestroy(alloc_graph));

    unsigned long long graph_mem_value = 0;
    CHECK_CUDA(cudaDeviceGetGraphMemAttribute(
        0, cudaGraphMemAttrUsedMemCurrent, &graph_mem_value));
    CHECK_CUDA(cudaDeviceGetGraphMemAttribute(
        0, cudaGraphMemAttrReservedMemCurrent, &graph_mem_value));
    graph_mem_value = 0;
    CHECK_CUDA(cudaDeviceSetGraphMemAttribute(
        0, cudaGraphMemAttrUsedMemHigh, &graph_mem_value));
    CHECK_CUDA(cudaDeviceSetGraphMemAttribute(
        0, cudaGraphMemAttrReservedMemHigh, &graph_mem_value));
    CHECK_CUDA(cudaDeviceGraphMemTrim(0));
    CHECK_CUDA(cudaFree(copy_dst));
    CHECK_CUDA(cudaFree(copy_src));

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
    cudaGraphNode_t root_node_list[1] = {};
    size_t root_node_capacity = 1;
    CHECK_CUDA(cudaGraphGetRootNodes(manual_graph, root_node_list,
                                     &root_node_capacity));
    if (root_node_capacity != 1 || root_node_list[0] != node_a) {
        std::fprintf(stderr, "unexpected root node query result\n");
        return 1;
    }

    size_t edges = 0;
    CHECK_CUDA(cudaGraphGetEdges(manual_graph, nullptr, nullptr, nullptr, &edges));
    if (edges != 1) {
        std::fprintf(stderr, "unexpected edge count: %zu\n", edges);
        return 1;
    }
    cudaGraphNode_t edge_from[1] = {};
    cudaGraphNode_t edge_to[1] = {};
    cudaGraphEdgeData edge_data[1] = {};
    size_t edge_capacity = 1;
    CHECK_CUDA(cudaGraphGetEdges(manual_graph, edge_from, edge_to, edge_data,
                                 &edge_capacity));
    if (edge_capacity != 1 || edge_from[0] != node_a || edge_to[0] != node_b) {
        std::fprintf(stderr, "unexpected edge query result\n");
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
    cudaGraphNode_t dependencies[1] = {};
    cudaGraphEdgeData dependency_edge_data[1] = {};
    size_t dependency_capacity = 1;
    CHECK_CUDA(cudaGraphNodeGetDependencies(node_b, dependencies,
                                            dependency_edge_data,
                                            &dependency_capacity));
    if (dependency_capacity != 1 || dependencies[0] != node_a) {
        std::fprintf(stderr, "unexpected dependency query result\n");
        return 1;
    }

    size_t dependent_count = 0;
    CHECK_CUDA(cudaGraphNodeGetDependentNodes(node_a, nullptr, nullptr, &dependent_count));
    if (dependent_count != 1) {
        std::fprintf(stderr, "unexpected dependent count: %zu\n", dependent_count);
        return 1;
    }
    cudaGraphNode_t dependents[1] = {};
    cudaGraphEdgeData dependent_edge_data[1] = {};
    size_t dependent_capacity = 1;
    CHECK_CUDA(cudaGraphNodeGetDependentNodes(node_a, dependents,
                                              dependent_edge_data,
                                              &dependent_capacity));
    if (dependent_capacity != 1 || dependents[0] != node_b) {
        std::fprintf(stderr, "unexpected dependent query result\n");
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
    cudaGraphNode_t child_node_list[1] = {};
    size_t child_node_capacity = 1;
    CHECK_CUDA(cudaGraphGetNodes(containing_graph, child_node_list,
                                 &child_node_capacity));
    if (child_node_capacity != 1 || child_node_list[0] == nullptr) {
        std::fprintf(stderr, "unexpected child graph node query result\n");
        return 1;
    }
    CHECK_CUDA(cudaGraphDestroy(child_graph));
    cudaGraph_t replacement_child_graph = nullptr;
    CHECK_CUDA(cudaGraphCreate(&replacement_child_graph, 0));
    cudaGraphNode_t replacement_child_node = nullptr;
    CHECK_CUDA(cudaGraphAddEmptyNode(&replacement_child_node,
                                     replacement_child_graph, nullptr, 0));

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
    CHECK_CUDA(cudaGraphExecChildGraphNodeSetParams(
        exec, child_graph_node, replacement_child_graph));
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
    CHECK_CUDA(cudaGraphDestroy(replacement_child_graph));

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
    cudaGraphNode_t remaining_nodes[1] = {};
    size_t remaining_node_capacity = 1;
    CHECK_CUDA(cudaGraphGetNodes(manual_graph, remaining_nodes,
                                 &remaining_node_capacity));
    if (remaining_node_capacity != 1 || remaining_nodes[0] != node_a) {
        std::fprintf(stderr, "unexpected remaining node query result\n");
        return 1;
    }
    char dot_path[128] = {};
    std::snprintf(dot_path, sizeof(dot_path),
                  "/tmp/gpu_remoting_graph_extra_%d.dot",
                  static_cast<int>(getpid()));
    std::remove(dot_path);
    CHECK_CUDA(cudaGraphDebugDotPrint(manual_graph, dot_path, 0));
    FILE *dot_file = std::fopen(dot_path, "r");
    if (dot_file == nullptr) {
        std::fprintf(stderr, "cudaGraphDebugDotPrint did not create output\n");
        return 1;
    }
    std::fclose(dot_file);
    std::remove(dot_path);
    CHECK_CUDA(cudaGraphDestroy(manual_graph));
    CHECK_CUDA(cudaStreamDestroy(stream));

    std::puts("graph API test passed");
    return 0;
}
