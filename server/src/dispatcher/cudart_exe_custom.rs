#![expect(non_snake_case)]

use super::*;
use cudasys::cudart::*;
use cudasys::types::cuda::{CUDA_KERNEL_NODE_PARAMS, CUgraph, CUgraphExec, CUgraphNode, CUresult};
use std::collections::BTreeMap;
use std::os::raw::c_char;
use std::sync::{Mutex, OnceLock};

#[derive(Default)]
struct RuntimeIpcEventState {
    handle_to_event: BTreeMap<[c_char; 64], usize>,
    event_refs: BTreeMap<usize, usize>,
}

fn runtime_ipc_events() -> &'static Mutex<RuntimeIpcEventState> {
    static STATE: OnceLock<Mutex<RuntimeIpcEventState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(RuntimeIpcEventState::default()))
}

fn recv_output_request<C: CommChannel>(channel_receiver: &C) -> (bool, usize) {
    let mut requested = false;
    requested.recv(channel_receiver).unwrap();
    let mut capacity = 0usize;
    capacity.recv(channel_receiver).unwrap();
    (requested, capacity)
}

fn send_result<C: CommChannel>(
    target: &'static str,
    server_id: i32,
    result: cudaError_t,
    channel_sender: &C,
) {
    if result.is_error() {
        log::error!(target: target, "[#{}] returned error: {:?}", server_id, result);
    }
    result.send(channel_sender).unwrap();
    channel_sender.flush_out().unwrap();
}

fn output_len(requested: bool, result: cudaError_t, count: usize, capacity: usize) -> usize {
    if requested && result == cudaError_t::cudaSuccess {
        count.min(capacity)
    } else {
        0
    }
}

fn driver_result_to_runtime(result: CUresult) -> cudaError_t {
    if result == CUresult::CUDA_SUCCESS {
        cudaError_t::cudaSuccess
    } else {
        cudaError_t::cudaErrorUnknown
    }
}

fn recv_kernel_node_params<C: CommChannel>(
    channel_receiver: &C,
) -> (CUDA_KERNEL_NODE_PARAMS, Box<[u32]>, Box<[u8]>) {
    let mut packed_params = std::mem::MaybeUninit::<CUDA_KERNEL_NODE_PARAMS>::uninit();
    packed_params.recv(channel_receiver).unwrap();
    let packed_params = unsafe { packed_params.assume_init() };
    let arg_offsets = recv_slice::<u32, _>(channel_receiver).unwrap();
    let args = recv_slice::<u8, _>(channel_receiver).unwrap();
    (packed_params, arg_offsets, args)
}

pub fn cudaEventDestroyExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cudaEventDestroy", "[#{}]", server.id);

    let mut event = std::mem::MaybeUninit::<cudaEvent_t>::uninit();
    event.recv(channel_receiver).unwrap();
    let event = unsafe { event.assume_init() };
    channel_receiver.recv_ts().unwrap();

    let event_key = event as usize;
    let should_destroy = {
        let mut state = runtime_ipc_events().lock().unwrap();
        match state.event_refs.get_mut(&event_key) {
            Some(refs) if *refs > 1 => {
                *refs -= 1;
                false
            }
            Some(_) => {
                state.event_refs.remove(&event_key);
                state
                    .handle_to_event
                    .retain(|_, mapped_event| *mapped_event != event_key);
                true
            }
            None => true,
        }
    };
    let result = if should_destroy {
        unsafe { cudaEventDestroy(event) }
    } else {
        cudaError_t::cudaSuccess
    };

    send_result("cudaEventDestroy", server.id, result, channel_sender);
}

pub fn cudaIpcGetEventHandleExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cudaIpcGetEventHandle", "[#{}]", server.id);

    let mut event = std::mem::MaybeUninit::<cudaEvent_t>::uninit();
    event.recv(channel_receiver).unwrap();
    let event = unsafe { event.assume_init() };
    channel_receiver.recv_ts().unwrap();

    let mut handle: cudaIpcEventHandle_t = unsafe { std::mem::zeroed() };
    let result = unsafe { cudaIpcGetEventHandle(&raw mut handle, event) };
    if result == cudaError_t::cudaSuccess {
        let mut state = runtime_ipc_events().lock().unwrap();
        state
            .handle_to_event
            .insert(handle.reserved, event as usize);
        state.event_refs.entry(event as usize).or_insert(1);
    }

    handle.send(channel_sender).unwrap();
    send_result("cudaIpcGetEventHandle", server.id, result, channel_sender);
}

pub fn cudaIpcOpenEventHandleExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cudaIpcOpenEventHandle", "[#{}]", server.id);

    let mut handle = std::mem::MaybeUninit::<cudaIpcEventHandle_t>::uninit();
    handle.recv(channel_receiver).unwrap();
    let handle = unsafe { handle.assume_init() };
    channel_receiver.recv_ts().unwrap();

    let mut event: cudaEvent_t = std::ptr::null_mut();
    let mapped_event = {
        let mut state = runtime_ipc_events().lock().unwrap();
        let mapped = state.handle_to_event.get(&handle.reserved).copied();
        if let Some(event_key) = mapped {
            *state.event_refs.entry(event_key).or_insert(1) += 1;
            event = event_key as cudaEvent_t;
        }
        mapped
    };
    let result = if mapped_event.is_some() {
        cudaError_t::cudaSuccess
    } else {
        unsafe { cudaIpcOpenEventHandle(&raw mut event, handle) }
    };

    event.send(channel_sender).unwrap();
    send_result("cudaIpcOpenEventHandle", server.id, result, channel_sender);
}

fn materialize_kernel_params(
    packed_params: &mut CUDA_KERNEL_NODE_PARAMS,
    args: &[u8],
    arg_offsets: &[u32],
) -> Vec<*mut std::ffi::c_void> {
    let mut kernel_params =
        super::cuda_exe_utils::kernel_params_from_packed_args(args, arg_offsets);
    packed_params.kernelParams = if kernel_params.is_empty() {
        std::ptr::null_mut()
    } else {
        kernel_params.as_mut_ptr()
    };
    kernel_params
}

fn runtime_kernel_params_from_driver(
    driver_params: CUDA_KERNEL_NODE_PARAMS,
) -> cudaKernelNodeParams {
    cudaKernelNodeParams {
        func: driver_params.func.cast(),
        gridDim: dim3 {
            x: driver_params.gridDimX,
            y: driver_params.gridDimY,
            z: driver_params.gridDimZ,
        },
        blockDim: dim3 {
            x: driver_params.blockDimX,
            y: driver_params.blockDimY,
            z: driver_params.blockDimZ,
        },
        sharedMemBytes: driver_params.sharedMemBytes,
        kernelParams: std::ptr::null_mut(),
        extra: std::ptr::null_mut(),
    }
}

fn make_edge_data_buffer(capacity: usize) -> Vec<cudaGraphEdgeData> {
    (0..capacity)
        .map(|_| unsafe { std::mem::zeroed() })
        .collect()
}

pub fn cudaGraphGetNodesExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cudaGraphGetNodes", "[#{}]", server.id);

    let mut graph = std::mem::MaybeUninit::<cudaGraph_t>::uninit();
    graph.recv(channel_receiver).unwrap();
    let graph = unsafe { graph.assume_init() };
    let (requested, capacity) = recv_output_request(channel_receiver);
    channel_receiver.recv_ts().unwrap();

    let mut count = capacity;
    let mut nodes = vec![std::ptr::null_mut(); if requested { capacity } else { 0 }];
    let nodes_ptr = if requested {
        nodes.as_mut_ptr()
    } else {
        std::ptr::null_mut()
    };
    let result = unsafe { cudaGraphGetNodes(graph, nodes_ptr, &raw mut count) };

    if requested {
        send_slice(
            &nodes[..output_len(requested, result, count, capacity)],
            channel_sender,
        )
        .unwrap();
    }
    count.send(channel_sender).unwrap();
    send_result("cudaGraphGetNodes", server.id, result, channel_sender);
}

pub fn cudaGraphGetRootNodesExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cudaGraphGetRootNodes", "[#{}]", server.id);

    let mut graph = std::mem::MaybeUninit::<cudaGraph_t>::uninit();
    graph.recv(channel_receiver).unwrap();
    let graph = unsafe { graph.assume_init() };
    let (requested, capacity) = recv_output_request(channel_receiver);
    channel_receiver.recv_ts().unwrap();

    let mut count = capacity;
    let mut nodes = vec![std::ptr::null_mut(); if requested { capacity } else { 0 }];
    let nodes_ptr = if requested {
        nodes.as_mut_ptr()
    } else {
        std::ptr::null_mut()
    };
    let result = unsafe { cudaGraphGetRootNodes(graph, nodes_ptr, &raw mut count) };

    if requested {
        send_slice(
            &nodes[..output_len(requested, result, count, capacity)],
            channel_sender,
        )
        .unwrap();
    }
    count.send(channel_sender).unwrap();
    send_result("cudaGraphGetRootNodes", server.id, result, channel_sender);
}

pub fn cudaGraphGetEdgesExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cudaGraphGetEdges", "[#{}]", server.id);

    let mut graph = std::mem::MaybeUninit::<cudaGraph_t>::uninit();
    graph.recv(channel_receiver).unwrap();
    let graph = unsafe { graph.assume_init() };
    let mut wants_from = false;
    wants_from.recv(channel_receiver).unwrap();
    let mut wants_to = false;
    wants_to.recv(channel_receiver).unwrap();
    let (wants_edge_data, capacity) = recv_output_request(channel_receiver);
    channel_receiver.recv_ts().unwrap();

    let mut count = capacity;
    let mut from = vec![std::ptr::null_mut(); if wants_from { capacity } else { 0 }];
    let mut to = vec![std::ptr::null_mut(); if wants_to { capacity } else { 0 }];
    let mut edge_data = make_edge_data_buffer(if wants_edge_data { capacity } else { 0 });
    let from_ptr = if wants_from {
        from.as_mut_ptr()
    } else {
        std::ptr::null_mut()
    };
    let to_ptr = if wants_to {
        to.as_mut_ptr()
    } else {
        std::ptr::null_mut()
    };
    let edge_data_ptr = if wants_edge_data {
        edge_data.as_mut_ptr()
    } else {
        std::ptr::null_mut()
    };
    let result =
        unsafe { cudaGraphGetEdges(graph, from_ptr, to_ptr, edge_data_ptr, &raw mut count) };
    let len = output_len(
        wants_from || wants_to || wants_edge_data,
        result,
        count,
        capacity,
    );

    if wants_from {
        send_slice(&from[..len], channel_sender).unwrap();
    }
    if wants_to {
        send_slice(&to[..len], channel_sender).unwrap();
    }
    if wants_edge_data {
        send_slice(&edge_data[..len], channel_sender).unwrap();
    }
    count.send(channel_sender).unwrap();
    send_result("cudaGraphGetEdges", server.id, result, channel_sender);
}

pub fn cudaGraphAddKernelNodeExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cudaGraphAddKernelNode", "[#{}]", server.id);

    let mut graph = std::mem::MaybeUninit::<cudaGraph_t>::uninit();
    graph.recv(channel_receiver).unwrap();
    let graph = unsafe { graph.assume_init() };
    let (mut packed_params, arg_offsets, args) = recv_kernel_node_params(channel_receiver);
    channel_receiver.recv_ts().unwrap();

    let _kernel_params = materialize_kernel_params(&mut packed_params, &args, &arg_offsets);
    let mut node: CUgraphNode = std::ptr::null_mut();
    let driver_result = unsafe {
        cudasys::cuda::cuGraphAddKernelNode_v2(
            &raw mut node,
            graph as CUgraph,
            std::ptr::null(),
            0,
            &raw const packed_params,
        )
    };
    let result = driver_result_to_runtime(driver_result);

    (node as cudaGraphNode_t).send(channel_sender).unwrap();
    send_result("cudaGraphAddKernelNode", server.id, result, channel_sender);
}

pub fn cudaGraphKernelNodeGetParamsExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cudaGraphKernelNodeGetParams", "[#{}]", server.id);

    let mut node = std::mem::MaybeUninit::<cudaGraphNode_t>::uninit();
    node.recv(channel_receiver).unwrap();
    let node = unsafe { node.assume_init() };
    channel_receiver.recv_ts().unwrap();

    let mut driver_params = std::mem::MaybeUninit::<CUDA_KERNEL_NODE_PARAMS>::uninit();
    let driver_result = unsafe {
        cudasys::cuda::cuGraphKernelNodeGetParams_v2(
            node as CUgraphNode,
            driver_params.as_mut_ptr(),
        )
    };
    let result = driver_result_to_runtime(driver_result);
    let runtime_params = if result == cudaError_t::cudaSuccess {
        runtime_kernel_params_from_driver(unsafe { driver_params.assume_init() })
    } else {
        unsafe { std::mem::zeroed() }
    };

    runtime_params.send(channel_sender).unwrap();
    send_result(
        "cudaGraphKernelNodeGetParams",
        server.id,
        result,
        channel_sender,
    );
}

pub fn cudaGraphKernelNodeSetParamsExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cudaGraphKernelNodeSetParams", "[#{}]", server.id);

    let mut node = std::mem::MaybeUninit::<cudaGraphNode_t>::uninit();
    node.recv(channel_receiver).unwrap();
    let node = unsafe { node.assume_init() };
    let (mut packed_params, arg_offsets, args) = recv_kernel_node_params(channel_receiver);
    channel_receiver.recv_ts().unwrap();

    let _kernel_params = materialize_kernel_params(&mut packed_params, &args, &arg_offsets);
    let driver_result = unsafe {
        cudasys::cuda::cuGraphKernelNodeSetParams_v2(node as CUgraphNode, &raw const packed_params)
    };
    send_result(
        "cudaGraphKernelNodeSetParams",
        server.id,
        driver_result_to_runtime(driver_result),
        channel_sender,
    );
}

pub fn cudaGraphExecKernelNodeSetParamsExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cudaGraphExecKernelNodeSetParams", "[#{}]", server.id);

    let mut graph_exec = std::mem::MaybeUninit::<cudaGraphExec_t>::uninit();
    graph_exec.recv(channel_receiver).unwrap();
    let graph_exec = unsafe { graph_exec.assume_init() };
    let mut node = std::mem::MaybeUninit::<cudaGraphNode_t>::uninit();
    node.recv(channel_receiver).unwrap();
    let node = unsafe { node.assume_init() };
    let (mut packed_params, arg_offsets, args) = recv_kernel_node_params(channel_receiver);
    channel_receiver.recv_ts().unwrap();

    let _kernel_params = materialize_kernel_params(&mut packed_params, &args, &arg_offsets);
    let driver_result = unsafe {
        cudasys::cuda::cuGraphExecKernelNodeSetParams_v2(
            graph_exec as CUgraphExec,
            node as CUgraphNode,
            &raw const packed_params,
        )
    };
    send_result(
        "cudaGraphExecKernelNodeSetParams",
        server.id,
        driver_result_to_runtime(driver_result),
        channel_sender,
    );
}

pub fn cudaGraphNodeGetDependenciesExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cudaGraphNodeGetDependencies", "[#{}]", server.id);

    let mut node = std::mem::MaybeUninit::<cudaGraphNode_t>::uninit();
    node.recv(channel_receiver).unwrap();
    let node = unsafe { node.assume_init() };
    let mut wants_dependencies = false;
    wants_dependencies.recv(channel_receiver).unwrap();
    let mut wants_to = false;
    wants_to.recv(channel_receiver).unwrap();
    assert!(!wants_to);
    let (wants_edge_data, capacity) = recv_output_request(channel_receiver);
    channel_receiver.recv_ts().unwrap();

    let mut count = capacity;
    let mut dependencies =
        vec![std::ptr::null_mut(); if wants_dependencies { capacity } else { 0 }];
    let mut edge_data = make_edge_data_buffer(if wants_edge_data { capacity } else { 0 });
    let dependencies_ptr = if wants_dependencies {
        dependencies.as_mut_ptr()
    } else {
        std::ptr::null_mut()
    };
    let edge_data_ptr = if wants_edge_data {
        edge_data.as_mut_ptr()
    } else {
        std::ptr::null_mut()
    };
    let result = unsafe {
        cudaGraphNodeGetDependencies(node, dependencies_ptr, edge_data_ptr, &raw mut count)
    };
    let len = output_len(
        wants_dependencies || wants_edge_data,
        result,
        count,
        capacity,
    );

    if wants_dependencies {
        send_slice(&dependencies[..len], channel_sender).unwrap();
    }
    if wants_edge_data {
        send_slice(&edge_data[..len], channel_sender).unwrap();
    }
    count.send(channel_sender).unwrap();
    send_result(
        "cudaGraphNodeGetDependencies",
        server.id,
        result,
        channel_sender,
    );
}

pub fn cudaGraphNodeGetDependentNodesExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cudaGraphNodeGetDependentNodes", "[#{}]", server.id);

    let mut node = std::mem::MaybeUninit::<cudaGraphNode_t>::uninit();
    node.recv(channel_receiver).unwrap();
    let node = unsafe { node.assume_init() };
    let mut wants_dependents = false;
    wants_dependents.recv(channel_receiver).unwrap();
    let mut wants_to = false;
    wants_to.recv(channel_receiver).unwrap();
    assert!(!wants_to);
    let (wants_edge_data, capacity) = recv_output_request(channel_receiver);
    channel_receiver.recv_ts().unwrap();

    let mut count = capacity;
    let mut dependents = vec![std::ptr::null_mut(); if wants_dependents { capacity } else { 0 }];
    let mut edge_data = make_edge_data_buffer(if wants_edge_data { capacity } else { 0 });
    let dependents_ptr = if wants_dependents {
        dependents.as_mut_ptr()
    } else {
        std::ptr::null_mut()
    };
    let edge_data_ptr = if wants_edge_data {
        edge_data.as_mut_ptr()
    } else {
        std::ptr::null_mut()
    };
    let result = unsafe {
        cudaGraphNodeGetDependentNodes(node, dependents_ptr, edge_data_ptr, &raw mut count)
    };
    let len = output_len(wants_dependents || wants_edge_data, result, count, capacity);

    if wants_dependents {
        send_slice(&dependents[..len], channel_sender).unwrap();
    }
    if wants_edge_data {
        send_slice(&edge_data[..len], channel_sender).unwrap();
    }
    count.send(channel_sender).unwrap();
    send_result(
        "cudaGraphNodeGetDependentNodes",
        server.id,
        result,
        channel_sender,
    );
}
