#![expect(non_snake_case)]

use crate::ServerWorker;
use cudasys::cuda::*;
use network::type_impl::{recv_slice, send_slice};
use network::{CommChannel, Transportable};
use std::collections::BTreeMap;
use std::os::raw::c_char;
use std::sync::{Mutex, OnceLock};

#[derive(Default)]
struct DriverIpcEventState {
    handle_to_event: BTreeMap<[c_char; 64], usize>,
    event_refs: BTreeMap<usize, usize>,
}

fn driver_ipc_events() -> &'static Mutex<DriverIpcEventState> {
    static STATE: OnceLock<Mutex<DriverIpcEventState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(DriverIpcEventState::default()))
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
    result: CUresult,
    channel_sender: &C,
) {
    if result != CUresult::CUDA_SUCCESS {
        log::error!(target: target, "[#{}] returned error: {:?}", server_id, result);
    }
    result.send(channel_sender).unwrap();
    channel_sender.flush_out().unwrap();
}

fn output_len(requested: bool, result: CUresult, count: usize, capacity: usize) -> usize {
    if requested && result == CUresult::CUDA_SUCCESS {
        count.min(capacity)
    } else {
        0
    }
}

fn make_edge_data_buffer(capacity: usize) -> Vec<CUgraphEdgeData> {
    (0..capacity)
        .map(|_| unsafe { std::mem::zeroed() })
        .collect()
}

pub fn cuTexObjectCreateExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuTexObjectCreate", "[#{}]", server.id);

    let mut res_desc = std::mem::MaybeUninit::<CUDA_RESOURCE_DESC>::uninit();
    res_desc.recv(channel_receiver).unwrap();
    let res_desc = unsafe { res_desc.assume_init() };

    let mut tex_desc = std::mem::MaybeUninit::<CUDA_TEXTURE_DESC>::uninit();
    tex_desc.recv(channel_receiver).unwrap();
    let tex_desc = unsafe { tex_desc.assume_init() };

    let mut has_view_desc = false;
    has_view_desc.recv(channel_receiver).unwrap();
    let mut view_desc = std::mem::MaybeUninit::<CUDA_RESOURCE_VIEW_DESC>::uninit();
    let view_desc_ptr = if has_view_desc {
        view_desc.recv(channel_receiver).unwrap();
        unsafe { view_desc.assume_init_ref() as *const CUDA_RESOURCE_VIEW_DESC }
    } else {
        std::ptr::null()
    };
    channel_receiver.recv_ts().unwrap();

    let mut tex_object = 0;
    let result = unsafe {
        cuTexObjectCreate(
            &raw mut tex_object,
            &raw const res_desc,
            &raw const tex_desc,
            view_desc_ptr,
        )
    };

    tex_object.send(channel_sender).unwrap();
    send_result("cuTexObjectCreate", server.id, result, channel_sender);
}

pub fn cuTexObjectGetResourceDescExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuTexObjectGetResourceDesc", "[#{}]", server.id);

    let mut tex_object = std::mem::MaybeUninit::<CUtexObject>::uninit();
    tex_object.recv(channel_receiver).unwrap();
    let tex_object = unsafe { tex_object.assume_init() };
    channel_receiver.recv_ts().unwrap();

    let mut res_desc = unsafe { std::mem::zeroed::<CUDA_RESOURCE_DESC>() };
    let result = unsafe { cuTexObjectGetResourceDesc(&raw mut res_desc, tex_object) };

    res_desc.send(channel_sender).unwrap();
    send_result(
        "cuTexObjectGetResourceDesc",
        server.id,
        result,
        channel_sender,
    );
}

pub fn cuSurfObjectCreateExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuSurfObjectCreate", "[#{}]", server.id);

    let mut res_desc = std::mem::MaybeUninit::<CUDA_RESOURCE_DESC>::uninit();
    res_desc.recv(channel_receiver).unwrap();
    let res_desc = unsafe { res_desc.assume_init() };
    channel_receiver.recv_ts().unwrap();

    let mut surf_object = 0;
    let result = unsafe { cuSurfObjectCreate(&raw mut surf_object, &raw const res_desc) };

    surf_object.send(channel_sender).unwrap();
    send_result("cuSurfObjectCreate", server.id, result, channel_sender);
}

pub fn cuSurfObjectGetResourceDescExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuSurfObjectGetResourceDesc", "[#{}]", server.id);

    let mut surf_object = std::mem::MaybeUninit::<CUsurfObject>::uninit();
    surf_object.recv(channel_receiver).unwrap();
    let surf_object = unsafe { surf_object.assume_init() };
    channel_receiver.recv_ts().unwrap();

    let mut res_desc = unsafe { std::mem::zeroed::<CUDA_RESOURCE_DESC>() };
    let result = unsafe { cuSurfObjectGetResourceDesc(&raw mut res_desc, surf_object) };

    res_desc.send(channel_sender).unwrap();
    send_result(
        "cuSurfObjectGetResourceDesc",
        server.id,
        result,
        channel_sender,
    );
}

pub fn cuEventDestroy_v2Exe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuEventDestroy_v2", "[#{}]", server.id);

    let mut event = std::mem::MaybeUninit::<CUevent>::uninit();
    event.recv(channel_receiver).unwrap();
    let event = unsafe { event.assume_init() };
    channel_receiver.recv_ts().unwrap();

    let event_key = event as usize;
    let should_destroy = {
        let mut state = driver_ipc_events().lock().unwrap();
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
        unsafe { cuEventDestroy_v2(event) }
    } else {
        CUresult::CUDA_SUCCESS
    };

    send_result("cuEventDestroy_v2", server.id, result, channel_sender);
}

pub fn cuIpcGetEventHandleExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuIpcGetEventHandle", "[#{}]", server.id);

    let mut event = std::mem::MaybeUninit::<CUevent>::uninit();
    event.recv(channel_receiver).unwrap();
    let event = unsafe { event.assume_init() };
    channel_receiver.recv_ts().unwrap();

    let mut handle: CUipcEventHandle = unsafe { std::mem::zeroed() };
    let result = unsafe { cuIpcGetEventHandle(&raw mut handle, event) };
    if result == CUresult::CUDA_SUCCESS {
        let mut state = driver_ipc_events().lock().unwrap();
        state
            .handle_to_event
            .insert(handle.reserved, event as usize);
        state.event_refs.entry(event as usize).or_insert(1);
    }

    handle.send(channel_sender).unwrap();
    send_result("cuIpcGetEventHandle", server.id, result, channel_sender);
}

pub fn cuIpcOpenEventHandleExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuIpcOpenEventHandle", "[#{}]", server.id);

    let mut handle = std::mem::MaybeUninit::<CUipcEventHandle>::uninit();
    handle.recv(channel_receiver).unwrap();
    let handle = unsafe { handle.assume_init() };
    channel_receiver.recv_ts().unwrap();

    let mut event: CUevent = std::ptr::null_mut();
    let mapped_event = {
        let mut state = driver_ipc_events().lock().unwrap();
        let mapped = state.handle_to_event.get(&handle.reserved).copied();
        if let Some(event_key) = mapped {
            *state.event_refs.entry(event_key).or_insert(1) += 1;
            event = event_key as CUevent;
        }
        mapped
    };
    let result = if mapped_event.is_some() {
        CUresult::CUDA_SUCCESS
    } else {
        unsafe { cuIpcOpenEventHandle(&raw mut event, handle) }
    };

    event.send(channel_sender).unwrap();
    send_result("cuIpcOpenEventHandle", server.id, result, channel_sender);
}

pub fn cuCtxCreate_v4Exe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuCtxCreate_v4", "[#{}]", server.id);

    let mut has_params = false;
    has_params.recv(channel_receiver).unwrap();
    let mut params = unsafe { std::mem::zeroed::<CUctxCreateParams>() };
    let mut _affinity_params = None;
    if has_params {
        params.recv(channel_receiver).unwrap();
        let mut affinity_params = recv_slice::<CUexecAffinityParam, _>(channel_receiver).unwrap();
        params.execAffinityParams = if affinity_params.is_empty() {
            std::ptr::null_mut()
        } else {
            affinity_params.as_mut_ptr()
        };
        params.numExecAffinityParams = affinity_params.len() as _;
        _affinity_params = Some(affinity_params);
    }
    let mut flags = 0u32;
    flags.recv(channel_receiver).unwrap();
    let mut dev = 0;
    dev.recv(channel_receiver).unwrap();
    channel_receiver.recv_ts().unwrap();

    let mut context: CUcontext = std::ptr::null_mut();
    let params_ptr = if has_params {
        &raw mut params
    } else {
        std::ptr::null_mut()
    };
    let result = unsafe { cuCtxCreate_v4(&raw mut context, params_ptr, flags, dev) };

    context.send(channel_sender).unwrap();
    send_result("cuCtxCreate_v4", server.id, result, channel_sender);
}

pub fn cuGraphGetNodesExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuGraphGetNodes", "[#{}]", server.id);

    let mut graph = std::mem::MaybeUninit::<CUgraph>::uninit();
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
    let result = unsafe { cuGraphGetNodes(graph, nodes_ptr, &raw mut count) };

    if requested {
        send_slice(
            &nodes[..output_len(requested, result, count, capacity)],
            channel_sender,
        )
        .unwrap();
    }
    count.send(channel_sender).unwrap();
    send_result("cuGraphGetNodes", server.id, result, channel_sender);
}

pub fn cuGraphGetRootNodesExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuGraphGetRootNodes", "[#{}]", server.id);

    let mut graph = std::mem::MaybeUninit::<CUgraph>::uninit();
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
    let result = unsafe { cuGraphGetRootNodes(graph, nodes_ptr, &raw mut count) };

    if requested {
        send_slice(
            &nodes[..output_len(requested, result, count, capacity)],
            channel_sender,
        )
        .unwrap();
    }
    count.send(channel_sender).unwrap();
    send_result("cuGraphGetRootNodes", server.id, result, channel_sender);
}

pub fn cuGraphGetEdges_v2Exe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuGraphGetEdges_v2", "[#{}]", server.id);

    let mut graph = std::mem::MaybeUninit::<CUgraph>::uninit();
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
        unsafe { cuGraphGetEdges_v2(graph, from_ptr, to_ptr, edge_data_ptr, &raw mut count) };
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
    send_result("cuGraphGetEdges_v2", server.id, result, channel_sender);
}

pub fn cuGraphNodeGetDependencies_v2Exe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuGraphNodeGetDependencies_v2", "[#{}]", server.id);

    let mut node = std::mem::MaybeUninit::<CUgraphNode>::uninit();
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
        cuGraphNodeGetDependencies_v2(node, dependencies_ptr, edge_data_ptr, &raw mut count)
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
        "cuGraphNodeGetDependencies_v2",
        server.id,
        result,
        channel_sender,
    );
}

pub fn cuGraphNodeGetDependentNodes_v2Exe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuGraphNodeGetDependentNodes_v2", "[#{}]", server.id);

    let mut node = std::mem::MaybeUninit::<CUgraphNode>::uninit();
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
        cuGraphNodeGetDependentNodes_v2(node, dependents_ptr, edge_data_ptr, &raw mut count)
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
        "cuGraphNodeGetDependentNodes_v2",
        server.id,
        result,
        channel_sender,
    );
}
