#![expect(non_snake_case)]

use crate::ServerWorker;
use cudasys::cuda::*;
use network::type_impl::send_slice;
use network::{CommChannel, Transportable};

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
