#![expect(non_snake_case)]

use super::cuda_exe_utils;
use crate::ServerWorker;
use cudasys::cuda::*;
use network::type_impl::{recv_slice, send_slice};
use network::{CommChannel, Transportable};
use std::collections::BTreeMap;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_uint, c_ulonglong, c_void};
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

const GRAPH_LAUNCH_RESULT_MESSAGE: c_int = 0;
const GRAPH_HOST_CALLBACK_MESSAGE: c_int = 1;
pub(super) const GRAPH_HOST_FAMILY_DRIVER: c_int = 1;
pub(super) const GRAPH_HOST_FAMILY_RUNTIME: c_int = 2;

#[derive(Default)]
struct GraphHostNodeState {
    graph_has_host_nodes: BTreeMap<usize, bool>,
    graph_exec_has_host_nodes: BTreeMap<usize, bool>,
    payloads: Vec<usize>,
}

fn graph_host_nodes() -> &'static Mutex<GraphHostNodeState> {
    static STATE: OnceLock<Mutex<GraphHostNodeState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(GraphHostNodeState::default()))
}

pub(super) struct GraphHostCallbackPayload {
    pub(super) family: c_int,
    pub(super) node: usize,
    pub(super) graph_exec: usize,
    sender: *const c_void,
    receiver: *const c_void,
    send_request: unsafe fn(*const c_void, &GraphHostCallbackPayload),
    recv_ack: unsafe fn(*const c_void) -> CUresult,
}

static GRAPH_HOST_CALLBACK_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn graph_host_callback_lock() -> &'static Mutex<()> {
    GRAPH_HOST_CALLBACK_LOCK.get_or_init(|| Mutex::new(()))
}

pub(super) fn mark_graph_has_host_nodes(graph: CUgraph) {
    graph_host_nodes()
        .lock()
        .unwrap()
        .graph_has_host_nodes
        .insert(graph as usize, true);
}

pub(super) fn mark_graph_exec_has_host_nodes(graph_exec: CUgraphExec) {
    graph_host_nodes()
        .lock()
        .unwrap()
        .graph_exec_has_host_nodes
        .insert(graph_exec as usize, true);
}

pub(super) fn remember_graph_exec_host_nodes(graph_exec: CUgraphExec, graph: CUgraph) {
    let mut state = graph_host_nodes().lock().unwrap();
    if state
        .graph_has_host_nodes
        .get(&(graph as usize))
        .copied()
        .unwrap_or(false)
    {
        state
            .graph_exec_has_host_nodes
            .insert(graph_exec as usize, true);
    }
}

fn graph_exec_has_host_nodes(graph_exec: CUgraphExec) -> bool {
    graph_host_nodes()
        .lock()
        .unwrap()
        .graph_exec_has_host_nodes
        .get(&(graph_exec as usize))
        .copied()
        .unwrap_or(false)
}

pub(super) fn graph_host_callback_user_data<C: CommChannel>(
    server: &ServerWorker<C>,
    family: c_int,
    node: usize,
    graph_exec: usize,
) -> *mut c_void {
    let payload = Box::new(GraphHostCallbackPayload {
        family,
        node,
        graph_exec,
        sender: &server.channel_sender as *const C as *const c_void,
        receiver: &server.channel_receiver as *const C as *const c_void,
        send_request: send_graph_host_callback_request::<C>,
        recv_ack: recv_graph_host_callback_ack::<C>,
    });
    let ptr = Box::into_raw(payload);
    graph_host_nodes()
        .lock()
        .unwrap()
        .payloads
        .push(ptr as usize);
    ptr.cast()
}

unsafe fn send_graph_host_callback_request<C: CommChannel>(
    sender: *const c_void,
    payload: &GraphHostCallbackPayload,
) {
    let sender = unsafe { &*(sender as *const C) };
    GRAPH_HOST_CALLBACK_MESSAGE.send(sender).unwrap();
    payload.family.send(sender).unwrap();
    payload.node.send(sender).unwrap();
    payload.graph_exec.send(sender).unwrap();
    sender.flush_out().unwrap();
}

unsafe fn recv_graph_host_callback_ack<C: CommChannel>(receiver: *const c_void) -> CUresult {
    let receiver = unsafe { &*(receiver as *const C) };
    let mut ack = CUresult::CUDA_SUCCESS;
    ack.recv(receiver).unwrap();
    receiver.recv_ts().unwrap();
    ack
}

pub(super) unsafe extern "C" fn graph_host_callback(user_data: *mut c_void) {
    if user_data.is_null() {
        return;
    }
    let payload = unsafe { &*(user_data as *const GraphHostCallbackPayload) };
    let _guard = graph_host_callback_lock().lock().unwrap();
    unsafe {
        (payload.send_request)(payload.sender, payload);
    }
    let ack = unsafe { (payload.recv_ack)(payload.receiver) };
    if ack != CUresult::CUDA_SUCCESS {
        log::error!(target: "graphHostCallback", "client callback returned {:?}", ack);
    }
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

fn recv_checkpoint_pid<C: CommChannel>(channel_receiver: &C) -> c_int {
    let mut pid = 0;
    pid.recv(channel_receiver).unwrap();
    let mut is_client_process = false;
    is_client_process.recv(channel_receiver).unwrap();
    if is_client_process {
        std::process::id() as c_int
    } else {
        pid
    }
}

fn recv_optional_checkpoint_args<T: Transportable, C: CommChannel>(
    channel_receiver: &C,
) -> Option<T> {
    let mut args_present = false;
    args_present.recv(channel_receiver).unwrap();
    if !args_present {
        return None;
    }

    let mut args = unsafe { std::mem::zeroed::<T>() };
    args.recv(channel_receiver).unwrap();
    Some(args)
}

pub fn cuCheckpointProcessGetRestoreThreadIdExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuCheckpointProcessGetRestoreThreadId", "[#{}]", server.id);

    let pid = recv_checkpoint_pid(channel_receiver);
    channel_receiver.recv_ts().unwrap();

    let mut tid = 0;
    let result = unsafe { cuCheckpointProcessGetRestoreThreadId(pid, &raw mut tid) };
    tid.send(channel_sender).unwrap();
    send_result(
        "cuCheckpointProcessGetRestoreThreadId",
        server.id,
        result,
        channel_sender,
    );
}

pub fn cuCheckpointProcessGetStateExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuCheckpointProcessGetState", "[#{}]", server.id);

    let pid = recv_checkpoint_pid(channel_receiver);
    channel_receiver.recv_ts().unwrap();

    let mut state = CUprocessState::CU_PROCESS_STATE_RUNNING;
    let result = unsafe { cuCheckpointProcessGetState(pid, &raw mut state) };
    state.send(channel_sender).unwrap();
    send_result(
        "cuCheckpointProcessGetState",
        server.id,
        result,
        channel_sender,
    );
}

pub fn cuCheckpointProcessLockExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuCheckpointProcessLock", "[#{}]", server.id);

    let pid = recv_checkpoint_pid(channel_receiver);
    let mut args = recv_optional_checkpoint_args::<CUcheckpointLockArgs, _>(channel_receiver);
    channel_receiver.recv_ts().unwrap();

    let args_ptr = args
        .as_mut()
        .map_or(std::ptr::null_mut(), |args| args as *mut _);
    let result = unsafe { cuCheckpointProcessLock(pid, args_ptr) };
    send_result("cuCheckpointProcessLock", server.id, result, channel_sender);
}

pub fn cuCheckpointProcessCheckpointExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuCheckpointProcessCheckpoint", "[#{}]", server.id);

    let pid = recv_checkpoint_pid(channel_receiver);
    let mut args = recv_optional_checkpoint_args::<CUcheckpointCheckpointArgs, _>(channel_receiver);
    channel_receiver.recv_ts().unwrap();

    let args_ptr = args
        .as_mut()
        .map_or(std::ptr::null_mut(), |args| args as *mut _);
    let result = unsafe { cuCheckpointProcessCheckpoint(pid, args_ptr) };
    send_result(
        "cuCheckpointProcessCheckpoint",
        server.id,
        result,
        channel_sender,
    );
}

pub fn cuCheckpointProcessRestoreExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuCheckpointProcessRestore", "[#{}]", server.id);

    let pid = recv_checkpoint_pid(channel_receiver);
    let mut args = recv_optional_checkpoint_args::<CUcheckpointRestoreArgs, _>(channel_receiver);
    let mut gpu_pairs = if args.is_some() {
        recv_slice::<CUcheckpointGpuPair, _>(channel_receiver).unwrap()
    } else {
        Box::default()
    };
    channel_receiver.recv_ts().unwrap();

    if let Some(args) = args.as_mut() {
        if !gpu_pairs.is_empty() {
            args.gpuPairs = gpu_pairs.as_mut_ptr();
            args.gpuPairsCount = gpu_pairs.len() as c_uint;
        }
    }
    let args_ptr = args
        .as_mut()
        .map_or(std::ptr::null_mut(), |args| args as *mut _);
    let result = unsafe { cuCheckpointProcessRestore(pid, args_ptr) };
    send_result(
        "cuCheckpointProcessRestore",
        server.id,
        result,
        channel_sender,
    );
}

pub fn cuCheckpointProcessUnlockExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuCheckpointProcessUnlock", "[#{}]", server.id);

    let pid = recv_checkpoint_pid(channel_receiver);
    let mut args = recv_optional_checkpoint_args::<CUcheckpointUnlockArgs, _>(channel_receiver);
    channel_receiver.recv_ts().unwrap();

    let args_ptr = args
        .as_mut()
        .map_or(std::ptr::null_mut(), |args| args as *mut _);
    let result = unsafe { cuCheckpointProcessUnlock(pid, args_ptr) };
    send_result(
        "cuCheckpointProcessUnlock",
        server.id,
        result,
        channel_sender,
    );
}

pub fn cuTensorMapReplaceAddressExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuTensorMapReplaceAddress", "[#{}]", server.id);

    let mut tensor_map = std::mem::MaybeUninit::<CUtensorMap>::uninit();
    tensor_map.recv(channel_receiver).unwrap();
    let mut tensor_map = unsafe { tensor_map.assume_init() };
    let mut global_address = std::mem::MaybeUninit::<*mut c_void>::uninit();
    global_address.recv(channel_receiver).unwrap();
    let global_address = unsafe { global_address.assume_init() };
    channel_receiver.recv_ts().unwrap();

    let result = unsafe { cuTensorMapReplaceAddress(&raw mut tensor_map, global_address) };
    tensor_map.send(channel_sender).unwrap();
    send_result("cuTensorMapReplaceAddress", server.id, result, channel_sender);
}

pub fn cuMemGetHandleForAddressRangeExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    log::debug!(target: "cuMemGetHandleForAddressRange", "[#{}]", server.id);

    let mut dptr = 0;
    dptr.recv(&server.channel_receiver).unwrap();
    let mut size = 0usize;
    size.recv(&server.channel_receiver).unwrap();
    let mut handle_type = CUmemRangeHandleType::CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD;
    handle_type.recv(&server.channel_receiver).unwrap();
    let mut flags = 0u64;
    flags.recv(&server.channel_receiver).unwrap();
    let socket_path = recv_slice::<u8, _>(&server.channel_receiver).unwrap();
    server.channel_receiver.recv_ts().unwrap();

    let mut fd = -1 as c_int;
    let mut result = unsafe {
        cuMemGetHandleForAddressRange((&raw mut fd).cast(), dptr, size, handle_type, flags)
    };
    if result == CUresult::CUDA_SUCCESS {
        if let Err(error) = cuda_exe_utils::send_fd(&socket_path, fd) {
            log::error!(target: "cuMemGetHandleForAddressRange", "failed to send fd: {error}");
            unsafe {
                libc::close(fd);
            }
            result = CUresult::CUDA_ERROR_UNKNOWN;
        }
    }

    send_result(
        "cuMemGetHandleForAddressRange",
        server.id,
        result,
        &server.channel_sender,
    );
}

fn send_graph_launch_result<C: CommChannel>(
    target: &'static str,
    server_id: i32,
    result: CUresult,
    channel_sender: &C,
) {
    if result != CUresult::CUDA_SUCCESS {
        log::error!(target: target, "[#{}] returned error: {:?}", server_id, result);
    }
    GRAPH_LAUNCH_RESULT_MESSAGE.send(channel_sender).unwrap();
    result.send(channel_sender).unwrap();
    channel_sender.flush_out().unwrap();
}

fn compact_row_bytes(width: usize, height: usize) -> Result<usize, CUresult> {
    width
        .checked_mul(height)
        .ok_or(CUresult::CUDA_ERROR_INVALID_VALUE)
}

fn allocate_compact_rows(width: usize, height: usize) -> Result<Vec<u8>, CUresult> {
    let len = compact_row_bytes(width, height)?;
    let mut rows = Vec::new();
    rows.try_reserve_exact(len)
        .map_err(|_| CUresult::CUDA_ERROR_OUT_OF_MEMORY)?;
    rows.resize(len, 0);
    Ok(rows)
}

fn run_memcpy_2d<C, F>(
    server: &mut ServerWorker<C>,
    target: &'static str,
    receive_stream: bool,
    call: F,
) where
    C: CommChannel,
    F: FnOnce(*const CUDA_MEMCPY2D, Option<CUstream>) -> CUresult,
{
    log::debug!(target: target, "[#{}]", server.id);

    let mut copy = std::mem::MaybeUninit::<CUDA_MEMCPY2D>::uninit();
    copy.recv(&server.channel_receiver).unwrap();
    let mut copy = unsafe { copy.assume_init() };
    let stream = if receive_stream {
        let mut stream = std::mem::MaybeUninit::<CUstream>::uninit();
        stream.recv(&server.channel_receiver).unwrap();
        Some(unsafe { stream.assume_init() })
    } else {
        None
    };
    let host_src_rows = recv_slice::<u8, _>(&server.channel_receiver).unwrap();
    server.channel_receiver.recv_ts().unwrap();

    let has_host_src = copy.srcMemoryType == CUmemorytype::CU_MEMORYTYPE_HOST;
    let has_host_dst = copy.dstMemoryType == CUmemorytype::CU_MEMORYTYPE_HOST;

    let mut host_dst_rows = Vec::new();
    let result = if has_host_src || has_host_dst {
        let expected_len = match compact_row_bytes(copy.WidthInBytes, copy.Height) {
            Ok(len) => len,
            Err(result) => {
                send_slice::<u8, _>(&[], &server.channel_sender).unwrap();
                send_result(target, server.id, result, &server.channel_sender);
                return;
            }
        };
        if has_host_src {
            if host_src_rows.len() != expected_len {
                send_slice::<u8, _>(&[], &server.channel_sender).unwrap();
                send_result(
                    target,
                    server.id,
                    CUresult::CUDA_ERROR_INVALID_VALUE,
                    &server.channel_sender,
                );
                return;
            }
            copy.srcHost = host_src_rows.as_ptr().cast::<c_void>();
        } else if !host_src_rows.is_empty() {
            send_slice::<u8, _>(&[], &server.channel_sender).unwrap();
            send_result(
                target,
                server.id,
                CUresult::CUDA_ERROR_INVALID_VALUE,
                &server.channel_sender,
            );
            return;
        }
        if has_host_dst {
            host_dst_rows = match allocate_compact_rows(copy.WidthInBytes, copy.Height) {
                Ok(rows) => rows,
                Err(result) => {
                    send_slice::<u8, _>(&[], &server.channel_sender).unwrap();
                    send_result(target, server.id, result, &server.channel_sender);
                    return;
                }
            };
            copy.dstHost = host_dst_rows.as_mut_ptr().cast::<c_void>();
        }

        let result = call(&copy as *const _, stream);
        if result == CUresult::CUDA_SUCCESS {
            if let Some(stream) = stream {
                unsafe { cuStreamSynchronize(stream) }
            } else {
                result
            }
        } else {
            result
        }
    } else {
        if !host_src_rows.is_empty() {
            CUresult::CUDA_ERROR_INVALID_VALUE
        } else {
            call(&copy as *const _, stream)
        }
    };

    if has_host_dst && result == CUresult::CUDA_SUCCESS {
        send_slice(&host_dst_rows, &server.channel_sender).unwrap();
    } else {
        send_slice::<u8, _>(&[], &server.channel_sender).unwrap();
    }
    send_result(target, server.id, result, &server.channel_sender);
}

pub fn cuMemcpy2D_v2Exe<C: CommChannel>(server: &mut ServerWorker<C>) {
    run_memcpy_2d(server, "cuMemcpy2D_v2", false, |copy, _| unsafe {
        cuMemcpy2D_v2(copy)
    });
}

pub fn cuMemcpy2DUnaligned_v2Exe<C: CommChannel>(server: &mut ServerWorker<C>) {
    run_memcpy_2d(server, "cuMemcpy2DUnaligned_v2", false, |copy, _| unsafe {
        cuMemcpy2DUnaligned_v2(copy)
    });
}

pub fn cuMemcpy2DAsync_v2Exe<C: CommChannel>(server: &mut ServerWorker<C>) {
    run_memcpy_2d(server, "cuMemcpy2DAsync_v2", true, |copy, stream| unsafe {
        cuMemcpy2DAsync_v2(copy, stream.unwrap())
    });
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

pub fn cuPointerGetAttributesExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuPointerGetAttributes", "[#{}]", server.id);

    let mut attributes = recv_slice::<CUpointer_attribute, _>(channel_receiver).unwrap();
    let mut ptr = 0;
    ptr.recv(channel_receiver).unwrap();
    channel_receiver.recv_ts().unwrap();

    let mut data = attributes
        .iter()
        .map(|attr| vec![0u8; attr.data_size()])
        .collect::<Vec<_>>();
    let mut data_ptrs = data
        .iter_mut()
        .map(|buffer| buffer.as_mut_ptr().cast::<c_void>())
        .collect::<Vec<_>>();

    let result = unsafe {
        cuPointerGetAttributes(
            attributes.len() as _,
            attributes.as_mut_ptr(),
            data_ptrs.as_mut_ptr(),
            ptr,
        )
    };

    for buffer in data {
        let len = if result == CUresult::CUDA_SUCCESS {
            buffer.len()
        } else {
            0
        };
        send_slice(&buffer[..len], channel_sender).unwrap();
    }
    send_result("cuPointerGetAttributes", server.id, result, channel_sender);
}

pub fn cuMemPrefetchAsync_v2Exe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuMemPrefetchAsync_v2", "[#{}]", server.id);

    let mut dev_ptr = 0;
    dev_ptr.recv(channel_receiver).unwrap();
    let mut count = 0usize;
    count.recv(channel_receiver).unwrap();
    let mut location = std::mem::MaybeUninit::<CUmemLocation>::uninit();
    location.recv(channel_receiver).unwrap();
    let location = unsafe { location.assume_init() };
    let mut flags = 0u32;
    flags.recv(channel_receiver).unwrap();
    let mut stream = std::mem::MaybeUninit::<CUstream>::uninit();
    stream.recv(channel_receiver).unwrap();
    let stream = unsafe { stream.assume_init() };
    channel_receiver.recv_ts().unwrap();

    let result = unsafe { cuMemPrefetchAsync_v2(dev_ptr, count, location, flags, stream) };
    send_result("cuMemPrefetchAsync_v2", server.id, result, channel_sender);
}

pub fn cuMemAdvise_v2Exe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuMemAdvise_v2", "[#{}]", server.id);

    let mut dev_ptr = 0;
    dev_ptr.recv(channel_receiver).unwrap();
    let mut count = 0usize;
    count.recv(channel_receiver).unwrap();
    let mut advice = CUmem_advise::CU_MEM_ADVISE_SET_READ_MOSTLY;
    advice.recv(channel_receiver).unwrap();
    let mut location = std::mem::MaybeUninit::<CUmemLocation>::uninit();
    location.recv(channel_receiver).unwrap();
    let location = unsafe { location.assume_init() };
    channel_receiver.recv_ts().unwrap();

    let result = unsafe { cuMemAdvise_v2(dev_ptr, count, advice, location) };
    send_result("cuMemAdvise_v2", server.id, result, channel_sender);
}

pub fn cuMemRangeGetAttributesExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuMemRangeGetAttributes", "[#{}]", server.id);

    let mut data_sizes = recv_slice::<usize, _>(channel_receiver).unwrap();
    let mut attributes = recv_slice::<CUmem_range_attribute, _>(channel_receiver).unwrap();
    let mut dev_ptr = 0;
    dev_ptr.recv(channel_receiver).unwrap();
    let mut count = 0usize;
    count.recv(channel_receiver).unwrap();
    channel_receiver.recv_ts().unwrap();

    let mut data = data_sizes
        .iter()
        .map(|size| vec![0u8; *size])
        .collect::<Vec<_>>();
    let mut data_ptrs = data
        .iter_mut()
        .map(|buffer| buffer.as_mut_ptr().cast::<c_void>())
        .collect::<Vec<_>>();
    let result = unsafe {
        cuMemRangeGetAttributes(
            data_ptrs.as_mut_ptr(),
            data_sizes.as_mut_ptr(),
            attributes.as_mut_ptr(),
            attributes.len(),
            dev_ptr,
            count,
        )
    };

    for buffer in data {
        let len = if result == CUresult::CUDA_SUCCESS {
            buffer.len()
        } else {
            0
        };
        send_slice(&buffer[..len], channel_sender).unwrap();
    }
    send_result("cuMemRangeGetAttributes", server.id, result, channel_sender);
}

pub fn cuMemBatchDecompressAsyncExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuMemBatchDecompressAsync", "[#{}]", server.id);

    let mut has_params = false;
    has_params.recv(channel_receiver).unwrap();
    let mut params = if has_params {
        recv_slice::<CUmemDecompressParams, _>(channel_receiver).unwrap()
    } else {
        Vec::new().into_boxed_slice()
    };
    let mut count = 0usize;
    count.recv(channel_receiver).unwrap();
    let mut flags = 0u32;
    flags.recv(channel_receiver).unwrap();
    let mut has_error_index = false;
    has_error_index.recv(channel_receiver).unwrap();
    let mut stream = std::mem::MaybeUninit::<CUstream>::uninit();
    stream.recv(channel_receiver).unwrap();
    let stream = unsafe { stream.assume_init() };
    channel_receiver.recv_ts().unwrap();

    let params_ptr = if has_params {
        params.as_mut_ptr()
    } else {
        std::ptr::null_mut()
    };
    let mut error_index = usize::MAX;
    let error_index_ptr = if has_error_index {
        &raw mut error_index
    } else {
        std::ptr::null_mut()
    };
    let result =
        unsafe { cuMemBatchDecompressAsync(params_ptr, count, flags, error_index_ptr, stream) };

    if has_error_index {
        error_index.send(channel_sender).unwrap();
    }
    send_result(
        "cuMemBatchDecompressAsync",
        server.id,
        result,
        channel_sender,
    );
}

pub fn cuMemExportToShareableHandleExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    log::debug!(target: "cuMemExportToShareableHandle", "[#{}]", server.id);

    let mut handle = 0;
    handle.recv(&server.channel_receiver).unwrap();
    let mut handle_type = CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
    handle_type.recv(&server.channel_receiver).unwrap();
    let mut flags = 0u64;
    flags.recv(&server.channel_receiver).unwrap();
    server.channel_receiver.recv_ts().unwrap();

    let mut synthetic_fd = -1 as c_int;
    let result =
        if handle_type == CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR {
            let mut server_fd = -1 as c_int;
            let result = unsafe {
                cuMemExportToShareableHandle(
                    (&mut server_fd as *mut c_int).cast(),
                    handle,
                    handle_type,
                    flags,
                )
            };
            if result == CUresult::CUDA_SUCCESS {
                synthetic_fd = server.insert_shareable_handle(server_fd);
            }
            result
        } else {
            CUresult::CUDA_ERROR_NOT_SUPPORTED
        };

    synthetic_fd.send(&server.channel_sender).unwrap();
    send_result(
        "cuMemExportToShareableHandle",
        server.id,
        result,
        &server.channel_sender,
    );
}

pub fn cuMemImportFromShareableHandleExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    log::debug!(target: "cuMemImportFromShareableHandle", "[#{}]", server.id);

    let mut synthetic_fd = -1 as c_int;
    synthetic_fd.recv(&server.channel_receiver).unwrap();
    let mut handle_type = CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
    handle_type.recv(&server.channel_receiver).unwrap();
    server.channel_receiver.recv_ts().unwrap();

    let mut handle = 0;
    let result =
        if handle_type == CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR {
            if let Some(server_fd) = server.take_shareable_handle(synthetic_fd) {
                let result = unsafe {
                    cuMemImportFromShareableHandle(
                        &mut handle,
                        server_fd as isize as *mut c_void,
                        handle_type,
                    )
                };
                if result == CUresult::CUDA_SUCCESS {
                    unsafe {
                        libc::close(server_fd);
                    }
                } else {
                    server.restore_shareable_handle(synthetic_fd, server_fd);
                }
                result
            } else {
                CUresult::CUDA_ERROR_INVALID_VALUE
            }
        } else {
            CUresult::CUDA_ERROR_NOT_SUPPORTED
        };

    handle.send(&server.channel_sender).unwrap();
    send_result(
        "cuMemImportFromShareableHandle",
        server.id,
        result,
        &server.channel_sender,
    );
}

pub fn cuImportExternalMemoryExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    log::debug!(target: "cuImportExternalMemory", "[#{}]", server.id);

    let mut desc = std::mem::MaybeUninit::<CUDA_EXTERNAL_MEMORY_HANDLE_DESC>::uninit();
    desc.recv(&server.channel_receiver).unwrap();
    let mut desc = unsafe { desc.assume_init() };
    server.channel_receiver.recv_ts().unwrap();

    let mut external_memory = std::ptr::null_mut();
    let result =
        if desc.type_ == CUexternalMemoryHandleType::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD {
            let synthetic_fd = unsafe { desc.handle.fd };
            if let Some(server_fd) = server.take_shareable_handle(synthetic_fd) {
                desc.handle.fd = server_fd;
                let result = unsafe { cuImportExternalMemory(&mut external_memory, &desc) };
                if result != CUresult::CUDA_SUCCESS {
                    server.restore_shareable_handle(synthetic_fd, server_fd);
                }
                result
            } else {
                CUresult::CUDA_ERROR_INVALID_VALUE
            }
        } else {
            CUresult::CUDA_ERROR_NOT_SUPPORTED
        };

    external_memory.send(&server.channel_sender).unwrap();
    send_result(
        "cuImportExternalMemory",
        server.id,
        result,
        &server.channel_sender,
    );
}

pub fn cuImportExternalSemaphoreExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    log::debug!(target: "cuImportExternalSemaphore", "[#{}]", server.id);

    let mut desc = std::mem::MaybeUninit::<CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC>::uninit();
    desc.recv(&server.channel_receiver).unwrap();
    let mut desc = unsafe { desc.assume_init() };
    server.channel_receiver.recv_ts().unwrap();

    let mut external_semaphore = std::ptr::null_mut();
    let result = if matches!(
        desc.type_,
        CUexternalSemaphoreHandleType::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD
            | CUexternalSemaphoreHandleType::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD
    ) {
        match cuda_exe_utils::receive_client_fd(
            server,
            "cuImportExternalSemaphore",
            CUresult::CUDA_ERROR_INVALID_VALUE,
        ) {
            Ok(server_fd) => {
                desc.handle.fd = server_fd;
                let result = unsafe { cuImportExternalSemaphore(&mut external_semaphore, &desc) };
                if result != CUresult::CUDA_SUCCESS {
                    unsafe {
                        libc::close(server_fd);
                    }
                }
                result
            }
            Err(result) => result,
        }
    } else {
        send_slice::<u8, _>(&[], &server.channel_sender).unwrap();
        server.channel_sender.flush_out().unwrap();
        CUresult::CUDA_ERROR_NOT_SUPPORTED
    };

    external_semaphore.send(&server.channel_sender).unwrap();
    send_result(
        "cuImportExternalSemaphore",
        server.id,
        result,
        &server.channel_sender,
    );
}

pub fn cuMemPoolCreateExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuMemPoolCreate", "[#{}]", server.id);

    let mut pool_props = std::mem::MaybeUninit::<CUmemPoolProps>::uninit();
    pool_props.recv(channel_receiver).unwrap();
    let pool_props = unsafe { pool_props.assume_init() };
    channel_receiver.recv_ts().unwrap();

    let mut pool: CUmemoryPool = std::ptr::null_mut();
    let result = unsafe { cuMemPoolCreate(&raw mut pool, &raw const pool_props) };
    pool.send(channel_sender).unwrap();
    send_result("cuMemPoolCreate", server.id, result, channel_sender);
}

fn cu_mem_get_pool<C: CommChannel>(
    server: &mut ServerWorker<C>,
    target: &'static str,
    func: unsafe extern "C" fn(
        *mut CUmemoryPool,
        *mut CUmemLocation,
        CUmemAllocationType,
    ) -> CUresult,
) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: target, "[#{}]", server.id);

    let mut location = std::mem::MaybeUninit::<CUmemLocation>::uninit();
    location.recv(channel_receiver).unwrap();
    let mut location = unsafe { location.assume_init() };
    let mut type_ = CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    type_.recv(channel_receiver).unwrap();
    channel_receiver.recv_ts().unwrap();

    let mut pool: CUmemoryPool = std::ptr::null_mut();
    let result = unsafe { func(&raw mut pool, &raw mut location, type_) };
    pool.send(channel_sender).unwrap();
    send_result(target, server.id, result, channel_sender);
}

pub fn cuMemGetDefaultMemPoolExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    cu_mem_get_pool(server, "cuMemGetDefaultMemPool", cuMemGetDefaultMemPool);
}

pub fn cuMemGetMemPoolExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    cu_mem_get_pool(server, "cuMemGetMemPool", cuMemGetMemPool);
}

pub fn cuMemSetMemPoolExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuMemSetMemPool", "[#{}]", server.id);

    let mut location = std::mem::MaybeUninit::<CUmemLocation>::uninit();
    location.recv(channel_receiver).unwrap();
    let mut location = unsafe { location.assume_init() };
    let mut type_ = CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    type_.recv(channel_receiver).unwrap();
    let mut pool = std::mem::MaybeUninit::<CUmemoryPool>::uninit();
    pool.recv(channel_receiver).unwrap();
    let pool = unsafe { pool.assume_init() };
    channel_receiver.recv_ts().unwrap();

    let result = unsafe { cuMemSetMemPool(&raw mut location, type_, pool) };
    send_result("cuMemSetMemPool", server.id, result, channel_sender);
}

pub fn cuMemPoolSetAccessExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuMemPoolSetAccess", "[#{}]", server.id);

    let mut pool = std::mem::MaybeUninit::<CUmemoryPool>::uninit();
    pool.recv(channel_receiver).unwrap();
    let pool = unsafe { pool.assume_init() };
    let mut access = recv_slice::<CUmemAccessDesc, _>(channel_receiver).unwrap();
    channel_receiver.recv_ts().unwrap();

    let result = unsafe { cuMemPoolSetAccess(pool, access.as_mut_ptr(), access.len()) };
    send_result("cuMemPoolSetAccess", server.id, result, channel_sender);
}

pub fn cuMemPoolGetAccessExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuMemPoolGetAccess", "[#{}]", server.id);

    let mut pool = std::mem::MaybeUninit::<CUmemoryPool>::uninit();
    pool.recv(channel_receiver).unwrap();
    let pool = unsafe { pool.assume_init() };
    let mut location = std::mem::MaybeUninit::<CUmemLocation>::uninit();
    location.recv(channel_receiver).unwrap();
    let mut location = unsafe { location.assume_init() };
    channel_receiver.recv_ts().unwrap();

    let mut flags = CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_NONE;
    let result = unsafe { cuMemPoolGetAccess(&raw mut flags, pool, &raw mut location) };
    flags.send(channel_sender).unwrap();
    send_result("cuMemPoolGetAccess", server.id, result, channel_sender);
}

pub fn cuMemPoolExportToShareableHandleExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    log::debug!(target: "cuMemPoolExportToShareableHandle", "[#{}]", server.id);

    let mut pool = std::mem::MaybeUninit::<CUmemoryPool>::uninit();
    pool.recv(&server.channel_receiver).unwrap();
    let pool = unsafe { pool.assume_init() };
    let mut handle_type = CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
    handle_type.recv(&server.channel_receiver).unwrap();
    let mut flags = 0u64;
    flags.recv(&server.channel_receiver).unwrap();
    server.channel_receiver.recv_ts().unwrap();

    let mut synthetic_fd = -1 as c_int;
    let result =
        if handle_type == CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR {
            let mut server_fd = -1 as c_int;
            let result = unsafe {
                cuMemPoolExportToShareableHandle(
                    (&mut server_fd as *mut c_int).cast(),
                    pool,
                    handle_type,
                    flags,
                )
            };
            if result == CUresult::CUDA_SUCCESS {
                synthetic_fd = server.insert_shareable_handle(server_fd);
            }
            result
        } else {
            CUresult::CUDA_ERROR_NOT_SUPPORTED
        };

    synthetic_fd.send(&server.channel_sender).unwrap();
    send_result(
        "cuMemPoolExportToShareableHandle",
        server.id,
        result,
        &server.channel_sender,
    );
}

pub fn cuMemPoolImportFromShareableHandleExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    log::debug!(target: "cuMemPoolImportFromShareableHandle", "[#{}]", server.id);

    let mut synthetic_fd = -1 as c_int;
    synthetic_fd.recv(&server.channel_receiver).unwrap();
    let mut handle_type = CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
    handle_type.recv(&server.channel_receiver).unwrap();
    let mut flags = 0u64;
    flags.recv(&server.channel_receiver).unwrap();
    server.channel_receiver.recv_ts().unwrap();

    let mut pool = std::ptr::null_mut();
    let result =
        if handle_type == CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR {
            if let Some(server_fd) = server.take_shareable_handle(synthetic_fd) {
                let result = unsafe {
                    cuMemPoolImportFromShareableHandle(
                        &mut pool,
                        server_fd as isize as *mut c_void,
                        handle_type,
                        flags,
                    )
                };
                if result == CUresult::CUDA_SUCCESS {
                    unsafe {
                        libc::close(server_fd);
                    }
                } else {
                    server.restore_shareable_handle(synthetic_fd, server_fd);
                }
                result
            } else {
                CUresult::CUDA_ERROR_INVALID_VALUE
            }
        } else {
            CUresult::CUDA_ERROR_NOT_SUPPORTED
        };

    pool.send(&server.channel_sender).unwrap();
    send_result(
        "cuMemPoolImportFromShareableHandle",
        server.id,
        result,
        &server.channel_sender,
    );
}

pub fn cuMemPoolExportPointerExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    log::debug!(target: "cuMemPoolExportPointer", "[#{}]", server.id);

    let mut ptr = 0;
    ptr.recv(&server.channel_receiver).unwrap();
    server.channel_receiver.recv_ts().unwrap();

    let mut share_data = unsafe { std::mem::zeroed::<CUmemPoolPtrExportData>() };
    let result = unsafe { cuMemPoolExportPointer(&mut share_data, ptr) };
    share_data.send(&server.channel_sender).unwrap();
    send_result(
        "cuMemPoolExportPointer",
        server.id,
        result,
        &server.channel_sender,
    );
}

pub fn cuMemPoolImportPointerExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    log::debug!(target: "cuMemPoolImportPointer", "[#{}]", server.id);

    let mut pool = std::mem::MaybeUninit::<CUmemoryPool>::uninit();
    pool.recv(&server.channel_receiver).unwrap();
    let pool = unsafe { pool.assume_init() };
    let mut share_data = std::mem::MaybeUninit::<CUmemPoolPtrExportData>::uninit();
    share_data.recv(&server.channel_receiver).unwrap();
    let mut share_data = unsafe { share_data.assume_init() };
    server.channel_receiver.recv_ts().unwrap();

    let mut ptr = 0;
    let result = unsafe { cuMemPoolImportPointer(&mut ptr, pool, &mut share_data) };
    ptr.send(&server.channel_sender).unwrap();
    send_result(
        "cuMemPoolImportPointer",
        server.id,
        result,
        &server.channel_sender,
    );
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

fn driver_graph_dependencies_ptr(dependencies: &[CUgraphNode]) -> *const CUgraphNode {
    if dependencies.is_empty() {
        std::ptr::null()
    } else {
        dependencies.as_ptr()
    }
}

fn driver_signal_node_params(
    semaphores: &mut [CUexternalSemaphore],
    params: &[CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS],
) -> CUDA_EXT_SEM_SIGNAL_NODE_PARAMS {
    CUDA_EXT_SEM_SIGNAL_NODE_PARAMS {
        extSemArray: if semaphores.is_empty() {
            std::ptr::null_mut()
        } else {
            semaphores.as_mut_ptr()
        },
        paramsArray: if params.is_empty() {
            std::ptr::null()
        } else {
            params.as_ptr()
        },
        numExtSems: semaphores.len() as c_uint,
    }
}

fn driver_wait_node_params(
    semaphores: &mut [CUexternalSemaphore],
    params: &[CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS],
) -> CUDA_EXT_SEM_WAIT_NODE_PARAMS {
    CUDA_EXT_SEM_WAIT_NODE_PARAMS {
        extSemArray: if semaphores.is_empty() {
            std::ptr::null_mut()
        } else {
            semaphores.as_mut_ptr()
        },
        paramsArray: if params.is_empty() {
            std::ptr::null()
        } else {
            params.as_ptr()
        },
        numExtSems: semaphores.len() as c_uint,
    }
}

fn driver_signal_params_from_node(
    node_params: &CUDA_EXT_SEM_SIGNAL_NODE_PARAMS,
    result: CUresult,
) -> (
    Vec<CUexternalSemaphore>,
    Vec<CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS>,
) {
    if result != CUresult::CUDA_SUCCESS || node_params.numExtSems == 0 {
        return (Vec::new(), Vec::new());
    }
    let count = node_params.numExtSems as usize;
    let semaphores = if node_params.extSemArray.is_null() {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(node_params.extSemArray, count) }.to_vec()
    };
    let params = if node_params.paramsArray.is_null() {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(node_params.paramsArray, count) }.to_vec()
    };
    (semaphores, params)
}

fn driver_wait_params_from_node(
    node_params: &CUDA_EXT_SEM_WAIT_NODE_PARAMS,
    result: CUresult,
) -> (
    Vec<CUexternalSemaphore>,
    Vec<CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS>,
) {
    if result != CUresult::CUDA_SUCCESS || node_params.numExtSems == 0 {
        return (Vec::new(), Vec::new());
    }
    let count = node_params.numExtSems as usize;
    let semaphores = if node_params.extSemArray.is_null() {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(node_params.extSemArray, count) }.to_vec()
    };
    let params = if node_params.paramsArray.is_null() {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(node_params.paramsArray, count) }.to_vec()
    };
    (semaphores, params)
}

pub fn cuGraphInstantiateWithFlagsExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuGraphInstantiateWithFlags", "[#{}]", server.id);

    let mut graph = std::mem::MaybeUninit::<CUgraph>::uninit();
    graph.recv(channel_receiver).unwrap();
    let graph = unsafe { graph.assume_init() };
    let mut flags = 0 as c_ulonglong;
    flags.recv(channel_receiver).unwrap();
    channel_receiver.recv_ts().unwrap();

    let mut graph_exec: CUgraphExec = std::ptr::null_mut();
    let result = unsafe { cuGraphInstantiateWithFlags(&raw mut graph_exec, graph, flags) };
    if result == CUresult::CUDA_SUCCESS {
        remember_graph_exec_host_nodes(graph_exec, graph);
    }

    graph_exec.send(channel_sender).unwrap();
    send_result(
        "cuGraphInstantiateWithFlags",
        server.id,
        result,
        channel_sender,
    );
}

pub fn cuGraphInstantiateWithParamsExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuGraphInstantiateWithParams", "[#{}]", server.id);

    let mut graph = std::mem::MaybeUninit::<CUgraph>::uninit();
    graph.recv(channel_receiver).unwrap();
    let graph = unsafe { graph.assume_init() };
    let mut instantiate_params = std::mem::MaybeUninit::<CUDA_GRAPH_INSTANTIATE_PARAMS>::uninit();
    instantiate_params.recv(channel_receiver).unwrap();
    let mut instantiate_params = unsafe { instantiate_params.assume_init() };
    channel_receiver.recv_ts().unwrap();

    let mut graph_exec: CUgraphExec = std::ptr::null_mut();
    let result = unsafe {
        cuGraphInstantiateWithParams(&raw mut graph_exec, graph, &raw mut instantiate_params)
    };
    if result == CUresult::CUDA_SUCCESS {
        remember_graph_exec_host_nodes(graph_exec, graph);
    }

    graph_exec.send(channel_sender).unwrap();
    instantiate_params.send(channel_sender).unwrap();
    send_result(
        "cuGraphInstantiateWithParams",
        server.id,
        result,
        channel_sender,
    );
}

pub fn cuGraphLaunchExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuGraphLaunch", "[#{}]", server.id);

    let mut graph_exec = std::mem::MaybeUninit::<CUgraphExec>::uninit();
    graph_exec.recv(channel_receiver).unwrap();
    let graph_exec = unsafe { graph_exec.assume_init() };
    let mut stream = std::mem::MaybeUninit::<CUstream>::uninit();
    stream.recv(channel_receiver).unwrap();
    let stream = unsafe { stream.assume_init() };
    channel_receiver.recv_ts().unwrap();

    let mut result = unsafe { cuGraphLaunch(graph_exec, stream) };
    if result == CUresult::CUDA_SUCCESS && graph_exec_has_host_nodes(graph_exec) {
        result = unsafe { cuStreamSynchronize(stream) };
    }

    send_graph_launch_result("cuGraphLaunch", server.id, result, channel_sender);
}

pub fn cuGraphAddHostNodeExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    log::debug!(target: "cuGraphAddHostNode", "[#{}]", server.id);

    let mut graph = std::mem::MaybeUninit::<CUgraph>::uninit();
    graph.recv(&server.channel_receiver).unwrap();
    let graph = unsafe { graph.assume_init() };
    let dependencies = recv_slice::<CUgraphNode, _>(&server.channel_receiver).unwrap();
    let mut client_params = std::mem::MaybeUninit::<CUDA_HOST_NODE_PARAMS>::uninit();
    client_params.recv(&server.channel_receiver).unwrap();
    let client_params = unsafe { client_params.assume_init() };
    server.channel_receiver.recv_ts().unwrap();

    let mut node: CUgraphNode = std::ptr::null_mut();
    let user_data = graph_host_callback_user_data(server, GRAPH_HOST_FAMILY_DRIVER, 0, 0);
    let server_params = CUDA_HOST_NODE_PARAMS {
        fn_: Some(graph_host_callback),
        userData: user_data,
    };
    let result = if client_params.fn_.is_some() {
        unsafe {
            cuGraphAddHostNode(
                &raw mut node,
                graph,
                driver_graph_dependencies_ptr(&dependencies),
                dependencies.len(),
                &raw const server_params,
            )
        }
    } else {
        CUresult::CUDA_ERROR_INVALID_VALUE
    };
    if result == CUresult::CUDA_SUCCESS {
        unsafe {
            (*(user_data as *mut GraphHostCallbackPayload)).node = node as usize;
        }
        mark_graph_has_host_nodes(graph);
    }

    node.send(&server.channel_sender).unwrap();
    send_result(
        "cuGraphAddHostNode",
        server.id,
        result,
        &server.channel_sender,
    );
}

pub fn cuGraphHostNodeGetParamsExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuGraphHostNodeGetParams", "[#{}]", server.id);

    let mut node = std::mem::MaybeUninit::<CUgraphNode>::uninit();
    node.recv(channel_receiver).unwrap();
    let node = unsafe { node.assume_init() };
    channel_receiver.recv_ts().unwrap();

    let mut params = std::mem::MaybeUninit::<CUDA_HOST_NODE_PARAMS>::uninit();
    let result = unsafe { cuGraphHostNodeGetParams(node, params.as_mut_ptr()) };
    send_result(
        "cuGraphHostNodeGetParams",
        server.id,
        result,
        channel_sender,
    );
}

pub fn cuGraphHostNodeSetParamsExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    log::debug!(target: "cuGraphHostNodeSetParams", "[#{}]", server.id);

    let mut node = std::mem::MaybeUninit::<CUgraphNode>::uninit();
    node.recv(&server.channel_receiver).unwrap();
    let node = unsafe { node.assume_init() };
    let mut client_params = std::mem::MaybeUninit::<CUDA_HOST_NODE_PARAMS>::uninit();
    client_params.recv(&server.channel_receiver).unwrap();
    let client_params = unsafe { client_params.assume_init() };
    server.channel_receiver.recv_ts().unwrap();

    let user_data =
        graph_host_callback_user_data(server, GRAPH_HOST_FAMILY_DRIVER, node as usize, 0);
    let server_params = CUDA_HOST_NODE_PARAMS {
        fn_: Some(graph_host_callback),
        userData: user_data,
    };
    let result = if client_params.fn_.is_some() {
        unsafe { cuGraphHostNodeSetParams(node, &raw const server_params) }
    } else {
        CUresult::CUDA_ERROR_INVALID_VALUE
    };

    send_result(
        "cuGraphHostNodeSetParams",
        server.id,
        result,
        &server.channel_sender,
    );
}

pub fn cuGraphExecHostNodeSetParamsExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    log::debug!(target: "cuGraphExecHostNodeSetParams", "[#{}]", server.id);

    let mut graph_exec = std::mem::MaybeUninit::<CUgraphExec>::uninit();
    graph_exec.recv(&server.channel_receiver).unwrap();
    let graph_exec = unsafe { graph_exec.assume_init() };
    let mut node = std::mem::MaybeUninit::<CUgraphNode>::uninit();
    node.recv(&server.channel_receiver).unwrap();
    let node = unsafe { node.assume_init() };
    let mut client_params = std::mem::MaybeUninit::<CUDA_HOST_NODE_PARAMS>::uninit();
    client_params.recv(&server.channel_receiver).unwrap();
    let client_params = unsafe { client_params.assume_init() };
    server.channel_receiver.recv_ts().unwrap();

    let user_data = graph_host_callback_user_data(
        server,
        GRAPH_HOST_FAMILY_DRIVER,
        node as usize,
        graph_exec as usize,
    );
    let server_params = CUDA_HOST_NODE_PARAMS {
        fn_: Some(graph_host_callback),
        userData: user_data,
    };
    let result = if client_params.fn_.is_some() {
        unsafe { cuGraphExecHostNodeSetParams(graph_exec, node, &raw const server_params) }
    } else {
        CUresult::CUDA_ERROR_INVALID_VALUE
    };
    if result == CUresult::CUDA_SUCCESS {
        mark_graph_exec_has_host_nodes(graph_exec);
    }

    send_result(
        "cuGraphExecHostNodeSetParams",
        server.id,
        result,
        &server.channel_sender,
    );
}

pub fn cuGraphAddExternalSemaphoresSignalNodeExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuGraphAddExternalSemaphoresSignalNode", "[#{}]", server.id);

    let mut graph = std::mem::MaybeUninit::<CUgraph>::uninit();
    graph.recv(channel_receiver).unwrap();
    let graph = unsafe { graph.assume_init() };
    let dependencies = recv_slice::<CUgraphNode, _>(channel_receiver).unwrap();
    let mut semaphores = recv_slice::<CUexternalSemaphore, _>(channel_receiver).unwrap();
    let params = recv_slice::<CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS, _>(channel_receiver).unwrap();
    channel_receiver.recv_ts().unwrap();

    let mut node: CUgraphNode = std::ptr::null_mut();
    let result = if semaphores.len() == params.len() {
        let node_params = driver_signal_node_params(&mut semaphores, &params);
        unsafe {
            cuGraphAddExternalSemaphoresSignalNode(
                &raw mut node,
                graph,
                driver_graph_dependencies_ptr(&dependencies),
                dependencies.len(),
                &raw const node_params,
            )
        }
    } else {
        CUresult::CUDA_ERROR_INVALID_VALUE
    };

    node.send(channel_sender).unwrap();
    send_result(
        "cuGraphAddExternalSemaphoresSignalNode",
        server.id,
        result,
        channel_sender,
    );
}

pub fn cuGraphExternalSemaphoresSignalNodeGetParamsExe<C: CommChannel>(
    server: &mut ServerWorker<C>,
) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuGraphExternalSemaphoresSignalNodeGetParams", "[#{}]", server.id);

    let mut node = std::mem::MaybeUninit::<CUgraphNode>::uninit();
    node.recv(channel_receiver).unwrap();
    let node = unsafe { node.assume_init() };
    channel_receiver.recv_ts().unwrap();

    let mut node_params = unsafe { std::mem::zeroed::<CUDA_EXT_SEM_SIGNAL_NODE_PARAMS>() };
    let result =
        unsafe { cuGraphExternalSemaphoresSignalNodeGetParams(node, &raw mut node_params) };
    let (semaphores, params) = driver_signal_params_from_node(&node_params, result);
    send_slice(&semaphores, channel_sender).unwrap();
    send_slice(&params, channel_sender).unwrap();
    send_result(
        "cuGraphExternalSemaphoresSignalNodeGetParams",
        server.id,
        result,
        channel_sender,
    );
}

pub fn cuGraphExternalSemaphoresSignalNodeSetParamsExe<C: CommChannel>(
    server: &mut ServerWorker<C>,
) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuGraphExternalSemaphoresSignalNodeSetParams", "[#{}]", server.id);

    let mut node = std::mem::MaybeUninit::<CUgraphNode>::uninit();
    node.recv(channel_receiver).unwrap();
    let node = unsafe { node.assume_init() };
    let mut semaphores = recv_slice::<CUexternalSemaphore, _>(channel_receiver).unwrap();
    let params = recv_slice::<CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS, _>(channel_receiver).unwrap();
    channel_receiver.recv_ts().unwrap();

    let result = if semaphores.len() == params.len() {
        let node_params = driver_signal_node_params(&mut semaphores, &params);
        unsafe { cuGraphExternalSemaphoresSignalNodeSetParams(node, &raw const node_params) }
    } else {
        CUresult::CUDA_ERROR_INVALID_VALUE
    };
    send_result(
        "cuGraphExternalSemaphoresSignalNodeSetParams",
        server.id,
        result,
        channel_sender,
    );
}

pub fn cuGraphExecExternalSemaphoresSignalNodeSetParamsExe<C: CommChannel>(
    server: &mut ServerWorker<C>,
) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuGraphExecExternalSemaphoresSignalNodeSetParams", "[#{}]", server.id);

    let mut graph_exec = std::mem::MaybeUninit::<CUgraphExec>::uninit();
    graph_exec.recv(channel_receiver).unwrap();
    let graph_exec = unsafe { graph_exec.assume_init() };
    let mut node = std::mem::MaybeUninit::<CUgraphNode>::uninit();
    node.recv(channel_receiver).unwrap();
    let node = unsafe { node.assume_init() };
    let mut semaphores = recv_slice::<CUexternalSemaphore, _>(channel_receiver).unwrap();
    let params = recv_slice::<CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS, _>(channel_receiver).unwrap();
    channel_receiver.recv_ts().unwrap();

    let result = if semaphores.len() == params.len() {
        let node_params = driver_signal_node_params(&mut semaphores, &params);
        unsafe {
            cuGraphExecExternalSemaphoresSignalNodeSetParams(
                graph_exec,
                node,
                &raw const node_params,
            )
        }
    } else {
        CUresult::CUDA_ERROR_INVALID_VALUE
    };
    send_result(
        "cuGraphExecExternalSemaphoresSignalNodeSetParams",
        server.id,
        result,
        channel_sender,
    );
}

pub fn cuGraphAddExternalSemaphoresWaitNodeExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuGraphAddExternalSemaphoresWaitNode", "[#{}]", server.id);

    let mut graph = std::mem::MaybeUninit::<CUgraph>::uninit();
    graph.recv(channel_receiver).unwrap();
    let graph = unsafe { graph.assume_init() };
    let dependencies = recv_slice::<CUgraphNode, _>(channel_receiver).unwrap();
    let mut semaphores = recv_slice::<CUexternalSemaphore, _>(channel_receiver).unwrap();
    let params = recv_slice::<CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS, _>(channel_receiver).unwrap();
    channel_receiver.recv_ts().unwrap();

    let mut node: CUgraphNode = std::ptr::null_mut();
    let result = if semaphores.len() == params.len() {
        let node_params = driver_wait_node_params(&mut semaphores, &params);
        unsafe {
            cuGraphAddExternalSemaphoresWaitNode(
                &raw mut node,
                graph,
                driver_graph_dependencies_ptr(&dependencies),
                dependencies.len(),
                &raw const node_params,
            )
        }
    } else {
        CUresult::CUDA_ERROR_INVALID_VALUE
    };

    node.send(channel_sender).unwrap();
    send_result(
        "cuGraphAddExternalSemaphoresWaitNode",
        server.id,
        result,
        channel_sender,
    );
}

pub fn cuGraphExternalSemaphoresWaitNodeGetParamsExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuGraphExternalSemaphoresWaitNodeGetParams", "[#{}]", server.id);

    let mut node = std::mem::MaybeUninit::<CUgraphNode>::uninit();
    node.recv(channel_receiver).unwrap();
    let node = unsafe { node.assume_init() };
    channel_receiver.recv_ts().unwrap();

    let mut node_params = unsafe { std::mem::zeroed::<CUDA_EXT_SEM_WAIT_NODE_PARAMS>() };
    let result = unsafe { cuGraphExternalSemaphoresWaitNodeGetParams(node, &raw mut node_params) };
    let (semaphores, params) = driver_wait_params_from_node(&node_params, result);
    send_slice(&semaphores, channel_sender).unwrap();
    send_slice(&params, channel_sender).unwrap();
    send_result(
        "cuGraphExternalSemaphoresWaitNodeGetParams",
        server.id,
        result,
        channel_sender,
    );
}

pub fn cuGraphExternalSemaphoresWaitNodeSetParamsExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuGraphExternalSemaphoresWaitNodeSetParams", "[#{}]", server.id);

    let mut node = std::mem::MaybeUninit::<CUgraphNode>::uninit();
    node.recv(channel_receiver).unwrap();
    let node = unsafe { node.assume_init() };
    let mut semaphores = recv_slice::<CUexternalSemaphore, _>(channel_receiver).unwrap();
    let params = recv_slice::<CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS, _>(channel_receiver).unwrap();
    channel_receiver.recv_ts().unwrap();

    let result = if semaphores.len() == params.len() {
        let node_params = driver_wait_node_params(&mut semaphores, &params);
        unsafe { cuGraphExternalSemaphoresWaitNodeSetParams(node, &raw const node_params) }
    } else {
        CUresult::CUDA_ERROR_INVALID_VALUE
    };
    send_result(
        "cuGraphExternalSemaphoresWaitNodeSetParams",
        server.id,
        result,
        channel_sender,
    );
}

pub fn cuGraphExecExternalSemaphoresWaitNodeSetParamsExe<C: CommChannel>(
    server: &mut ServerWorker<C>,
) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuGraphExecExternalSemaphoresWaitNodeSetParams", "[#{}]", server.id);

    let mut graph_exec = std::mem::MaybeUninit::<CUgraphExec>::uninit();
    graph_exec.recv(channel_receiver).unwrap();
    let graph_exec = unsafe { graph_exec.assume_init() };
    let mut node = std::mem::MaybeUninit::<CUgraphNode>::uninit();
    node.recv(channel_receiver).unwrap();
    let node = unsafe { node.assume_init() };
    let mut semaphores = recv_slice::<CUexternalSemaphore, _>(channel_receiver).unwrap();
    let params = recv_slice::<CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS, _>(channel_receiver).unwrap();
    channel_receiver.recv_ts().unwrap();

    let result = if semaphores.len() == params.len() {
        let node_params = driver_wait_node_params(&mut semaphores, &params);
        unsafe {
            cuGraphExecExternalSemaphoresWaitNodeSetParams(graph_exec, node, &raw const node_params)
        }
    } else {
        CUresult::CUDA_ERROR_INVALID_VALUE
    };
    send_result(
        "cuGraphExecExternalSemaphoresWaitNodeSetParams",
        server.id,
        result,
        channel_sender,
    );
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

pub fn cuModuleEnumerateFunctionsExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuModuleEnumerateFunctions", "[#{}]", server.id);

    let mut num_functions = 0u32;
    num_functions.recv(channel_receiver).unwrap();
    let mut module: CUmodule = std::ptr::null_mut();
    module.recv(channel_receiver).unwrap();
    channel_receiver.recv_ts().unwrap();

    let mut functions = vec![std::ptr::null_mut(); num_functions as usize];
    let functions_ptr = if functions.is_empty() {
        std::ptr::null_mut()
    } else {
        functions.as_mut_ptr()
    };
    let result = unsafe { cuModuleEnumerateFunctions(functions_ptr, num_functions, module) };

    let functions = if result == CUresult::CUDA_SUCCESS {
        functions
    } else {
        Vec::new()
    };
    send_slice(&functions, channel_sender).unwrap();
    for function in &functions {
        let mut name = std::ptr::null();
        let name_result = unsafe { cuFuncGetName(&raw mut name, *function) };
        let bytes = if name_result == CUresult::CUDA_SUCCESS && !name.is_null() {
            unsafe { CStr::from_ptr(name).to_bytes().to_vec() }
        } else {
            Vec::new()
        };
        send_slice(&bytes, channel_sender).unwrap();
    }
    send_result(
        "cuModuleEnumerateFunctions",
        server.id,
        result,
        channel_sender,
    );
}

pub fn cuLibraryEnumerateKernelsExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker {
        channel_sender,
        channel_receiver,
        ..
    } = server;
    log::debug!(target: "cuLibraryEnumerateKernels", "[#{}]", server.id);

    let mut num_kernels = 0u32;
    num_kernels.recv(channel_receiver).unwrap();
    let mut library: CUlibrary = std::ptr::null_mut();
    library.recv(channel_receiver).unwrap();
    channel_receiver.recv_ts().unwrap();

    let mut kernels = vec![std::ptr::null_mut(); num_kernels as usize];
    let kernels_ptr = if kernels.is_empty() {
        std::ptr::null_mut()
    } else {
        kernels.as_mut_ptr()
    };
    let result = unsafe { cuLibraryEnumerateKernels(kernels_ptr, num_kernels, library) };

    let kernels = if result == CUresult::CUDA_SUCCESS {
        kernels
    } else {
        Vec::new()
    };
    send_slice(&kernels, channel_sender).unwrap();
    for kernel in &kernels {
        let mut name = std::ptr::null();
        let name_result = unsafe { cuKernelGetName(&raw mut name, *kernel) };
        let bytes = if name_result == CUresult::CUDA_SUCCESS && !name.is_null() {
            unsafe { CStr::from_ptr(name).to_bytes().to_vec() }
        } else {
            Vec::new()
        };
        send_slice(&bytes, channel_sender).unwrap();
    }
    send_result(
        "cuLibraryEnumerateKernels",
        server.id,
        result,
        channel_sender,
    );
}
