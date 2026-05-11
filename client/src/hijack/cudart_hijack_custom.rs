#![expect(non_snake_case)]
use super::*;
use cudasys::types::cuda::{
    CUDA_KERNEL_NODE_PARAMS, CUdeviceptr, CUgraphNode, CUlaunchAttribute, CUlaunchConfig, CUmodule,
    CUresult,
};
use cudasys::types::cudart::*;
use network::type_impl::recv_slice;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::*;
use std::sync::{Mutex, OnceLock};

fn host_allocations() -> &'static Mutex<BTreeSet<usize>> {
    static ALLOCATIONS: OnceLock<Mutex<BTreeSet<usize>>> = OnceLock::new();
    ALLOCATIONS.get_or_init(|| Mutex::new(BTreeSet::new()))
}

fn cuda_error_text(error: cudaError_t, include_code: bool) -> *const c_char {
    static ERROR_TEXTS: OnceLock<Mutex<BTreeMap<(c_int, bool), CString>>> = OnceLock::new();
    let error_code = error as c_int;
    let mut texts = ERROR_TEXTS
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .unwrap();
    let text = texts.entry((error_code, include_code)).or_insert_with(|| {
        let text = if include_code {
            format!("{error:?} ({error_code})")
        } else {
            format!("{error:?}")
        };
        CString::new(text).unwrap()
    });
    text.as_ptr()
}

fn recv_cuda_result<C: CommChannel>(
    target: &'static str,
    client_id: i32,
    channel_receiver: &C,
) -> cudaError_t {
    let mut result = cudaError_t::cudaSuccess;
    result.recv(channel_receiver).unwrap();
    channel_receiver.recv_ts().unwrap();
    if result.is_error() {
        log::error!(
            target: target,
            "[#{}] returned error: {:?}\n{}",
            client_id,
            result,
            std::backtrace::Backtrace::force_capture(),
        );
    }
    result
}

fn recv_graph_node_slice<C: CommChannel>(
    target: &'static str,
    dst: *mut cudaGraphNode_t,
    capacity: usize,
    channel_receiver: &C,
) -> Option<cudaError_t> {
    if dst.is_null() {
        return None;
    }
    let nodes = recv_slice::<cudaGraphNode_t, _>(channel_receiver).unwrap();
    if nodes.len() > capacity {
        log::error!(
            target: target,
            "server returned {} graph nodes for client capacity {}",
            nodes.len(),
            capacity,
        );
        return Some(cudaError_t::cudaErrorInvalidValue);
    }
    unsafe {
        std::ptr::copy_nonoverlapping(nodes.as_ptr(), dst, nodes.len());
    }
    None
}

fn recv_graph_edge_data_slice<C: CommChannel>(
    target: &'static str,
    dst: *mut cudaGraphEdgeData,
    capacity: usize,
    channel_receiver: &C,
) -> Option<cudaError_t> {
    if dst.is_null() {
        return None;
    }
    let edge_data = recv_slice::<cudaGraphEdgeData, _>(channel_receiver).unwrap();
    if edge_data.len() > capacity {
        log::error!(
            target: target,
            "server returned {} graph edges for client capacity {}",
            edge_data.len(),
            capacity,
        );
        return Some(cudaError_t::cudaErrorInvalidValue);
    }
    unsafe {
        std::ptr::copy_nonoverlapping(edge_data.as_ptr(), dst, edge_data.len());
    }
    None
}

fn cuda_graph_get_node_list<T: Transportable>(
    proc_id: i32,
    target: &'static str,
    graph_or_node: T,
    nodes: *mut cudaGraphNode_t,
    count: *mut usize,
) -> cudaError_t {
    if count.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }
    let has_nodes = !nodes.is_null();
    let capacity = if has_nodes { unsafe { *count } } else { 0 };
    if has_nodes && capacity == 0 {
        return cudaError_t::cudaErrorInvalidValue;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: target, "[#{}]", client.id);
        let ClientThread {
            channel_sender,
            channel_receiver,
            ..
        } = client;

        proc_id.send(channel_sender).unwrap();
        graph_or_node.send(channel_sender).unwrap();
        has_nodes.send(channel_sender).unwrap();
        capacity.send(channel_sender).unwrap();
        channel_sender.flush_out().unwrap();

        let local_error = recv_graph_node_slice(target, nodes, capacity, channel_receiver);
        unsafe { &mut *count }.recv(channel_receiver).unwrap();
        let result = recv_cuda_result(target, client.id, channel_receiver);
        local_error.unwrap_or(result)
    })
}

fn cuda_graph_get_edge_list<T: Transportable>(
    proc_id: i32,
    target: &'static str,
    graph_or_node: T,
    from: *mut cudaGraphNode_t,
    to: *mut cudaGraphNode_t,
    edge_data: *mut cudaGraphEdgeData,
    count: *mut usize,
) -> cudaError_t {
    if count.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }
    let has_from = !from.is_null();
    let has_to = !to.is_null();
    let has_edge_data = !edge_data.is_null();
    let has_outputs = has_from || has_to || has_edge_data;
    let capacity = if has_outputs { unsafe { *count } } else { 0 };
    if has_outputs && capacity == 0 {
        return cudaError_t::cudaErrorInvalidValue;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: target, "[#{}]", client.id);
        let ClientThread {
            channel_sender,
            channel_receiver,
            ..
        } = client;

        proc_id.send(channel_sender).unwrap();
        graph_or_node.send(channel_sender).unwrap();
        has_from.send(channel_sender).unwrap();
        has_to.send(channel_sender).unwrap();
        has_edge_data.send(channel_sender).unwrap();
        capacity.send(channel_sender).unwrap();
        channel_sender.flush_out().unwrap();

        let mut local_error = recv_graph_node_slice(target, from, capacity, channel_receiver);
        let to_error = recv_graph_node_slice(target, to, capacity, channel_receiver);
        if local_error.is_none() {
            local_error = to_error;
        }
        let edge_data_error =
            recv_graph_edge_data_slice(target, edge_data, capacity, channel_receiver);
        if local_error.is_none() {
            local_error = edge_data_error;
        }
        unsafe { &mut *count }.recv(channel_receiver).unwrap();
        let result = recv_cuda_result(target, client.id, channel_receiver);
        local_error.unwrap_or(result)
    })
}

fn classify_pointer(ptr: *const c_void) -> Option<cudaMemoryType> {
    if ptr.is_null() {
        return None;
    }

    let mut attributes = std::mem::MaybeUninit::<cudaPointerAttributes>::uninit();
    let result = super::cudart_hijack::cudaPointerGetAttributes(attributes.as_mut_ptr(), ptr);
    if result == cudaError_t::cudaSuccess {
        Some(unsafe { attributes.assume_init() }.type_)
    } else {
        None
    }
}

fn is_device_pointer(ptr: *const c_void) -> bool {
    matches!(
        classify_pointer(ptr),
        Some(cudaMemoryType::cudaMemoryTypeDevice | cudaMemoryType::cudaMemoryTypeManaged)
    )
}

fn resolve_memcpy_kind(
    dst: *mut c_void,
    src: *const c_void,
    kind: cudaMemcpyKind,
) -> cudaMemcpyKind {
    if kind != cudaMemcpyKind::cudaMemcpyDefault {
        return kind;
    }

    match (
        is_device_pointer(dst as *const c_void),
        is_device_pointer(src),
    ) {
        (true, true) => cudaMemcpyKind::cudaMemcpyDeviceToDevice,
        (true, false) => cudaMemcpyKind::cudaMemcpyHostToDevice,
        (false, true) => cudaMemcpyKind::cudaMemcpyDeviceToHost,
        (false, false) => cudaMemcpyKind::cudaMemcpyHostToHost,
    }
}

fn resolve_to_array_kind(src: *const c_void, kind: cudaMemcpyKind) -> cudaMemcpyKind {
    if kind != cudaMemcpyKind::cudaMemcpyDefault {
        return kind;
    }

    if is_device_pointer(src) {
        cudaMemcpyKind::cudaMemcpyDeviceToDevice
    } else {
        cudaMemcpyKind::cudaMemcpyHostToDevice
    }
}

fn resolve_from_array_kind(dst: *mut c_void, kind: cudaMemcpyKind) -> cudaMemcpyKind {
    if kind != cudaMemcpyKind::cudaMemcpyDefault {
        return kind;
    }

    if is_device_pointer(dst.cast_const()) {
        cudaMemcpyKind::cudaMemcpyDeviceToDevice
    } else {
        cudaMemcpyKind::cudaMemcpyDeviceToHost
    }
}

fn ensure_runtime_device(runtime: &mut RuntimeCache) -> cudaError_t {
    if let Some(device) = runtime.cuda_device {
        let current_device = CLIENT_THREAD.with_borrow_mut(|client| {
            client.ensure_current_process();
            client.cuda_device
        });
        if current_device != Some(device) {
            return cudaError_t::cudaErrorInvalidDevice;
        }
        return cudaError_t::cudaSuccess;
    }

    let mut device = 0;
    let result = super::cudart_hijack::cudaGetDevice(&mut device);
    if result != cudaError_t::cudaSuccess {
        return result;
    }
    let result = super::cudart_hijack::cudaSetDevice(device);
    if result != cudaError_t::cudaSuccess {
        return result;
    }
    runtime.cuda_device = Some(device);
    cudaError_t::cudaSuccess
}

fn load_module_for_fatbin(runtime: &mut RuntimeCache, fatCubinHandle: FatBinaryHandle) -> CUmodule {
    *runtime
        .loaded_modules
        .entry(fatCubinHandle)
        .or_insert_with(|| {
            // See our implementation of `__cudaRegisterFatBinary`.
            let index = (fatCubinHandle >> 4) - 1;
            log::debug!("registering fatbin #{index}");
            let image = runtime.lazy_fatbins[index];
            let mut module = std::ptr::null_mut();
            assert_eq!(
                super::cuda_hijack::cuModuleLoadDataInternal(&raw mut module, image.cast(), true),
                CUresult::CUDA_SUCCESS,
            );
            module
        })
}

fn allocate_host(ptr_out: *mut *mut c_void, size: usize) -> cudaError_t {
    if ptr_out.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }

    if size == 0 {
        unsafe {
            *ptr_out = std::ptr::null_mut();
        }
        return cudaError_t::cudaSuccess;
    }

    let ptr = unsafe { libc::malloc(size) };
    if ptr.is_null() {
        return cudaError_t::cudaErrorMemoryAllocation;
    }

    host_allocations().lock().unwrap().insert(ptr as usize);
    unsafe {
        *ptr_out = ptr;
    }
    cudaError_t::cudaSuccess
}

#[no_mangle]
pub extern "C" fn cudaMemcpy(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    log::debug!(target: "cudaMemcpy", "kind = {kind:?}");
    if count > 0 && (dst.is_null() || src.is_null()) {
        return cudaError_t::cudaErrorInvalidValue;
    }

    let kind = resolve_memcpy_kind(dst, src, kind);
    match kind {
        cudaMemcpyKind::cudaMemcpyHostToHost => unsafe {
            std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, count);
            cudaError_t::cudaSuccess
        },
        cudaMemcpyKind::cudaMemcpyHostToDevice => {
            super::cudart_hijack::cudaMemcpyHtod(dst, src.cast(), count, kind)
        }
        cudaMemcpyKind::cudaMemcpyDeviceToHost => {
            super::cudart_hijack::cudaMemcpyDtoh(dst.cast(), src, count, kind)
        }
        cudaMemcpyKind::cudaMemcpyDeviceToDevice => {
            super::cudart_hijack::cudaMemcpyDtod(dst, src, count, kind)
        }
        cudaMemcpyKind::cudaMemcpyDefault => unreachable!(),
    }
}

#[no_mangle]
pub extern "C" fn cudaMemcpyAsync(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    log::debug!(target: "cudaMemcpyAsync", "kind = {kind:?}");
    if count > 0 && (dst.is_null() || src.is_null()) {
        return cudaError_t::cudaErrorInvalidValue;
    }

    let kind = resolve_memcpy_kind(dst, src, kind);
    match kind {
        cudaMemcpyKind::cudaMemcpyHostToHost => unsafe {
            super::cudart_hijack::cudaStreamSynchronize(stream);
            std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, count);
            cudaError_t::cudaSuccess
        },
        cudaMemcpyKind::cudaMemcpyHostToDevice => {
            super::cudart_hijack::cudaMemcpyAsyncHtod(dst, src.cast(), count, kind, stream)
        }
        cudaMemcpyKind::cudaMemcpyDeviceToHost => {
            super::cudart_hijack::cudaMemcpyAsyncDtoh(dst.cast(), src, count, kind, stream)
        }
        cudaMemcpyKind::cudaMemcpyDeviceToDevice => {
            super::cudart_hijack::cudaMemcpyAsyncDtod(dst, src, count, kind, stream)
        }
        cudaMemcpyKind::cudaMemcpyDefault => unreachable!(),
    }
}

#[no_mangle]
pub extern "C" fn cudaMemcpy2D(
    dst: *mut c_void,
    dpitch: usize,
    src: *const c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    log::debug!(
        target: "cudaMemcpy2D",
        "width = {width}, height = {height}, kind = {kind:?}",
    );

    if height == 0 || width == 0 {
        return cudaError_t::cudaSuccess;
    }

    if dst.is_null() || src.is_null() || width > dpitch || width > spitch {
        return cudaError_t::cudaErrorInvalidValue;
    }

    for row in 0..height {
        let Some(dst_offset) = row.checked_mul(dpitch) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let Some(src_offset) = row.checked_mul(spitch) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let row_dst = unsafe { (dst as *mut u8).add(dst_offset).cast() };
        let row_src = unsafe { (src as *const u8).add(src_offset).cast() };
        let result = cudaMemcpy(row_dst, row_src, width, kind);
        if result != cudaError_t::cudaSuccess {
            return result;
        }
    }

    cudaError_t::cudaSuccess
}

#[no_mangle]
pub extern "C" fn cudaMemcpy2DAsync(
    dst: *mut c_void,
    dpitch: usize,
    src: *const c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    log::debug!(
        target: "cudaMemcpy2DAsync",
        "width = {width}, height = {height}, kind = {kind:?}",
    );

    if height == 0 || width == 0 {
        return cudaError_t::cudaSuccess;
    }

    if dst.is_null() || src.is_null() || width > dpitch || width > spitch {
        return cudaError_t::cudaErrorInvalidValue;
    }

    for row in 0..height {
        let Some(dst_offset) = row.checked_mul(dpitch) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let Some(src_offset) = row.checked_mul(spitch) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let row_dst = unsafe { (dst as *mut u8).add(dst_offset).cast() };
        let row_src = unsafe { (src as *const u8).add(src_offset).cast() };
        let result = cudaMemcpyAsync(row_dst, row_src, width, kind, stream);
        if result != cudaError_t::cudaSuccess {
            return result;
        }
    }

    cudaError_t::cudaSuccess
}

#[no_mangle]
pub extern "C" fn cudaMemcpyToArray(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    log::debug!(target: "cudaMemcpyToArray", "count = {count}, kind = {kind:?}");
    if count > 0 && (dst.is_null() || src.is_null()) {
        return cudaError_t::cudaErrorInvalidValue;
    }

    let kind = resolve_to_array_kind(src, kind);

    match kind {
        cudaMemcpyKind::cudaMemcpyHostToDevice => super::cudart_hijack::cudaMemcpyToArrayHtod(
            dst,
            wOffset,
            hOffset,
            src.cast::<u8>(),
            count,
            kind,
        ),
        cudaMemcpyKind::cudaMemcpyDeviceToDevice => {
            super::cudart_hijack::cudaMemcpyToArrayDtod(dst, wOffset, hOffset, src, count, kind)
        }
        _ => cudaError_t::cudaErrorInvalidValue,
    }
}

#[no_mangle]
pub extern "C" fn cudaMemcpyFromArray(
    dst: *mut c_void,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    log::debug!(target: "cudaMemcpyFromArray", "count = {count}, kind = {kind:?}");
    if count > 0 && (dst.is_null() || src.is_null()) {
        return cudaError_t::cudaErrorInvalidValue;
    }

    let kind = resolve_from_array_kind(dst, kind);

    match kind {
        cudaMemcpyKind::cudaMemcpyDeviceToHost => super::cudart_hijack::cudaMemcpyFromArrayDtoh(
            dst.cast::<u8>(),
            src,
            wOffset,
            hOffset,
            count,
            kind,
        ),
        cudaMemcpyKind::cudaMemcpyDeviceToDevice => {
            super::cudart_hijack::cudaMemcpyFromArrayDtod(dst, src, wOffset, hOffset, count, kind)
        }
        _ => cudaError_t::cudaErrorInvalidValue,
    }
}

#[no_mangle]
pub extern "C" fn cudaMemcpyToArrayAsync(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    log::debug!(target: "cudaMemcpyToArrayAsync", "count = {count}, kind = {kind:?}");
    if count > 0 && (dst.is_null() || src.is_null()) {
        return cudaError_t::cudaErrorInvalidValue;
    }

    let kind = resolve_to_array_kind(src, kind);
    match kind {
        cudaMemcpyKind::cudaMemcpyHostToDevice => super::cudart_hijack::cudaMemcpyToArrayAsyncHtod(
            dst,
            wOffset,
            hOffset,
            src.cast::<u8>(),
            count,
            kind,
            stream,
        ),
        cudaMemcpyKind::cudaMemcpyDeviceToDevice => {
            super::cudart_hijack::cudaMemcpyToArrayAsyncDtod(
                dst, wOffset, hOffset, src, count, kind, stream,
            )
        }
        _ => cudaError_t::cudaErrorInvalidValue,
    }
}

#[no_mangle]
pub extern "C" fn cudaMemcpyFromArrayAsync(
    dst: *mut c_void,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    log::debug!(target: "cudaMemcpyFromArrayAsync", "count = {count}, kind = {kind:?}");
    if count > 0 && (dst.is_null() || src.is_null()) {
        return cudaError_t::cudaErrorInvalidValue;
    }

    let kind = resolve_from_array_kind(dst, kind);
    match kind {
        cudaMemcpyKind::cudaMemcpyDeviceToHost => {
            super::cudart_hijack::cudaMemcpyFromArrayAsyncDtoh(
                dst.cast::<u8>(),
                src,
                wOffset,
                hOffset,
                count,
                kind,
                stream,
            )
        }
        cudaMemcpyKind::cudaMemcpyDeviceToDevice => {
            super::cudart_hijack::cudaMemcpyFromArrayAsyncDtod(
                dst, src, wOffset, hOffset, count, kind, stream,
            )
        }
        _ => cudaError_t::cudaErrorInvalidValue,
    }
}

#[no_mangle]
pub extern "C" fn cudaMemcpy2DToArray(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    src: *const c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    log::debug!(
        target: "cudaMemcpy2DToArray",
        "width = {width}, height = {height}, kind = {kind:?}",
    );

    if height == 0 || width == 0 {
        return cudaError_t::cudaSuccess;
    }
    if dst.is_null() || src.is_null() || width > spitch {
        return cudaError_t::cudaErrorInvalidValue;
    }

    for row in 0..height {
        let Some(src_offset) = row.checked_mul(spitch) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let Some(row_h_offset) = hOffset.checked_add(row) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let row_src = unsafe { (src as *const u8).add(src_offset).cast() };
        let result = cudaMemcpyToArray(dst, wOffset, row_h_offset, row_src, width, kind);
        if result != cudaError_t::cudaSuccess {
            return result;
        }
    }

    cudaError_t::cudaSuccess
}

#[no_mangle]
pub extern "C" fn cudaMemcpy2DFromArray(
    dst: *mut c_void,
    dpitch: usize,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    log::debug!(
        target: "cudaMemcpy2DFromArray",
        "width = {width}, height = {height}, kind = {kind:?}",
    );

    if height == 0 || width == 0 {
        return cudaError_t::cudaSuccess;
    }
    if dst.is_null() || src.is_null() || width > dpitch {
        return cudaError_t::cudaErrorInvalidValue;
    }

    for row in 0..height {
        let Some(dst_offset) = row.checked_mul(dpitch) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let Some(row_h_offset) = hOffset.checked_add(row) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let row_dst = unsafe { (dst as *mut u8).add(dst_offset).cast() };
        let result = cudaMemcpyFromArray(row_dst, src, wOffset, row_h_offset, width, kind);
        if result != cudaError_t::cudaSuccess {
            return result;
        }
    }

    cudaError_t::cudaSuccess
}

#[no_mangle]
pub extern "C" fn cudaMemcpy2DToArrayAsync(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    src: *const c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    log::debug!(
        target: "cudaMemcpy2DToArrayAsync",
        "width = {width}, height = {height}, kind = {kind:?}",
    );

    if height == 0 || width == 0 {
        return cudaError_t::cudaSuccess;
    }
    if dst.is_null() || src.is_null() || width > spitch {
        return cudaError_t::cudaErrorInvalidValue;
    }

    for row in 0..height {
        let Some(src_offset) = row.checked_mul(spitch) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let Some(row_h_offset) = hOffset.checked_add(row) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let row_src = unsafe { (src as *const u8).add(src_offset).cast() };
        let result =
            cudaMemcpyToArrayAsync(dst, wOffset, row_h_offset, row_src, width, kind, stream);
        if result != cudaError_t::cudaSuccess {
            return result;
        }
    }

    cudaError_t::cudaSuccess
}

#[no_mangle]
pub extern "C" fn cudaMemcpy2DFromArrayAsync(
    dst: *mut c_void,
    dpitch: usize,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    log::debug!(
        target: "cudaMemcpy2DFromArrayAsync",
        "width = {width}, height = {height}, kind = {kind:?}",
    );

    if height == 0 || width == 0 {
        return cudaError_t::cudaSuccess;
    }
    if dst.is_null() || src.is_null() || width > dpitch {
        return cudaError_t::cudaErrorInvalidValue;
    }

    for row in 0..height {
        let Some(dst_offset) = row.checked_mul(dpitch) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let Some(row_h_offset) = hOffset.checked_add(row) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let row_dst = unsafe { (dst as *mut u8).add(dst_offset).cast() };
        let result =
            cudaMemcpyFromArrayAsync(row_dst, src, wOffset, row_h_offset, width, kind, stream);
        if result != cudaError_t::cudaSuccess {
            return result;
        }
    }

    cudaError_t::cudaSuccess
}

fn get_cufunction(func: HostPtr) -> cudasys::cuda::CUfunction {
    if !CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        client.cuda_device_init
    }) {
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#initialization
        assert_eq!(
            super::cudart_hijack::cudaFree(std::ptr::null_mut()),
            Default::default()
        );
        CLIENT_THREAD.with_borrow_mut(|client| client.cuda_device_init = true);
    }

    if let Some(&cufunc) = RUNTIME_CACHE.read().unwrap().loaded_functions.get(&func) {
        return cufunc;
    }

    let runtime = &mut *RUNTIME_CACHE.write().unwrap();

    // TODO: In CUDA 12, use `cuLibrary{LoadData,GetKernel}` to avoid pinning device.
    if runtime.cuda_device.is_none() {
        log::info!(
            "#fatbins = {}, #functions = {}",
            runtime.lazy_fatbins.len(),
            runtime.lazy_functions.len(),
        );
    }
    assert_eq!(ensure_runtime_device(runtime), cudaError_t::cudaSuccess);

    let (fatCubinHandle, deviceName) = *runtime.lazy_functions.get(&func).unwrap();
    let module = load_module_for_fatbin(runtime, fatCubinHandle);
    log::debug!("registering function {:?}", unsafe {
        CStr::from_ptr(deviceName)
    });
    let mut cufunc = std::ptr::null_mut();
    assert_eq!(
        super::cuda_hijack::cuModuleGetFunction(&raw mut cufunc, module, deviceName),
        Default::default(),
    );
    runtime.loaded_functions.insert(func, cufunc);
    cufunc
}

fn pack_runtime_kernel_node_params(
    pNodeParams: *const cudaKernelNodeParams,
) -> Result<
    (
        cudaKernelNodeParams,
        CUDA_KERNEL_NODE_PARAMS,
        Box<[u8]>,
        Box<[u32]>,
    ),
    cudaError_t,
> {
    if pNodeParams.is_null() {
        return Err(cudaError_t::cudaErrorInvalidValue);
    }
    let node_params = unsafe { &*pNodeParams };
    if node_params.func.is_null() {
        return Err(cudaError_t::cudaErrorInvalidDeviceFunction);
    }
    if !node_params.extra.is_null() {
        return Err(cudaError_t::cudaErrorNotSupported);
    }

    let host_func = node_params.func as HostPtr;
    if !RUNTIME_CACHE
        .read()
        .unwrap()
        .lazy_functions
        .contains_key(&host_func)
    {
        return Err(cudaError_t::cudaErrorInvalidDeviceFunction);
    }
    let cufunc = get_cufunction(host_func);
    let driver = DRIVER_CACHE.read().unwrap();
    let Some(param_info) = driver.function_params.get(&cufunc) else {
        return Err(cudaError_t::cudaErrorInvalidDeviceFunction);
    };
    let (args, arg_offsets) = super::cuda_hijack_utils::pack_kernel_args_with_offsets(
        node_params.kernelParams,
        param_info,
    );
    let packed_params = CUDA_KERNEL_NODE_PARAMS {
        func: cufunc,
        gridDimX: node_params.gridDim.x,
        gridDimY: node_params.gridDim.y,
        gridDimZ: node_params.gridDim.z,
        blockDimX: node_params.blockDim.x,
        blockDimY: node_params.blockDim.y,
        blockDimZ: node_params.blockDim.z,
        sharedMemBytes: node_params.sharedMemBytes,
        kernelParams: std::ptr::null_mut(),
        extra: std::ptr::null_mut(),
        kern: std::ptr::null_mut(),
        ctx: std::ptr::null_mut(),
    };
    Ok((*node_params, packed_params, args, arg_offsets))
}

fn cache_runtime_graph_kernel_node(
    node: cudaGraphNode_t,
    cached_params: cudaKernelNodeParams,
    packed_params: CUDA_KERNEL_NODE_PARAMS,
    args: Box<[u8]>,
    arg_offsets: Box<[u32]>,
) {
    RUNTIME_CACHE.write().unwrap().graph_kernel_nodes.insert(
        node,
        crate::RuntimeGraphKernelNodeCache::new(cached_params, args.clone(), arg_offsets.clone()),
    );
    DRIVER_CACHE.write().unwrap().graph_kernel_nodes.insert(
        node as CUgraphNode,
        crate::GraphKernelNodeCache::new(packed_params, args, arg_offsets),
    );
}

fn runtime_symbol_metadata(symbol: *const c_void) -> Result<(CUdeviceptr, usize), cudaError_t> {
    if symbol.is_null() {
        return Err(cudaError_t::cudaErrorInvalidValue);
    }

    let runtime = &mut *RUNTIME_CACHE.write().unwrap();
    let Some(&(fatCubinHandle, deviceName)) = runtime.lazy_variables.get(&(symbol as HostPtr))
    else {
        return Err(cudaError_t::cudaErrorInvalidSymbol);
    };
    let result = ensure_runtime_device(runtime);
    if result != cudaError_t::cudaSuccess {
        return Err(result);
    }
    let module = load_module_for_fatbin(runtime, fatCubinHandle);

    let mut dptr: CUdeviceptr = 0;
    let mut bytes: usize = 0;
    let result =
        super::cuda_hijack::cuModuleGetGlobal_v2(&raw mut dptr, &raw mut bytes, module, deviceName);
    if result != CUresult::CUDA_SUCCESS {
        return Err(cudaError_t::cudaErrorInvalidSymbol);
    }
    Ok((dptr, bytes))
}

fn symbol_device_address(symbol: *const c_void, offset: usize) -> Result<*mut c_void, cudaError_t> {
    let (base, _) = runtime_symbol_metadata(symbol)?;
    Ok(unsafe { (base as *mut u8).add(offset).cast() })
}

fn symbol_device_range(
    symbol: *const c_void,
    offset: usize,
    count: usize,
) -> Result<*mut c_void, cudaError_t> {
    let (base, bytes) = runtime_symbol_metadata(symbol)?;
    let Some(end) = offset.checked_add(count) else {
        return Err(cudaError_t::cudaErrorInvalidValue);
    };
    if end > bytes {
        return Err(cudaError_t::cudaErrorInvalidValue);
    }
    Ok(unsafe { (base as *mut u8).add(offset).cast() })
}

fn symbol_copy_to_kind(src: *const c_void, kind: cudaMemcpyKind) -> cudaMemcpyKind {
    if kind != cudaMemcpyKind::cudaMemcpyDefault {
        return kind;
    }
    if is_device_pointer(src) {
        cudaMemcpyKind::cudaMemcpyDeviceToDevice
    } else {
        cudaMemcpyKind::cudaMemcpyHostToDevice
    }
}

fn symbol_copy_from_kind(dst: *const c_void, kind: cudaMemcpyKind) -> cudaMemcpyKind {
    if kind != cudaMemcpyKind::cudaMemcpyDefault {
        return kind;
    }
    if is_device_pointer(dst) {
        cudaMemcpyKind::cudaMemcpyDeviceToDevice
    } else {
        cudaMemcpyKind::cudaMemcpyDeviceToHost
    }
}

fn write_driver_entry_point(
    symbol: *const c_char,
    funcPtr: *mut *mut c_void,
    driverStatus: *mut cudaDriverEntryPointQueryResult,
) -> cudaError_t {
    if symbol.is_null() || funcPtr.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }

    let func = unsafe { libc::dlsym(libc::RTLD_DEFAULT, symbol) };
    unsafe {
        *funcPtr = func;
        if !driverStatus.is_null() {
            *driverStatus = if func.is_null() {
                cudaDriverEntryPointQueryResult::cudaDriverEntryPointSymbolNotFound
            } else {
                cudaDriverEntryPointQueryResult::cudaDriverEntryPointSuccess
            };
        }
    }
    cudaError_t::cudaSuccess
}

#[no_mangle]
pub extern "C" fn cudaGetDriverEntryPoint(
    symbol: *const c_char,
    funcPtr: *mut *mut c_void,
    _flags: c_ulonglong,
    driverStatus: *mut cudaDriverEntryPointQueryResult,
) -> cudaError_t {
    log::debug!(target: "cudaGetDriverEntryPoint", "");
    write_driver_entry_point(symbol, funcPtr, driverStatus)
}

#[no_mangle]
pub extern "C" fn cudaGetDriverEntryPointByVersion(
    symbol: *const c_char,
    funcPtr: *mut *mut c_void,
    _cudaVersion: c_uint,
    _flags: c_ulonglong,
    driverStatus: *mut cudaDriverEntryPointQueryResult,
) -> cudaError_t {
    log::debug!(target: "cudaGetDriverEntryPointByVersion", "");
    write_driver_entry_point(symbol, funcPtr, driverStatus)
}

#[no_mangle]
pub extern "C" fn cudaGetSymbolAddress(
    devPtr: *mut *mut c_void,
    symbol: *const c_void,
) -> cudaError_t {
    log::debug!(target: "cudaGetSymbolAddress", "");
    if devPtr.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }

    let (dptr, _) = match runtime_symbol_metadata(symbol) {
        Ok(metadata) => metadata,
        Err(error) => return error,
    };

    unsafe {
        *devPtr = dptr as *mut c_void;
    }
    cudaError_t::cudaSuccess
}

#[no_mangle]
pub extern "C" fn cudaGetSymbolSize(size: *mut usize, symbol: *const c_void) -> cudaError_t {
    log::debug!(target: "cudaGetSymbolSize", "");
    if size.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }

    let (_, bytes) = match runtime_symbol_metadata(symbol) {
        Ok(metadata) => metadata,
        Err(error) => return error,
    };

    unsafe {
        *size = bytes;
    }
    cudaError_t::cudaSuccess
}

#[no_mangle]
pub extern "C" fn cudaMemcpyToSymbol(
    symbol: *const c_void,
    src: *const c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    log::debug!(target: "cudaMemcpyToSymbol", "count = {count}, offset = {offset}, kind = {kind:?}");
    let dst = match symbol_device_range(symbol, offset, count) {
        Ok(ptr) => ptr,
        Err(error) => return error,
    };
    let kind = symbol_copy_to_kind(src, kind);
    cudaMemcpy(dst, src, count, kind)
}

#[no_mangle]
pub extern "C" fn cudaMemcpyFromSymbol(
    dst: *mut c_void,
    symbol: *const c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    log::debug!(target: "cudaMemcpyFromSymbol", "count = {count}, offset = {offset}, kind = {kind:?}");
    let src = match symbol_device_range(symbol, offset, count) {
        Ok(ptr) => ptr.cast_const(),
        Err(error) => return error,
    };
    let kind = symbol_copy_from_kind(dst.cast_const(), kind);
    cudaMemcpy(dst, src, count, kind)
}

#[no_mangle]
pub extern "C" fn cudaMemcpyToSymbolAsync(
    symbol: *const c_void,
    src: *const c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    log::debug!(target: "cudaMemcpyToSymbolAsync", "count = {count}, offset = {offset}, kind = {kind:?}");
    let dst = match symbol_device_range(symbol, offset, count) {
        Ok(ptr) => ptr,
        Err(error) => return error,
    };
    let kind = symbol_copy_to_kind(src, kind);
    cudaMemcpyAsync(dst, src, count, kind, stream)
}

#[no_mangle]
pub extern "C" fn cudaMemcpyFromSymbolAsync(
    dst: *mut c_void,
    symbol: *const c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    log::debug!(target: "cudaMemcpyFromSymbolAsync", "count = {count}, offset = {offset}, kind = {kind:?}");
    let src = match symbol_device_range(symbol, offset, count) {
        Ok(ptr) => ptr.cast_const(),
        Err(error) => return error,
    };
    let kind = symbol_copy_from_kind(dst.cast_const(), kind);
    cudaMemcpyAsync(dst, src, count, kind, stream)
}

fn write_kernel_handle(kernel: *mut cudaKernel_t, entry_func_addr: MemPtr) -> cudaError_t {
    if kernel.is_null() || entry_func_addr == 0 {
        return cudaError_t::cudaErrorInvalidValue;
    }

    unsafe {
        *kernel = entry_func_addr as cudaKernel_t;
    }
    cudaError_t::cudaSuccess
}

#[no_mangle]
pub extern "C" fn __cudaGetKernel(kernel: *mut cudaKernel_t, entryFuncAddr: MemPtr) -> cudaError_t {
    log::debug!(target: "__cudaGetKernel", "");
    write_kernel_handle(kernel, entryFuncAddr)
}

#[no_mangle]
pub extern "C" fn cudaGetKernel(kernel: *mut cudaKernel_t, entryFuncAddr: MemPtr) -> cudaError_t {
    log::debug!(target: "cudaGetKernel", "");
    write_kernel_handle(kernel, entryFuncAddr)
}

#[no_mangle]
pub extern "C" fn cudaLaunchKernel(
    func: MemPtr,
    gridDim: dim3,
    blockDim: dim3,
    args: *mut *mut ::std::os::raw::c_void,
    sharedMem: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    log::debug!(target: "cudaLaunchKernel", "");

    let cufunc = get_cufunction(func);

    unsafe {
        std::mem::transmute(super::cuda_hijack::cuLaunchKernel(
            cufunc,
            gridDim.x,
            gridDim.y,
            gridDim.z,
            blockDim.x,
            blockDim.y,
            blockDim.z,
            sharedMem.try_into().unwrap(),
            stream.cast(),
            args,
            std::ptr::null_mut(),
        ))
    }
}

#[no_mangle]
pub extern "C" fn __cudaLaunchKernel(
    kernel: cudaKernel_t,
    gridDim: dim3,
    blockDim: dim3,
    args: *mut *mut ::std::os::raw::c_void,
    sharedMem: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    log::debug!(target: "__cudaLaunchKernel", "");
    cudaLaunchKernel(kernel as MemPtr, gridDim, blockDim, args, sharedMem, stream)
}

#[no_mangle]
pub extern "C" fn __cudaLaunchKernel_ptsz(
    kernel: cudaKernel_t,
    gridDim: dim3,
    blockDim: dim3,
    args: *mut *mut ::std::os::raw::c_void,
    sharedMem: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    log::debug!(target: "__cudaLaunchKernel_ptsz", "");
    cudaLaunchKernel(kernel as MemPtr, gridDim, blockDim, args, sharedMem, stream)
}

#[no_mangle]
pub extern "C" fn cudaLaunchKernelExC(
    config: *const cudaLaunchConfig_t,
    func: MemPtr,
    args: *mut *mut ::std::os::raw::c_void,
) -> cudaError_t {
    log::debug!(target: "cudaLaunchKernelExC", "");
    if config.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }

    let config = unsafe { &*config };
    let shared_mem = match config.dynamicSmemBytes.try_into() {
        Ok(shared_mem) => shared_mem,
        Err(_) => return cudaError_t::cudaErrorInvalidValue,
    };
    let launch_config = CUlaunchConfig {
        gridDimX: config.gridDim.x,
        gridDimY: config.gridDim.y,
        gridDimZ: config.gridDim.z,
        blockDimX: config.blockDim.x,
        blockDimY: config.blockDim.y,
        blockDimZ: config.blockDim.z,
        sharedMemBytes: shared_mem,
        hStream: config.stream.cast(),
        attrs: config.attrs.cast::<CUlaunchAttribute>(),
        numAttrs: config.numAttrs,
    };
    let cufunc = get_cufunction(func);

    unsafe {
        std::mem::transmute(super::cuda_hijack::cuLaunchKernelEx(
            &launch_config,
            cufunc,
            args,
            std::ptr::null_mut(),
        ))
    }
}

#[no_mangle]
pub extern "C" fn cudaHostAlloc(
    pHost: *mut *mut ::std::os::raw::c_void,
    size: usize,
    flags: c_uint,
) -> cudaError_t {
    log::debug!(target: "cudaHostAlloc", "size = {size}, flags = {flags}");
    allocate_host(pHost, size)
}

#[no_mangle]
pub extern "C" fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> cudaError_t {
    log::debug!(target: "cudaMallocHost", "size = {size}");
    allocate_host(ptr, size)
}

#[no_mangle]
pub extern "C" fn cudaFreeHost(ptr: *mut c_void) -> cudaError_t {
    log::debug!(target: "cudaFreeHost", "");
    if ptr.is_null() {
        return cudaError_t::cudaSuccess;
    }

    if !host_allocations().lock().unwrap().remove(&(ptr as usize)) {
        return cudaError_t::cudaErrorInvalidValue;
    }

    unsafe {
        libc::free(ptr);
    }
    cudaError_t::cudaSuccess
}

#[no_mangle]
pub extern "C" fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: c_uint) -> cudaError_t {
    log::debug!(target: "cudaHostRegister", "size = {size}, flags = {flags}");
    if ptr.is_null() {
        cudaError_t::cudaErrorInvalidValue
    } else {
        cudaError_t::cudaSuccess
    }
}

#[no_mangle]
pub extern "C" fn cudaHostUnregister(ptr: *mut c_void) -> cudaError_t {
    log::debug!(target: "cudaHostUnregister", "");
    if ptr.is_null() {
        cudaError_t::cudaErrorInvalidValue
    } else {
        cudaError_t::cudaSuccess
    }
}

#[no_mangle]
pub extern "C" fn cudaHostGetDevicePointer(
    pDevice: *mut *mut c_void,
    pHost: *mut c_void,
    flags: c_uint,
) -> cudaError_t {
    log::debug!(target: "cudaHostGetDevicePointer", "flags = {flags}");
    if pDevice.is_null() || pHost.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }
    unsafe {
        *pDevice = std::ptr::null_mut();
    }
    cudaError_t::cudaErrorNotSupported
}

#[no_mangle]
pub extern "C" fn cudaLaunchHostFunc(
    stream: cudaStream_t,
    fn_: cudaHostFn_t,
    userData: *mut c_void,
) -> cudaError_t {
    log::debug!(target: "cudaLaunchHostFunc", "");
    let Some(callback) = fn_ else {
        return cudaError_t::cudaErrorInvalidValue;
    };
    let result = super::cudart_hijack::cudaStreamSynchronize(stream);
    if result.is_error() {
        return result;
    }
    unsafe {
        callback(userData);
    }
    cudaError_t::cudaSuccess
}

#[no_mangle]
pub extern "C" fn cudaLaunchHostFunc_v2(
    stream: cudaStream_t,
    fn_: cudaHostFn_t,
    userData: *mut c_void,
    syncMode: c_uint,
) -> cudaError_t {
    log::debug!(target: "cudaLaunchHostFunc_v2", "syncMode = {syncMode}");
    cudaLaunchHostFunc(stream, fn_, userData)
}

#[no_mangle]
pub extern "C" fn cudaStreamAddCallback(
    stream: cudaStream_t,
    callback: cudaStreamCallback_t,
    userData: *mut c_void,
    flags: c_uint,
) -> cudaError_t {
    log::debug!(target: "cudaStreamAddCallback", "flags = {flags}");
    if flags != 0 {
        return cudaError_t::cudaErrorInvalidValue;
    }
    let Some(callback) = callback else {
        return cudaError_t::cudaErrorInvalidValue;
    };
    let status = super::cudart_hijack::cudaStreamSynchronize(stream);
    unsafe {
        callback(stream, status, userData);
    }
    if status.is_error() {
        status
    } else {
        cudaError_t::cudaSuccess
    }
}

#[no_mangle]
pub extern "C" fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t {
    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cudaEventDestroy", "[#{}]", client.id);

        202.send(&client.channel_sender).unwrap();
        event.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        recv_cuda_result("cudaEventDestroy", client.id, &client.channel_receiver)
    })
}

#[no_mangle]
pub extern "C" fn cudaIpcGetEventHandle(
    handle: *mut cudaIpcEventHandle_t,
    event: cudaEvent_t,
) -> cudaError_t {
    if handle.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cudaIpcGetEventHandle", "[#{}]", client.id);

        900914.send(&client.channel_sender).unwrap();
        event.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        unsafe { &mut *handle }
            .recv(&client.channel_receiver)
            .unwrap();
        recv_cuda_result("cudaIpcGetEventHandle", client.id, &client.channel_receiver)
    })
}

#[no_mangle]
pub extern "C" fn cudaIpcOpenEventHandle(
    event: *mut cudaEvent_t,
    handle: cudaIpcEventHandle_t,
) -> cudaError_t {
    if event.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cudaIpcOpenEventHandle", "[#{}]", client.id);

        900915.send(&client.channel_sender).unwrap();
        handle.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        unsafe { &mut *event }
            .recv(&client.channel_receiver)
            .unwrap();
        recv_cuda_result(
            "cudaIpcOpenEventHandle",
            client.id,
            &client.channel_receiver,
        )
    })
}

#[no_mangle]
pub extern "C" fn cudaGraphGetNodes(
    graph: cudaGraph_t,
    nodes: *mut cudaGraphNode_t,
    numNodes: *mut usize,
) -> cudaError_t {
    cuda_graph_get_node_list(563, "cudaGraphGetNodes", graph, nodes, numNodes)
}

#[no_mangle]
pub extern "C" fn cudaGraphGetRootNodes(
    graph: cudaGraph_t,
    pRootNodes: *mut cudaGraphNode_t,
    pNumRootNodes: *mut usize,
) -> cudaError_t {
    cuda_graph_get_node_list(
        900509,
        "cudaGraphGetRootNodes",
        graph,
        pRootNodes,
        pNumRootNodes,
    )
}

#[no_mangle]
pub extern "C" fn cudaGraphGetEdges(
    graph: cudaGraph_t,
    from: *mut cudaGraphNode_t,
    to: *mut cudaGraphNode_t,
    edgeData: *mut cudaGraphEdgeData,
    numEdges: *mut usize,
) -> cudaError_t {
    cuda_graph_get_edge_list(
        900510,
        "cudaGraphGetEdges",
        graph,
        from,
        to,
        edgeData,
        numEdges,
    )
}

#[no_mangle]
pub extern "C" fn cudaGraphAddKernelNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    pNodeParams: *const cudaKernelNodeParams,
) -> cudaError_t {
    if pGraphNode.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }
    if !pDependencies.is_null() || numDependencies != 0 {
        return cudaError_t::cudaErrorNotSupported;
    }
    let (cached_params, packed_params, args, arg_offsets) =
        match pack_runtime_kernel_node_params(pNodeParams) {
            Ok(params) => params,
            Err(error) => return error,
        };

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cudaGraphAddKernelNode", "[#{}]", client.id);

        900910.send(&client.channel_sender).unwrap();
        graph.send(&client.channel_sender).unwrap();
        packed_params.send(&client.channel_sender).unwrap();
        send_slice(&arg_offsets, &client.channel_sender).unwrap();
        send_slice(&args, &client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        let mut node: cudaGraphNode_t = std::ptr::null_mut();
        node.recv(&client.channel_receiver).unwrap();
        let result = recv_cuda_result(
            "cudaGraphAddKernelNode",
            client.id,
            &client.channel_receiver,
        );
        if result == cudaError_t::cudaSuccess {
            unsafe {
                *pGraphNode = node;
            }
            cache_runtime_graph_kernel_node(node, cached_params, packed_params, args, arg_offsets);
        }
        result
    })
}

#[no_mangle]
pub extern "C" fn cudaGraphKernelNodeGetParams(
    node: cudaGraphNode_t,
    pNodeParams: *mut cudaKernelNodeParams,
) -> cudaError_t {
    if pNodeParams.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cudaGraphKernelNodeGetParams", "[#{}]", client.id);

        900911.send(&client.channel_sender).unwrap();
        node.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        let mut server_params = std::mem::MaybeUninit::<cudaKernelNodeParams>::uninit();
        server_params.recv(&client.channel_receiver).unwrap();
        let result = recv_cuda_result(
            "cudaGraphKernelNodeGetParams",
            client.id,
            &client.channel_receiver,
        );
        if result == cudaError_t::cudaSuccess {
            unsafe {
                *pNodeParams = server_params.assume_init();
            }
            if let Some(cache) = RUNTIME_CACHE
                .write()
                .unwrap()
                .graph_kernel_nodes
                .get_mut(&node)
            {
                unsafe {
                    *pNodeParams = cache.params_for_client();
                }
            }
        }
        result
    })
}

#[no_mangle]
pub extern "C" fn cudaGraphKernelNodeSetParams(
    node: cudaGraphNode_t,
    pNodeParams: *const cudaKernelNodeParams,
) -> cudaError_t {
    let (cached_params, packed_params, args, arg_offsets) =
        match pack_runtime_kernel_node_params(pNodeParams) {
            Ok(params) => params,
            Err(error) => return error,
        };

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cudaGraphKernelNodeSetParams", "[#{}]", client.id);

        900912.send(&client.channel_sender).unwrap();
        node.send(&client.channel_sender).unwrap();
        packed_params.send(&client.channel_sender).unwrap();
        send_slice(&arg_offsets, &client.channel_sender).unwrap();
        send_slice(&args, &client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        let result = recv_cuda_result(
            "cudaGraphKernelNodeSetParams",
            client.id,
            &client.channel_receiver,
        );
        if result == cudaError_t::cudaSuccess {
            cache_runtime_graph_kernel_node(node, cached_params, packed_params, args, arg_offsets);
        }
        result
    })
}

#[no_mangle]
pub extern "C" fn cudaGraphExecKernelNodeSetParams(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    pNodeParams: *const cudaKernelNodeParams,
) -> cudaError_t {
    let (_cached_params, packed_params, args, arg_offsets) =
        match pack_runtime_kernel_node_params(pNodeParams) {
            Ok(params) => params,
            Err(error) => return error,
        };

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cudaGraphExecKernelNodeSetParams", "[#{}]", client.id);

        900913.send(&client.channel_sender).unwrap();
        hGraphExec.send(&client.channel_sender).unwrap();
        node.send(&client.channel_sender).unwrap();
        packed_params.send(&client.channel_sender).unwrap();
        send_slice(&arg_offsets, &client.channel_sender).unwrap();
        send_slice(&args, &client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        recv_cuda_result(
            "cudaGraphExecKernelNodeSetParams",
            client.id,
            &client.channel_receiver,
        )
    })
}

#[no_mangle]
pub extern "C" fn cudaGraphAddMemcpyNodeToSymbol(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    symbol: *const c_void,
    src: *const c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = symbol_copy_to_kind(src, kind);
    if kind != cudaMemcpyKind::cudaMemcpyDeviceToDevice {
        return cudaError_t::cudaErrorNotSupported;
    }
    let dst = match symbol_device_address(symbol, offset) {
        Ok(ptr) => ptr,
        Err(error) => return error,
    };
    super::cudart_hijack::cudaGraphAddMemcpyNode1D(
        pGraphNode,
        graph,
        pDependencies,
        numDependencies,
        dst,
        src,
        count,
        kind,
    )
}

#[no_mangle]
pub extern "C" fn cudaGraphAddMemcpyNodeFromSymbol(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    dst: *mut c_void,
    symbol: *const c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = symbol_copy_from_kind(dst.cast_const(), kind);
    if kind != cudaMemcpyKind::cudaMemcpyDeviceToDevice {
        return cudaError_t::cudaErrorNotSupported;
    }
    let src = match symbol_device_address(symbol, offset) {
        Ok(ptr) => ptr.cast_const(),
        Err(error) => return error,
    };
    super::cudart_hijack::cudaGraphAddMemcpyNode1D(
        pGraphNode,
        graph,
        pDependencies,
        numDependencies,
        dst,
        src,
        count,
        kind,
    )
}

#[no_mangle]
pub extern "C" fn cudaGraphMemcpyNodeSetParamsToSymbol(
    node: cudaGraphNode_t,
    symbol: *const c_void,
    src: *const c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = symbol_copy_to_kind(src, kind);
    if kind != cudaMemcpyKind::cudaMemcpyDeviceToDevice {
        return cudaError_t::cudaErrorNotSupported;
    }
    let dst = match symbol_device_address(symbol, offset) {
        Ok(ptr) => ptr,
        Err(error) => return error,
    };
    super::cudart_hijack::cudaGraphMemcpyNodeSetParams1D(node, dst, src, count, kind)
}

#[no_mangle]
pub extern "C" fn cudaGraphMemcpyNodeSetParamsFromSymbol(
    node: cudaGraphNode_t,
    dst: *mut c_void,
    symbol: *const c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = symbol_copy_from_kind(dst.cast_const(), kind);
    if kind != cudaMemcpyKind::cudaMemcpyDeviceToDevice {
        return cudaError_t::cudaErrorNotSupported;
    }
    let src = match symbol_device_address(symbol, offset) {
        Ok(ptr) => ptr.cast_const(),
        Err(error) => return error,
    };
    super::cudart_hijack::cudaGraphMemcpyNodeSetParams1D(node, dst, src, count, kind)
}

#[no_mangle]
pub extern "C" fn cudaGraphExecMemcpyNodeSetParamsToSymbol(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    symbol: *const c_void,
    src: *const c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = symbol_copy_to_kind(src, kind);
    if kind != cudaMemcpyKind::cudaMemcpyDeviceToDevice {
        return cudaError_t::cudaErrorNotSupported;
    }
    let dst = match symbol_device_address(symbol, offset) {
        Ok(ptr) => ptr,
        Err(error) => return error,
    };
    super::cudart_hijack::cudaGraphExecMemcpyNodeSetParams1D(
        hGraphExec, node, dst, src, count, kind,
    )
}

#[no_mangle]
pub extern "C" fn cudaGraphExecMemcpyNodeSetParamsFromSymbol(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    dst: *mut c_void,
    symbol: *const c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = symbol_copy_from_kind(dst.cast_const(), kind);
    if kind != cudaMemcpyKind::cudaMemcpyDeviceToDevice {
        return cudaError_t::cudaErrorNotSupported;
    }
    let src = match symbol_device_address(symbol, offset) {
        Ok(ptr) => ptr.cast_const(),
        Err(error) => return error,
    };
    super::cudart_hijack::cudaGraphExecMemcpyNodeSetParams1D(
        hGraphExec, node, dst, src, count, kind,
    )
}

#[no_mangle]
pub extern "C" fn cudaGraphNodeGetDependencies(
    node: cudaGraphNode_t,
    pDependencies: *mut cudaGraphNode_t,
    edgeData: *mut cudaGraphEdgeData,
    pNumDependencies: *mut usize,
) -> cudaError_t {
    cuda_graph_get_edge_list(
        900523,
        "cudaGraphNodeGetDependencies",
        node,
        pDependencies,
        std::ptr::null_mut(),
        edgeData,
        pNumDependencies,
    )
}

#[no_mangle]
pub extern "C" fn cudaGraphNodeGetDependentNodes(
    node: cudaGraphNode_t,
    pDependentNodes: *mut cudaGraphNode_t,
    edgeData: *mut cudaGraphEdgeData,
    pNumDependentNodes: *mut usize,
) -> cudaError_t {
    cuda_graph_get_edge_list(
        900524,
        "cudaGraphNodeGetDependentNodes",
        node,
        pDependentNodes,
        std::ptr::null_mut(),
        edgeData,
        pNumDependentNodes,
    )
}

#[no_mangle]
pub extern "C" fn cudaGetErrorString(cudaError: cudaError_t) -> *const ::std::os::raw::c_char {
    log::debug!(target: "cudaGetErrorString", "{cudaError:?}");
    cuda_error_text(cudaError, true)
}

#[no_mangle]
pub extern "C" fn cudaGetErrorName(cudaError: cudaError_t) -> *const ::std::os::raw::c_char {
    log::debug!(target: "cudaGetErrorName", "{cudaError:?}");
    cuda_error_text(cudaError, false)
}

#[no_mangle]
pub extern "C" fn cudaCreateChannelDesc(
    x: c_int,
    y: c_int,
    z: c_int,
    w: c_int,
    f: cudaChannelFormatKind,
) -> cudaChannelFormatDesc {
    cudaChannelFormatDesc { x, y, z, w, f }
}

struct CallConfiguration {
    gridDim: dim3,
    blockDim: dim3,
    sharedMem: usize,
    stream: MemPtr,
}

thread_local! {
    static CALL_CONFIGURATIONS: RefCell<Vec<CallConfiguration>> = const {
        RefCell::new(Vec::new())
    };
}

#[no_mangle]
pub extern "C" fn __cudaPushCallConfiguration(
    gridDim: dim3,
    blockDim: dim3,
    sharedMem: usize,
    stream: MemPtr,
) -> cudaError_t {
    CALL_CONFIGURATIONS.with_borrow_mut(|v| {
        v.push(CallConfiguration {
            gridDim,
            blockDim,
            sharedMem,
            stream,
        });
    });
    cudaError_t::cudaSuccess
}

#[no_mangle]
pub extern "C" fn __cudaPopCallConfiguration(
    gridDim: *mut dim3,
    blockDim: *mut dim3,
    sharedMem: *mut usize,
    stream: *mut MemPtr,
) -> cudaError_t {
    if let Some(config) = CALL_CONFIGURATIONS.with_borrow_mut(Vec::pop) {
        unsafe {
            *gridDim = config.gridDim;
            *blockDim = config.blockDim;
            *sharedMem = config.sharedMem;
            *stream = config.stream;
        }
        cudaError_t::cudaSuccess
    } else {
        cudaError_t::cudaErrorMissingConfiguration
    }
}

#[no_mangle]
extern "C" fn cudaFuncGetAttributes(
    attr: *mut cudaFuncAttributes,
    func: *const c_void,
) -> cudaError_t {
    log::debug!(target: "cudaFuncGetAttributes", "");
    let func = get_cufunction(func as HostPtr);
    super::cudart_hijack::cudaFuncGetAttributesInternal(attr, func)
}

#[no_mangle]
extern "C" fn cudaFuncGetName(name: *mut *const c_char, func: *const c_void) -> cudaError_t {
    log::debug!(target: "cudaFuncGetName", "");
    if name.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }
    #[expect(clippy::missing_transmute_annotations)]
    unsafe {
        std::mem::transmute(super::cuda_hijack_custom::cuFuncGetName(
            name,
            get_cufunction(func as HostPtr),
        ))
    }
}

#[no_mangle]
extern "C" fn cudaFuncGetParamInfo(
    func: *const c_void,
    paramIndex: usize,
    paramOffset: *mut usize,
    paramSize: *mut usize,
) -> cudaError_t {
    log::debug!(target: "cudaFuncGetParamInfo", "");
    #[expect(clippy::missing_transmute_annotations)]
    unsafe {
        std::mem::transmute(super::cuda_hijack::cuFuncGetParamInfo(
            get_cufunction(func as HostPtr),
            paramIndex,
            paramOffset,
            paramSize,
        ))
    }
}

#[no_mangle]
extern "C" fn cudaFuncGetParamCount(func: *const c_void, paramCount: *mut usize) -> cudaError_t {
    log::debug!(target: "cudaFuncGetParamCount", "");
    #[expect(clippy::missing_transmute_annotations)]
    unsafe {
        std::mem::transmute(super::cuda_hijack::cuFuncGetParamCount(
            get_cufunction(func as HostPtr),
            paramCount,
        ))
    }
}

#[no_mangle]
extern "C" fn cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    numBlocks: *mut c_int,
    func: *const c_void,
    blockSize: c_int,
    dynamicSMemSize: usize,
    flags: c_uint,
) -> cudaError_t {
    log::debug!(target: "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", "");
    let result = super::cuda_hijack::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks,
        get_cufunction(func as HostPtr),
        blockSize,
        dynamicSMemSize,
        flags,
    );
    unsafe { std::mem::transmute(result) }
}

#[no_mangle]
extern "C" fn cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    numBlocks: *mut c_int,
    func: *const c_void,
    blockSize: c_int,
    dynamicSMemSize: usize,
) -> cudaError_t {
    log::debug!(target: "cudaOccupancyMaxActiveBlocksPerMultiprocessor", "");
    let result = super::cuda_hijack::cuOccupancyMaxActiveBlocksPerMultiprocessor(
        numBlocks,
        get_cufunction(func as HostPtr),
        blockSize,
        dynamicSMemSize,
    );
    unsafe { std::mem::transmute(result) }
}

#[no_mangle]
extern "C" fn cudaOccupancyAvailableDynamicSMemPerBlock(
    dynamicSmemSize: *mut usize,
    func: *const c_void,
    numBlocks: c_int,
    blockSize: c_int,
) -> cudaError_t {
    log::debug!(target: "cudaOccupancyAvailableDynamicSMemPerBlock", "");
    let result = super::cuda_hijack::cuOccupancyAvailableDynamicSMemPerBlock(
        dynamicSmemSize,
        get_cufunction(func as HostPtr),
        numBlocks,
        blockSize,
    );
    unsafe { std::mem::transmute(result) }
}

#[no_mangle]
extern "C" fn cudaFuncSetAttribute(
    func: *const c_void,
    attr: cudaFuncAttribute,
    value: c_int,
) -> cudaError_t {
    log::debug!(target: "cudaFuncSetAttribute", "");
    #[expect(clippy::missing_transmute_annotations)]
    unsafe {
        std::mem::transmute(super::cuda_hijack::cuFuncSetAttribute(
            get_cufunction(func as _),
            std::mem::transmute(attr),
            value,
        ))
    }
}

#[no_mangle]
extern "C" fn cudaFuncSetCacheConfig(
    func: *const c_void,
    cacheConfig: cudaFuncCache,
) -> cudaError_t {
    log::debug!(target: "cudaFuncSetCacheConfig", "");
    #[expect(clippy::missing_transmute_annotations)]
    unsafe {
        std::mem::transmute(super::cuda_hijack::cuFuncSetCacheConfig(
            get_cufunction(func as _),
            std::mem::transmute(cacheConfig),
        ))
    }
}

#[no_mangle]
extern "C" fn cudaFuncSetSharedMemConfig(
    func: *const c_void,
    config: cudaSharedMemConfig,
) -> cudaError_t {
    log::debug!(target: "cudaFuncSetSharedMemConfig", "");
    #[expect(clippy::missing_transmute_annotations)]
    unsafe {
        std::mem::transmute(super::cuda_hijack::cuFuncSetSharedMemConfig(
            get_cufunction(func as _),
            std::mem::transmute(config),
        ))
    }
}
