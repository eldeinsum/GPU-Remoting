#![expect(non_snake_case)]
use super::*;
use super::cuda_hijack_utils::send_fd;
use cudasys::types::cuda::{
    CUDA_KERNEL_NODE_PARAMS, CUdeviceptr, CUgraphNode, CUkernel, CUlaunchAttribute, CUlaunchConfig,
    CUmodule, CUresult,
};
use cudasys::types::cudart::*;
use network::type_impl::recv_slice;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::*;
use std::sync::{Mutex, OnceLock};

fn host_memory_error(error: super::host_memory::HostMemoryError) -> cudaError_t {
    match error {
        super::host_memory::HostMemoryError::InvalidValue => cudaError_t::cudaErrorInvalidValue,
        super::host_memory::HostMemoryError::MemoryAllocation => {
            cudaError_t::cudaErrorMemoryAllocation
        }
    }
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

#[no_mangle]
pub extern "C" fn cudaMemRangeGetAttributes(
    data: *mut *mut c_void,
    dataSizes: *mut usize,
    attributes: *mut cudaMemRangeAttribute,
    numAttributes: usize,
    devPtr: *const c_void,
    count: usize,
) -> cudaError_t {
    if data.is_null()
        || dataSizes.is_null()
        || attributes.is_null()
        || numAttributes == 0
        || devPtr.is_null()
        || count == 0
    {
        return cudaError_t::cudaErrorInvalidValue;
    }

    let data_sizes = unsafe { std::slice::from_raw_parts(dataSizes, numAttributes) };
    let attrs = unsafe { std::slice::from_raw_parts(attributes, numAttributes) };
    for (idx, size) in data_sizes.iter().enumerate() {
        if *size == 0 || unsafe { *data.add(idx) }.is_null() {
            return cudaError_t::cudaErrorInvalidValue;
        }
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cudaMemRangeGetAttributes", "[#{}]", client.id);

        900994.send(&client.channel_sender).unwrap();
        send_slice(data_sizes, &client.channel_sender).unwrap();
        send_slice(attrs, &client.channel_sender).unwrap();
        devPtr.send(&client.channel_sender).unwrap();
        count.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        let mut local_error = None;
        for (idx, capacity) in data_sizes.iter().copied().enumerate() {
            let bytes = recv_slice::<u8, _>(&client.channel_receiver).unwrap();
            if bytes.len() > capacity {
                log::error!(
                    target: "cudaMemRangeGetAttributes",
                    "server returned {} bytes for client capacity {}",
                    bytes.len(),
                    capacity,
                );
                local_error = Some(cudaError_t::cudaErrorInvalidValue);
                continue;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr().cast(), *data.add(idx), bytes.len());
            }
        }
        let result = recv_cuda_result(
            "cudaMemRangeGetAttributes",
            client.id,
            &client.channel_receiver,
        );
        local_error.unwrap_or(result)
    })
}

#[no_mangle]
pub extern "C" fn cudaMemPoolExportToShareableHandle(
    shareableHandle: *mut c_void,
    memPool: cudaMemPool_t,
    handleType: cudaMemAllocationHandleType,
    flags: c_uint,
) -> cudaError_t {
    if shareableHandle.is_null() || memPool.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }
    if handleType != cudaMemAllocationHandleType::cudaMemHandleTypePosixFileDescriptor {
        return cudaError_t::cudaErrorNotSupported;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cudaMemPoolExportToShareableHandle", "[#{}]", client.id);

        901112.send(&client.channel_sender).unwrap();
        memPool.send(&client.channel_sender).unwrap();
        handleType.send(&client.channel_sender).unwrap();
        flags.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        let mut synthetic_fd = -1 as c_int;
        synthetic_fd.recv(&client.channel_receiver).unwrap();
        let result = recv_cuda_result(
            "cudaMemPoolExportToShareableHandle",
            client.id,
            &client.channel_receiver,
        );
        if result == cudaError_t::cudaSuccess {
            unsafe {
                *shareableHandle.cast::<c_int>() = synthetic_fd;
            }
        }
        result
    })
}

#[no_mangle]
pub extern "C" fn cudaMemPoolImportFromShareableHandle(
    memPool: *mut cudaMemPool_t,
    shareableHandle: *mut c_void,
    handleType: cudaMemAllocationHandleType,
    flags: c_uint,
) -> cudaError_t {
    if memPool.is_null() || shareableHandle.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }
    if handleType != cudaMemAllocationHandleType::cudaMemHandleTypePosixFileDescriptor {
        return cudaError_t::cudaErrorNotSupported;
    }
    let synthetic_fd = shareableHandle as isize as c_int;

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cudaMemPoolImportFromShareableHandle", "[#{}]", client.id);

        901113.send(&client.channel_sender).unwrap();
        synthetic_fd.send(&client.channel_sender).unwrap();
        handleType.send(&client.channel_sender).unwrap();
        flags.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        unsafe { &mut *memPool }
            .recv(&client.channel_receiver)
            .unwrap();
        recv_cuda_result(
            "cudaMemPoolImportFromShareableHandle",
            client.id,
            &client.channel_receiver,
        )
    })
}

#[no_mangle]
pub extern "C" fn cudaImportExternalMemory(
    extMem_out: *mut cudaExternalMemory_t,
    memHandleDesc: *const cudaExternalMemoryHandleDesc,
) -> cudaError_t {
    if extMem_out.is_null() || memHandleDesc.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }

    let desc = unsafe { *memHandleDesc };
    if desc.type_ != cudaExternalMemoryHandleType::cudaExternalMemoryHandleTypeOpaqueFd {
        return cudaError_t::cudaErrorNotSupported;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cudaImportExternalMemory", "[#{}]", client.id);

        901121.send(&client.channel_sender).unwrap();
        desc.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        unsafe { &mut *extMem_out }
            .recv(&client.channel_receiver)
            .unwrap();
        recv_cuda_result(
            "cudaImportExternalMemory",
            client.id,
            &client.channel_receiver,
        )
    })
}

fn supported_runtime_external_semaphore_type(type_: cudaExternalSemaphoreHandleType) -> bool {
    matches!(
        type_,
        cudaExternalSemaphoreHandleType::cudaExternalSemaphoreHandleTypeOpaqueFd
            | cudaExternalSemaphoreHandleType::cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd
    )
}

#[no_mangle]
pub extern "C" fn cudaImportExternalSemaphore(
    extSem_out: *mut cudaExternalSemaphore_t,
    semHandleDesc: *const cudaExternalSemaphoreHandleDesc,
) -> cudaError_t {
    if extSem_out.is_null() || semHandleDesc.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }

    let desc = unsafe { *semHandleDesc };
    if !supported_runtime_external_semaphore_type(desc.type_) {
        return cudaError_t::cudaErrorNotSupported;
    }
    let client_fd = unsafe { desc.handle.fd };
    if client_fd < 0 {
        return cudaError_t::cudaErrorInvalidValue;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cudaImportExternalSemaphore", "[#{}]", client.id);

        901129.send(&client.channel_sender).unwrap();
        desc.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        let socket_path = recv_slice::<u8, _>(&client.channel_receiver).unwrap();
        if !socket_path.is_empty() && send_fd(&socket_path, client_fd).is_err() {
            return cudaError_t::cudaErrorInvalidValue;
        }

        unsafe { &mut *extSem_out }
            .recv(&client.channel_receiver)
            .unwrap();
        let result = recv_cuda_result(
            "cudaImportExternalSemaphore",
            client.id,
            &client.channel_receiver,
        );
        if result == cudaError_t::cudaSuccess {
            unsafe {
                libc::close(client_fd);
            }
        }
        result
    })
}

#[no_mangle]
pub extern "C" fn cudaMemPoolExportPointer(
    exportData: *mut cudaMemPoolPtrExportData,
    ptr: *mut c_void,
) -> cudaError_t {
    if exportData.is_null() || ptr.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cudaMemPoolExportPointer", "[#{}]", client.id);

        901114.send(&client.channel_sender).unwrap();
        ptr.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        unsafe { &mut *exportData }
            .recv(&client.channel_receiver)
            .unwrap();
        recv_cuda_result(
            "cudaMemPoolExportPointer",
            client.id,
            &client.channel_receiver,
        )
    })
}

#[no_mangle]
pub extern "C" fn cudaMemPoolImportPointer(
    ptr: *mut *mut c_void,
    memPool: cudaMemPool_t,
    exportData: *mut cudaMemPoolPtrExportData,
) -> cudaError_t {
    if ptr.is_null() || memPool.is_null() || exportData.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cudaMemPoolImportPointer", "[#{}]", client.id);

        901115.send(&client.channel_sender).unwrap();
        memPool.send(&client.channel_sender).unwrap();
        unsafe { &*exportData }.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        unsafe { &mut *ptr }.recv(&client.channel_receiver).unwrap();
        recv_cuda_result(
            "cudaMemPoolImportPointer",
            client.id,
            &client.channel_receiver,
        )
    })
}

#[no_mangle]
pub extern "C" fn cudaCreateTextureObject(
    pTexObject: *mut cudaTextureObject_t,
    pResDesc: *const cudaResourceDesc,
    pTexDesc: *const cudaTextureDesc,
    pResViewDesc: *const cudaResourceViewDesc,
) -> cudaError_t {
    if pTexObject.is_null() || pResDesc.is_null() || pTexDesc.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cudaCreateTextureObject", "[#{}]", client.id);

        900967.send(&client.channel_sender).unwrap();
        unsafe { &*pResDesc }.send(&client.channel_sender).unwrap();
        unsafe { &*pTexDesc }.send(&client.channel_sender).unwrap();
        let has_view_desc = !pResViewDesc.is_null();
        has_view_desc.send(&client.channel_sender).unwrap();
        if has_view_desc {
            unsafe { &*pResViewDesc }
                .send(&client.channel_sender)
                .unwrap();
        }
        client.channel_sender.flush_out().unwrap();

        unsafe { &mut *pTexObject }
            .recv(&client.channel_receiver)
            .unwrap();
        recv_cuda_result(
            "cudaCreateTextureObject",
            client.id,
            &client.channel_receiver,
        )
    })
}

#[no_mangle]
pub extern "C" fn cudaGetTextureObjectResourceDesc(
    pResDesc: *mut cudaResourceDesc,
    texObject: cudaTextureObject_t,
) -> cudaError_t {
    if pResDesc.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(
            target: "cudaGetTextureObjectResourceDesc",
            "[#{}]",
            client.id
        );

        900969.send(&client.channel_sender).unwrap();
        texObject.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        unsafe { &mut *pResDesc }
            .recv(&client.channel_receiver)
            .unwrap();
        recv_cuda_result(
            "cudaGetTextureObjectResourceDesc",
            client.id,
            &client.channel_receiver,
        )
    })
}

#[no_mangle]
pub extern "C" fn cudaCreateSurfaceObject(
    pSurfObject: *mut cudaSurfaceObject_t,
    pResDesc: *const cudaResourceDesc,
) -> cudaError_t {
    if pSurfObject.is_null() || pResDesc.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cudaCreateSurfaceObject", "[#{}]", client.id);

        900972.send(&client.channel_sender).unwrap();
        unsafe { &*pResDesc }.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        unsafe { &mut *pSurfObject }
            .recv(&client.channel_receiver)
            .unwrap();
        recv_cuda_result(
            "cudaCreateSurfaceObject",
            client.id,
            &client.channel_receiver,
        )
    })
}

#[no_mangle]
pub extern "C" fn cudaGetSurfaceObjectResourceDesc(
    pResDesc: *mut cudaResourceDesc,
    surfObject: cudaSurfaceObject_t,
) -> cudaError_t {
    if pResDesc.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(
            target: "cudaGetSurfaceObjectResourceDesc",
            "[#{}]",
            client.id
        );

        900974.send(&client.channel_sender).unwrap();
        surfObject.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        unsafe { &mut *pResDesc }
            .recv(&client.channel_receiver)
            .unwrap();
        recv_cuda_result(
            "cudaGetSurfaceObjectResourceDesc",
            client.id,
            &client.channel_receiver,
        )
    })
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

fn cached_driver_function(func: HostPtr) -> Option<cudasys::types::cuda::CUfunction> {
    let cufunc = func as cudasys::types::cuda::CUfunction;
    DRIVER_CACHE
        .read()
        .unwrap()
        .function_params
        .contains_key(&cufunc)
        .then_some(cufunc)
}

fn driver_library(library: cudaLibrary_t) -> cudasys::types::cuda::CUlibrary {
    library.cast::<cudasys::types::cuda::CUlib_st>()
}

fn runtime_result(result: CUresult) -> cudaError_t {
    unsafe { std::mem::transmute(result) }
}

fn ensure_runtime_context_for_driver_sync() -> cudaError_t {
    let mut runtime = RUNTIME_CACHE.write().unwrap();
    ensure_runtime_device(&mut runtime)
}

fn synchronize_managed_variables_with_driver<F>(driver_call: F) -> cudaError_t
where
    F: FnOnce() -> CUresult,
{
    let result = ensure_runtime_context_for_driver_sync();
    if result != cudaError_t::cudaSuccess {
        return result;
    }

    let result = flush_managed_variables_to_device();
    if result != cudaError_t::cudaSuccess {
        return result;
    }

    let result = runtime_result(driver_call());
    if result != cudaError_t::cudaSuccess {
        return result;
    }

    refresh_managed_variables_from_device()
}

#[no_mangle]
pub extern "C" fn cudaDeviceSynchronize() -> cudaError_t {
    log::debug!(target: "cudaDeviceSynchronize", "");
    synchronize_managed_variables_with_driver(|| super::cuda_hijack::cuCtxSynchronize())
}

#[no_mangle]
pub extern "C" fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t {
    log::debug!(target: "cudaStreamSynchronize", "");
    synchronize_managed_variables_with_driver(|| {
        super::cuda_hijack::cuStreamSynchronize(stream.cast())
    })
}

#[no_mangle]
pub extern "C" fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t {
    log::debug!(target: "cudaEventSynchronize", "");
    synchronize_managed_variables_with_driver(|| {
        super::cuda_hijack::cuEventSynchronize(event.cast())
    })
}

#[no_mangle]
pub extern "C" fn cudaGraphLaunch(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t {
    log::debug!(target: "cudaGraphLaunch", "");
    let result = ensure_runtime_context_for_driver_sync();
    if result != cudaError_t::cudaSuccess {
        return result;
    }

    let result = flush_managed_variables_to_device();
    if result != cudaError_t::cudaSuccess {
        return result;
    }
    runtime_result(super::cuda_hijack::cuGraphLaunch(
        graphExec.cast(),
        stream.cast(),
    ))
}

fn cache_runtime_library_kernel(kernel: cudaKernel_t) -> cudaError_t {
    if kernel.is_null() {
        return cudaError_t::cudaErrorInvalidResourceHandle;
    }

    let mut function = std::ptr::null_mut();
    let result = super::cuda_hijack::cuKernelGetFunction(
        &raw mut function,
        kernel.cast::<cudasys::types::cuda::CUkern_st>(),
    );
    if result == CUresult::CUDA_SUCCESS {
        RUNTIME_CACHE
            .write()
            .unwrap()
            .loaded_functions
            .insert(kernel as HostPtr, function);
    }
    runtime_result(result)
}

fn runtime_library_load_data(
    library: *mut cudaLibrary_t,
    code: *const c_void,
    num_jit_options: c_uint,
    num_library_options: c_uint,
) -> cudaError_t {
    if library.is_null() || code.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }
    if num_jit_options != 0 || num_library_options != 0 {
        return cudaError_t::cudaErrorNotSupported;
    }

    runtime_result(super::cuda_hijack::cuLibraryLoadDataInternal(
        library.cast::<cudasys::types::cuda::CUlibrary>(),
        code.cast(),
    ))
}

#[no_mangle]
extern "C" fn cudaLibraryLoadData(
    library: *mut cudaLibrary_t,
    code: *const c_void,
    _jitOptions: *mut cudaJitOption,
    _jitOptionsValues: *mut *mut c_void,
    numJitOptions: c_uint,
    _libraryOptions: *mut cudaLibraryOption,
    _libraryOptionValues: *mut *mut c_void,
    numLibraryOptions: c_uint,
) -> cudaError_t {
    log::debug!(target: "cudaLibraryLoadData", "");
    runtime_library_load_data(library, code, numJitOptions, numLibraryOptions)
}

#[no_mangle]
extern "C" fn cudaLibraryLoadFromFile(
    library: *mut cudaLibrary_t,
    fileName: *const c_char,
    _jitOptions: *mut cudaJitOption,
    _jitOptionsValues: *mut *mut c_void,
    numJitOptions: c_uint,
    _libraryOptions: *mut cudaLibraryOption,
    _libraryOptionValues: *mut *mut c_void,
    numLibraryOptions: c_uint,
) -> cudaError_t {
    log::debug!(target: "cudaLibraryLoadFromFile", "");
    if library.is_null() || fileName.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }
    if numJitOptions != 0 || numLibraryOptions != 0 {
        return cudaError_t::cudaErrorNotSupported;
    }

    let path = unsafe { CStr::from_ptr(fileName) };
    let Ok(mut image) = std::fs::read(path.to_string_lossy().as_ref()) else {
        return cudaError_t::cudaErrorFileNotFound;
    };
    if image.last().copied() != Some(0) {
        image.push(0);
    }

    runtime_library_load_data(library, image.as_ptr().cast(), 0, 0)
}

#[no_mangle]
extern "C" fn cudaLibraryUnload(library: cudaLibrary_t) -> cudaError_t {
    log::debug!(target: "cudaLibraryUnload", "");
    let driver_library = driver_library(library);
    let runtime_kernels = DRIVER_CACHE
        .read()
        .unwrap()
        .kernel_libraries
        .iter()
        .filter_map(|(kernel, owner)| (*owner == driver_library).then_some(*kernel as HostPtr))
        .collect::<Vec<_>>();
    let result = runtime_result(super::cuda_hijack::cuLibraryUnload(driver_library));
    if result == cudaError_t::cudaSuccess {
        let mut runtime = RUNTIME_CACHE.write().unwrap();
        for kernel in runtime_kernels {
            runtime.loaded_functions.remove(&kernel);
        }
    }
    result
}

#[no_mangle]
extern "C" fn cudaLibraryGetKernel(
    pKernel: *mut cudaKernel_t,
    library: cudaLibrary_t,
    name: *const c_char,
) -> cudaError_t {
    log::debug!(target: "cudaLibraryGetKernel", "");
    if pKernel.is_null() || name.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }

    let result = runtime_result(super::cuda_hijack::cuLibraryGetKernel(
        pKernel.cast::<CUkernel>(),
        driver_library(library),
        name,
    ));
    if result == cudaError_t::cudaSuccess {
        unsafe { cache_runtime_library_kernel(*pKernel) }
    } else {
        result
    }
}

fn runtime_library_get_device_symbol(
    dptr: *mut *mut c_void,
    bytes: *mut usize,
    library: cudaLibrary_t,
    name: *const c_char,
    get_symbol: unsafe extern "C" fn(
        *mut CUdeviceptr,
        *mut usize,
        cudasys::types::cuda::CUlibrary,
        *const c_char,
    ) -> CUresult,
) -> cudaError_t {
    if dptr.is_null() && bytes.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }
    if name.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }

    let mut device_ptr = 0;
    let device_ptr_out = if dptr.is_null() {
        std::ptr::null_mut()
    } else {
        &raw mut device_ptr
    };
    let result = unsafe { get_symbol(device_ptr_out, bytes, driver_library(library), name) };
    if result == CUresult::CUDA_SUCCESS && !dptr.is_null() {
        unsafe {
            *dptr = device_ptr as *mut c_void;
        }
    }
    runtime_result(result)
}

#[no_mangle]
extern "C" fn cudaLibraryGetGlobal(
    dptr: *mut *mut c_void,
    bytes: *mut usize,
    library: cudaLibrary_t,
    name: *const c_char,
) -> cudaError_t {
    log::debug!(target: "cudaLibraryGetGlobal", "");
    runtime_library_get_device_symbol(
        dptr,
        bytes,
        library,
        name,
        super::cuda_hijack::cuLibraryGetGlobal,
    )
}

#[no_mangle]
extern "C" fn cudaLibraryGetManaged(
    dptr: *mut *mut c_void,
    bytes: *mut usize,
    library: cudaLibrary_t,
    name: *const c_char,
) -> cudaError_t {
    log::debug!(target: "cudaLibraryGetManaged", "");
    runtime_library_get_device_symbol(
        dptr,
        bytes,
        library,
        name,
        super::cuda_hijack::cuLibraryGetManaged,
    )
}

#[no_mangle]
extern "C" fn cudaLibraryGetUnifiedFunction(
    fptr: *mut *mut c_void,
    library: cudaLibrary_t,
    symbol: *const c_char,
) -> cudaError_t {
    log::debug!(target: "cudaLibraryGetUnifiedFunction", "");
    runtime_result(super::cuda_hijack::cuLibraryGetUnifiedFunction(
        fptr,
        driver_library(library),
        symbol,
    ))
}

#[no_mangle]
extern "C" fn cudaLibraryGetKernelCount(count: *mut c_uint, lib: cudaLibrary_t) -> cudaError_t {
    log::debug!(target: "cudaLibraryGetKernelCount", "");
    runtime_result(super::cuda_hijack::cuLibraryGetKernelCount(
        count,
        driver_library(lib),
    ))
}

#[no_mangle]
extern "C" fn cudaLibraryEnumerateKernels(
    kernels: *mut cudaKernel_t,
    numKernels: c_uint,
    lib: cudaLibrary_t,
) -> cudaError_t {
    log::debug!(target: "cudaLibraryEnumerateKernels", "");
    let result = runtime_result(super::cuda_hijack_custom::cuLibraryEnumerateKernels(
        kernels.cast::<CUkernel>(),
        numKernels,
        driver_library(lib),
    ));
    if result != cudaError_t::cudaSuccess || kernels.is_null() || numKernels == 0 {
        return result;
    }

    for kernel in unsafe { std::slice::from_raw_parts(kernels, numKernels as usize) } {
        let cache_result = cache_runtime_library_kernel(*kernel);
        if cache_result != cudaError_t::cudaSuccess {
            return cache_result;
        }
    }
    cudaError_t::cudaSuccess
}

fn allocate_host(ptr_out: *mut *mut c_void, size: usize, flags: c_uint) -> cudaError_t {
    super::host_memory::allocate(ptr_out, size, flags)
        .map(|_| cudaError_t::cudaSuccess)
        .unwrap_or_else(host_memory_error)
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
    if matches!(
        kind,
        cudaMemcpyKind::cudaMemcpyDeviceToHost | cudaMemcpyKind::cudaMemcpyDeviceToDevice
    ) {
        let result = flush_managed_variables_to_device();
        if result != cudaError_t::cudaSuccess {
            return result;
        }
    }

    let result = match kind {
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
    };
    if result != cudaError_t::cudaSuccess {
        return result;
    }

    match kind {
        cudaMemcpyKind::cudaMemcpyHostToDevice => update_managed_shadow_from_host(dst, src, count),
        cudaMemcpyKind::cudaMemcpyDeviceToHost => update_managed_shadow_from_host(src, dst, count),
        cudaMemcpyKind::cudaMemcpyDeviceToDevice => refresh_managed_variables_from_device(),
        cudaMemcpyKind::cudaMemcpyHostToHost => cudaError_t::cudaSuccess,
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
    if matches!(
        kind,
        cudaMemcpyKind::cudaMemcpyDeviceToHost | cudaMemcpyKind::cudaMemcpyDeviceToDevice
    ) {
        let result = flush_managed_variables_to_device();
        if result != cudaError_t::cudaSuccess {
            return result;
        }
    }

    let result = match kind {
        cudaMemcpyKind::cudaMemcpyHostToHost => unsafe {
            let result = cudaStreamSynchronize(stream);
            if result != cudaError_t::cudaSuccess {
                return result;
            }
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
    };
    if result != cudaError_t::cudaSuccess {
        return result;
    }

    match kind {
        cudaMemcpyKind::cudaMemcpyHostToDevice => update_managed_shadow_from_host(dst, src, count),
        cudaMemcpyKind::cudaMemcpyDeviceToDevice => mark_managed_device_write_pending(dst, count),
        cudaMemcpyKind::cudaMemcpyDeviceToHost | cudaMemcpyKind::cudaMemcpyHostToHost => {
            cudaError_t::cudaSuccess
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

    if let Some(cufunc) = cached_driver_function(func) {
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

type PackedRuntimeKernelNodeParams = (
    cudaKernelNodeParams,
    CUDA_KERNEL_NODE_PARAMS,
    Box<[u8]>,
    Box<[u32]>,
);

fn pack_runtime_kernel_node_params(
    pNodeParams: *const cudaKernelNodeParams,
) -> Result<PackedRuntimeKernelNodeParams, cudaError_t> {
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
    let has_lazy_function = RUNTIME_CACHE
        .read()
        .unwrap()
        .lazy_functions
        .contains_key(&host_func);
    let cufunc = if has_lazy_function {
        get_cufunction(host_func)
    } else if let Some(cufunc) = cached_driver_function(host_func) {
        cufunc
    } else {
        return Err(cudaError_t::cudaErrorInvalidDeviceFunction);
    };
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

struct ManagedVariableDeviceEntry {
    host_key: HostPtr,
    device_ptr: CUdeviceptr,
    bytes: usize,
}

fn managed_variable_device_entries(
    runtime: &mut RuntimeCache,
) -> Result<Vec<ManagedVariableDeviceEntry>, cudaError_t> {
    if runtime.managed_variable_shadows.is_empty() {
        return Ok(Vec::new());
    }

    let result = ensure_runtime_device(runtime);
    if result != cudaError_t::cudaSuccess {
        return Err(result);
    }

    let keys = runtime
        .managed_variable_shadows
        .keys()
        .copied()
        .collect::<Vec<_>>();
    let mut entries = Vec::with_capacity(keys.len());

    for host_key in keys {
        let Some(shadow) = runtime.managed_variable_shadows.get(&host_key) else {
            continue;
        };
        let shadow_ptr = shadow.ptr();
        let shadow_size = shadow.size;
        if shadow_size == 0 {
            continue;
        }

        let Some(&(fatCubinHandle, deviceName)) = runtime
            .lazy_variables
            .get(&shadow_ptr)
            .or_else(|| runtime.lazy_variables.get(&host_key))
        else {
            continue;
        };

        let module = load_module_for_fatbin(runtime, fatCubinHandle);
        let mut device_ptr: CUdeviceptr = 0;
        let mut device_bytes: usize = 0;
        let result = super::cuda_hijack::cuModuleGetGlobal_v2(
            &raw mut device_ptr,
            &raw mut device_bytes,
            module,
            deviceName,
        );
        if result != CUresult::CUDA_SUCCESS {
            return Err(cudaError_t::cudaErrorInvalidSymbol);
        }
        if device_bytes != shadow_size {
            log::warn!("managed variable size mismatch: host={shadow_size}, device={device_bytes}");
        }
        entries.push(ManagedVariableDeviceEntry {
            host_key,
            device_ptr,
            bytes: std::cmp::min(shadow_size, device_bytes),
        });
    }

    Ok(entries)
}

pub(crate) fn flush_managed_variables_to_device() -> cudaError_t {
    let plans = {
        let mut runtime = RUNTIME_CACHE.write().unwrap();
        let entries = match managed_variable_device_entries(&mut runtime) {
            Ok(entries) => entries,
            Err(error) => return error,
        };
        let mut plans = Vec::new();
        for entry in entries {
            let Some(shadow) = runtime.managed_variable_shadows.get(&entry.host_key) else {
                continue;
            };
            let bytes = entry.bytes;
            if bytes == 0
                || shadow.pending_device_write
                || shadow.bytes[..bytes] == shadow.synced_bytes[..bytes]
            {
                continue;
            }
            plans.push((
                entry.host_key,
                entry.device_ptr,
                shadow.bytes[..bytes].to_vec(),
            ));
        }
        plans
    };

    for (host_key, device_ptr, bytes) in plans {
        let result = super::cudart_hijack::cudaMemcpyHtod(
            device_ptr as *mut c_void,
            bytes.as_ptr().cast(),
            bytes.len(),
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        );
        if result != cudaError_t::cudaSuccess {
            return result;
        }

        let mut runtime = RUNTIME_CACHE.write().unwrap();
        if let Some(shadow) = runtime.managed_variable_shadows.get_mut(&host_key) {
            let len = bytes.len();
            if shadow.bytes[..len] == bytes[..] {
                shadow.synced_bytes[..len].copy_from_slice(&bytes);
            }
        }
    }

    cudaError_t::cudaSuccess
}

pub(crate) fn refresh_managed_variables_from_device() -> cudaError_t {
    let entries = {
        let mut runtime = RUNTIME_CACHE.write().unwrap();
        match managed_variable_device_entries(&mut runtime) {
            Ok(entries) => entries,
            Err(error) => return error,
        }
    };

    for entry in entries {
        if entry.bytes == 0 {
            continue;
        }

        let mut bytes = vec![0u8; entry.bytes];
        let result = super::cudart_hijack::cudaMemcpyDtoh(
            bytes.as_mut_ptr().cast(),
            entry.device_ptr as *const c_void,
            bytes.len(),
            cudaMemcpyKind::cudaMemcpyDeviceToHost,
        );
        if result != cudaError_t::cudaSuccess {
            return result;
        }

        let mut runtime = RUNTIME_CACHE.write().unwrap();
        if let Some(shadow) = runtime.managed_variable_shadows.get_mut(&entry.host_key) {
            let len = std::cmp::min(bytes.len(), shadow.size);
            shadow.bytes[..len].copy_from_slice(&bytes[..len]);
            shadow.synced_bytes[..len].copy_from_slice(&bytes[..len]);
            shadow.pending_device_write = false;
        }
    }

    cudaError_t::cudaSuccess
}

struct ManagedVariableHostCopy {
    host_key: HostPtr,
    host_offset: usize,
    shadow_offset: usize,
    bytes: usize,
}

fn managed_variable_overlaps(
    device_ptr: *const c_void,
    count: usize,
) -> Result<Vec<ManagedVariableHostCopy>, cudaError_t> {
    if count == 0 {
        return Ok(Vec::new());
    }

    let start = device_ptr as usize;
    let Some(end) = start.checked_add(count) else {
        return Err(cudaError_t::cudaErrorInvalidValue);
    };

    let mut runtime = RUNTIME_CACHE.write().unwrap();
    let entries = managed_variable_device_entries(&mut runtime)?;
    let mut copies = Vec::new();
    for entry in entries {
        let entry_start = entry.device_ptr as usize;
        let Some(entry_end) = entry_start.checked_add(entry.bytes) else {
            return Err(cudaError_t::cudaErrorInvalidValue);
        };
        let overlap_start = std::cmp::max(start, entry_start);
        let overlap_end = std::cmp::min(end, entry_end);
        if overlap_start >= overlap_end {
            continue;
        }
        copies.push(ManagedVariableHostCopy {
            host_key: entry.host_key,
            host_offset: overlap_start - start,
            shadow_offset: overlap_start - entry_start,
            bytes: overlap_end - overlap_start,
        });
    }
    Ok(copies)
}

fn update_managed_shadow_from_host(
    device_ptr: *const c_void,
    host_ptr: *const c_void,
    count: usize,
) -> cudaError_t {
    if host_ptr.is_null() && count > 0 {
        return cudaError_t::cudaErrorInvalidValue;
    }

    let copies = match managed_variable_overlaps(device_ptr, count) {
        Ok(copies) => copies,
        Err(error) => return error,
    };
    if copies.is_empty() {
        return cudaError_t::cudaSuccess;
    }

    let mut runtime = RUNTIME_CACHE.write().unwrap();
    for copy in copies {
        let Some(shadow) = runtime.managed_variable_shadows.get_mut(&copy.host_key) else {
            continue;
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                host_ptr.cast::<u8>().add(copy.host_offset),
                shadow.bytes.as_mut_ptr().add(copy.shadow_offset),
                copy.bytes,
            );
        }
        let shadow_range = copy.shadow_offset..copy.shadow_offset + copy.bytes;
        shadow.synced_bytes[shadow_range.clone()].copy_from_slice(&shadow.bytes[shadow_range]);
        shadow.pending_device_write = false;
    }

    cudaError_t::cudaSuccess
}

fn mark_managed_device_write_pending(device_ptr: *const c_void, count: usize) -> cudaError_t {
    let copies = match managed_variable_overlaps(device_ptr, count) {
        Ok(copies) => copies,
        Err(error) => return error,
    };
    if copies.is_empty() {
        return cudaError_t::cudaSuccess;
    }

    let mut runtime = RUNTIME_CACHE.write().unwrap();
    for copy in copies {
        if let Some(shadow) = runtime.managed_variable_shadows.get_mut(&copy.host_key) {
            shadow.pending_device_write = true;
        }
    }

    cudaError_t::cudaSuccess
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
pub extern "C" fn cudaGetExportTable(
    ppExportTable: *mut *const c_void,
    pExportTableId: *const cudaUUID_t,
) -> cudaError_t {
    log::debug!(target: "cudaGetExportTable", "");
    if ppExportTable.is_null() || pExportTableId.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }
    unsafe {
        *ppExportTable = std::ptr::null();
    }
    runtime_result(super::cuda_hijack_custom::cuGetExportTable(
        ppExportTable,
        pExportTableId.cast(),
    ))
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
extern "C" fn cudaGetFuncBySymbol(
    functionPtr: *mut cudaFunction_t,
    symbolPtr: *const c_void,
) -> cudaError_t {
    log::debug!(target: "cudaGetFuncBySymbol", "");
    if functionPtr.is_null() || symbolPtr.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }

    unsafe {
        *functionPtr = get_cufunction(symbolPtr as HostPtr).cast();
    }
    cudaError_t::cudaSuccess
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

    let result = flush_managed_variables_to_device();
    if result != cudaError_t::cudaSuccess {
        return result;
    }

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
pub extern "C" fn cudaLaunchCooperativeKernel(
    func: MemPtr,
    gridDim: dim3,
    blockDim: dim3,
    args: *mut *mut ::std::os::raw::c_void,
    sharedMem: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    log::debug!(target: "cudaLaunchCooperativeKernel", "");

    let result = flush_managed_variables_to_device();
    if result != cudaError_t::cudaSuccess {
        return result;
    }

    let cufunc = get_cufunction(func);

    unsafe {
        std::mem::transmute(super::cuda_hijack::cuLaunchCooperativeKernel(
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

fn driver_launch_config(config: *const cudaLaunchConfig_t) -> Result<CUlaunchConfig, cudaError_t> {
    if config.is_null() {
        return Err(cudaError_t::cudaErrorInvalidValue);
    }

    let config = unsafe { &*config };
    let shared_mem = match config.dynamicSmemBytes.try_into() {
        Ok(shared_mem) => shared_mem,
        Err(_) => return Err(cudaError_t::cudaErrorInvalidValue),
    };

    Ok(CUlaunchConfig {
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
    })
}

#[no_mangle]
pub extern "C" fn cudaLaunchKernelExC(
    config: *const cudaLaunchConfig_t,
    func: MemPtr,
    args: *mut *mut ::std::os::raw::c_void,
) -> cudaError_t {
    log::debug!(target: "cudaLaunchKernelExC", "");
    let result = flush_managed_variables_to_device();
    if result != cudaError_t::cudaSuccess {
        return result;
    }

    let launch_config = match driver_launch_config(config) {
        Ok(config) => config,
        Err(error) => return error,
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
    allocate_host(pHost, size, flags)
}

#[no_mangle]
pub extern "C" fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> cudaError_t {
    log::debug!(target: "cudaMallocHost", "size = {size}");
    allocate_host(ptr, size, 0)
}

#[no_mangle]
pub extern "C" fn cudaFreeHost(ptr: *mut c_void) -> cudaError_t {
    log::debug!(target: "cudaFreeHost", "");
    super::host_memory::free(ptr)
        .map(|_| cudaError_t::cudaSuccess)
        .unwrap_or_else(host_memory_error)
}

#[no_mangle]
pub extern "C" fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: c_uint) -> cudaError_t {
    log::debug!(target: "cudaHostRegister", "size = {size}, flags = {flags}");
    super::host_memory::register(ptr, size, flags)
        .map(|_| cudaError_t::cudaSuccess)
        .unwrap_or_else(host_memory_error)
}

#[no_mangle]
pub extern "C" fn cudaHostUnregister(ptr: *mut c_void) -> cudaError_t {
    log::debug!(target: "cudaHostUnregister", "");
    super::host_memory::unregister(ptr)
        .map(|_| cudaError_t::cudaSuccess)
        .unwrap_or_else(host_memory_error)
}

#[no_mangle]
pub extern "C" fn cudaHostGetFlags(pFlags: *mut c_uint, pHost: *mut c_void) -> cudaError_t {
    log::debug!(target: "cudaHostGetFlags", "");
    super::host_memory::get_flags(pFlags, pHost)
        .map(|_| cudaError_t::cudaSuccess)
        .unwrap_or_else(host_memory_error)
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
    let result = cudaStreamSynchronize(stream);
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
extern "C" fn cudaUserObjectCreate(
    object_out: *mut cudaUserObject_t,
    ptr: *mut c_void,
    destroy: cudaHostFn_t,
    initialRefcount: c_uint,
    flags: c_uint,
) -> cudaError_t {
    log::debug!(target: "cudaUserObjectCreate", "");
    runtime_result(super::user_object::create(
        object_out.cast(),
        ptr,
        destroy,
        initialRefcount,
        flags,
    ))
}

#[no_mangle]
extern "C" fn cudaUserObjectRetain(object: cudaUserObject_t, count: c_uint) -> cudaError_t {
    log::debug!(target: "cudaUserObjectRetain", "");
    runtime_result(super::user_object::retain(object.cast(), count))
}

#[no_mangle]
extern "C" fn cudaUserObjectRelease(object: cudaUserObject_t, count: c_uint) -> cudaError_t {
    log::debug!(target: "cudaUserObjectRelease", "");
    runtime_result(super::user_object::release(object.cast(), count))
}

#[no_mangle]
extern "C" fn cudaGraphRetainUserObject(
    graph: cudaGraph_t,
    object: cudaUserObject_t,
    count: c_uint,
    flags: c_uint,
) -> cudaError_t {
    log::debug!(target: "cudaGraphRetainUserObject", "");
    runtime_result(super::user_object::graph_retain(
        graph.cast(),
        object.cast(),
        count,
        flags,
    ))
}

#[no_mangle]
extern "C" fn cudaGraphReleaseUserObject(
    graph: cudaGraph_t,
    object: cudaUserObject_t,
    count: c_uint,
) -> cudaError_t {
    log::debug!(target: "cudaGraphReleaseUserObject", "");
    runtime_result(super::user_object::graph_release(
        graph.cast(),
        object.cast(),
        count,
    ))
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
    let status = cudaStreamSynchronize(stream);
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
    runtime_result(super::cuda_hijack_custom::cuFuncGetName(
        name,
        get_cufunction(func as HostPtr),
    ))
}

#[no_mangle]
extern "C" fn cudaFuncGetParamInfo(
    func: *const c_void,
    paramIndex: usize,
    paramOffset: *mut usize,
    paramSize: *mut usize,
) -> cudaError_t {
    log::debug!(target: "cudaFuncGetParamInfo", "");
    runtime_result(super::cuda_hijack::cuFuncGetParamInfo(
        get_cufunction(func as HostPtr),
        paramIndex,
        paramOffset,
        paramSize,
    ))
}

#[no_mangle]
extern "C" fn cudaFuncGetParamCount(func: *const c_void, paramCount: *mut usize) -> cudaError_t {
    log::debug!(target: "cudaFuncGetParamCount", "");
    runtime_result(super::cuda_hijack::cuFuncGetParamCount(
        get_cufunction(func as HostPtr),
        paramCount,
    ))
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
extern "C" fn cudaOccupancyMaxPotentialClusterSize(
    clusterSize: *mut c_int,
    func: *const c_void,
    launchConfig: *const cudaLaunchConfig_t,
) -> cudaError_t {
    log::debug!(target: "cudaOccupancyMaxPotentialClusterSize", "");
    let launch_config = match driver_launch_config(launchConfig) {
        Ok(config) => config,
        Err(error) => return error,
    };
    let result = super::cuda_hijack::cuOccupancyMaxPotentialClusterSize(
        clusterSize,
        get_cufunction(func as HostPtr),
        &launch_config,
    );
    unsafe { std::mem::transmute(result) }
}

#[no_mangle]
extern "C" fn cudaOccupancyMaxActiveClusters(
    numClusters: *mut c_int,
    func: *const c_void,
    launchConfig: *const cudaLaunchConfig_t,
) -> cudaError_t {
    log::debug!(target: "cudaOccupancyMaxActiveClusters", "");
    let launch_config = match driver_launch_config(launchConfig) {
        Ok(config) => config,
        Err(error) => return error,
    };
    let result = super::cuda_hijack::cuOccupancyMaxActiveClusters(
        numClusters,
        get_cufunction(func as HostPtr),
        &launch_config,
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
extern "C" fn cudaKernelSetAttributeForDevice(
    kernel: cudaKernel_t,
    attr: cudaFuncAttribute,
    value: c_int,
    device: c_int,
) -> cudaError_t {
    log::debug!(target: "cudaKernelSetAttributeForDevice", "");
    #[expect(clippy::missing_transmute_annotations)]
    unsafe {
        std::mem::transmute(super::cuda_hijack::cuKernelSetAttribute(
            std::mem::transmute(attr),
            value,
            kernel.cast::<cudasys::types::cuda::CUkern_st>(),
            device,
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
