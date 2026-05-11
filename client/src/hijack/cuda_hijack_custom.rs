use super::*;
use cudasys::types::cuda::*;
use network::type_impl::{recv_slice, send_slice};
use network::{CommChannel, Transportable};
use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::os::raw::*;
use std::sync::{Mutex, OnceLock};

fn recv_cu_result<C: CommChannel>(
    target: &'static str,
    client_id: i32,
    channel_receiver: &C,
) -> CUresult {
    let mut result = CUresult::CUDA_SUCCESS;
    result.recv(channel_receiver).unwrap();
    channel_receiver.recv_ts().unwrap();
    if result != CUresult::CUDA_SUCCESS {
        log::error!(target: target, "[#{}] returned error: {:?}", client_id, result);
    }
    result
}

fn recv_cu_graph_node_slice<C: CommChannel>(
    target: &'static str,
    dst: *mut CUgraphNode,
    capacity: usize,
    channel_receiver: &C,
) -> Option<CUresult> {
    if dst.is_null() {
        return None;
    }
    let nodes = recv_slice::<CUgraphNode, _>(channel_receiver).unwrap();
    if nodes.len() > capacity {
        log::error!(
            target: target,
            "server returned {} graph nodes for client capacity {}",
            nodes.len(),
            capacity,
        );
        return Some(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    unsafe {
        std::ptr::copy_nonoverlapping(nodes.as_ptr(), dst, nodes.len());
    }
    None
}

fn recv_cu_graph_edge_data_slice<C: CommChannel>(
    target: &'static str,
    dst: *mut CUgraphEdgeData,
    capacity: usize,
    channel_receiver: &C,
) -> Option<CUresult> {
    if dst.is_null() {
        return None;
    }
    let edge_data = recv_slice::<CUgraphEdgeData, _>(channel_receiver).unwrap();
    if edge_data.len() > capacity {
        log::error!(
            target: target,
            "server returned {} graph edges for client capacity {}",
            edge_data.len(),
            capacity,
        );
        return Some(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    unsafe {
        std::ptr::copy_nonoverlapping(edge_data.as_ptr(), dst, edge_data.len());
    }
    None
}

fn host_memory_error(error: super::host_memory::HostMemoryError) -> CUresult {
    match error {
        super::host_memory::HostMemoryError::InvalidValue => CUresult::CUDA_ERROR_INVALID_VALUE,
        super::host_memory::HostMemoryError::MemoryAllocation => CUresult::CUDA_ERROR_OUT_OF_MEMORY,
    }
}

#[no_mangle]
extern "C" fn cuMemAllocHost_v2(pp: *mut *mut c_void, bytesize: usize) -> CUresult {
    log::debug!(target: "cuMemAllocHost_v2", "bytesize = {bytesize}");
    super::host_memory::allocate(pp, bytesize, 0)
        .map(|_| CUresult::CUDA_SUCCESS)
        .unwrap_or_else(host_memory_error)
}

#[no_mangle]
extern "C" fn cuMemHostAlloc(pp: *mut *mut c_void, bytesize: usize, Flags: c_uint) -> CUresult {
    log::debug!(target: "cuMemHostAlloc", "bytesize = {bytesize}, flags = {Flags}");
    super::host_memory::allocate(pp, bytesize, Flags)
        .map(|_| CUresult::CUDA_SUCCESS)
        .unwrap_or_else(host_memory_error)
}

#[no_mangle]
extern "C" fn cuMemFreeHost(p: *mut c_void) -> CUresult {
    log::debug!(target: "cuMemFreeHost", "");
    super::host_memory::free(p)
        .map(|_| CUresult::CUDA_SUCCESS)
        .unwrap_or_else(host_memory_error)
}

#[no_mangle]
extern "C" fn cuMemHostRegister_v2(p: *mut c_void, bytesize: usize, Flags: c_uint) -> CUresult {
    log::debug!(target: "cuMemHostRegister_v2", "bytesize = {bytesize}, flags = {Flags}");
    super::host_memory::register(p, bytesize, Flags)
        .map(|_| CUresult::CUDA_SUCCESS)
        .unwrap_or_else(host_memory_error)
}

#[no_mangle]
extern "C" fn cuMemHostUnregister(p: *mut c_void) -> CUresult {
    log::debug!(target: "cuMemHostUnregister", "");
    super::host_memory::unregister(p)
        .map(|_| CUresult::CUDA_SUCCESS)
        .unwrap_or_else(host_memory_error)
}

#[no_mangle]
extern "C" fn cuMemHostGetFlags(pFlags: *mut c_uint, p: *mut c_void) -> CUresult {
    log::debug!(target: "cuMemHostGetFlags", "");
    super::host_memory::get_flags(pFlags, p)
        .map(|_| CUresult::CUDA_SUCCESS)
        .unwrap_or_else(host_memory_error)
}

#[no_mangle]
extern "C" fn cuMemHostGetDevicePointer_v2(
    pdptr: *mut CUdeviceptr,
    p: *mut c_void,
    Flags: c_uint,
) -> CUresult {
    log::debug!(target: "cuMemHostGetDevicePointer_v2", "flags = {Flags}");
    if pdptr.is_null() || p.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }
    unsafe {
        *pdptr = 0;
    }
    CUresult::CUDA_ERROR_NOT_SUPPORTED
}

#[no_mangle]
extern "C" fn cuMemExportToShareableHandle(
    shareableHandle: *mut c_void,
    handle: CUmemGenericAllocationHandle,
    handleType: CUmemAllocationHandleType,
    flags: c_ulonglong,
) -> CUresult {
    if shareableHandle.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }
    if handleType != CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR {
        return CUresult::CUDA_ERROR_NOT_SUPPORTED;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuMemExportToShareableHandle", "[#{}]", client.id);

        901106.send(&client.channel_sender).unwrap();
        handle.send(&client.channel_sender).unwrap();
        handleType.send(&client.channel_sender).unwrap();
        flags.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        let mut synthetic_fd = -1 as c_int;
        synthetic_fd.recv(&client.channel_receiver).unwrap();
        let result = recv_cu_result(
            "cuMemExportToShareableHandle",
            client.id,
            &client.channel_receiver,
        );
        if result == CUresult::CUDA_SUCCESS {
            unsafe {
                *shareableHandle.cast::<c_int>() = synthetic_fd;
            }
        }
        result
    })
}

#[no_mangle]
extern "C" fn cuMemImportFromShareableHandle(
    handle: *mut CUmemGenericAllocationHandle,
    osHandle: *mut c_void,
    shHandleType: CUmemAllocationHandleType,
) -> CUresult {
    if handle.is_null() || osHandle.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }
    if shHandleType != CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR {
        return CUresult::CUDA_ERROR_NOT_SUPPORTED;
    }
    let synthetic_fd = osHandle as isize as c_int;

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuMemImportFromShareableHandle", "[#{}]", client.id);

        901107.send(&client.channel_sender).unwrap();
        synthetic_fd.send(&client.channel_sender).unwrap();
        shHandleType.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        let mut imported_handle = 0;
        imported_handle.recv(&client.channel_receiver).unwrap();
        let result = recv_cu_result(
            "cuMemImportFromShareableHandle",
            client.id,
            &client.channel_receiver,
        );
        if result == CUresult::CUDA_SUCCESS {
            unsafe {
                *handle = imported_handle;
            }
        }
        result
    })
}

#[no_mangle]
extern "C" fn cuPointerGetAttributes(
    numAttributes: c_uint,
    attributes: *mut CUpointer_attribute,
    data: *mut *mut c_void,
    ptr: CUdeviceptr,
) -> CUresult {
    if numAttributes == 0 || attributes.is_null() || data.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    let num_attributes = numAttributes as usize;
    let attrs = unsafe { std::slice::from_raw_parts(attributes, num_attributes) };
    let data_sizes = attrs
        .iter()
        .map(CUpointer_attribute::data_size)
        .collect::<Vec<_>>();
    for (idx, size) in data_sizes.iter().enumerate() {
        if *size == 0 || unsafe { *data.add(idx) }.is_null() {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        }
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuPointerGetAttributes", "[#{}]", client.id);

        901016.send(&client.channel_sender).unwrap();
        send_slice(attrs, &client.channel_sender).unwrap();
        ptr.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        let mut local_error = None;
        for (idx, capacity) in data_sizes.iter().copied().enumerate() {
            let bytes = recv_slice::<u8, _>(&client.channel_receiver).unwrap();
            if bytes.len() > capacity {
                log::error!(
                    target: "cuPointerGetAttributes",
                    "server returned {} bytes for client capacity {}",
                    bytes.len(),
                    capacity,
                );
                local_error = Some(CUresult::CUDA_ERROR_INVALID_VALUE);
                continue;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr().cast(), *data.add(idx), bytes.len());
            }
        }
        let result = recv_cu_result(
            "cuPointerGetAttributes",
            client.id,
            &client.channel_receiver,
        );
        local_error.unwrap_or(result)
    })
}

fn cu_graph_get_node_list<T: Transportable>(
    proc_id: i32,
    target: &'static str,
    graph_or_node: T,
    nodes: *mut CUgraphNode,
    count: *mut usize,
) -> CUresult {
    if count.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }
    let has_nodes = !nodes.is_null();
    let capacity = if has_nodes { unsafe { *count } } else { 0 };
    if has_nodes && capacity == 0 {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
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

        let local_error = recv_cu_graph_node_slice(target, nodes, capacity, channel_receiver);
        unsafe { &mut *count }.recv(channel_receiver).unwrap();
        let result = recv_cu_result(target, client.id, channel_receiver);
        local_error.unwrap_or(result)
    })
}

fn cu_graph_get_edge_list<T: Transportable>(
    proc_id: i32,
    target: &'static str,
    graph_or_node: T,
    from: *mut CUgraphNode,
    to: *mut CUgraphNode,
    edge_data: *mut CUgraphEdgeData,
    count: *mut usize,
) -> CUresult {
    if count.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }
    let has_from = !from.is_null();
    let has_to = !to.is_null();
    let has_edge_data = !edge_data.is_null();
    let has_outputs = has_from || has_to || has_edge_data;
    let capacity = if has_outputs { unsafe { *count } } else { 0 };
    if has_outputs && capacity == 0 {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
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

        let mut local_error = recv_cu_graph_node_slice(target, from, capacity, channel_receiver);
        let to_error = recv_cu_graph_node_slice(target, to, capacity, channel_receiver);
        if local_error.is_none() {
            local_error = to_error;
        }
        let edge_data_error =
            recv_cu_graph_edge_data_slice(target, edge_data, capacity, channel_receiver);
        if local_error.is_none() {
            local_error = edge_data_error;
        }
        unsafe { &mut *count }.recv(channel_receiver).unwrap();
        let result = recv_cu_result(target, client.id, channel_receiver);
        local_error.unwrap_or(result)
    })
}

#[no_mangle]
extern "C" fn cuMemPrefetchAsync_v2(
    devPtr: CUdeviceptr,
    count: usize,
    location: CUmemLocation,
    flags: c_uint,
    hStream: CUstream,
) -> CUresult {
    if devPtr == 0 || count == 0 {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuMemPrefetchAsync_v2", "[#{}]", client.id);

        900984.send(&client.channel_sender).unwrap();
        devPtr.send(&client.channel_sender).unwrap();
        count.send(&client.channel_sender).unwrap();
        location.send(&client.channel_sender).unwrap();
        flags.send(&client.channel_sender).unwrap();
        hStream.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        recv_cu_result("cuMemPrefetchAsync_v2", client.id, &client.channel_receiver)
    })
}

#[no_mangle]
extern "C" fn cuMemAdvise_v2(
    devPtr: CUdeviceptr,
    count: usize,
    advice: CUmem_advise,
    location: CUmemLocation,
) -> CUresult {
    if devPtr == 0 || count == 0 {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuMemAdvise_v2", "[#{}]", client.id);

        900985.send(&client.channel_sender).unwrap();
        devPtr.send(&client.channel_sender).unwrap();
        count.send(&client.channel_sender).unwrap();
        advice.send(&client.channel_sender).unwrap();
        location.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        recv_cu_result("cuMemAdvise_v2", client.id, &client.channel_receiver)
    })
}

#[no_mangle]
extern "C" fn cuMemRangeGetAttributes(
    data: *mut *mut c_void,
    dataSizes: *mut usize,
    attributes: *mut CUmem_range_attribute,
    numAttributes: usize,
    devPtr: CUdeviceptr,
    count: usize,
) -> CUresult {
    if data.is_null()
        || dataSizes.is_null()
        || attributes.is_null()
        || numAttributes == 0
        || devPtr == 0
        || count == 0
    {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    let data_sizes = unsafe { std::slice::from_raw_parts(dataSizes, numAttributes) };
    let attrs = unsafe { std::slice::from_raw_parts(attributes, numAttributes) };
    for (idx, size) in data_sizes.iter().enumerate() {
        if *size == 0 || unsafe { *data.add(idx) }.is_null() {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        }
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuMemRangeGetAttributes", "[#{}]", client.id);

        900987.send(&client.channel_sender).unwrap();
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
                    target: "cuMemRangeGetAttributes",
                    "server returned {} bytes for client capacity {}",
                    bytes.len(),
                    capacity,
                );
                local_error = Some(CUresult::CUDA_ERROR_INVALID_VALUE);
                continue;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr().cast(), *data.add(idx), bytes.len());
            }
        }
        let result = recv_cu_result(
            "cuMemRangeGetAttributes",
            client.id,
            &client.channel_receiver,
        );
        local_error.unwrap_or(result)
    })
}

#[no_mangle]
extern "C" fn cuMemBatchDecompressAsync(
    paramsArray: *mut CUmemDecompressParams,
    count: usize,
    flags: c_uint,
    errorIndex: *mut usize,
    stream: CUstream,
) -> CUresult {
    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuMemBatchDecompressAsync", "[#{}]", client.id);

        901105.send(&client.channel_sender).unwrap();
        let has_params = !paramsArray.is_null();
        has_params.send(&client.channel_sender).unwrap();
        if has_params {
            let params = unsafe { std::slice::from_raw_parts(paramsArray, count) };
            send_slice(params, &client.channel_sender).unwrap();
        }
        count.send(&client.channel_sender).unwrap();
        flags.send(&client.channel_sender).unwrap();
        let has_error_index = !errorIndex.is_null();
        has_error_index.send(&client.channel_sender).unwrap();
        stream.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        let mut remote_error_index = usize::MAX;
        if has_error_index {
            remote_error_index.recv(&client.channel_receiver).unwrap();
        }
        let result = recv_cu_result(
            "cuMemBatchDecompressAsync",
            client.id,
            &client.channel_receiver,
        );
        if has_error_index
            && result != CUresult::CUDA_SUCCESS
            && result != CUresult::CUDA_ERROR_NOT_SUPPORTED
        {
            unsafe {
                *errorIndex = remote_error_index;
            }
        }
        result
    })
}

#[no_mangle]
extern "C" fn cuMemPoolCreate(
    pool: *mut CUmemoryPool,
    poolProps: *const CUmemPoolProps,
) -> CUresult {
    if pool.is_null() || poolProps.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuMemPoolCreate", "[#{}]", client.id);

        901002.send(&client.channel_sender).unwrap();
        unsafe { &*poolProps }.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        unsafe { &mut *pool }
            .recv(&client.channel_receiver)
            .unwrap();
        recv_cu_result("cuMemPoolCreate", client.id, &client.channel_receiver)
    })
}

fn cu_mem_get_pool(
    proc_id: i32,
    target: &'static str,
    pool: *mut CUmemoryPool,
    location: *mut CUmemLocation,
    type_: CUmemAllocationType,
) -> CUresult {
    if pool.is_null() || location.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: target, "[#{}]", client.id);

        proc_id.send(&client.channel_sender).unwrap();
        unsafe { &*location }.send(&client.channel_sender).unwrap();
        type_.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        unsafe { &mut *pool }
            .recv(&client.channel_receiver)
            .unwrap();
        recv_cu_result(target, client.id, &client.channel_receiver)
    })
}

#[no_mangle]
extern "C" fn cuMemGetDefaultMemPool(
    pool_out: *mut CUmemoryPool,
    location: *mut CUmemLocation,
    type_: CUmemAllocationType,
) -> CUresult {
    cu_mem_get_pool(901004, "cuMemGetDefaultMemPool", pool_out, location, type_)
}

#[no_mangle]
extern "C" fn cuMemGetMemPool(
    pool: *mut CUmemoryPool,
    location: *mut CUmemLocation,
    type_: CUmemAllocationType,
) -> CUresult {
    cu_mem_get_pool(901005, "cuMemGetMemPool", pool, location, type_)
}

#[no_mangle]
extern "C" fn cuMemSetMemPool(
    location: *mut CUmemLocation,
    type_: CUmemAllocationType,
    pool: CUmemoryPool,
) -> CUresult {
    if location.is_null() || pool.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuMemSetMemPool", "[#{}]", client.id);

        901006.send(&client.channel_sender).unwrap();
        unsafe { &*location }.send(&client.channel_sender).unwrap();
        type_.send(&client.channel_sender).unwrap();
        pool.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        recv_cu_result("cuMemSetMemPool", client.id, &client.channel_receiver)
    })
}

#[no_mangle]
extern "C" fn cuMemPoolSetAccess(
    pool: CUmemoryPool,
    map: *const CUmemAccessDesc,
    count: usize,
) -> CUresult {
    if pool.is_null() || map.is_null() || count == 0 {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuMemPoolSetAccess", "[#{}]", client.id);

        901007.send(&client.channel_sender).unwrap();
        pool.send(&client.channel_sender).unwrap();
        let access = unsafe { std::slice::from_raw_parts(map, count) };
        send_slice(access, &client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        recv_cu_result("cuMemPoolSetAccess", client.id, &client.channel_receiver)
    })
}

#[no_mangle]
extern "C" fn cuMemPoolGetAccess(
    flags: *mut CUmemAccess_flags,
    memPool: CUmemoryPool,
    location: *mut CUmemLocation,
) -> CUresult {
    if flags.is_null() || memPool.is_null() || location.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuMemPoolGetAccess", "[#{}]", client.id);

        901008.send(&client.channel_sender).unwrap();
        memPool.send(&client.channel_sender).unwrap();
        unsafe { &*location }.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        unsafe { &mut *flags }
            .recv(&client.channel_receiver)
            .unwrap();
        recv_cu_result("cuMemPoolGetAccess", client.id, &client.channel_receiver)
    })
}

#[no_mangle]
extern "C" fn cuMemPoolExportToShareableHandle(
    handle_out: *mut c_void,
    pool: CUmemoryPool,
    handleType: CUmemAllocationHandleType,
    flags: c_ulonglong,
) -> CUresult {
    if handle_out.is_null() || pool.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }
    if handleType != CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR {
        return CUresult::CUDA_ERROR_NOT_SUPPORTED;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuMemPoolExportToShareableHandle", "[#{}]", client.id);

        901108.send(&client.channel_sender).unwrap();
        pool.send(&client.channel_sender).unwrap();
        handleType.send(&client.channel_sender).unwrap();
        flags.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        let mut synthetic_fd = -1 as c_int;
        synthetic_fd.recv(&client.channel_receiver).unwrap();
        let result = recv_cu_result(
            "cuMemPoolExportToShareableHandle",
            client.id,
            &client.channel_receiver,
        );
        if result == CUresult::CUDA_SUCCESS {
            unsafe {
                *handle_out.cast::<c_int>() = synthetic_fd;
            }
        }
        result
    })
}

#[no_mangle]
extern "C" fn cuMemPoolImportFromShareableHandle(
    pool_out: *mut CUmemoryPool,
    handle: *mut c_void,
    handleType: CUmemAllocationHandleType,
    flags: c_ulonglong,
) -> CUresult {
    if pool_out.is_null() || handle.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }
    if handleType != CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR {
        return CUresult::CUDA_ERROR_NOT_SUPPORTED;
    }
    let synthetic_fd = handle as isize as c_int;

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuMemPoolImportFromShareableHandle", "[#{}]", client.id);

        901109.send(&client.channel_sender).unwrap();
        synthetic_fd.send(&client.channel_sender).unwrap();
        handleType.send(&client.channel_sender).unwrap();
        flags.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        unsafe { &mut *pool_out }
            .recv(&client.channel_receiver)
            .unwrap();
        recv_cu_result(
            "cuMemPoolImportFromShareableHandle",
            client.id,
            &client.channel_receiver,
        )
    })
}

#[no_mangle]
extern "C" fn cuMemPoolExportPointer(
    shareData_out: *mut CUmemPoolPtrExportData,
    ptr: CUdeviceptr,
) -> CUresult {
    if shareData_out.is_null() || ptr == 0 {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuMemPoolExportPointer", "[#{}]", client.id);

        901110.send(&client.channel_sender).unwrap();
        ptr.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        unsafe { &mut *shareData_out }
            .recv(&client.channel_receiver)
            .unwrap();
        recv_cu_result("cuMemPoolExportPointer", client.id, &client.channel_receiver)
    })
}

#[no_mangle]
extern "C" fn cuMemPoolImportPointer(
    ptr_out: *mut CUdeviceptr,
    pool: CUmemoryPool,
    shareData: *mut CUmemPoolPtrExportData,
) -> CUresult {
    if ptr_out.is_null() || pool.is_null() || shareData.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuMemPoolImportPointer", "[#{}]", client.id);

        901111.send(&client.channel_sender).unwrap();
        pool.send(&client.channel_sender).unwrap();
        unsafe { &*shareData }.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        unsafe { &mut *ptr_out }
            .recv(&client.channel_receiver)
            .unwrap();
        recv_cu_result("cuMemPoolImportPointer", client.id, &client.channel_receiver)
    })
}

#[no_mangle]
extern "C" fn cuTexObjectCreate(
    pTexObject: *mut CUtexObject,
    pResDesc: *const CUDA_RESOURCE_DESC,
    pTexDesc: *const CUDA_TEXTURE_DESC,
    pResViewDesc: *const CUDA_RESOURCE_VIEW_DESC,
) -> CUresult {
    if pTexObject.is_null() || pResDesc.is_null() || pTexDesc.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuTexObjectCreate", "[#{}]", client.id);

        900975.send(&client.channel_sender).unwrap();
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
        recv_cu_result("cuTexObjectCreate", client.id, &client.channel_receiver)
    })
}

#[no_mangle]
extern "C" fn cuTexObjectGetResourceDesc(
    pResDesc: *mut CUDA_RESOURCE_DESC,
    texObject: CUtexObject,
) -> CUresult {
    if pResDesc.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuTexObjectGetResourceDesc", "[#{}]", client.id);

        900977.send(&client.channel_sender).unwrap();
        texObject.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        unsafe { &mut *pResDesc }
            .recv(&client.channel_receiver)
            .unwrap();
        recv_cu_result(
            "cuTexObjectGetResourceDesc",
            client.id,
            &client.channel_receiver,
        )
    })
}

#[no_mangle]
extern "C" fn cuSurfObjectCreate(
    pSurfObject: *mut CUsurfObject,
    pResDesc: *const CUDA_RESOURCE_DESC,
) -> CUresult {
    if pSurfObject.is_null() || pResDesc.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuSurfObjectCreate", "[#{}]", client.id);

        900980.send(&client.channel_sender).unwrap();
        unsafe { &*pResDesc }.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        unsafe { &mut *pSurfObject }
            .recv(&client.channel_receiver)
            .unwrap();
        recv_cu_result("cuSurfObjectCreate", client.id, &client.channel_receiver)
    })
}

#[no_mangle]
extern "C" fn cuSurfObjectGetResourceDesc(
    pResDesc: *mut CUDA_RESOURCE_DESC,
    surfObject: CUsurfObject,
) -> CUresult {
    if pResDesc.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuSurfObjectGetResourceDesc", "[#{}]", client.id);

        900982.send(&client.channel_sender).unwrap();
        surfObject.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        unsafe { &mut *pResDesc }
            .recv(&client.channel_receiver)
            .unwrap();
        recv_cu_result(
            "cuSurfObjectGetResourceDesc",
            client.id,
            &client.channel_receiver,
        )
    })
}

#[no_mangle]
extern "C" fn cuGraphGetNodes(
    hGraph: CUgraph,
    nodes: *mut CUgraphNode,
    numNodes: *mut usize,
) -> CUresult {
    cu_graph_get_node_list(900806, "cuGraphGetNodes", hGraph, nodes, numNodes)
}

#[no_mangle]
extern "C" fn cuGraphGetRootNodes(
    hGraph: CUgraph,
    rootNodes: *mut CUgraphNode,
    numRootNodes: *mut usize,
) -> CUresult {
    cu_graph_get_node_list(
        900807,
        "cuGraphGetRootNodes",
        hGraph,
        rootNodes,
        numRootNodes,
    )
}

#[no_mangle]
extern "C" fn cuGraphGetEdges_v2(
    hGraph: CUgraph,
    from: *mut CUgraphNode,
    to: *mut CUgraphNode,
    edgeData: *mut CUgraphEdgeData,
    numEdges: *mut usize,
) -> CUresult {
    cu_graph_get_edge_list(
        900808,
        "cuGraphGetEdges_v2",
        hGraph,
        from,
        to,
        edgeData,
        numEdges,
    )
}

#[no_mangle]
extern "C" fn cuGraphNodeGetDependencies_v2(
    hNode: CUgraphNode,
    dependencies: *mut CUgraphNode,
    edgeData: *mut CUgraphEdgeData,
    numDependencies: *mut usize,
) -> CUresult {
    cu_graph_get_edge_list(
        900809,
        "cuGraphNodeGetDependencies_v2",
        hNode,
        dependencies,
        std::ptr::null_mut(),
        edgeData,
        numDependencies,
    )
}

#[no_mangle]
extern "C" fn cuGraphNodeGetDependentNodes_v2(
    hNode: CUgraphNode,
    dependentNodes: *mut CUgraphNode,
    edgeData: *mut CUgraphEdgeData,
    numDependentNodes: *mut usize,
) -> CUresult {
    cu_graph_get_edge_list(
        900810,
        "cuGraphNodeGetDependentNodes_v2",
        hNode,
        dependentNodes,
        std::ptr::null_mut(),
        edgeData,
        numDependentNodes,
    )
}

#[no_mangle]
pub extern "C" fn cuStreamAddCallback(
    hStream: CUstream,
    callback: CUstreamCallback,
    userData: *mut c_void,
    flags: c_uint,
) -> CUresult {
    log::debug!(target: "cuStreamAddCallback", "flags = {flags}");
    if flags != 0 {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }
    let Some(callback) = callback else {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    };
    let status = super::cuda_hijack::cuStreamSynchronize(hStream);
    unsafe {
        callback(hStream, status, userData);
    }
    if status.is_error() {
        status
    } else {
        CUresult::CUDA_SUCCESS
    }
}

#[no_mangle]
pub extern "C" fn cuLaunchHostFunc(
    hStream: CUstream,
    fn_: CUhostFn,
    userData: *mut c_void,
) -> CUresult {
    log::debug!(target: "cuLaunchHostFunc", "");
    let Some(callback) = fn_ else {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    };
    let result = super::cuda_hijack::cuStreamSynchronize(hStream);
    if result.is_error() {
        return result;
    }
    unsafe {
        callback(userData);
    }
    CUresult::CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuLaunchHostFunc_v2(
    hStream: CUstream,
    fn_: CUhostFn,
    userData: *mut c_void,
    syncMode: c_uint,
) -> CUresult {
    log::debug!(target: "cuLaunchHostFunc_v2", "syncMode = {syncMode}");
    cuLaunchHostFunc(hStream, fn_, userData)
}

#[no_mangle]
extern "C" fn cuUserObjectCreate(
    object_out: *mut CUuserObject,
    ptr: *mut c_void,
    destroy: CUhostFn,
    initialRefcount: c_uint,
    flags: c_uint,
) -> CUresult {
    log::debug!(target: "cuUserObjectCreate", "");
    super::user_object::create(object_out, ptr, destroy, initialRefcount, flags)
}

#[no_mangle]
extern "C" fn cuUserObjectRetain(object: CUuserObject, count: c_uint) -> CUresult {
    log::debug!(target: "cuUserObjectRetain", "");
    super::user_object::retain(object, count)
}

#[no_mangle]
extern "C" fn cuUserObjectRelease(object: CUuserObject, count: c_uint) -> CUresult {
    log::debug!(target: "cuUserObjectRelease", "");
    super::user_object::release(object, count)
}

#[no_mangle]
extern "C" fn cuGraphRetainUserObject(
    graph: CUgraph,
    object: CUuserObject,
    count: c_uint,
    flags: c_uint,
) -> CUresult {
    log::debug!(target: "cuGraphRetainUserObject", "");
    super::user_object::graph_retain(graph, object, count, flags)
}

#[no_mangle]
extern "C" fn cuGraphReleaseUserObject(
    graph: CUgraph,
    object: CUuserObject,
    count: c_uint,
) -> CUresult {
    log::debug!(target: "cuGraphReleaseUserObject", "");
    super::user_object::graph_release(graph, object, count)
}

#[no_mangle]
pub extern "C" fn cuEventDestroy_v2(hEvent: CUevent) -> CUresult {
    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuEventDestroy_v2", "[#{}]", client.id);

        900214.send(&client.channel_sender).unwrap();
        hEvent.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        recv_cu_result("cuEventDestroy_v2", client.id, &client.channel_receiver)
    })
}

#[no_mangle]
pub extern "C" fn cuIpcGetEventHandle(pHandle: *mut CUipcEventHandle, event: CUevent) -> CUresult {
    if pHandle.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuIpcGetEventHandle", "[#{}]", client.id);

        900916.send(&client.channel_sender).unwrap();
        event.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        unsafe { &mut *pHandle }
            .recv(&client.channel_receiver)
            .unwrap();
        recv_cu_result("cuIpcGetEventHandle", client.id, &client.channel_receiver)
    })
}

#[no_mangle]
pub extern "C" fn cuIpcOpenEventHandle(
    phEvent: *mut CUevent,
    handle: CUipcEventHandle,
) -> CUresult {
    if phEvent.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuIpcOpenEventHandle", "[#{}]", client.id);

        900917.send(&client.channel_sender).unwrap();
        handle.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        unsafe { &mut *phEvent }
            .recv(&client.channel_receiver)
            .unwrap();
        recv_cu_result("cuIpcOpenEventHandle", client.id, &client.channel_receiver)
    })
}

#[no_mangle]
pub extern "C" fn cuCtxCreate_v4(
    pctx: *mut CUcontext,
    ctxCreateParams: *mut CUctxCreateParams,
    flags: c_uint,
    dev: CUdevice,
) -> CUresult {
    if pctx.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    let has_params = !ctxCreateParams.is_null();
    let mut params = unsafe { std::mem::zeroed::<CUctxCreateParams>() };
    let mut affinity_params = Vec::new();
    if has_params {
        params = unsafe { *ctxCreateParams };
        if !params.cigParams.is_null() {
            return CUresult::CUDA_ERROR_NOT_SUPPORTED;
        }
        if params.numExecAffinityParams < 0 {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        }
        if params.numExecAffinityParams > 0 {
            if params.execAffinityParams.is_null() {
                return CUresult::CUDA_ERROR_INVALID_VALUE;
            }
            let affinity_len = params.numExecAffinityParams as usize;
            affinity_params.extend_from_slice(unsafe {
                std::slice::from_raw_parts(params.execAffinityParams, affinity_len)
            });
            params.execAffinityParams = std::ptr::null_mut();
        }
    }

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuCtxCreate_v4", "[#{}]", client.id);

        900923.send(&client.channel_sender).unwrap();
        has_params.send(&client.channel_sender).unwrap();
        if has_params {
            params.send(&client.channel_sender).unwrap();
            send_slice(&affinity_params, &client.channel_sender).unwrap();
        }
        flags.send(&client.channel_sender).unwrap();
        dev.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        unsafe { &mut *pctx }
            .recv(&client.channel_receiver)
            .unwrap();
        recv_cu_result("cuCtxCreate_v4", client.id, &client.channel_receiver)
    })
}

#[no_mangle]
extern "C" fn cuModuleLoad(module: *mut CUmodule, fname: *const c_char) -> CUresult {
    if module.is_null() || fname.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    let path = unsafe { CStr::from_ptr(fname) };
    let Ok(mut image) = std::fs::read(path.to_string_lossy().as_ref()) else {
        return CUresult::CUDA_ERROR_FILE_NOT_FOUND;
    };
    if image.last().copied() != Some(0) {
        image.push(0);
    }

    super::cuda_hijack::cuModuleLoadDataInternal(module, image.as_ptr().cast(), false)
}

#[no_mangle]
extern "C" fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult {
    super::cuda_hijack::cuModuleLoadDataInternal(module, image.cast(), false)
}

#[no_mangle]
extern "C" fn cuModuleLoadDataEx(
    module: *mut CUmodule,
    image: *const c_void,
    numOptions: c_uint,
    _options: *mut CUjit_option,
    _optionValues: *mut *mut c_void,
) -> CUresult {
    if module.is_null() || image.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }
    if numOptions != 0 {
        return CUresult::CUDA_ERROR_NOT_SUPPORTED;
    }

    super::cuda_hijack::cuModuleLoadDataInternal(module, image.cast(), false)
}

fn cache_module_function_metadata(hmod: CUmodule, hfunc: CUfunction, name_bytes: &[u8]) {
    if name_bytes.is_empty() {
        return;
    }
    let Ok(function_name) = std::str::from_utf8(name_bytes) else {
        return;
    };
    let Ok(c_name) = CString::new(name_bytes) else {
        return;
    };

    let target_arch = DRIVER_CACHE.read().unwrap().device_arch;
    let mut driver = DRIVER_CACHE.write().unwrap();
    let Some(image) = driver.images.get(&hmod) else {
        return;
    };
    let params = if FatBinaryHeader::is_fat_binary(image.as_ptr()) {
        let fatbin: &FatBinaryHeader = unsafe { &*image.as_ptr().cast() };
        fatbin.find_kernel_params(function_name, target_arch)
    } else {
        crate::elf::find_kernel_params_or_empty(image, function_name)
    };
    driver.function_params.insert(hfunc, params);
    driver.function_names.insert(hfunc, c_name);
}

#[no_mangle]
extern "C" fn cuModuleEnumerateFunctions(
    functions: *mut CUfunction,
    numFunctions: c_uint,
    mod_: CUmodule,
) -> CUresult {
    if numFunctions != 0 && functions.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    let capacity = numFunctions as usize;
    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuModuleEnumerateFunctions", "[#{}]", client.id);

        901022.send(&client.channel_sender).unwrap();
        numFunctions.send(&client.channel_sender).unwrap();
        mod_.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        let returned = recv_slice::<CUfunction, _>(&client.channel_receiver).unwrap();
        let mut names = Vec::with_capacity(returned.len());
        for _ in 0..returned.len() {
            names.push(recv_slice::<u8, _>(&client.channel_receiver).unwrap());
        }
        let result = recv_cu_result(
            "cuModuleEnumerateFunctions",
            client.id,
            &client.channel_receiver,
        );

        if returned.len() > capacity {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        }
        if result == CUresult::CUDA_SUCCESS {
            if !returned.is_empty() {
                unsafe {
                    std::ptr::copy_nonoverlapping(returned.as_ptr(), functions, returned.len());
                }
            }
            for (function, name) in returned.iter().copied().zip(names.iter()) {
                cache_module_function_metadata(mod_, function, name);
            }
        }
        result
    })
}

fn has_unsupported_library_options(numJitOptions: c_uint, numLibraryOptions: c_uint) -> bool {
    numJitOptions != 0 || numLibraryOptions != 0
}

#[no_mangle]
extern "C" fn cuLibraryLoadData(
    library: *mut CUlibrary,
    code: *const c_void,
    _jitOptions: *mut CUjit_option,
    _jitOptionsValues: *mut *mut c_void,
    numJitOptions: c_uint,
    _libraryOptions: *mut CUlibraryOption,
    _libraryOptionValues: *mut *mut c_void,
    numLibraryOptions: c_uint,
) -> CUresult {
    if library.is_null() || code.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }
    if has_unsupported_library_options(numJitOptions, numLibraryOptions) {
        return CUresult::CUDA_ERROR_NOT_SUPPORTED;
    }

    super::cuda_hijack::cuLibraryLoadDataInternal(library, code.cast())
}

#[no_mangle]
extern "C" fn cuLibraryLoadFromFile(
    library: *mut CUlibrary,
    fileName: *const c_char,
    _jitOptions: *mut CUjit_option,
    _jitOptionsValues: *mut *mut c_void,
    numJitOptions: c_uint,
    _libraryOptions: *mut CUlibraryOption,
    _libraryOptionValues: *mut *mut c_void,
    numLibraryOptions: c_uint,
) -> CUresult {
    if library.is_null() || fileName.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }
    if has_unsupported_library_options(numJitOptions, numLibraryOptions) {
        return CUresult::CUDA_ERROR_NOT_SUPPORTED;
    }

    let path = unsafe { CStr::from_ptr(fileName) };
    let Ok(mut image) = std::fs::read(path.to_string_lossy().as_ref()) else {
        return CUresult::CUDA_ERROR_FILE_NOT_FOUND;
    };
    if image.last().copied() != Some(0) {
        image.push(0);
    }

    super::cuda_hijack::cuLibraryLoadDataInternal(library, image.as_ptr().cast())
}

fn cache_library_kernel_metadata(library: CUlibrary, kernel: CUkernel, name_bytes: &[u8]) {
    if name_bytes.is_empty() {
        return;
    }
    let Ok(kernel_name) = std::str::from_utf8(name_bytes) else {
        return;
    };
    let Ok(c_name) = CString::new(name_bytes) else {
        return;
    };

    let target_arch = DRIVER_CACHE.read().unwrap().device_arch;
    let mut driver = DRIVER_CACHE.write().unwrap();
    let Some(image) = driver.library_images.get(&library) else {
        return;
    };
    let params = if FatBinaryHeader::is_fat_binary(image.as_ptr()) {
        let fatbin: &FatBinaryHeader = unsafe { &*image.as_ptr().cast() };
        fatbin.find_kernel_params(kernel_name, target_arch)
    } else {
        crate::elf::find_kernel_params_or_empty(image, kernel_name)
    };
    driver
        .function_params
        .insert(kernel.cast::<CUfunc_st>(), params);
    driver.kernel_names.insert(kernel, c_name);
    driver.kernel_libraries.insert(kernel, library);
}

#[no_mangle]
pub(super) extern "C" fn cuLibraryEnumerateKernels(
    kernels: *mut CUkernel,
    numKernels: c_uint,
    lib: CUlibrary,
) -> CUresult {
    if numKernels != 0 && kernels.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    let capacity = numKernels as usize;
    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cuLibraryEnumerateKernels", "[#{}]", client.id);

        901024.send(&client.channel_sender).unwrap();
        numKernels.send(&client.channel_sender).unwrap();
        lib.send(&client.channel_sender).unwrap();
        client.channel_sender.flush_out().unwrap();

        let returned = recv_slice::<CUkernel, _>(&client.channel_receiver).unwrap();
        let mut names = Vec::with_capacity(returned.len());
        for _ in 0..returned.len() {
            names.push(recv_slice::<u8, _>(&client.channel_receiver).unwrap());
        }
        let result = recv_cu_result(
            "cuLibraryEnumerateKernels",
            client.id,
            &client.channel_receiver,
        );

        if returned.len() > capacity {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        }
        if result == CUresult::CUDA_SUCCESS {
            if !returned.is_empty() {
                unsafe {
                    std::ptr::copy_nonoverlapping(returned.as_ptr(), kernels, returned.len());
                }
            }
            for (kernel, name) in returned.iter().copied().zip(names.iter()) {
                cache_library_kernel_metadata(lib, kernel, name);
            }
        }
        result
    })
}

#[no_mangle]
extern "C" fn cuKernelGetName(name: *mut *const c_char, hfunc: CUkernel) -> CUresult {
    if name.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    let driver = crate::DRIVER_CACHE.read().unwrap();
    let Some(kernel_name) = driver.kernel_names.get(&hfunc) else {
        unsafe {
            *name = std::ptr::null();
        }
        return CUresult::CUDA_ERROR_INVALID_HANDLE;
    };
    unsafe {
        *name = kernel_name.as_ptr();
    }
    CUresult::CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuFuncGetName(name: *mut *const c_char, hfunc: CUfunction) -> CUresult {
    if name.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    let driver = crate::DRIVER_CACHE.read().unwrap();
    let Some(function_name) = driver.function_names.get(&hfunc) else {
        unsafe {
            *name = std::ptr::null();
        }
        return CUresult::CUDA_ERROR_INVALID_HANDLE;
    };
    unsafe {
        *name = function_name.as_ptr();
    }
    CUresult::CUDA_SUCCESS
}

fn write_error_text(error: CUresult, pStr: *mut *const c_char) -> CUresult {
    if pStr.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    static ERROR_TEXTS: OnceLock<Mutex<BTreeMap<c_int, CString>>> = OnceLock::new();
    let error_code = error as c_int;
    let mut texts = ERROR_TEXTS
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .unwrap();
    let text = texts
        .entry(error_code)
        .or_insert_with(|| CString::new(format!("{error:?}")).unwrap());
    unsafe {
        *pStr = text.as_ptr();
    }
    CUresult::CUDA_SUCCESS
}

#[no_mangle]
extern "C" fn cuGetErrorName(error: CUresult, pStr: *mut *const c_char) -> CUresult {
    write_error_text(error, pStr)
}

#[no_mangle]
extern "C" fn cuGetErrorString(error: CUresult, pStr: *mut *const c_char) -> CUresult {
    write_error_text(error, pStr)
}

fn real_cuda_handle() -> *mut c_void {
    static HANDLE: OnceLock<usize> = OnceLock::new();
    let handle = *HANDLE.get_or_init(|| {
        let mut candidates = Vec::new();
        if let Ok(path) = std::env::var("GPU_REMOTING_REAL_CUDA") {
            candidates.push(path);
        }
        candidates.extend([
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1".to_string(),
            "/usr/local/cuda/compat/libcuda.so.1".to_string(),
        ]);

        let dlopen = crate::dl::original_dlopen();
        for path in &candidates {
            let c_path = CString::new(path.as_str()).unwrap();
            let handle = dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
            if !handle.is_null() {
                return handle as usize;
            }
        }
        panic!(
            "failed to load real libcuda for NVRTC; tried: {}",
            candidates.join(", ")
        );
    });
    handle as *mut c_void
}

#[no_mangle]
pub extern "C" fn cuGetExportTable(
    ppExportTable: *mut *const c_void,
    pExportTableId: *const CUuuid,
) -> CUresult {
    if !crate::dl::real_cuda_dlopen_active() {
        if ppExportTable.is_null() || pExportTableId.is_null() {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        }
        unsafe {
            *ppExportTable = std::ptr::null();
        }
        return CUresult::CUDA_ERROR_NOT_SUPPORTED;
    }
    type FnTy = extern "C" fn(*mut *const c_void, *const CUuuid) -> CUresult;
    static FN: OnceLock<usize> = OnceLock::new();
    let ptr = *FN.get_or_init(|| crate::dl::dlsym_handle(real_cuda_handle(), "cuGetExportTable"));
    let func: FnTy = unsafe { std::mem::transmute(ptr) };
    func(ppExportTable, pExportTableId)
}

#[no_mangle]
extern "C" fn cuGetProcAddress_v2(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    _cudaVersion: c_int,
    _flags: cuuint64_t,
    symbolStatus: *mut CUdriverProcAddressQueryResult,
) -> CUresult {
    if symbol.is_null() || pfn.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    let func = unsafe { libc::dlsym(libc::RTLD_DEFAULT, symbol) };
    unsafe {
        if func.is_null() {
            *pfn = std::ptr::null_mut();
            if !symbolStatus.is_null() {
                *symbolStatus =
                    CUdriverProcAddressQueryResult::CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
            }
            log::debug!(
                target: "cuGetProcAddress_v2",
                "symbol not found: {:?}",
                CStr::from_ptr(symbol),
            );
            return CUresult::CUDA_ERROR_NOT_FOUND;
        }

        *pfn = func;
        if !symbolStatus.is_null() {
            *symbolStatus = CUdriverProcAddressQueryResult::CU_GET_PROC_ADDRESS_SUCCESS;
        }
    }

    CUresult::CUDA_SUCCESS
}
