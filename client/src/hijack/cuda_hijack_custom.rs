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
extern "C" fn cuGetExportTable(
    ppExportTable: *mut *const c_void,
    pExportTableId: *const CUuuid,
) -> CUresult {
    if !crate::dl::real_cuda_dlopen_active() {
        unimplemented!("cuGetExportTable")
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
