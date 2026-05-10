use crate::types::cudart::*;
use codegen::{cuda_custom_hook, cuda_hook};
use std::os::raw::*;

// Hacked via type alias because imports are rewritten in generated code.
type CUfunction = crate::types::cuda::CUfunction;

#[cuda_hook(proc_id = 120)]
fn cudaGetDevice(device: *mut c_int) -> cudaError_t {
    'client_before_send: {
        if let (true, Some(val)) = (client.opt_local, client.cuda_device) {
            unsafe {
                *device = val;
            }
            return cudaError_t::cudaSuccess;
        }
    }
    'client_after_recv: {
        {
            client.cuda_device = Some(*device);
        }
    }
}

#[cuda_hook(proc_id = 129)]
fn cudaSetDevice(device: c_int) -> cudaError_t {
    'client_after_recv: {
        if result == Default::default() {
            client.cuda_device = Some(device);
        } else {
            client.cuda_device = None;
        }
    }
}

#[cuda_hook(proc_id = 121)]
fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t;

#[cuda_hook(proc_id = 900350)]
fn cudaDeviceCanAccessPeer(
    canAccessPeer: *mut c_int,
    device: c_int,
    peerDevice: c_int,
) -> cudaError_t;

#[cuda_hook(proc_id = 900447)]
fn cudaDeviceGetP2PAttribute(
    value: *mut c_int,
    attr: cudaDeviceP2PAttr,
    srcDevice: c_int,
    dstDevice: c_int,
) -> cudaError_t;

#[cuda_hook(proc_id = 900448)]
fn cudaDeviceGetP2PAtomicCapabilities(
    #[host(output, len = count as usize)] capabilities: *mut c_uint,
    #[host(len = count as usize)] operations: *const cudaAtomicOperation,
    count: c_uint,
    srcDevice: c_int,
    dstDevice: c_int,
) -> cudaError_t;

#[cuda_hook(proc_id = 900440)]
fn cudaDeviceGetDefaultMemPool(memPool: *mut cudaMemPool_t, device: c_int) -> cudaError_t;

#[cuda_hook(proc_id = 900441)]
fn cudaDeviceGetMemPool(memPool: *mut cudaMemPool_t, device: c_int) -> cudaError_t;

#[cuda_hook(proc_id = 900442, async_api = false)]
fn cudaDeviceSetMemPool(device: c_int, memPool: cudaMemPool_t) -> cudaError_t;

#[cuda_hook(proc_id = 900351)]
fn cudaDeviceGetByPCIBusId(device: *mut c_int, pciBusId: *const c_char) -> cudaError_t;

#[cuda_hook(proc_id = 900352)]
fn cudaDeviceGetCacheConfig(pCacheConfig: *mut cudaFuncCache) -> cudaError_t;

#[cuda_hook(proc_id = 900353)]
fn cudaDeviceGetLimit(pValue: *mut usize, limit: cudaLimit) -> cudaError_t;

#[cuda_hook(proc_id = 900354)]
fn cudaDeviceGetPCIBusId(
    #[host(output, len = len)] pciBusId: *mut c_char,
    len: c_int,
    device: c_int,
) -> cudaError_t;

#[cuda_hook(proc_id = 900355)]
fn cudaDeviceGetSharedMemConfig(pConfig: *mut cudaSharedMemConfig) -> cudaError_t;

#[cuda_hook(proc_id = 900356, async_api = false)]
fn cudaDeviceSetCacheConfig(cacheConfig: cudaFuncCache) -> cudaError_t;

#[cuda_hook(proc_id = 900357, async_api = false)]
fn cudaDeviceSetLimit(limit: cudaLimit, value: usize) -> cudaError_t;

#[cuda_hook(proc_id = 900358, async_api = false)]
fn cudaDeviceSetSharedMemConfig(config: cudaSharedMemConfig) -> cudaError_t;

#[cuda_hook(proc_id = 152, async_api = false)]
fn cudaGetLastError() -> cudaError_t;

#[cuda_hook(proc_id = 153, async_api = false)]
fn cudaPeekAtLastError() -> cudaError_t;

#[cuda_hook(proc_id = 178, async_api = false)]
fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;

#[cuda_hook(proc_id = 900100)]
fn cudaStreamCreate(pStream: *mut cudaStream_t) -> cudaError_t;

#[cuda_hook(proc_id = 900422)]
fn cudaStreamCreateWithFlags(pStream: *mut cudaStream_t, flags: c_uint) -> cudaError_t;

#[cuda_hook(proc_id = 900101, async_api = false)]
fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;

#[cuda_hook(proc_id = 900423)]
fn cudaStreamGetFlags(hStream: cudaStream_t, flags: *mut c_uint) -> cudaError_t;

#[cuda_hook(proc_id = 900424)]
fn cudaStreamGetPriority(hStream: cudaStream_t, priority: *mut c_int) -> cudaError_t;

#[cuda_hook(proc_id = 900429)]
fn cudaStreamGetId(hStream: cudaStream_t, streamId: *mut c_ulonglong) -> cudaError_t;

#[cuda_hook(proc_id = 900430)]
fn cudaStreamGetDevice(hStream: cudaStream_t, device: *mut c_int) -> cudaError_t;

#[cuda_hook(proc_id = 900431)]
fn cudaStreamCopyAttributes(dst: cudaStream_t, src: cudaStream_t) -> cudaError_t;

#[cuda_hook(proc_id = 900425, async_api = false)]
fn cudaStreamQuery(stream: cudaStream_t) -> cudaError_t;

#[cuda_hook(proc_id = 265)]
fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> cudaError_t;

#[cuda_hook(proc_id = 900420)]
fn cudaMallocAsync(devPtr: *mut *mut c_void, size: usize, hStream: cudaStream_t) -> cudaError_t;

#[cuda_hook(proc_id = 900102)]
fn cudaMallocPitch(
    devPtr: *mut *mut c_void,
    pitch: *mut usize,
    width: usize,
    height: usize,
) -> cudaError_t;

#[cuda_custom_hook] // calls one of the following internal APIs
fn cudaMemcpy(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_hook(proc_id = 320, async_api, parent = cudaMemcpy)]
fn cudaMemcpyHtod(
    #[device] dst: *mut c_void,
    #[host(len = count)] src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_hook(proc_id = 321, parent = cudaMemcpy)]
fn cudaMemcpyDtoh(
    #[host(output, len = count)] dst: *mut c_void,
    #[device] src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_hook(proc_id = 322, async_api, parent = cudaMemcpy)]
fn cudaMemcpyDtod(
    #[device] dst: *mut c_void,
    #[device] src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_custom_hook] // calls one of the following internal APIs
fn cudaMemcpyAsync(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 323, async_api, parent = cudaMemcpyAsync)]
fn cudaMemcpyAsyncHtod(
    #[device] dst: *mut c_void,
    #[host(len = count)] src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    'server_execution: {
        // FIXME: can't async because server deallocates memory after calling
        let result = unsafe {
            assert_eq!(cudaStreamSynchronize(stream), Default::default());
            cudaMemcpy(dst, src__ptr.cast(), count, kind)
        };
    }
}

#[cuda_hook(proc_id = 324, parent = cudaMemcpyAsync)]
fn cudaMemcpyAsyncDtoh(
    #[host(output, len = count)] dst: *mut c_void,
    #[device] src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    'server_execution: {
        // FIXME: can't async because server can't send data back async
        let result = unsafe {
            assert_eq!(cudaStreamSynchronize(stream), Default::default());
            cudaMemcpy(dst__ptr.cast(), src, count, kind)
        };
    }
}

#[cuda_hook(proc_id = 325, async_api, parent = cudaMemcpyAsync)]
fn cudaMemcpyAsyncDtod(
    #[device] dst: *mut c_void,
    #[device] src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_custom_hook] // calls cudaMemcpy row by row
fn cudaMemcpy2D(
    dst: *mut c_void,
    dpitch: usize,
    src: *const c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_custom_hook] // calls cudaMemcpyAsync row by row
fn cudaMemcpy2DAsync(
    dst: *mut c_void,
    dpitch: usize,
    src: *const c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 253, async_api)]
fn cudaFree(#[device] devPtr: *mut c_void) -> cudaError_t;

#[cuda_hook(proc_id = 900421, async_api = false)]
fn cudaFreeAsync(#[device] devPtr: *mut c_void, hStream: cudaStream_t) -> cudaError_t;

#[cuda_hook(proc_id = 175)]
fn cudaStreamIsCapturing(
    stream: cudaStream_t,
    pCaptureStatus: *mut cudaStreamCaptureStatus,
) -> cudaError_t;

#[cuda_hook(proc_id = 123)]
fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: c_int) -> cudaError_t;

#[cuda_hook(proc_id = 400)]
fn cudaPointerGetAttributes(
    attributes: *mut cudaPointerAttributes,
    #[device] ptr: *const c_void,
) -> cudaError_t;

#[cuda_custom_hook] // local
fn cudaHostAlloc(pHost: *mut *mut c_void, size: usize, flags: c_uint) -> cudaError_t;

#[cuda_custom_hook] // local
fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> cudaError_t;

#[cuda_custom_hook] // local
fn cudaFreeHost(ptr: *mut c_void) -> cudaError_t;

#[cuda_custom_hook] // local
fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: c_uint) -> cudaError_t;

#[cuda_custom_hook] // local
fn cudaHostUnregister(ptr: *mut c_void) -> cudaError_t;

#[cuda_custom_hook] // calls the internal API below
fn cudaFuncGetAttributes(attr: *mut cudaFuncAttributes, func: *const c_void) -> cudaError_t;

#[cuda_hook(proc_id = 230, parent = cudaFuncGetAttributes)]
fn cudaFuncGetAttributesInternal(attr: *mut cudaFuncAttributes, func: CUfunction) -> cudaError_t {
    'server_execution: {
        unsafe { attr__ptr.write_bytes(0u8, 1) };
        let result = super::cuda_exe_utils::cu_func_get_attributes(attr__ptr, func);
    }
}

#[cuda_custom_hook] // local
fn __cudaRegisterFatBinary(fatCubin: *mut c_void) -> *mut *mut c_void;

#[cuda_custom_hook] // local
fn __cudaRegisterFatBinaryEnd(fatCubinHandle: *mut *mut c_void);

#[cuda_custom_hook] // local
fn __cudaUnregisterFatBinary(fatCubinHandle: *mut *mut c_void);

#[cuda_custom_hook] // local
fn __cudaRegisterFunction(
    fatCubinHandle: *mut *mut c_void,
    hostFun: *const c_char,
    deviceFun: *mut c_char,
    deviceName: *const c_char,
    thread_limit: c_int,
    tid: *mut uint3,
    bid: *mut uint3,
    bDim: *mut dim3,
    gDim: *mut dim3,
    wSize: *mut c_int,
);

#[cuda_custom_hook] // local
fn __cudaRegisterVar(
    fatCubinHandle: *mut *mut c_void,
    hostVar: *mut c_char,
    deviceAddress: *mut c_char,
    deviceName: *const c_char,
    ext: c_int,
    size: usize,
    constant: c_int,
    global: c_int,
);

#[cuda_custom_hook] // calls driver API
fn cudaLaunchKernel(
    func: *const c_void,
    gridDim: dim3,
    blockDim: dim3,
    args: *mut *mut c_void,
    sharedMem: usize,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 112)]
fn cudaDeviceGetStreamPriorityRange(
    leastPriority: *mut c_int,
    greatestPriority: *mut c_int,
) -> cudaError_t;

#[cuda_hook(proc_id = 302, async_api)]
fn cudaMemsetAsync(
    #[device] devPtr: *mut c_void,
    value: c_int,
    count: usize,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_custom_hook] // local
fn cudaGetErrorString(error: cudaError_t) -> *const c_char;

#[cuda_custom_hook] // local
fn cudaGetErrorName(error: cudaError_t) -> *const c_char;

#[cuda_hook(proc_id = 274)]
fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;

#[cuda_hook(proc_id = 900443, async_api = false)]
fn cudaMemPoolTrimTo(memPool: cudaMemPool_t, minBytesToKeep: usize) -> cudaError_t;

#[cuda_hook(proc_id = 900444)]
fn cudaMemPoolGetAttribute(
    memPool: cudaMemPool_t,
    attr: cudaMemPoolAttr,
    #[host(output, len = attr.data_size())] value: *mut c_void,
) -> cudaError_t;

#[cuda_hook(proc_id = 900445)]
fn cudaMemPoolSetAttribute(
    memPool: cudaMemPool_t,
    attr: cudaMemPoolAttr,
    #[host(input, len = attr.data_size())] value: *mut c_void,
) -> cudaError_t;

#[cuda_hook(proc_id = 900446)]
fn cudaMallocFromPoolAsync(
    ptr: *mut *mut c_void,
    size: usize,
    memPool: cudaMemPool_t,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_custom_hook] // local
fn __cudaPushCallConfiguration(
    gridDim: dim3,
    blockDim: dim3,
    sharedMem: usize,
    stream: *mut CUstream_st,
) -> c_uint;

#[cuda_custom_hook] // local
fn __cudaPopCallConfiguration(
    gridDim: *mut dim3,
    blockDim: *mut dim3,
    sharedMem: *mut usize,
    stream: *mut c_void,
) -> cudaError_t;

#[cuda_hook(proc_id = 167)]
fn cudaStreamCreateWithPriority(
    pStream: *mut cudaStream_t,
    flags: c_uint,
    priority: c_int,
) -> cudaError_t;

#[cuda_hook(proc_id = 201)]
fn cudaEventCreateWithFlags(event: *mut cudaEvent_t, flags: c_uint) -> cudaError_t;

#[cuda_hook(proc_id = 900103)]
fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t;

#[cuda_hook(proc_id = 205)]
fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;

#[cuda_hook(proc_id = 900426)]
fn cudaEventRecordWithFlags(event: cudaEvent_t, stream: cudaStream_t, flags: c_uint)
    -> cudaError_t;

#[cuda_hook(proc_id = 900104, async_api = false)]
fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t;

#[cuda_hook(proc_id = 180, async_api = false)]
fn cudaStreamWaitEvent(stream: cudaStream_t, event: cudaEvent_t, flags: c_uint) -> cudaError_t;

#[cuda_hook(proc_id = 202)]
fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t;

#[cuda_hook(proc_id = 102)]
fn cudaDeviceGetAttribute(value: *mut c_int, attr: cudaDeviceAttr, device: c_int) -> cudaError_t;

#[cuda_custom_hook] // calls driver API
fn cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    numBlocks: *mut c_int,
    func: *const c_void,
    blockSize: c_int,
    dynamicSMemSize: usize,
    flags: c_uint,
) -> cudaError_t;

#[cuda_hook(proc_id = 126)]
fn cudaIpcGetMemHandle(
    handle: *mut cudaIpcMemHandle_t,
    #[device] devPtr: *mut c_void,
) -> cudaError_t;

#[cuda_hook(proc_id = 128)]
fn cudaIpcOpenMemHandle(
    devPtr: *mut *mut c_void,
    handle: cudaIpcMemHandle_t,
    flags: c_uint,
) -> cudaError_t;

#[cuda_hook(proc_id = 204, async_api = false)]
fn cudaEventQuery(event: cudaEvent_t) -> cudaError_t;

#[cuda_hook(proc_id = 119, async_api = false)]
fn cudaDeviceSynchronize() -> cudaError_t;

#[cuda_custom_hook] // calls driver API
fn cudaFuncSetAttribute(func: *const c_void, attr: cudaFuncAttribute, value: c_int) -> cudaError_t;

#[cuda_hook(proc_id = 203)]
fn cudaEventElapsedTime(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> cudaError_t;

#[cuda_hook(proc_id = 412, async_api = false)]
fn cudaDeviceEnablePeerAccess(peerDevice: c_int, flags: c_uint) -> cudaError_t;

#[cuda_hook(proc_id = 297, async_api)]
fn cudaMemset(#[device] devPtr: *mut c_void, value: c_int, count: usize) -> cudaError_t;

#[cuda_hook(proc_id = 114, async_api = false)]
fn cudaDeviceReset() -> cudaError_t;

#[cuda_hook(proc_id = 900427, async_api = false)]
fn cudaCtxResetPersistingL2Cache() -> cudaError_t;

#[cuda_hook(proc_id = 163, async_api = false)]
fn cudaStreamBeginCapture(stream: cudaStream_t, mode: cudaStreamCaptureMode) -> cudaError_t;

#[cuda_hook(proc_id = 169)]
fn cudaStreamEndCapture(stream: cudaStream_t, pGraph: *mut cudaGraph_t) -> cudaError_t;

#[cuda_hook(proc_id = 172)]
fn cudaStreamGetCaptureInfo(
    stream: cudaStream_t,
    captureStatus_out: *mut cudaStreamCaptureStatus,
    id_out: *mut c_ulonglong,
    #[device] graph_out: *mut cudaGraph_t, // null
    #[device] dependencies_out: *mut *const cudaGraphNode_t, // null
    #[device] edgeData_out: *mut *const cudaGraphEdgeData, // null
    #[device] numDependencies_out: *mut usize, // null
) -> cudaError_t {
    'client_before_send: {
        assert!(!id_out.is_null());
        assert!(graph_out.is_null());
        assert!(dependencies_out.is_null());
        assert!(edgeData_out.is_null());
        assert!(numDependencies_out.is_null());
    }
}

// We use hooks to implement the inout parameter `mode` for now.
#[cuda_hook(proc_id = 181)]
fn cudaThreadExchangeStreamCaptureMode(mode: *mut cudaStreamCaptureMode) -> cudaError_t {
    'client_extra_send: {
        unsafe { *mode }.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut mode_in = cudaStreamCaptureMode::cudaStreamCaptureModeGlobal;
        mode_in.recv(channel_receiver).unwrap();
    }
    'server_before_execution: {
        mode.write(mode_in);
    }
}

#[cuda_hook(proc_id = 510)]
fn cudaDriverGetVersion(driverVersion: *mut c_int) -> cudaError_t;

#[cuda_hook(proc_id = 900359)]
fn cudaGetDeviceFlags(flags: *mut c_uint) -> cudaError_t;

#[cuda_hook(proc_id = 900360)]
fn cudaRuntimeGetVersion(runtimeVersion: *mut c_int) -> cudaError_t;

#[cuda_hook(proc_id = 538, async_api = false)]
fn cudaGraphDestroy(graph: cudaGraph_t) -> cudaError_t;

#[cuda_hook(proc_id = 545, async_api = false)]
fn cudaGraphExecDestroy(graphExec: cudaGraphExec_t) -> cudaError_t;

#[cuda_hook(proc_id = 563)]
fn cudaGraphGetNodes(
    graph: cudaGraph_t,
    #[device] nodes: *mut cudaGraphNode_t, // null
    numNodes: *mut usize,
) -> cudaError_t {
    'client_before_send: {
        assert!(nodes.is_null());
    }
}

#[cuda_hook(proc_id = 567, min_cuda_version = 12)]
fn cudaGraphInstantiate(
    pGraphExec: *mut cudaGraphExec_t,
    graph: cudaGraph_t,
    flags: c_ulonglong,
) -> cudaError_t;

#[cuda_hook(proc_id = 999567, min_cuda_version = 12)]
fn cudaGraphInstantiateWithFlags(
    pGraphExec: *mut cudaGraphExec_t,
    graph: cudaGraph_t,
    flags: c_ulonglong,
) -> cudaError_t;

#[cuda_hook(proc_id = 573, async_api)]
fn cudaGraphLaunch(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t;
