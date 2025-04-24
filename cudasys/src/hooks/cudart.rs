use crate::types::cudart::*;
use codegen::{cuda_custom_hook, cuda_hook};
use std::os::raw::*;

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

#[cuda_hook(proc_id = 152)]
fn cudaGetLastError() -> cudaError_t;

#[cuda_hook(proc_id = 153)]
fn cudaPeekAtLastError() -> cudaError_t;

#[cuda_hook(proc_id = 178)]
fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;

#[cuda_hook(proc_id = 265)]
fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> cudaError_t;

#[cuda_custom_hook(proc_id = 278)]
fn cudaMemcpy(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_custom_hook] // calls cudaMemcpy (wrong implementation)
fn cudaMemcpyAsync(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 253, async_api)]
fn cudaFree(#[device] devPtr: *mut c_void) -> cudaError_t;

#[cuda_hook(proc_id = 175)]
fn cudaStreamIsCapturing(
    stream: cudaStream_t,
    pCaptureStatus: *mut cudaStreamCaptureStatus,
) -> cudaError_t;

// This function is hidden and superseded by `cudaGetDeviceProperties_v2` in CUDA 12.
// The change is that `cudaDeviceProp` grew bigger. We don't hook it in CUDA 12
// to prevent reading or writing past the end of allocated memory when sending or receiving data.
#[cuda_hook(proc_id = 123, max_cuda_version = 11)]
fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: c_int) -> cudaError_t;

#[cuda_hook(proc_id = 400)]
fn cudaPointerGetAttributes(
    attributes: *mut cudaPointerAttributes,
    #[device] ptr: *const c_void,
) -> cudaError_t;

#[cuda_custom_hook] // local
fn cudaHostAlloc(pHost: *mut *mut c_void, size: usize, flags: c_uint) -> cudaError_t;

#[cuda_custom_hook] // calls driver API
fn cudaFuncGetAttributes(attr: *mut cudaFuncAttributes, func: *const c_void) -> cudaError_t;

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

#[cuda_custom_hook(proc_id = 151)]
fn cudaGetErrorString(error: cudaError_t) -> *const c_char;

#[cuda_hook(proc_id = 274)]
fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;

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

#[cuda_hook(proc_id = 999123, min_cuda_version = 12)]
fn cudaGetDeviceProperties_v2(prop: *mut cudaDeviceProp, device: c_int) -> cudaError_t;

#[cuda_hook(proc_id = 167)]
fn cudaStreamCreateWithPriority(
    pStream: *mut cudaStream_t,
    flags: c_uint,
    priority: c_int,
) -> cudaError_t;

#[cuda_hook(proc_id = 201)]
fn cudaEventCreateWithFlags(event: *mut cudaEvent_t, flags: c_uint) -> cudaError_t;

#[cuda_hook(proc_id = 205)]
fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;

#[cuda_hook(proc_id = 180)]
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

#[cuda_hook(proc_id = 204)]
fn cudaEventQuery(event: cudaEvent_t) -> cudaError_t;

#[cuda_hook(proc_id = 119)]
fn cudaDeviceSynchronize() -> cudaError_t;

#[cuda_custom_hook] // calls driver API
fn cudaFuncSetAttribute(func: *const c_void, attr: cudaFuncAttribute, value: c_int) -> cudaError_t;

#[cuda_hook(proc_id = 203)]
fn cudaEventElapsedTime(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> cudaError_t;

#[cuda_hook(proc_id = 412)]
fn cudaDeviceEnablePeerAccess(peerDevice: c_int, flags: c_uint) -> cudaError_t;

#[cuda_hook(proc_id = 297, async_api)]
fn cudaMemset(#[device] devPtr: *mut c_void, value: c_int, count: usize) -> cudaError_t;

#[cuda_hook(proc_id = 114)]
fn cudaDeviceReset() -> cudaError_t;
