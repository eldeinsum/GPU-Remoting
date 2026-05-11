use crate::types::cudart::*;
use codegen::{cuda_custom_hook, cuda_hook};
use std::os::raw::*;

// Hacked via type alias because imports are rewritten in generated code.
type CUfunction = crate::types::cuda::CUfunction;

#[cuda_hook(proc_id = 900468)]
fn cudaMallocArray(
    array: *mut cudaArray_t,
    #[host(len = 1)] desc: *const cudaChannelFormatDesc,
    width: usize,
    height: usize,
    flags: c_uint,
) -> cudaError_t;

#[cuda_hook(proc_id = 900469)]
fn cudaArrayGetInfo(
    desc: *mut cudaChannelFormatDesc,
    extent: *mut cudaExtent,
    flags: *mut c_uint,
    array: cudaArray_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 900578)]
fn cudaArrayGetPlane(
    pPlaneArray: *mut cudaArray_t,
    hArray: cudaArray_t,
    planeIdx: c_uint,
) -> cudaError_t;

#[cuda_hook(proc_id = 900481)]
fn cudaGetChannelDesc(desc: *mut cudaChannelFormatDesc, array: cudaArray_const_t) -> cudaError_t;

#[cuda_hook(proc_id = 900470, async_api = false)]
fn cudaFreeArray(array: cudaArray_t) -> cudaError_t;

#[cuda_hook(proc_id = 900569)]
fn cudaMallocMipmappedArray(
    mipmappedArray: *mut cudaMipmappedArray_t,
    #[host(len = 1)] desc: *const cudaChannelFormatDesc,
    extent: cudaExtent,
    numLevels: c_uint,
    flags: c_uint,
) -> cudaError_t;

#[cuda_hook(proc_id = 900570)]
fn cudaGetMipmappedArrayLevel(
    levelArray: *mut cudaArray_t,
    mipmappedArray: cudaMipmappedArray_const_t,
    level: c_uint,
) -> cudaError_t;

#[cuda_hook(proc_id = 900571, async_api = false)]
fn cudaFreeMipmappedArray(mipmappedArray: cudaMipmappedArray_t) -> cudaError_t;

#[cuda_hook(proc_id = 900572)]
fn cudaArrayGetMemoryRequirements(
    memoryRequirements: *mut cudaArrayMemoryRequirements,
    array: cudaArray_t,
    device: c_int,
) -> cudaError_t;

#[cuda_hook(proc_id = 900573)]
fn cudaMipmappedArrayGetMemoryRequirements(
    memoryRequirements: *mut cudaArrayMemoryRequirements,
    mipmap: cudaMipmappedArray_t,
    device: c_int,
) -> cudaError_t;

#[cuda_hook(proc_id = 900574)]
fn cudaArrayGetSparseProperties(
    sparseProperties: *mut cudaArraySparseProperties,
    array: cudaArray_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 900575)]
fn cudaMipmappedArrayGetSparseProperties(
    sparseProperties: *mut cudaArraySparseProperties,
    mipmap: cudaMipmappedArray_t,
) -> cudaError_t;

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
            client.cuda_device_init = true;
        } else {
            client.cuda_device = None;
        }
    }
}

#[cuda_hook(proc_id = 121)]
fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t;

#[cuda_hook(proc_id = 900457)]
fn cudaChooseDevice(
    device: *mut c_int,
    #[host(len = 1)] prop: *const cudaDeviceProp,
) -> cudaError_t;

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

#[cuda_hook(proc_id = 900921, async_api = false)]
fn cudaDeviceFlushGPUDirectRDMAWrites(
    target: cudaFlushGPUDirectRDMAWritesTarget,
    scope: cudaFlushGPUDirectRDMAWritesScope,
) -> cudaError_t;

#[cuda_hook(proc_id = 900448)]
fn cudaDeviceGetP2PAtomicCapabilities(
    #[host(output, len = count as usize)] capabilities: *mut c_uint,
    #[host(len = count as usize)] operations: *const cudaAtomicOperation,
    count: c_uint,
    srcDevice: c_int,
    dstDevice: c_int,
) -> cudaError_t;

#[cuda_hook(proc_id = 900458, async_api = false)]
fn cudaDeviceDisablePeerAccess(peerDevice: c_int) -> cudaError_t;

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

#[cuda_hook(proc_id = 900459)]
fn cudaDeviceGetHostAtomicCapabilities(
    #[host(output, len = count as usize)] capabilities: *mut c_uint,
    #[host(len = count as usize)] operations: *const cudaAtomicOperation,
    count: c_uint,
    device: c_int,
) -> cudaError_t;

#[cuda_hook(proc_id = 900893)]
fn cudaDeviceGetDevResource(
    device: c_int,
    #[host(output, len = 1)] resource: *mut cudaDevResource,
    type_: cudaDevResourceType,
) -> cudaError_t;

#[cuda_hook(proc_id = 900940)]
fn cudaDevResourceGenerateDesc(
    phDesc: *mut cudaDevResourceDesc_t,
    #[host(input, len = nbResources as usize)] resources: *mut cudaDevResource,
    nbResources: c_uint,
) -> cudaError_t {
    'client_before_send: {
        assert!(!resources.is_null());
        let resource_count = nbResources as usize;
        let resource_slice = unsafe { std::slice::from_raw_parts(resources, resource_count) };
        assert!(
            resource_slice
                .iter()
                .all(|resource| resource.nextResource.is_null())
        );
    }
}

#[cuda_hook(proc_id = 900945)]
fn cudaDevSmResourceSplitByCount(
    #[skip] result_out: *mut cudaDevResource,
    #[skip] nbGroups: *mut c_uint,
    #[skip] input: *const cudaDevResource,
    #[skip] remaining: *mut cudaDevResource,
    flags: c_uint,
    minCount: c_uint,
) -> cudaError_t {
    'client_before_send: {
        assert!(!nbGroups.is_null());
        assert!(!input.is_null());
        let requested_groups = unsafe { *nbGroups };
        let has_result = !result_out.is_null();
        let result_capacity = if has_result {
            assert!(requested_groups > 0);
            requested_groups as usize
        } else {
            0
        };
        let input_resource = unsafe { *input };
        assert!(input_resource.nextResource.is_null());
        let has_remaining = !remaining.is_null();
    }
    'client_extra_send: {
        has_result.send(channel_sender).unwrap();
        requested_groups.send(channel_sender).unwrap();
        input_resource.send(channel_sender).unwrap();
        has_remaining.send(channel_sender).unwrap();
    }
    'client_after_recv: {
        let mut returned_groups = 0 as c_uint;
        returned_groups.recv(channel_receiver).unwrap();
        unsafe {
            *nbGroups = returned_groups;
        }
        if result == cudaError_t::cudaSuccess {
            if has_result {
                let result_resources = recv_slice::<cudaDevResource, _>(channel_receiver).unwrap();
                assert!(result_resources.len() <= result_capacity);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        result_resources.as_ptr(),
                        result_out,
                        result_resources.len(),
                    );
                }
            }
            if has_remaining {
                let mut remaining_resource = std::mem::MaybeUninit::<cudaDevResource>::uninit();
                remaining_resource.recv(channel_receiver).unwrap();
                unsafe {
                    std::ptr::write(remaining, remaining_resource.assume_init());
                }
            }
        }
    }
    'server_extra_recv: {
        let mut has_result = false;
        has_result.recv(channel_receiver).unwrap();
        let mut requested_groups = 0 as c_uint;
        requested_groups.recv(channel_receiver).unwrap();
        let mut input_resource = std::mem::MaybeUninit::<cudaDevResource>::uninit();
        input_resource.recv(channel_receiver).unwrap();
        let input_resource = unsafe { input_resource.assume_init() };
        let mut has_remaining = false;
        has_remaining.recv(channel_receiver).unwrap();
    }
    'server_execution: {
        let mut returned_groups = requested_groups;
        let mut result_resources = if has_result {
            unsafe { vec![std::mem::zeroed::<cudaDevResource>(); requested_groups as usize] }
        } else {
            Vec::new()
        };
        let mut remaining_resource = unsafe { std::mem::zeroed::<cudaDevResource>() };
        let result_ptr = if has_result {
            result_resources.as_mut_ptr()
        } else {
            std::ptr::null_mut()
        };
        let remaining_ptr = if has_remaining {
            &raw mut remaining_resource
        } else {
            std::ptr::null_mut()
        };
        let result = unsafe {
            cudasys::cudart::cudaDevSmResourceSplitByCount(
                result_ptr,
                &mut returned_groups,
                &raw const input_resource,
                remaining_ptr,
                flags,
                minCount,
            )
        };
        if result == cudaError_t::cudaSuccess && has_result {
            assert!(returned_groups as usize <= result_resources.len());
        }
    }
    'server_after_send: {
        returned_groups.send(channel_sender).unwrap();
        if result == cudaError_t::cudaSuccess {
            if has_result {
                send_slice(
                    &result_resources[..returned_groups as usize],
                    channel_sender,
                )
                .unwrap();
            }
            if has_remaining {
                remaining_resource.send(channel_sender).unwrap();
            }
        }
        channel_sender.flush_out().unwrap();
    }
}

#[cuda_hook(proc_id = 900946)]
fn cudaDevSmResourceSplit(
    #[skip] result_out: *mut cudaDevResource,
    nbGroups: c_uint,
    #[skip] input: *const cudaDevResource,
    #[skip] remainder: *mut cudaDevResource,
    flags: c_uint,
    #[skip] groupParams: *mut cudaDevSmResourceGroupParams,
) -> cudaError_t {
    'client_before_send: {
        assert!(!input.is_null());
        assert!(!groupParams.is_null());
        let has_result = !result_out.is_null();
        let result_capacity = if has_result { nbGroups as usize } else { 0 };
        let input_resource = unsafe { *input };
        assert!(input_resource.nextResource.is_null());
        let has_remainder = !remainder.is_null();
        let group_params = unsafe { std::slice::from_raw_parts(groupParams, nbGroups as usize) };
    }
    'client_extra_send: {
        has_result.send(channel_sender).unwrap();
        input_resource.send(channel_sender).unwrap();
        has_remainder.send(channel_sender).unwrap();
        send_slice(group_params, channel_sender).unwrap();
    }
    'client_after_recv: {
        if result == cudaError_t::cudaSuccess {
            if has_result {
                let result_resources = recv_slice::<cudaDevResource, _>(channel_receiver).unwrap();
                assert!(result_resources.len() <= result_capacity);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        result_resources.as_ptr(),
                        result_out,
                        result_resources.len(),
                    );
                }
            }
            let returned_group_params =
                recv_slice::<cudaDevSmResourceGroupParams, _>(channel_receiver).unwrap();
            assert_eq!(returned_group_params.len(), nbGroups as usize);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    returned_group_params.as_ptr(),
                    groupParams,
                    returned_group_params.len(),
                );
            }
            if has_remainder {
                let mut remainder_resource = std::mem::MaybeUninit::<cudaDevResource>::uninit();
                remainder_resource.recv(channel_receiver).unwrap();
                unsafe {
                    std::ptr::write(remainder, remainder_resource.assume_init());
                }
            }
        }
    }
    'server_extra_recv: {
        let mut has_result = false;
        has_result.recv(channel_receiver).unwrap();
        let mut input_resource = std::mem::MaybeUninit::<cudaDevResource>::uninit();
        input_resource.recv(channel_receiver).unwrap();
        let input_resource = unsafe { input_resource.assume_init() };
        let mut has_remainder = false;
        has_remainder.recv(channel_receiver).unwrap();
        let mut group_params =
            recv_slice::<cudaDevSmResourceGroupParams, _>(channel_receiver).unwrap();
    }
    'server_execution: {
        let mut result_resources = if has_result {
            unsafe { vec![std::mem::zeroed::<cudaDevResource>(); nbGroups as usize] }
        } else {
            Vec::new()
        };
        let mut remainder_resource = unsafe { std::mem::zeroed::<cudaDevResource>() };
        let result_ptr = if has_result {
            result_resources.as_mut_ptr()
        } else {
            std::ptr::null_mut()
        };
        let remainder_ptr = if has_remainder {
            &raw mut remainder_resource
        } else {
            std::ptr::null_mut()
        };
        let result = unsafe {
            cudasys::cudart::cudaDevSmResourceSplit(
                result_ptr,
                nbGroups,
                &raw const input_resource,
                remainder_ptr,
                flags,
                group_params.as_mut_ptr(),
            )
        };
    }
    'server_after_send: {
        if result == cudaError_t::cudaSuccess {
            if has_result {
                send_slice(&result_resources, channel_sender).unwrap();
            }
            send_slice(&group_params, channel_sender).unwrap();
            if has_remainder {
                remainder_resource.send(channel_sender).unwrap();
            }
        }
        channel_sender.flush_out().unwrap();
    }
}

#[cuda_hook(proc_id = 900894)]
fn cudaDeviceGetExecutionCtx(ctx: *mut cudaExecutionContext_t, device: c_int) -> cudaError_t;

#[cuda_hook(proc_id = 900941)]
fn cudaGreenCtxCreate(
    phCtx: *mut cudaExecutionContext_t,
    desc: cudaDevResourceDesc_t,
    device: c_int,
    flags: c_uint,
) -> cudaError_t;

#[cuda_hook(proc_id = 900942, async_api = false)]
fn cudaExecutionCtxDestroy(ctx: cudaExecutionContext_t) -> cudaError_t;

#[cuda_hook(proc_id = 900895)]
fn cudaExecutionCtxGetDevResource(
    ctx: cudaExecutionContext_t,
    #[host(output, len = 1)] resource: *mut cudaDevResource,
    type_: cudaDevResourceType,
) -> cudaError_t;

#[cuda_hook(proc_id = 900896)]
fn cudaExecutionCtxGetDevice(device: *mut c_int, ctx: cudaExecutionContext_t) -> cudaError_t;

#[cuda_hook(proc_id = 900897)]
fn cudaExecutionCtxGetId(ctx: cudaExecutionContext_t, ctxId: *mut c_ulonglong) -> cudaError_t;

#[cuda_hook(proc_id = 900898)]
fn cudaExecutionCtxStreamCreate(
    phStream: *mut cudaStream_t,
    ctx: cudaExecutionContext_t,
    flags: c_uint,
    priority: c_int,
) -> cudaError_t;

#[cuda_hook(proc_id = 900899, async_api = false)]
fn cudaExecutionCtxSynchronize(ctx: cudaExecutionContext_t) -> cudaError_t;

#[cuda_hook(proc_id = 900900, async_api)]
fn cudaExecutionCtxRecordEvent(ctx: cudaExecutionContext_t, event: cudaEvent_t) -> cudaError_t;

#[cuda_hook(proc_id = 900901, async_api)]
fn cudaExecutionCtxWaitEvent(ctx: cudaExecutionContext_t, event: cudaEvent_t) -> cudaError_t;

#[cuda_hook(proc_id = 900460)]
fn cudaDeviceGetTexture1DLinearMaxWidth(
    maxWidthInElements: *mut usize,
    #[host(len = 1)] fmtDesc: *const cudaChannelFormatDesc,
    device: c_int,
) -> cudaError_t;

#[cuda_hook(proc_id = 900356, async_api = false)]
fn cudaDeviceSetCacheConfig(cacheConfig: cudaFuncCache) -> cudaError_t;

#[cuda_hook(proc_id = 900357, async_api = false)]
fn cudaDeviceSetLimit(limit: cudaLimit, value: usize) -> cudaError_t;

#[cuda_hook(proc_id = 900358, async_api = false)]
fn cudaDeviceSetSharedMemConfig(config: cudaSharedMemConfig) -> cudaError_t;

#[cuda_hook(proc_id = 900499, async_api = false)]
fn cudaSetDeviceFlags(flags: c_uint) -> cudaError_t;

#[cuda_hook(proc_id = 900500, async_api = false)]
fn cudaSetValidDevices(
    #[host(input, len = len as usize)] device_arr: *mut c_int,
    len: c_int,
) -> cudaError_t;

#[cuda_hook(proc_id = 900501, async_api = false)]
fn cudaInitDevice(device: c_int, deviceFlags: c_uint, flags: c_uint) -> cudaError_t {
    'client_after_recv: {
        if result == Default::default() {
            client.cuda_device = Some(device);
            client.cuda_device_init = true;
        }
    }
}

#[cuda_hook(proc_id = 152, async_api = false)]
fn cudaGetLastError() -> cudaError_t;

#[cuda_hook(proc_id = 153, async_api = false)]
fn cudaPeekAtLastError() -> cudaError_t;

#[cuda_hook(proc_id = 178, async_api = false)]
fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;

#[cuda_custom_hook]
fn cudaLaunchHostFunc(
    stream: cudaStream_t,
    fn_: cudaHostFn_t,
    userData: *mut c_void,
) -> cudaError_t;

#[cuda_custom_hook]
fn cudaLaunchHostFunc_v2(
    stream: cudaStream_t,
    fn_: cudaHostFn_t,
    userData: *mut c_void,
    syncMode: c_uint,
) -> cudaError_t;

#[cuda_custom_hook]
fn cudaStreamAddCallback(
    stream: cudaStream_t,
    callback: cudaStreamCallback_t,
    userData: *mut c_void,
    flags: c_uint,
) -> cudaError_t;

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

#[cuda_hook(proc_id = 900887)]
fn cudaStreamGetAttribute(
    hStream: cudaStream_t,
    attr: cudaLaunchAttributeID,
    #[host(output, len = 1)] value_out: *mut cudaLaunchAttributeValue,
) -> cudaError_t;

#[cuda_hook(proc_id = 900888)]
fn cudaStreamSetAttribute(
    hStream: cudaStream_t,
    attr: cudaLaunchAttributeID,
    #[host(len = 1)] value: *const cudaLaunchAttributeValue,
) -> cudaError_t;

#[cuda_hook(proc_id = 900902)]
fn cudaStreamGetDevResource(
    hStream: cudaStream_t,
    #[host(output, len = 1)] resource: *mut cudaDevResource,
    type_: cudaDevResourceType,
) -> cudaError_t;

#[cuda_hook(proc_id = 900425, async_api = false)]
fn cudaStreamQuery(stream: cudaStream_t) -> cudaError_t;

#[cuda_hook(proc_id = 900989, async_api)]
fn cudaStreamAttachMemAsync(
    stream: cudaStream_t,
    #[device] devPtr: *mut c_void,
    length: usize,
    flags: c_uint,
) -> cudaError_t;

#[cuda_hook(proc_id = 265)]
fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> cudaError_t;

#[cuda_hook(proc_id = 900990)]
fn cudaMallocManaged(devPtr: *mut *mut c_void, size: usize, flags: c_uint) -> cudaError_t;

#[cuda_hook(proc_id = 900420)]
fn cudaMallocAsync(devPtr: *mut *mut c_void, size: usize, hStream: cudaStream_t) -> cudaError_t;

#[cuda_hook(proc_id = 900102)]
fn cudaMallocPitch(
    devPtr: *mut *mut c_void,
    pitch: *mut usize,
    width: usize,
    height: usize,
) -> cudaError_t;

#[cuda_hook(proc_id = 900482)]
fn cudaMalloc3D(pitchedDevPtr: *mut cudaPitchedPtr, extent: cudaExtent) -> cudaError_t;

#[cuda_hook(proc_id = 900483)]
fn cudaMalloc3DArray(
    array: *mut cudaArray_t,
    #[host(len = 1)] desc: *const cudaChannelFormatDesc,
    extent: cudaExtent,
    flags: c_uint,
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

#[cuda_hook(proc_id = 900565)]
fn cudaMemcpy3D(#[host(len = 1)] p: *const cudaMemcpy3DParms) -> cudaError_t {
    'client_before_send: {
        let params = unsafe { &*p };
        assert!(params.srcArray.is_null());
        assert!(params.dstArray.is_null());
        assert!(!params.srcPtr.ptr.is_null());
        assert!(!params.dstPtr.ptr.is_null());
        assert_eq!(params.kind, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    }
}

#[cuda_hook(proc_id = 900566, async_api)]
fn cudaMemcpy3DAsync(
    #[host(len = 1)] p: *const cudaMemcpy3DParms,
    stream: cudaStream_t,
) -> cudaError_t {
    'client_before_send: {
        let params = unsafe { &*p };
        assert!(params.srcArray.is_null());
        assert!(params.dstArray.is_null());
        assert!(!params.srcPtr.ptr.is_null());
        assert!(!params.dstPtr.ptr.is_null());
        assert_eq!(params.kind, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    }
}

#[cuda_hook(proc_id = 900567)]
fn cudaMemcpy3DPeer(#[host(len = 1)] p: *const cudaMemcpy3DPeerParms) -> cudaError_t {
    'client_before_send: {
        let params = unsafe { &*p };
        assert!(params.srcArray.is_null());
        assert!(params.dstArray.is_null());
        assert!(!params.srcPtr.ptr.is_null());
        assert!(!params.dstPtr.ptr.is_null());
    }
    'server_execution: {
        let params = unsafe { &*p__ptr };
        let copy_params = cudaMemcpy3DParms {
            srcArray: params.srcArray,
            srcPos: params.srcPos,
            srcPtr: params.srcPtr,
            dstArray: params.dstArray,
            dstPos: params.dstPos,
            dstPtr: params.dstPtr,
            extent: params.extent,
            kind: cudaMemcpyKind::cudaMemcpyDeviceToDevice,
        };
        let set_result = unsafe { cudasys::cudart::cudaSetDevice(params.srcDevice) };
        let result = if set_result == cudaError_t::cudaSuccess {
            unsafe { cudasys::cudart::cudaMemcpy3D(&copy_params as *const _) }
        } else {
            set_result
        };
    }
}

#[cuda_hook(proc_id = 900568, async_api)]
fn cudaMemcpy3DPeerAsync(
    #[host(len = 1)] p: *const cudaMemcpy3DPeerParms,
    stream: cudaStream_t,
) -> cudaError_t {
    'client_before_send: {
        let params = unsafe { &*p };
        assert!(params.srcArray.is_null());
        assert!(params.dstArray.is_null());
        assert!(!params.srcPtr.ptr.is_null());
        assert!(!params.dstPtr.ptr.is_null());
    }
    'server_execution: {
        let params = unsafe { &*p__ptr };
        let copy_params = cudaMemcpy3DParms {
            srcArray: params.srcArray,
            srcPos: params.srcPos,
            srcPtr: params.srcPtr,
            dstArray: params.dstArray,
            dstPos: params.dstPos,
            dstPtr: params.dstPtr,
            extent: params.extent,
            kind: cudaMemcpyKind::cudaMemcpyDeviceToDevice,
        };
        let set_result = unsafe { cudasys::cudart::cudaSetDevice(params.srcDevice) };
        let result = if set_result == cudaError_t::cudaSuccess {
            unsafe { cudasys::cudart::cudaMemcpy3DAsync(&copy_params as *const _, stream) }
        } else {
            set_result
        };
    }
}

#[cuda_hook(proc_id = 900965, async_api)]
fn cudaMemcpyBatchAsync(
    #[host(len = count)] dsts: *const *mut c_void,
    #[host(len = count)] srcs: *const *const c_void,
    #[host(len = count)] sizes: *const usize,
    count: usize,
    #[host(input, len = numAttrs)] attrs: *mut cudaMemcpyAttributes,
    #[host(input, len = numAttrs)] attrsIdxs: *mut usize,
    numAttrs: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    'client_before_send: {
        assert!(count > 0);
        assert!(!dsts.is_null());
        assert!(!srcs.is_null());
        assert!(!sizes.is_null());
        assert!(numAttrs > 0);
        assert!(numAttrs <= count);
        assert!(!attrs.is_null());
        assert!(!attrsIdxs.is_null());
        let dsts_slice = unsafe { std::slice::from_raw_parts(dsts, count) };
        let srcs_slice = unsafe { std::slice::from_raw_parts(srcs, count) };
        let sizes_slice = unsafe { std::slice::from_raw_parts(sizes, count) };
        assert!(dsts_slice.iter().all(|dst| !dst.is_null()));
        assert!(srcs_slice.iter().all(|src| !src.is_null()));
        assert!(sizes_slice.iter().all(|size| *size > 0));
        let attrs_idx_slice = unsafe { std::slice::from_raw_parts(attrsIdxs, numAttrs) };
        assert_eq!(attrs_idx_slice[0], 0);
        assert!(attrs_idx_slice.iter().all(|idx| *idx < count));
        assert!(attrs_idx_slice.windows(2).all(|idxs| idxs[0] < idxs[1]));
    }
}

#[cuda_hook(proc_id = 900966, async_api)]
fn cudaMemcpyWithAttributesAsync(
    #[device] dst: *mut c_void,
    #[device] src: *const c_void,
    size: usize,
    #[host(input, len = 1)] attr: *mut cudaMemcpyAttributes,
    stream: cudaStream_t,
) -> cudaError_t {
    'client_before_send: {
        assert!(!dst.is_null());
        assert!(!src.is_null());
        assert!(size > 0);
        assert!(!attr.is_null());
    }
}

#[cuda_hook(proc_id = 900576, async_api)]
fn cudaMemcpy3DBatchAsync(
    numOps: usize,
    #[host(input, len = numOps)] opList: *mut cudaMemcpy3DBatchOp,
    flags: c_ulonglong,
    stream: cudaStream_t,
) -> cudaError_t {
    'client_before_send: {
        assert!(numOps > 0);
        assert!(!opList.is_null());
        assert_eq!(flags, 0);
        let ops = unsafe { std::slice::from_raw_parts(opList, numOps) };
        for op in ops {
            assert_eq!(
                op.src.type_,
                cudaMemcpy3DOperandType::cudaMemcpyOperandTypePointer
            );
            assert_eq!(
                op.dst.type_,
                cudaMemcpy3DOperandType::cudaMemcpyOperandTypePointer
            );
            let src = unsafe { op.src.op.ptr };
            let dst = unsafe { op.dst.op.ptr };
            assert!(!src.ptr.is_null());
            assert!(!dst.ptr.is_null());
            assert!(op.extent.width > 0);
            assert!(op.extent.height > 0);
            assert!(op.extent.depth > 0);
        }
    }
}

#[cuda_hook(proc_id = 900577, async_api)]
fn cudaMemcpy3DWithAttributesAsync(
    #[host(input, len = 1)] op: *mut cudaMemcpy3DBatchOp,
    flags: c_ulonglong,
    stream: cudaStream_t,
) -> cudaError_t {
    'client_before_send: {
        assert!(!op.is_null());
        assert_eq!(flags, 0);
        let op_ref = unsafe { &*op };
        assert_eq!(
            op_ref.src.type_,
            cudaMemcpy3DOperandType::cudaMemcpyOperandTypePointer
        );
        assert_eq!(
            op_ref.dst.type_,
            cudaMemcpy3DOperandType::cudaMemcpyOperandTypePointer
        );
        let src = unsafe { op_ref.src.op.ptr };
        let dst = unsafe { op_ref.dst.op.ptr };
        assert!(!src.ptr.is_null());
        assert!(!dst.ptr.is_null());
        assert!(op_ref.extent.width > 0);
        assert!(op_ref.extent.height > 0);
        assert!(op_ref.extent.depth > 0);
    }
}

#[cuda_custom_hook] // calls one of the following internal APIs
fn cudaMemcpyToArray(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_hook(proc_id = 900471, async_api, parent = cudaMemcpyToArray)]
fn cudaMemcpyToArrayHtod(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    #[host(len = count)] src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_hook(proc_id = 900472, async_api, parent = cudaMemcpyToArray)]
fn cudaMemcpyToArrayDtod(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    #[device] src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_custom_hook] // calls one of the following internal APIs
fn cudaMemcpyFromArray(
    dst: *mut c_void,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_hook(proc_id = 900473, parent = cudaMemcpyFromArray)]
fn cudaMemcpyFromArrayDtoh(
    #[host(output, len = count)] dst: *mut c_void,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_hook(proc_id = 900474, async_api, parent = cudaMemcpyFromArray)]
fn cudaMemcpyFromArrayDtod(
    #[device] dst: *mut c_void,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_custom_hook] // calls one of the following internal APIs
fn cudaMemcpyToArrayAsync(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 900477, parent = cudaMemcpyToArrayAsync)]
fn cudaMemcpyToArrayAsyncHtod(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    #[host(len = count)] src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    'server_execution: {
        let result = unsafe {
            assert_eq!(cudaStreamSynchronize(stream), Default::default());
            cudaMemcpyToArray(dst, wOffset, hOffset, src__ptr.cast(), count, kind)
        };
    }
}

#[cuda_hook(proc_id = 900478, async_api, parent = cudaMemcpyToArrayAsync)]
fn cudaMemcpyToArrayAsyncDtod(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    #[device] src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_custom_hook] // calls one of the following internal APIs
fn cudaMemcpyFromArrayAsync(
    dst: *mut c_void,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 900479, parent = cudaMemcpyFromArrayAsync)]
fn cudaMemcpyFromArrayAsyncDtoh(
    #[host(output, len = count)] dst: *mut c_void,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    'server_execution: {
        let result = unsafe {
            assert_eq!(cudaStreamSynchronize(stream), Default::default());
            cudaMemcpyFromArray(dst__ptr.cast(), src, wOffset, hOffset, count, kind)
        };
    }
}

#[cuda_hook(proc_id = 900480, async_api, parent = cudaMemcpyFromArrayAsync)]
fn cudaMemcpyFromArrayAsyncDtod(
    #[device] dst: *mut c_void,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_custom_hook] // calls cudaMemcpyToArray row by row
fn cudaMemcpy2DToArray(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    src: *const c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_custom_hook] // calls cudaMemcpyFromArray row by row
fn cudaMemcpy2DFromArray(
    dst: *mut c_void,
    dpitch: usize,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_custom_hook] // calls cudaMemcpyToArrayAsync row by row
fn cudaMemcpy2DToArrayAsync(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    src: *const c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_custom_hook] // calls cudaMemcpyFromArrayAsync row by row
fn cudaMemcpy2DFromArrayAsync(
    dst: *mut c_void,
    dpitch: usize,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 900484)]
fn cudaMemcpyArrayToArray(
    dst: cudaArray_t,
    wOffsetDst: usize,
    hOffsetDst: usize,
    src: cudaArray_const_t,
    wOffsetSrc: usize,
    hOffsetSrc: usize,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_hook(proc_id = 900485)]
fn cudaMemcpy2DArrayToArray(
    dst: cudaArray_t,
    wOffsetDst: usize,
    hOffsetDst: usize,
    src: cudaArray_const_t,
    wOffsetSrc: usize,
    hOffsetSrc: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_hook(proc_id = 900530)]
fn cudaMemcpyPeer(
    #[device] dst: *mut c_void,
    dstDevice: c_int,
    #[device] src: *const c_void,
    srcDevice: c_int,
    count: usize,
) -> cudaError_t {
    'server_execution: {
        let _ = dstDevice;
        let set_result = unsafe { cudasys::cudart::cudaSetDevice(srcDevice) };
        let result = if set_result == cudaError_t::cudaSuccess {
            unsafe {
                cudasys::cudart::cudaMemcpy(
                    dst,
                    src,
                    count,
                    cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                )
            }
        } else {
            set_result
        };
    }
}

#[cuda_hook(proc_id = 900531, async_api)]
fn cudaMemcpyPeerAsync(
    #[device] dst: *mut c_void,
    dstDevice: c_int,
    #[device] src: *const c_void,
    srcDevice: c_int,
    count: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    'server_execution: {
        let _ = dstDevice;
        let set_result = unsafe { cudasys::cudart::cudaSetDevice(srcDevice) };
        let result = if set_result == cudaError_t::cudaSuccess {
            unsafe {
                cudasys::cudart::cudaMemcpyAsync(
                    dst,
                    src,
                    count,
                    cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                    stream,
                )
            }
        } else {
            set_result
        };
    }
}

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
fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: c_int) -> cudaError_t {
    'client_after_recv: {
        if result == cudaError_t::cudaSuccess {
            if prop.major > 0 && prop.minor >= 0 {
                DRIVER_CACHE.write().unwrap().device_arch =
                    Some(prop.major as u32 * 10 + prop.minor as u32);
            }
        }
    }
}

#[cuda_hook(proc_id = 400)]
fn cudaPointerGetAttributes(
    attributes: *mut cudaPointerAttributes,
    #[device] ptr: *const c_void,
) -> cudaError_t;

#[cuda_custom_hook] // local
fn cudaGetDriverEntryPoint(
    symbol: *const c_char,
    funcPtr: *mut *mut c_void,
    flags: c_ulonglong,
    driverStatus: *mut cudaDriverEntryPointQueryResult,
) -> cudaError_t;

#[cuda_custom_hook] // local
fn cudaGetDriverEntryPointByVersion(
    symbol: *const c_char,
    funcPtr: *mut *mut c_void,
    cudaVersion: c_uint,
    flags: c_ulonglong,
    driverStatus: *mut cudaDriverEntryPointQueryResult,
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

#[cuda_custom_hook] // unsupported across the remoting boundary
fn cudaHostGetDevicePointer(
    pDevice: *mut *mut c_void,
    pHost: *mut c_void,
    flags: c_uint,
) -> cudaError_t;

#[cuda_custom_hook] // calls driver API
fn cudaGetSymbolAddress(devPtr: *mut *mut c_void, symbol: *const c_void) -> cudaError_t;

#[cuda_custom_hook] // calls driver API
fn cudaGetSymbolSize(size: *mut usize, symbol: *const c_void) -> cudaError_t;

#[cuda_hook(proc_id = 900991, async_api)]
fn cudaMemPrefetchAsync(
    #[device] devPtr: *const c_void,
    count: usize,
    location: cudaMemLocation,
    flags: c_uint,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 900992)]
fn cudaMemAdvise(
    #[device] devPtr: *const c_void,
    count: usize,
    advice: cudaMemoryAdvise,
    location: cudaMemLocation,
) -> cudaError_t;

#[cuda_hook(proc_id = 900993)]
fn cudaMemRangeGetAttribute(
    #[host(output, len = dataSize)] data: *mut c_void,
    dataSize: usize,
    attribute: cudaMemRangeAttribute,
    #[device] devPtr: *const c_void,
    count: usize,
) -> cudaError_t;

#[cuda_custom_hook(proc_id = 900994)]
fn cudaMemRangeGetAttributes(
    data: *mut *mut c_void,
    dataSizes: *mut usize,
    attributes: *mut cudaMemRangeAttribute,
    numAttributes: usize,
    devPtr: *const c_void,
    count: usize,
) -> cudaError_t;

#[cuda_custom_hook] // calls cudaMemcpy
fn cudaMemcpyToSymbol(
    symbol: *const c_void,
    src: *const c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_custom_hook] // calls cudaMemcpy
fn cudaMemcpyFromSymbol(
    dst: *mut c_void,
    symbol: *const c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_custom_hook] // calls cudaMemcpyAsync
fn cudaMemcpyToSymbolAsync(
    symbol: *const c_void,
    src: *const c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_custom_hook] // calls cudaMemcpyAsync
fn cudaMemcpyFromSymbolAsync(
    dst: *mut c_void,
    symbol: *const c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_custom_hook] // calls the internal API below
fn cudaFuncGetAttributes(attr: *mut cudaFuncAttributes, func: *const c_void) -> cudaError_t;

#[cuda_custom_hook] // calls driver API
fn cudaFuncGetName(name: *mut *const c_char, func: *const c_void) -> cudaError_t;

#[cuda_custom_hook] // calls driver API
fn cudaFuncGetParamInfo(
    func: *const c_void,
    paramIndex: usize,
    paramOffset: *mut usize,
    paramSize: *mut usize,
) -> cudaError_t;

#[cuda_custom_hook] // calls driver API
fn cudaFuncGetParamCount(func: *const c_void, paramCount: *mut usize) -> cudaError_t;

#[cuda_custom_hook(proc_id = 900967)]
fn cudaCreateTextureObject(
    pTexObject: *mut cudaTextureObject_t,
    pResDesc: *const cudaResourceDesc,
    pTexDesc: *const cudaTextureDesc,
    pResViewDesc: *const cudaResourceViewDesc,
) -> cudaError_t;

#[cuda_hook(proc_id = 900968)]
fn cudaDestroyTextureObject(texObject: cudaTextureObject_t) -> cudaError_t;

#[cuda_custom_hook(proc_id = 900969)]
fn cudaGetTextureObjectResourceDesc(
    pResDesc: *mut cudaResourceDesc,
    texObject: cudaTextureObject_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 900970)]
fn cudaGetTextureObjectTextureDesc(
    #[host(output, len = 1)] pTexDesc: *mut cudaTextureDesc,
    texObject: cudaTextureObject_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 900971)]
fn cudaGetTextureObjectResourceViewDesc(
    #[host(output, len = 1)] pResViewDesc: *mut cudaResourceViewDesc,
    texObject: cudaTextureObject_t,
) -> cudaError_t;

#[cuda_custom_hook(proc_id = 900972)]
fn cudaCreateSurfaceObject(
    pSurfObject: *mut cudaSurfaceObject_t,
    pResDesc: *const cudaResourceDesc,
) -> cudaError_t;

#[cuda_hook(proc_id = 900973)]
fn cudaDestroySurfaceObject(surfObject: cudaSurfaceObject_t) -> cudaError_t;

#[cuda_custom_hook(proc_id = 900974)]
fn cudaGetSurfaceObjectResourceDesc(
    pResDesc: *mut cudaResourceDesc,
    surfObject: cudaSurfaceObject_t,
) -> cudaError_t;

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

#[cuda_custom_hook] // local
fn cudaGetKernel(kernelPtr: *mut cudaKernel_t, entryFuncAddr: *const c_void) -> cudaError_t;

#[cuda_custom_hook] // local
fn __cudaGetKernel(kernelPtr: *mut cudaKernel_t, entryFuncAddr: *const c_void) -> cudaError_t;

#[cuda_custom_hook] // calls driver API
fn __cudaLaunchKernel(
    kernel: cudaKernel_t,
    gridDim: dim3,
    blockDim: dim3,
    args: *mut *mut c_void,
    sharedMem: usize,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_custom_hook] // calls driver API
fn __cudaLaunchKernel_ptsz(
    kernel: cudaKernel_t,
    gridDim: dim3,
    blockDim: dim3,
    args: *mut *mut c_void,
    sharedMem: usize,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_custom_hook] // calls driver API
fn cudaLaunchKernelExC(
    config: *const cudaLaunchConfig_t,
    func: *const c_void,
    args: *mut *mut c_void,
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

#[cuda_hook(proc_id = 900486, async_api)]
fn cudaMemset2D(
    #[device] devPtr: *mut c_void,
    pitch: usize,
    value: c_int,
    width: usize,
    height: usize,
) -> cudaError_t;

#[cuda_hook(proc_id = 900487, async_api)]
fn cudaMemset2DAsync(
    #[device] devPtr: *mut c_void,
    pitch: usize,
    value: c_int,
    width: usize,
    height: usize,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 900488, async_api)]
fn cudaMemset3D(pitchedDevPtr: cudaPitchedPtr, value: c_int, extent: cudaExtent) -> cudaError_t;

#[cuda_hook(proc_id = 900489, async_api)]
fn cudaMemset3DAsync(
    pitchedDevPtr: cudaPitchedPtr,
    value: c_int,
    extent: cudaExtent,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_custom_hook] // local
fn cudaGetErrorString(error: cudaError_t) -> *const c_char;

#[cuda_custom_hook] // local
fn cudaGetErrorName(error: cudaError_t) -> *const c_char;

#[cuda_custom_hook] // local
fn cudaCreateChannelDesc(
    x: c_int,
    y: c_int,
    z: c_int,
    w: c_int,
    f: cudaChannelFormatKind,
) -> cudaChannelFormatDesc;

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

#[cuda_hook(proc_id = 900995)]
fn cudaMemPoolCreate(
    memPool: *mut cudaMemPool_t,
    #[host(len = 1)] poolProps: *const cudaMemPoolProps,
) -> cudaError_t;

#[cuda_hook(proc_id = 900996, async_api = false)]
fn cudaMemPoolDestroy(memPool: cudaMemPool_t) -> cudaError_t;

#[cuda_hook(proc_id = 900997)]
fn cudaMemGetDefaultMemPool(
    memPool: *mut cudaMemPool_t,
    #[host(input, len = 1)] location: *mut cudaMemLocation,
    type_: cudaMemAllocationType,
) -> cudaError_t;

#[cuda_hook(proc_id = 900998)]
fn cudaMemGetMemPool(
    memPool: *mut cudaMemPool_t,
    #[host(input, len = 1)] location: *mut cudaMemLocation,
    type_: cudaMemAllocationType,
) -> cudaError_t;

#[cuda_hook(proc_id = 900999)]
fn cudaMemSetMemPool(
    #[host(input, len = 1)] location: *mut cudaMemLocation,
    type_: cudaMemAllocationType,
    memPool: cudaMemPool_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 901000)]
fn cudaMemPoolSetAccess(
    memPool: cudaMemPool_t,
    #[host(len = count)] descList: *const cudaMemAccessDesc,
    count: usize,
) -> cudaError_t;

#[cuda_hook(proc_id = 901001)]
fn cudaMemPoolGetAccess(
    #[host(output, len = 1)] flags: *mut cudaMemAccessFlags,
    memPool: cudaMemPool_t,
    #[host(input, len = 1)] location: *mut cudaMemLocation,
) -> cudaError_t;

#[cuda_hook(proc_id = 900446)]
fn cudaMallocFromPoolAsync(
    ptr: *mut *mut c_void,
    size: usize,
    memPool: cudaMemPool_t,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 900562, async_api = false)]
fn cudaDeviceGraphMemTrim(device: c_int) -> cudaError_t;

#[cuda_hook(proc_id = 900563)]
fn cudaDeviceGetGraphMemAttribute(
    device: c_int,
    attr: cudaGraphMemAttributeType,
    #[host(output, len = attr.data_size())] value: *mut c_void,
) -> cudaError_t;

#[cuda_hook(proc_id = 900564)]
fn cudaDeviceSetGraphMemAttribute(
    device: c_int,
    attr: cudaGraphMemAttributeType,
    #[host(input, len = attr.data_size())] value: *mut c_void,
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

#[cuda_custom_hook(proc_id = 202)]
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

#[cuda_custom_hook] // calls driver API
fn cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    numBlocks: *mut c_int,
    func: *const c_void,
    blockSize: c_int,
    dynamicSMemSize: usize,
) -> cudaError_t;

#[cuda_custom_hook] // calls driver API
fn cudaOccupancyAvailableDynamicSMemPerBlock(
    dynamicSmemSize: *mut usize,
    func: *const c_void,
    numBlocks: c_int,
    blockSize: c_int,
) -> cudaError_t;

#[cuda_hook(proc_id = 126)]
fn cudaIpcGetMemHandle(
    handle: *mut cudaIpcMemHandle_t,
    #[device] devPtr: *mut c_void,
) -> cudaError_t;

#[cuda_custom_hook(proc_id = 900914)]
fn cudaIpcGetEventHandle(handle: *mut cudaIpcEventHandle_t, event: cudaEvent_t) -> cudaError_t;

#[cuda_custom_hook(proc_id = 900915)]
fn cudaIpcOpenEventHandle(event: *mut cudaEvent_t, handle: cudaIpcEventHandle_t) -> cudaError_t;

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

#[cuda_custom_hook] // calls driver API
fn cudaFuncSetCacheConfig(func: *const c_void, cacheConfig: cudaFuncCache) -> cudaError_t;

#[cuda_custom_hook] // calls driver API
fn cudaFuncSetSharedMemConfig(func: *const c_void, config: cudaSharedMemConfig) -> cudaError_t;

#[cuda_hook(proc_id = 203)]
fn cudaEventElapsedTime(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> cudaError_t;

#[cuda_hook(proc_id = 412, async_api = false)]
fn cudaDeviceEnablePeerAccess(peerDevice: c_int, flags: c_uint) -> cudaError_t;

#[cuda_hook(proc_id = 297, async_api)]
fn cudaMemset(#[device] devPtr: *mut c_void, value: c_int, count: usize) -> cudaError_t;

#[cuda_hook(proc_id = 114, async_api = false)]
fn cudaDeviceReset() -> cudaError_t;

#[cuda_hook(proc_id = 900502, async_api = false)]
fn cudaProfilerStart() -> cudaError_t;

#[cuda_hook(proc_id = 900503, async_api = false)]
fn cudaProfilerStop() -> cudaError_t;

#[cuda_hook(proc_id = 900427, async_api = false)]
fn cudaCtxResetPersistingL2Cache() -> cudaError_t;

#[cuda_hook(proc_id = 163, async_api = false)]
fn cudaStreamBeginCapture(stream: cudaStream_t, mode: cudaStreamCaptureMode) -> cudaError_t;

#[cuda_hook(proc_id = 900553)]
fn cudaStreamBeginCaptureToGraph(
    stream: cudaStream_t,
    graph: cudaGraph_t,
    #[host(len = numDependencies)] dependencies: *const cudaGraphNode_t,
    #[device] dependencyData: *const cudaGraphEdgeData, // null
    numDependencies: usize,
    mode: cudaStreamCaptureMode,
) -> cudaError_t {
    'client_before_send: {
        assert!(numDependencies > 0);
        assert!(dependencyData.is_null());
    }
}

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

#[cuda_hook(proc_id = 900555)]
fn cudaStreamUpdateCaptureDependencies(
    stream: cudaStream_t,
    #[host(input, len = numDependencies)] dependencies: *mut cudaGraphNode_t,
    #[device] dependencyData: *const cudaGraphEdgeData, // null
    numDependencies: usize,
    flags: c_uint,
) -> cudaError_t {
    'client_before_send: {
        assert!(numDependencies > 0);
        assert!(dependencyData.is_null());
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

#[cuda_hook(proc_id = 900504)]
fn cudaGraphCreate(pGraph: *mut cudaGraph_t, flags: c_uint) -> cudaError_t;

#[cuda_hook(proc_id = 900505)]
fn cudaGraphAddEmptyNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    #[device] pDependencies: *const cudaGraphNode_t, // null
    numDependencies: usize,
) -> cudaError_t {
    'client_before_send: {
        assert!(pDependencies.is_null());
        assert_eq!(numDependencies, 0);
    }
}

#[cuda_hook(proc_id = 900506)]
fn cudaGraphAddDependencies(
    graph: cudaGraph_t,
    #[host(len = numDependencies)] from: *const cudaGraphNode_t,
    #[host(len = numDependencies)] to: *const cudaGraphNode_t,
    #[device] edgeData: *const cudaGraphEdgeData, // null
    numDependencies: usize,
) -> cudaError_t {
    'client_before_send: {
        assert!(edgeData.is_null());
    }
}

#[cuda_hook(proc_id = 900507)]
fn cudaGraphRemoveDependencies(
    graph: cudaGraph_t,
    #[host(len = numDependencies)] from: *const cudaGraphNode_t,
    #[host(len = numDependencies)] to: *const cudaGraphNode_t,
    #[device] edgeData: *const cudaGraphEdgeData, // null
    numDependencies: usize,
) -> cudaError_t {
    'client_before_send: {
        assert!(edgeData.is_null());
    }
}

#[cuda_hook(proc_id = 900508, async_api = false)]
fn cudaGraphDestroyNode(node: cudaGraphNode_t) -> cudaError_t;

#[cuda_hook(proc_id = 900903)]
fn cudaGraphAddNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    #[device] pDependencies: *const cudaGraphNode_t,
    #[device] dependencyData: *const cudaGraphEdgeData,
    numDependencies: usize,
    #[host(input, len = 1)] nodeParams: *mut cudaGraphNodeParams,
) -> cudaError_t {
    'client_before_send: {
        assert!(pDependencies.is_null());
        assert!(dependencyData.is_null());
        assert_eq!(numDependencies, 0);
        assert!(!nodeParams.is_null());
        let params = unsafe { &*nodeParams };
        assert_eq!(params.type_, cudaGraphNodeType::cudaGraphNodeTypeMemset);
        let memset = unsafe { params.__bindgen_anon_1.memset };
        assert_eq!(memset.elementSize, 1);
        assert_eq!(memset.height, 1);
    }
}

#[cuda_hook(proc_id = 900904)]
fn cudaGraphNodeGetParams(
    node: cudaGraphNode_t,
    #[host(output, len = 1)] nodeParams: *mut cudaGraphNodeParams,
) -> cudaError_t {
    'server_execution: {
        let result = unsafe { cudaGraphNodeGetParams(node, nodeParams__ptr) };
        if result == cudaError_t::cudaSuccess {
            unsafe {
                assert_eq!(
                    (*nodeParams__ptr).type_,
                    cudaGraphNodeType::cudaGraphNodeTypeMemset
                );
            }
        }
    }
}

#[cuda_hook(proc_id = 900905)]
fn cudaGraphNodeSetParams(
    node: cudaGraphNode_t,
    #[host(input, len = 1)] nodeParams: *mut cudaGraphNodeParams,
) -> cudaError_t {
    'client_before_send: {
        assert!(!nodeParams.is_null());
        let params = unsafe { &*nodeParams };
        assert_eq!(params.type_, cudaGraphNodeType::cudaGraphNodeTypeMemset);
        let memset = unsafe { params.__bindgen_anon_1.memset };
        assert_eq!(memset.elementSize, 1);
        assert_eq!(memset.height, 1);
    }
}

#[cuda_hook(proc_id = 900906)]
fn cudaGraphExecNodeSetParams(
    graphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    #[host(input, len = 1)] nodeParams: *mut cudaGraphNodeParams,
) -> cudaError_t {
    'client_before_send: {
        assert!(!nodeParams.is_null());
        let params = unsafe { &*nodeParams };
        assert_eq!(params.type_, cudaGraphNodeType::cudaGraphNodeTypeMemset);
        let memset = unsafe { params.__bindgen_anon_1.memset };
        assert_eq!(memset.elementSize, 1);
        assert_eq!(memset.height, 1);
    }
}

#[cuda_custom_hook(proc_id = 900910)]
fn cudaGraphAddKernelNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    pNodeParams: *const cudaKernelNodeParams,
) -> cudaError_t;

#[cuda_custom_hook(proc_id = 900911)]
fn cudaGraphKernelNodeGetParams(
    node: cudaGraphNode_t,
    pNodeParams: *mut cudaKernelNodeParams,
) -> cudaError_t;

#[cuda_custom_hook(proc_id = 900912)]
fn cudaGraphKernelNodeSetParams(
    node: cudaGraphNode_t,
    pNodeParams: *const cudaKernelNodeParams,
) -> cudaError_t;

#[cuda_custom_hook(proc_id = 900913)]
fn cudaGraphExecKernelNodeSetParams(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    pNodeParams: *const cudaKernelNodeParams,
) -> cudaError_t;

#[cuda_hook(proc_id = 900907)]
fn cudaGraphKernelNodeCopyAttributes(hDst: cudaGraphNode_t, hSrc: cudaGraphNode_t) -> cudaError_t;

#[cuda_hook(proc_id = 900908)]
fn cudaGraphKernelNodeGetAttribute(
    hNode: cudaGraphNode_t,
    attr: cudaLaunchAttributeID,
    #[host(output, len = 1)] value_out: *mut cudaLaunchAttributeValue,
) -> cudaError_t;

#[cuda_hook(proc_id = 900909)]
fn cudaGraphKernelNodeSetAttribute(
    hNode: cudaGraphNode_t,
    attr: cudaLaunchAttributeID,
    #[host(len = 1)] value: *const cudaLaunchAttributeValue,
) -> cudaError_t;

#[cuda_hook(proc_id = 900515)]
fn cudaGraphAddChildGraphNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    #[device] pDependencies: *const cudaGraphNode_t, // null
    numDependencies: usize,
    childGraph: cudaGraph_t,
) -> cudaError_t {
    'client_before_send: {
        assert!(pDependencies.is_null());
        assert_eq!(numDependencies, 0);
    }
}

#[cuda_hook(proc_id = 900516)]
fn cudaGraphChildGraphNodeGetGraph(node: cudaGraphNode_t, pGraph: *mut cudaGraph_t) -> cudaError_t;

#[cuda_hook(proc_id = 900556)]
fn cudaGraphExecChildGraphNodeSetParams(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    childGraph: cudaGraph_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 900532)]
fn cudaGraphAddEventRecordNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    #[device] pDependencies: *const cudaGraphNode_t, // null
    numDependencies: usize,
    event: cudaEvent_t,
) -> cudaError_t {
    'client_before_send: {
        assert!(pDependencies.is_null());
        assert_eq!(numDependencies, 0);
    }
}

#[cuda_hook(proc_id = 900533)]
fn cudaGraphEventRecordNodeGetEvent(
    node: cudaGraphNode_t,
    event_out: *mut cudaEvent_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 900534)]
fn cudaGraphEventRecordNodeSetEvent(node: cudaGraphNode_t, event: cudaEvent_t) -> cudaError_t;

#[cuda_hook(proc_id = 900535)]
fn cudaGraphAddEventWaitNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    #[device] pDependencies: *const cudaGraphNode_t, // null
    numDependencies: usize,
    event: cudaEvent_t,
) -> cudaError_t {
    'client_before_send: {
        assert!(pDependencies.is_null());
        assert_eq!(numDependencies, 0);
    }
}

#[cuda_hook(proc_id = 900536)]
fn cudaGraphEventWaitNodeGetEvent(
    node: cudaGraphNode_t,
    event_out: *mut cudaEvent_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 900537)]
fn cudaGraphEventWaitNodeSetEvent(node: cudaGraphNode_t, event: cudaEvent_t) -> cudaError_t;

#[cuda_hook(proc_id = 900540)]
fn cudaGraphAddMemsetNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    #[device] pDependencies: *const cudaGraphNode_t, // null
    numDependencies: usize,
    #[host(len = 1)] pMemsetParams: *const cudaMemsetParams,
) -> cudaError_t {
    'client_before_send: {
        assert!(pDependencies.is_null());
        assert_eq!(numDependencies, 0);
    }
}

#[cuda_hook(proc_id = 900541)]
fn cudaGraphMemsetNodeGetParams(
    node: cudaGraphNode_t,
    #[host(output, len = 1)] pNodeParams: *mut cudaMemsetParams,
) -> cudaError_t;

#[cuda_hook(proc_id = 900542)]
fn cudaGraphMemsetNodeSetParams(
    node: cudaGraphNode_t,
    #[host(len = 1)] pNodeParams: *const cudaMemsetParams,
) -> cudaError_t;

#[cuda_hook(proc_id = 900547)]
fn cudaGraphAddMemcpyNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    #[device] pDependencies: *const cudaGraphNode_t, // null
    numDependencies: usize,
    #[host(len = 1)] pCopyParams: *const cudaMemcpy3DParms,
) -> cudaError_t {
    'client_before_send: {
        assert!(pDependencies.is_null());
        assert_eq!(numDependencies, 0);
        let params = unsafe { &*pCopyParams };
        assert!(params.srcArray.is_null());
        assert!(params.dstArray.is_null());
        assert!(!params.srcPtr.ptr.is_null());
        assert!(!params.dstPtr.ptr.is_null());
        assert_eq!(params.kind, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    }
}

#[cuda_hook(proc_id = 900548)]
fn cudaGraphMemcpyNodeGetParams(
    node: cudaGraphNode_t,
    #[host(output, len = 1)] pNodeParams: *mut cudaMemcpy3DParms,
) -> cudaError_t;

#[cuda_hook(proc_id = 900549)]
fn cudaGraphMemcpyNodeSetParams(
    node: cudaGraphNode_t,
    #[host(len = 1)] pNodeParams: *const cudaMemcpy3DParms,
) -> cudaError_t {
    'client_before_send: {
        let params = unsafe { &*pNodeParams };
        assert!(params.srcArray.is_null());
        assert!(params.dstArray.is_null());
        assert!(!params.srcPtr.ptr.is_null());
        assert!(!params.dstPtr.ptr.is_null());
        assert_eq!(params.kind, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    }
}

#[cuda_hook(proc_id = 900544)]
fn cudaGraphAddMemcpyNode1D(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    #[device] pDependencies: *const cudaGraphNode_t, // null
    numDependencies: usize,
    #[device] dst: *mut c_void,
    #[device] src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    'client_before_send: {
        assert!(pDependencies.is_null());
        assert_eq!(numDependencies, 0);
        assert_eq!(kind, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    }
}

#[cuda_custom_hook] // local, resolves symbol then calls 1D graph memcpy APIs
fn cudaGraphAddMemcpyNodeToSymbol(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    symbol: *const c_void,
    src: *const c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_custom_hook] // local, resolves symbol then calls 1D graph memcpy APIs
fn cudaGraphAddMemcpyNodeFromSymbol(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    dst: *mut c_void,
    symbol: *const c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_hook(proc_id = 900545)]
fn cudaGraphMemcpyNodeSetParams1D(
    node: cudaGraphNode_t,
    #[device] dst: *mut c_void,
    #[device] src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    'client_before_send: {
        assert_eq!(kind, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    }
}

#[cuda_custom_hook] // local, resolves symbol then calls 1D graph memcpy APIs
fn cudaGraphMemcpyNodeSetParamsToSymbol(
    node: cudaGraphNode_t,
    symbol: *const c_void,
    src: *const c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_custom_hook] // local, resolves symbol then calls 1D graph memcpy APIs
fn cudaGraphMemcpyNodeSetParamsFromSymbol(
    node: cudaGraphNode_t,
    dst: *mut c_void,
    symbol: *const c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_hook(proc_id = 545, async_api = false)]
fn cudaGraphExecDestroy(graphExec: cudaGraphExec_t) -> cudaError_t;

#[cuda_custom_hook(proc_id = 563)]
fn cudaGraphGetNodes(
    graph: cudaGraph_t,
    nodes: *mut cudaGraphNode_t,
    numNodes: *mut usize,
) -> cudaError_t;

#[cuda_custom_hook(proc_id = 900509)]
fn cudaGraphGetRootNodes(
    graph: cudaGraph_t,
    pRootNodes: *mut cudaGraphNode_t,
    pNumRootNodes: *mut usize,
) -> cudaError_t;

#[cuda_custom_hook(proc_id = 900510)]
fn cudaGraphGetEdges(
    graph: cudaGraph_t,
    from: *mut cudaGraphNode_t,
    to: *mut cudaGraphNode_t,
    edgeData: *mut cudaGraphEdgeData,
    numEdges: *mut usize,
) -> cudaError_t;

#[cuda_hook(proc_id = 900511)]
fn cudaGraphGetId(hGraph: cudaGraph_t, graphID: *mut c_uint) -> cudaError_t;

#[cuda_hook(proc_id = 900517)]
fn cudaGraphClone(pGraphClone: *mut cudaGraph_t, originalGraph: cudaGraph_t) -> cudaError_t;

#[cuda_hook(proc_id = 900518)]
fn cudaGraphNodeFindInClone(
    pNode: *mut cudaGraphNode_t,
    originalNode: cudaGraphNode_t,
    clonedGraph: cudaGraph_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 900519)]
fn cudaGraphNodeGetType(node: cudaGraphNode_t, pType: *mut cudaGraphNodeType) -> cudaError_t;

#[cuda_hook(proc_id = 900520)]
fn cudaGraphNodeGetContainingGraph(
    hNode: cudaGraphNode_t,
    phGraph: *mut cudaGraph_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 900521)]
fn cudaGraphNodeGetLocalId(hNode: cudaGraphNode_t, nodeId: *mut c_uint) -> cudaError_t;

#[cuda_hook(proc_id = 900522)]
fn cudaGraphNodeGetToolsId(hNode: cudaGraphNode_t, toolsNodeId: *mut c_ulonglong) -> cudaError_t;

#[cuda_custom_hook(proc_id = 900523)]
fn cudaGraphNodeGetDependencies(
    node: cudaGraphNode_t,
    pDependencies: *mut cudaGraphNode_t,
    edgeData: *mut cudaGraphEdgeData,
    pNumDependencies: *mut usize,
) -> cudaError_t;

#[cuda_custom_hook(proc_id = 900524)]
fn cudaGraphNodeGetDependentNodes(
    node: cudaGraphNode_t,
    pDependentNodes: *mut cudaGraphNode_t,
    edgeData: *mut cudaGraphEdgeData,
    pNumDependentNodes: *mut usize,
) -> cudaError_t;

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

#[cuda_hook(proc_id = 900554)]
fn cudaGraphInstantiateWithParams(
    pGraphExec: *mut cudaGraphExec_t,
    graph: cudaGraph_t,
    #[host(input, len = 1)] instantiateParams: *mut cudaGraphInstantiateParams,
) -> cudaError_t {
    'client_after_recv: {
        let mut instantiate_params_out: cudaGraphInstantiateParams = unsafe { std::mem::zeroed() };
        instantiate_params_out.recv(channel_receiver).unwrap();
        unsafe {
            std::ptr::write(
                instantiateParams.as_ptr().cast_mut(),
                instantiate_params_out,
            );
        }
    }
    'server_after_send: {
        instantiateParams[0].send(channel_sender).unwrap();
        channel_sender.flush_out().unwrap();
    }
}

#[cuda_hook(proc_id = 900512)]
fn cudaGraphExecGetFlags(graphExec: cudaGraphExec_t, flags: *mut c_ulonglong) -> cudaError_t;

#[cuda_hook(proc_id = 900513)]
fn cudaGraphExecGetId(hGraphExec: cudaGraphExec_t, graphID: *mut c_uint) -> cudaError_t;

#[cuda_hook(proc_id = 900525)]
fn cudaGraphExecUpdate(
    hGraphExec: cudaGraphExec_t,
    hGraph: cudaGraph_t,
    resultInfo: *mut cudaGraphExecUpdateResultInfo,
) -> cudaError_t;

#[cuda_hook(proc_id = 900538)]
fn cudaGraphExecEventRecordNodeSetEvent(
    hGraphExec: cudaGraphExec_t,
    hNode: cudaGraphNode_t,
    event: cudaEvent_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 900539)]
fn cudaGraphExecEventWaitNodeSetEvent(
    hGraphExec: cudaGraphExec_t,
    hNode: cudaGraphNode_t,
    event: cudaEvent_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 900543)]
fn cudaGraphExecMemsetNodeSetParams(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    #[host(len = 1)] pNodeParams: *const cudaMemsetParams,
) -> cudaError_t;

#[cuda_hook(proc_id = 900557)]
fn cudaGraphAddMemAllocNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    #[device] pDependencies: *const cudaGraphNode_t, // null
    numDependencies: usize,
    #[host(input, len = 1)] nodeParams: *mut cudaMemAllocNodeParams,
) -> cudaError_t {
    'client_before_send: {
        assert!(pDependencies.is_null());
        assert_eq!(numDependencies, 0);
        let params = unsafe { &*nodeParams };
        assert!(params.accessDescs.is_null());
        assert_eq!(params.accessDescCount, 0);
        assert_eq!(
            params.poolProps.allocType,
            cudaMemAllocationType::cudaMemAllocationTypePinned
        );
        assert_eq!(
            params.poolProps.handleTypes,
            cudaMemAllocationHandleType::cudaMemHandleTypeNone
        );
        assert_eq!(
            params.poolProps.location.type_,
            cudaMemLocationType::cudaMemLocationTypeDevice
        );
        assert!(params.poolProps.win32SecurityAttributes.is_null());
    }
    'client_after_recv: {
        let mut node_params_out: cudaMemAllocNodeParams = unsafe { std::mem::zeroed() };
        node_params_out.recv(channel_receiver).unwrap();
        unsafe {
            std::ptr::write(nodeParams.as_ptr().cast_mut(), node_params_out);
        }
    }
    'server_after_send: {
        nodeParams[0].send(channel_sender).unwrap();
        channel_sender.flush_out().unwrap();
    }
}

#[cuda_hook(proc_id = 900558)]
fn cudaGraphMemAllocNodeGetParams(
    node: cudaGraphNode_t,
    #[host(output, len = 1)] params_out: *mut cudaMemAllocNodeParams,
) -> cudaError_t;

#[cuda_hook(proc_id = 900559)]
fn cudaGraphAddMemFreeNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    #[device] pDependencies: *const cudaGraphNode_t, // null
    numDependencies: usize,
    #[device] dptr: *mut c_void,
) -> cudaError_t {
    'client_before_send: {
        assert!(pDependencies.is_null());
        assert_eq!(numDependencies, 0);
        assert!(!dptr.is_null());
    }
}

#[cuda_hook(proc_id = 900560)]
fn cudaGraphMemFreeNodeGetParams(
    node: cudaGraphNode_t,
    #[host(output, len = std::mem::size_of::<*mut c_void>())] dptr_out: *mut c_void,
) -> cudaError_t;

#[cuda_hook(proc_id = 900550)]
fn cudaGraphExecMemcpyNodeSetParams(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    #[host(len = 1)] pNodeParams: *const cudaMemcpy3DParms,
) -> cudaError_t {
    'client_before_send: {
        let params = unsafe { &*pNodeParams };
        assert!(params.srcArray.is_null());
        assert!(params.dstArray.is_null());
        assert!(!params.srcPtr.ptr.is_null());
        assert!(!params.dstPtr.ptr.is_null());
        assert_eq!(params.kind, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    }
}

#[cuda_hook(proc_id = 900546)]
fn cudaGraphExecMemcpyNodeSetParams1D(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    #[device] dst: *mut c_void,
    #[device] src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    'client_before_send: {
        assert_eq!(kind, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    }
}

#[cuda_custom_hook] // local, resolves symbol then calls 1D graph memcpy APIs
fn cudaGraphExecMemcpyNodeSetParamsToSymbol(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    symbol: *const c_void,
    src: *const c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_custom_hook] // local, resolves symbol then calls 1D graph memcpy APIs
fn cudaGraphExecMemcpyNodeSetParamsFromSymbol(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    dst: *mut c_void,
    symbol: *const c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_hook(proc_id = 900514, async_api)]
fn cudaGraphUpload(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t;

#[cuda_hook(proc_id = 573, async_api)]
fn cudaGraphLaunch(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t;

#[cuda_hook(proc_id = 900551)]
fn cudaGraphNodeSetEnabled(
    hGraphExec: cudaGraphExec_t,
    hNode: cudaGraphNode_t,
    isEnabled: c_uint,
) -> cudaError_t;

#[cuda_hook(proc_id = 900552)]
fn cudaGraphNodeGetEnabled(
    hGraphExec: cudaGraphExec_t,
    hNode: cudaGraphNode_t,
    isEnabled: *mut c_uint,
) -> cudaError_t;

#[cuda_hook(proc_id = 900561)]
fn cudaGraphDebugDotPrint(graph: cudaGraph_t, path: *const c_char, flags: c_uint) -> cudaError_t;

#[cuda_hook(proc_id = 900929)]
fn cudaGraphConditionalHandleCreate(
    pHandle_out: *mut cudaGraphConditionalHandle,
    graph: cudaGraph_t,
    defaultLaunchValue: c_uint,
    flags: c_uint,
) -> cudaError_t;

#[cuda_hook(proc_id = 900930)]
fn cudaGraphConditionalHandleCreate_v2(
    pHandle_out: *mut cudaGraphConditionalHandle,
    graph: cudaGraph_t,
    ctx: cudaExecutionContext_t,
    defaultLaunchValue: c_uint,
    flags: c_uint,
) -> cudaError_t;
