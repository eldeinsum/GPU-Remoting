use crate::types::nccl::*;
use codegen::{cuda_custom_hook, cuda_hook};
use std::os::raw::*;

#[cuda_hook(proc_id = 3200)]
fn ncclGetVersion(version: *mut c_int) -> ncclResult_t;

#[cuda_hook(proc_id = 3201)]
fn ncclGetUniqueId(uniqueId: *mut ncclUniqueId) -> ncclResult_t;

#[cuda_custom_hook] // local: returns a client-owned C string
fn ncclGetErrorString(result: ncclResult_t) -> *const c_char;

#[cuda_custom_hook(proc_id = 3238)] // remoted: returns a client-owned copy of server text
fn ncclGetLastError(comm: ncclComm_t) -> *const c_char;

#[cuda_hook(proc_id = 3202)]
fn ncclCommInitRank(
    comm: *mut ncclComm_t,
    nranks: c_int,
    commId: ncclUniqueId,
    rank: c_int,
) -> ncclResult_t;

#[cuda_hook(proc_id = 3203)]
fn ncclCommDestroy(comm: ncclComm_t) -> ncclResult_t;

#[cuda_hook(proc_id = 3216)]
fn ncclCommFinalize(comm: ncclComm_t) -> ncclResult_t;

#[cuda_hook(proc_id = 3204)]
fn ncclCommAbort(comm: ncclComm_t) -> ncclResult_t;

#[cuda_hook(proc_id = 3235)]
fn ncclCommRevoke(comm: ncclComm_t, revokeFlags: c_int) -> ncclResult_t;

#[cuda_hook(proc_id = 3205)]
fn ncclCommGetAsyncError(comm: ncclComm_t, asyncError: *mut ncclResult_t) -> ncclResult_t;

#[cuda_custom_hook(proc_id = 3206)] // remoted: tracks grouped output completion
fn ncclGroupStart() -> ncclResult_t;

#[cuda_custom_hook(proc_id = 3207)] // remoted: completes grouped output handles
fn ncclGroupEnd() -> ncclResult_t;

#[cuda_hook(proc_id = 3234)]
fn ncclGroupSimulateEnd(#[skip] simInfo: *mut ncclSimInfo_t) -> ncclResult_t {
    'client_before_send: {
        if simInfo.is_null() {
            return ncclResult_t::ncclInvalidArgument;
        }
        let sim_info_in = unsafe { *simInfo };
    }
    'client_extra_send: {
        sim_info_in.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut sim_info = std::mem::MaybeUninit::<ncclSimInfo_t>::uninit();
        sim_info.recv(channel_receiver).unwrap();
        let sim_info_ptr = sim_info.as_mut_ptr();
    }
    'server_execution: {
        let result = unsafe { ncclGroupSimulateEnd(sim_info_ptr) };
    }
    'server_after_send: {
        if result == ncclResult_t::ncclSuccess {
            unsafe { *sim_info_ptr }.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == ncclResult_t::ncclSuccess {
            let mut sim_info_out = std::mem::MaybeUninit::<ncclSimInfo_t>::uninit();
            sim_info_out.recv(channel_receiver).unwrap();
            unsafe {
                *simInfo = sim_info_out.assume_init();
            }
        }
    }
}

#[cuda_hook(proc_id = 3208, async_api)]
fn ncclSend(
    #[device] sendbuff: *const c_void,
    count: usize,
    datatype: ncclDataType_t,
    peer: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

#[cuda_hook(proc_id = 3209, async_api)]
fn ncclRecv(
    #[device] recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    peer: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

#[cuda_hook(proc_id = 3210, async_api)]
fn ncclAllReduce(
    #[device] sendbuff: *const c_void,
    #[device] recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    op: ncclRedOp_t,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

#[cuda_hook(proc_id = 3211)]
fn ncclCommInitRankConfig(
    comm: *mut ncclComm_t,
    nranks: c_int,
    commId: ncclUniqueId,
    rank: c_int,
    #[host(input)] config: *mut ncclConfig_t,
) -> ncclResult_t;

#[cuda_hook(proc_id = 3212)]
fn ncclCommCount(comm: ncclComm_t, count: *mut c_int) -> ncclResult_t;

#[cuda_hook(proc_id = 3213)]
fn ncclCommUserRank(comm: ncclComm_t, rank: *mut c_int) -> ncclResult_t;

#[cuda_hook(proc_id = 3214, async_api)]
fn ncclBcast(
    #[device] buff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    root: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

#[cuda_hook(proc_id = 3215, async_api)]
fn ncclAllGather(
    #[device] sendbuff: *const c_void,
    #[device] recvbuff: *mut c_void,
    sendcount: usize,
    datatype: ncclDataType_t,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

#[cuda_hook(proc_id = 3217)]
fn ncclCommInitAll(
    #[host(output, len = ndev)] comm: *mut ncclComm_t,
    ndev: c_int,
    #[host(len = ndev)] devlist: *const c_int,
) -> ncclResult_t;

#[cuda_hook(proc_id = 3218)]
fn ncclCommCuDevice(comm: ncclComm_t, device: *mut c_int) -> ncclResult_t;

#[cuda_hook(proc_id = 3219)]
fn ncclCommGetUniqueId(comm: ncclComm_t, uniqueId: *mut ncclUniqueId) -> ncclResult_t;

#[cuda_hook(proc_id = 3220, async_api)]
fn ncclBroadcast(
    #[device] sendbuff: *const c_void,
    #[device] recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    root: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

#[cuda_hook(proc_id = 3221, async_api)]
fn ncclReduce(
    #[device] sendbuff: *const c_void,
    #[device] recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    op: ncclRedOp_t,
    root: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

#[cuda_hook(proc_id = 3222, async_api)]
fn ncclReduceScatter(
    #[device] sendbuff: *const c_void,
    #[device] recvbuff: *mut c_void,
    recvcount: usize,
    datatype: ncclDataType_t,
    op: ncclRedOp_t,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

#[cuda_hook(proc_id = 3223, async_api)]
fn ncclAlltoAll(
    #[device] sendbuff: *const c_void,
    #[device] recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

#[cuda_hook(proc_id = 3224, async_api)]
fn ncclGather(
    #[device] sendbuff: *const c_void,
    #[device] recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    root: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

#[cuda_hook(proc_id = 3225, async_api)]
fn ncclScatter(
    #[device] sendbuff: *const c_void,
    #[device] recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    root: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

#[cuda_hook(proc_id = 3226)]
fn ncclMemAlloc(ptr: *mut *mut c_void, size: usize) -> ncclResult_t;

#[cuda_hook(proc_id = 3227, async_api = false)]
fn ncclMemFree(#[device] ptr: *mut c_void) -> ncclResult_t;

#[cuda_hook(proc_id = 3228)]
fn ncclCommRegister(
    comm: ncclComm_t,
    #[device] buff: *mut c_void,
    size: usize,
    handle: *mut *mut c_void,
) -> ncclResult_t;

#[cuda_hook(proc_id = 3229)]
fn ncclCommDeregister(comm: ncclComm_t, #[device] handle: *mut c_void) -> ncclResult_t;

#[cuda_hook(proc_id = 3230)]
fn ncclCommInitRankScalable(
    newcomm: *mut ncclComm_t,
    nranks: c_int,
    myrank: c_int,
    nId: c_int,
    #[host(input, len = nId)] commIds: *mut ncclUniqueId,
    #[skip] config: *mut ncclConfig_t,
) -> ncclResult_t {
    'client_before_send: {
        let config_present = !config.is_null();
        let config_value = if config_present {
            Some(unsafe { *config })
        } else {
            None
        };
    }
    'client_extra_send: {
        config_present.send(channel_sender).unwrap();
        if let Some(config_value) = config_value {
            config_value.send(channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut config_present = false;
        config_present.recv(channel_receiver).unwrap();
        let mut config_value = std::mem::MaybeUninit::<ncclConfig_t>::uninit();
        let config_arg = if config_present {
            config_value.recv(channel_receiver).unwrap();
            config_value.as_mut_ptr()
        } else {
            std::ptr::null_mut()
        };
    }
    'server_execution: {
        let result = unsafe {
            ncclCommInitRankScalable(newcomm__ptr, nranks, myrank, nId, commIds__ptr, config_arg)
        };
    }
}

#[cuda_hook(proc_id = 3231)]
fn ncclRedOpCreatePreMulSum(
    op: *mut ncclRedOp_t,
    #[skip] scalar: *mut c_void,
    datatype: ncclDataType_t,
    residence: ncclScalarResidence_t,
    comm: ncclComm_t,
) -> ncclResult_t {
    'client_before_send: {
        let scalar_is_host = matches!(residence, ncclScalarResidence_t::ncclScalarHostImmediate);
        let scalar_addr = scalar as usize;
        let scalar_bytes = if scalar_is_host {
            assert!(!scalar.is_null());
            let scalar_len = match datatype {
                ncclDataType_t::ncclInt8
                | ncclDataType_t::ncclUint8
                | ncclDataType_t::ncclFloat8e4m3
                | ncclDataType_t::ncclFloat8e5m2 => 1,
                ncclDataType_t::ncclFloat16 | ncclDataType_t::ncclBfloat16 => 2,
                ncclDataType_t::ncclInt32
                | ncclDataType_t::ncclUint32
                | ncclDataType_t::ncclFloat32 => 4,
                ncclDataType_t::ncclInt64
                | ncclDataType_t::ncclUint64
                | ncclDataType_t::ncclFloat64 => 8,
                ncclDataType_t::ncclNumTypes => 0,
            };
            unsafe { std::slice::from_raw_parts(scalar.cast::<u8>(), scalar_len).to_vec() }
        } else {
            Vec::<u8>::new()
        };
    }
    'client_extra_send: {
        scalar_is_host.send(channel_sender).unwrap();
        scalar_addr.send(channel_sender).unwrap();
        if scalar_is_host {
            send_slice(&scalar_bytes, channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut scalar_is_host = false;
        scalar_is_host.recv(channel_receiver).unwrap();
        let mut scalar_addr = 0usize;
        scalar_addr.recv(channel_receiver).unwrap();
        let scalar_bytes = if scalar_is_host {
            recv_slice::<u8, _>(channel_receiver).unwrap()
        } else {
            Box::<[u8]>::default()
        };
        let mut scalar_storage = [0u64; 1];
        if scalar_is_host {
            let scalar_dst = unsafe {
                std::slice::from_raw_parts_mut(
                    scalar_storage.as_mut_ptr().cast::<u8>(),
                    scalar_bytes.len(),
                )
            };
            scalar_dst.copy_from_slice(&scalar_bytes);
        }
        let scalar_arg = if scalar_is_host {
            scalar_storage.as_mut_ptr().cast::<c_void>()
        } else {
            scalar_addr as *mut c_void
        };
    }
    'server_execution: {
        let result =
            unsafe { ncclRedOpCreatePreMulSum(op__ptr, scalar_arg, datatype, residence, comm) };
    }
}

#[cuda_hook(proc_id = 3232)]
fn ncclRedOpDestroy(op: ncclRedOp_t, comm: ncclComm_t) -> ncclResult_t;

#[cuda_hook(proc_id = 3233)]
fn ncclCommMemStats(comm: ncclComm_t, stat: ncclCommMemStat_t, value: *mut u64) -> ncclResult_t;

#[cuda_custom_hook(proc_id = 3239)] // remoted: window handles may complete at group end
fn ncclCommWindowRegister(
    comm: ncclComm_t,
    buff: *mut c_void,
    size: usize,
    win: *mut ncclWindow_t,
    winFlags: c_int,
) -> ncclResult_t;

#[cuda_hook(proc_id = 3240)]
fn ncclCommWindowDeregister(comm: ncclComm_t, win: ncclWindow_t) -> ncclResult_t;

#[cuda_hook(proc_id = 3241)]
fn ncclWinGetUserPtr(
    comm: ncclComm_t,
    win: ncclWindow_t,
    outUserPtr: *mut *mut c_void,
) -> ncclResult_t;

#[cuda_hook(proc_id = 3236)]
fn ncclCommSuspend(comm: ncclComm_t, flags: c_int) -> ncclResult_t;

#[cuda_hook(proc_id = 3237)]
fn ncclCommResume(comm: ncclComm_t) -> ncclResult_t;

#[cuda_custom_hook] // local: NCCL parameter registry owns returned handles
fn ncclParamBind(out: *mut *mut ncclParamHandle_t, key: *const c_char) -> ncclResult_t;

#[cuda_custom_hook] // local
fn ncclParamGetI8(h: *mut ncclParamHandle_t, out: *mut i8) -> ncclResult_t;

#[cuda_custom_hook] // local
fn ncclParamGetI16(h: *mut ncclParamHandle_t, out: *mut i16) -> ncclResult_t;

#[cuda_custom_hook] // local
fn ncclParamGetI32(h: *mut ncclParamHandle_t, out: *mut i32) -> ncclResult_t;

#[cuda_custom_hook] // local
fn ncclParamGetI64(h: *mut ncclParamHandle_t, out: *mut i64) -> ncclResult_t;

#[cuda_custom_hook] // local
fn ncclParamGetU8(h: *mut ncclParamHandle_t, out: *mut u8) -> ncclResult_t;

#[cuda_custom_hook] // local
fn ncclParamGetU16(h: *mut ncclParamHandle_t, out: *mut u16) -> ncclResult_t;

#[cuda_custom_hook] // local
fn ncclParamGetU32(h: *mut ncclParamHandle_t, out: *mut u32) -> ncclResult_t;

#[cuda_custom_hook] // local
fn ncclParamGetU64(h: *mut ncclParamHandle_t, out: *mut u64) -> ncclResult_t;

#[cuda_custom_hook] // local: NCCL parameter registry owns returned string
fn ncclParamGetStr(h: *mut ncclParamHandle_t, out: *mut *const c_char) -> ncclResult_t;

#[cuda_custom_hook] // local
fn ncclParamGet(
    h: *mut ncclParamHandle_t,
    out: *mut c_void,
    maxLen: c_int,
    len: *mut c_int,
) -> ncclResult_t;

#[cuda_custom_hook] // local: NCCL parameter registry owns returned string
fn ncclParamGetParameter(
    key: *const c_char,
    value: *mut *const c_char,
    valueLen: *mut c_int,
) -> ncclResult_t;

#[cuda_custom_hook] // local: NCCL parameter registry owns returned table
fn ncclParamGetAllParameterKeys(
    table: *mut *mut *const c_char,
    tableLen: *mut c_int,
) -> ncclResult_t;

#[cuda_custom_hook] // local
fn ncclParamDumpAll();
