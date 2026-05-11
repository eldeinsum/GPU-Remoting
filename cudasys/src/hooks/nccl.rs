use crate::types::nccl::*;
use codegen::cuda_hook;
use std::os::raw::*;

#[cuda_hook(proc_id = 3200)]
fn ncclGetVersion(version: *mut c_int) -> ncclResult_t;

#[cuda_hook(proc_id = 3201)]
fn ncclGetUniqueId(uniqueId: *mut ncclUniqueId) -> ncclResult_t;

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

#[cuda_hook(proc_id = 3205)]
fn ncclCommGetAsyncError(comm: ncclComm_t, asyncError: *mut ncclResult_t) -> ncclResult_t;

#[cuda_hook(proc_id = 3206)]
fn ncclGroupStart() -> ncclResult_t;

#[cuda_hook(proc_id = 3207)]
fn ncclGroupEnd() -> ncclResult_t;

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
