#![expect(non_snake_case)]

use cudasys::nccl::*;
use log::error;
use network::type_impl::send_slice;
use network::{CommChannel, Transportable};
use std::ffi::CStr;

use crate::ServerWorker;

pub fn ncclGetLastErrorExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    log::debug!(target: "ncclGetLastError", "[#{}]", server.id);

    let mut comm = std::mem::MaybeUninit::<ncclComm_t>::uninit();
    comm.recv(&server.channel_receiver)
        .expect("failed to receive comm");
    let comm = unsafe { comm.assume_init() };
    server
        .channel_receiver
        .recv_ts()
        .expect("failed to receive timestamp");

    let text = unsafe {
        let ptr = ncclGetLastError(comm);
        if ptr.is_null() {
            &[][..]
        } else {
            CStr::from_ptr(ptr).to_bytes()
        }
    };

    if let Err(err) = send_slice(text, &server.channel_sender) {
        error!(target: "ncclGetLastError", "failed to send last error text: {err}");
    }
    server
        .channel_sender
        .flush_out()
        .expect("failed to flush ncclGetLastError response");
}
