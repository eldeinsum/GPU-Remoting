#![expect(non_snake_case)]

use cudasys::nccl::*;
use log::error;
use network::type_impl::send_slice;
use network::{CommChannel, Transportable};
use std::ffi::CStr;
use std::os::raw::{c_int, c_void};
use std::sync::{Mutex, OnceLock};

use crate::ServerWorker;

#[derive(Default)]
struct NcclGroupState {
    depth: usize,
    pending_window_outputs: Vec<PendingWindowOutput>,
}

struct PendingWindowOutput {
    value: Box<usize>,
}

impl PendingWindowOutput {
    fn new() -> Self {
        Self { value: Box::new(0) }
    }

    fn as_mut_ptr(&mut self) -> *mut ncclWindow_t {
        (&mut *self.value as *mut usize).cast::<ncclWindow_t>()
    }

    fn handle(&self) -> ncclWindow_t {
        *self.value as ncclWindow_t
    }
}

fn nccl_group_state() -> &'static Mutex<NcclGroupState> {
    static STATE: OnceLock<Mutex<NcclGroupState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(NcclGroupState::default()))
}

pub fn ncclGroupStartExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    log::debug!(target: "ncclGroupStart", "[#{}]", server.id);
    server
        .channel_receiver
        .recv_ts()
        .expect("failed to receive ncclGroupStart timestamp");

    let result = unsafe { ncclGroupStart() };
    if result == ncclResult_t::ncclSuccess {
        let mut state = nccl_group_state().lock().unwrap();
        state.depth += 1;
    }

    result
        .send(&server.channel_sender)
        .expect("failed to send ncclGroupStart result");
    server
        .channel_sender
        .flush_out()
        .expect("failed to flush ncclGroupStart response");
}

pub fn ncclGroupEndExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    log::debug!(target: "ncclGroupEnd", "[#{}]", server.id);
    server
        .channel_receiver
        .recv_ts()
        .expect("failed to receive ncclGroupEnd timestamp");

    let result = unsafe { ncclGroupEnd() };
    let handles = {
        let mut state = nccl_group_state().lock().unwrap();
        if result == ncclResult_t::ncclSuccess {
            if state.depth > 0 {
                state.depth -= 1;
            }
            if state.depth == 0 {
                state
                    .pending_window_outputs
                    .drain(..)
                    .map(|slot| slot.handle())
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        } else {
            state.depth = 0;
            state.pending_window_outputs.clear();
            Vec::new()
        }
    };

    result
        .send(&server.channel_sender)
        .expect("failed to send ncclGroupEnd result");
    handles
        .len()
        .send(&server.channel_sender)
        .expect("failed to send ncclGroupEnd completion count");
    for handle in handles {
        handle
            .send(&server.channel_sender)
            .expect("failed to send completed NCCL window handle");
    }
    server
        .channel_sender
        .flush_out()
        .expect("failed to flush ncclGroupEnd response");
}

pub fn ncclCommWindowRegisterExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    log::debug!(target: "ncclCommWindowRegister", "[#{}]", server.id);

    let mut comm = std::mem::MaybeUninit::<ncclComm_t>::uninit();
    comm.recv(&server.channel_receiver)
        .expect("failed to receive comm");
    let comm = unsafe { comm.assume_init() };

    let mut buff = std::mem::MaybeUninit::<*mut c_void>::uninit();
    buff.recv(&server.channel_receiver)
        .expect("failed to receive buff");
    let buff = unsafe { buff.assume_init() };

    let mut size = 0usize;
    size.recv(&server.channel_receiver)
        .expect("failed to receive size");

    let mut win_flags = 0 as c_int;
    win_flags
        .recv(&server.channel_receiver)
        .expect("failed to receive winFlags");

    server
        .channel_receiver
        .recv_ts()
        .expect("failed to receive ncclCommWindowRegister timestamp");

    let mut win_storage = PendingWindowOutput::new();
    let result =
        unsafe { ncclCommWindowRegister(comm, buff, size, win_storage.as_mut_ptr(), win_flags) };
    let deferred =
        result == ncclResult_t::ncclSuccess && nccl_group_state().lock().unwrap().depth > 0;
    let handle = if deferred {
        let mut state = nccl_group_state().lock().unwrap();
        state.pending_window_outputs.push(win_storage);
        std::ptr::null_mut()
    } else {
        win_storage.handle()
    };

    result
        .send(&server.channel_sender)
        .expect("failed to send ncclCommWindowRegister result");
    deferred
        .send(&server.channel_sender)
        .expect("failed to send ncclCommWindowRegister deferred flag");
    handle
        .send(&server.channel_sender)
        .expect("failed to send ncclCommWindowRegister handle");
    server
        .channel_sender
        .flush_out()
        .expect("failed to flush ncclCommWindowRegister response");
}

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
