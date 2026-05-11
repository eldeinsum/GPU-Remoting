#![expect(non_snake_case)]

use cudasys::nccl::*;
use log::error;
use network::type_impl::{recv_slice, send_slice};
use network::{CommChannel, Transportable};
use std::ffi::CStr;
use std::os::raw::{c_int, c_void};
use std::sync::{Mutex, OnceLock};

use crate::ServerWorker;

#[derive(Default)]
struct NcclGroupState {
    depth: usize,
    pending_window_outputs: Vec<PendingWindowOutput>,
    pending_comm_outputs: Vec<PendingCommOutput>,
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

struct PendingCommOutput {
    value: Box<usize>,
}

impl PendingCommOutput {
    fn new() -> Self {
        Self { value: Box::new(0) }
    }

    fn as_mut_ptr(&mut self) -> *mut ncclComm_t {
        (&mut *self.value as *mut usize).cast::<ncclComm_t>()
    }

    fn handle(&self) -> ncclComm_t {
        *self.value as ncclComm_t
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
    let (window_handles, comm_handles) = {
        let mut state = nccl_group_state().lock().unwrap();
        if result == ncclResult_t::ncclSuccess {
            if state.depth > 0 {
                state.depth -= 1;
            }
            if state.depth == 0 {
                let window_handles = state
                    .pending_window_outputs
                    .drain(..)
                    .map(|slot| slot.handle())
                    .collect::<Vec<_>>();
                let comm_handles = state
                    .pending_comm_outputs
                    .drain(..)
                    .map(|slot| slot.handle())
                    .collect::<Vec<_>>();
                (window_handles, comm_handles)
            } else {
                (Vec::new(), Vec::new())
            }
        } else {
            state.depth = 0;
            state.pending_window_outputs.clear();
            state.pending_comm_outputs.clear();
            (Vec::new(), Vec::new())
        }
    };

    result
        .send(&server.channel_sender)
        .expect("failed to send ncclGroupEnd result");
    window_handles
        .len()
        .send(&server.channel_sender)
        .expect("failed to send ncclGroupEnd window completion count");
    for handle in window_handles {
        handle
            .send(&server.channel_sender)
            .expect("failed to send completed NCCL window handle");
    }
    comm_handles
        .len()
        .send(&server.channel_sender)
        .expect("failed to send ncclGroupEnd communicator completion count");
    for handle in comm_handles {
        handle
            .send(&server.channel_sender)
            .expect("failed to send completed NCCL communicator handle");
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

pub fn ncclCommSplitExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    log::debug!(target: "ncclCommSplit", "[#{}]", server.id);

    let mut comm = std::mem::MaybeUninit::<ncclComm_t>::uninit();
    comm.recv(&server.channel_receiver)
        .expect("failed to receive comm");
    let comm = unsafe { comm.assume_init() };

    let mut color = 0 as c_int;
    color
        .recv(&server.channel_receiver)
        .expect("failed to receive color");

    let mut key = 0 as c_int;
    key.recv(&server.channel_receiver)
        .expect("failed to receive key");

    let mut config_present = false;
    config_present
        .recv(&server.channel_receiver)
        .expect("failed to receive config presence");
    let mut config_value = std::mem::MaybeUninit::<ncclConfig_t>::uninit();
    let config_arg = if config_present {
        config_value
            .recv(&server.channel_receiver)
            .expect("failed to receive config");
        config_value.as_mut_ptr()
    } else {
        std::ptr::null_mut()
    };

    server
        .channel_receiver
        .recv_ts()
        .expect("failed to receive ncclCommSplit timestamp");

    let mut comm_storage = PendingCommOutput::new();
    let result = unsafe { ncclCommSplit(comm, color, key, comm_storage.as_mut_ptr(), config_arg) };
    let deferred =
        result == ncclResult_t::ncclSuccess && nccl_group_state().lock().unwrap().depth > 0;
    let handle = if deferred {
        let mut state = nccl_group_state().lock().unwrap();
        state.pending_comm_outputs.push(comm_storage);
        std::ptr::null_mut()
    } else {
        comm_storage.handle()
    };

    result
        .send(&server.channel_sender)
        .expect("failed to send ncclCommSplit result");
    deferred
        .send(&server.channel_sender)
        .expect("failed to send ncclCommSplit deferred flag");
    handle
        .send(&server.channel_sender)
        .expect("failed to send ncclCommSplit handle");
    server
        .channel_sender
        .flush_out()
        .expect("failed to flush ncclCommSplit response");
}

pub fn ncclCommShrinkExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    log::debug!(target: "ncclCommShrink", "[#{}]", server.id);

    let mut comm = std::mem::MaybeUninit::<ncclComm_t>::uninit();
    comm.recv(&server.channel_receiver)
        .expect("failed to receive comm");
    let comm = unsafe { comm.assume_init() };

    let exclude_ranks = recv_slice::<c_int, _>(&server.channel_receiver)
        .expect("failed to receive excludeRanksList");
    let mut exclude_ranks = exclude_ranks.into_vec();
    let exclude_ranks_arg = if exclude_ranks.is_empty() {
        std::ptr::null_mut()
    } else {
        exclude_ranks.as_mut_ptr()
    };
    let exclude_ranks_count = exclude_ranks.len() as c_int;

    let mut shrink_flags = 0 as c_int;
    shrink_flags
        .recv(&server.channel_receiver)
        .expect("failed to receive shrinkFlags");

    let mut config_present = false;
    config_present
        .recv(&server.channel_receiver)
        .expect("failed to receive config presence");
    let mut config_value = std::mem::MaybeUninit::<ncclConfig_t>::uninit();
    let config_arg = if config_present {
        config_value
            .recv(&server.channel_receiver)
            .expect("failed to receive config");
        config_value.as_mut_ptr()
    } else {
        std::ptr::null_mut()
    };

    server
        .channel_receiver
        .recv_ts()
        .expect("failed to receive ncclCommShrink timestamp");

    let mut comm_storage = PendingCommOutput::new();
    let result = unsafe {
        ncclCommShrink(
            comm,
            exclude_ranks_arg,
            exclude_ranks_count,
            comm_storage.as_mut_ptr(),
            config_arg,
            shrink_flags,
        )
    };
    let deferred =
        result == ncclResult_t::ncclSuccess && nccl_group_state().lock().unwrap().depth > 0;
    let handle = if deferred {
        let mut state = nccl_group_state().lock().unwrap();
        state.pending_comm_outputs.push(comm_storage);
        std::ptr::null_mut()
    } else {
        comm_storage.handle()
    };

    result
        .send(&server.channel_sender)
        .expect("failed to send ncclCommShrink result");
    deferred
        .send(&server.channel_sender)
        .expect("failed to send ncclCommShrink deferred flag");
    handle
        .send(&server.channel_sender)
        .expect("failed to send ncclCommShrink handle");
    server
        .channel_sender
        .flush_out()
        .expect("failed to flush ncclCommShrink response");
}

pub fn ncclCommGrowExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    log::debug!(target: "ncclCommGrow", "[#{}]", server.id);

    let mut comm = std::mem::MaybeUninit::<ncclComm_t>::uninit();
    comm.recv(&server.channel_receiver)
        .expect("failed to receive comm");
    let comm = unsafe { comm.assume_init() };

    let mut n_ranks = 0 as c_int;
    n_ranks
        .recv(&server.channel_receiver)
        .expect("failed to receive nRanks");

    let mut unique_id_present = false;
    unique_id_present
        .recv(&server.channel_receiver)
        .expect("failed to receive uniqueId presence");
    let mut unique_id_value = std::mem::MaybeUninit::<ncclUniqueId>::uninit();
    let unique_id_arg = if unique_id_present {
        unique_id_value
            .recv(&server.channel_receiver)
            .expect("failed to receive uniqueId");
        unique_id_value.as_ptr()
    } else {
        std::ptr::null()
    };

    let mut rank = 0 as c_int;
    rank.recv(&server.channel_receiver)
        .expect("failed to receive rank");

    let mut config_present = false;
    config_present
        .recv(&server.channel_receiver)
        .expect("failed to receive config presence");
    let mut config_value = std::mem::MaybeUninit::<ncclConfig_t>::uninit();
    let config_arg = if config_present {
        config_value
            .recv(&server.channel_receiver)
            .expect("failed to receive config");
        config_value.as_mut_ptr()
    } else {
        std::ptr::null_mut()
    };

    server
        .channel_receiver
        .recv_ts()
        .expect("failed to receive ncclCommGrow timestamp");

    let mut comm_storage = PendingCommOutput::new();
    let result = unsafe {
        ncclCommGrow(
            comm,
            n_ranks,
            unique_id_arg,
            rank,
            comm_storage.as_mut_ptr(),
            config_arg,
        )
    };
    let deferred =
        result == ncclResult_t::ncclSuccess && nccl_group_state().lock().unwrap().depth > 0;
    let handle = if deferred {
        let mut state = nccl_group_state().lock().unwrap();
        state.pending_comm_outputs.push(comm_storage);
        std::ptr::null_mut()
    } else {
        comm_storage.handle()
    };

    result
        .send(&server.channel_sender)
        .expect("failed to send ncclCommGrow result");
    deferred
        .send(&server.channel_sender)
        .expect("failed to send ncclCommGrow deferred flag");
    handle
        .send(&server.channel_sender)
        .expect("failed to send ncclCommGrow handle");
    server
        .channel_sender
        .flush_out()
        .expect("failed to flush ncclCommGrow response");
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
