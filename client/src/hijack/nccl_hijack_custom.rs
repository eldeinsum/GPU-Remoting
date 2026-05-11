use cudasys::types::nccl::*;
use network::type_impl::recv_slice;
use network::{CommChannelInner, Transportable};
use std::collections::BTreeMap;
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::{Mutex, OnceLock};

use crate::{CLIENT_THREAD, ClientThread};

#[derive(Default)]
struct NcclGroupState {
    depth: usize,
    pending_window_outputs: Vec<usize>,
}

fn nccl_group_state() -> &'static Mutex<NcclGroupState> {
    static STATE: OnceLock<Mutex<NcclGroupState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(NcclGroupState::default()))
}

fn real_nccl_handle() -> *mut c_void {
    static HANDLE: OnceLock<usize> = OnceLock::new();
    let handle = *HANDLE.get_or_init(|| {
        let mut candidates = Vec::new();
        if let Ok(path) = std::env::var("GPU_REMOTING_REAL_NCCL") {
            candidates.push(path);
        }
        candidates.extend([
            "/usr/lib/x86_64-linux-gnu/libnccl.so.2".to_string(),
            "/lib/x86_64-linux-gnu/libnccl.so.2".to_string(),
            "/usr/local/cuda/lib64/libnccl.so.2".to_string(),
            "/usr/lib/x86_64-linux-gnu/libnccl.so".to_string(),
            "/lib/x86_64-linux-gnu/libnccl.so".to_string(),
            "/usr/local/cuda/lib64/libnccl.so".to_string(),
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
            "failed to load real libnccl; tried: {}",
            candidates.join(", ")
        );
    });
    handle as *mut c_void
}

fn load_nccl_symbol(name: &str) -> usize {
    crate::dl::dlsym_handle(real_nccl_handle(), name)
}

macro_rules! forward_nccl {
    ($name:ident($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty) => {
        #[no_mangle]
        pub extern "C" fn $name($($arg: $ty),*) -> $ret {
            type FnTy = extern "C" fn($($ty),*) -> $ret;
            static FN: OnceLock<usize> = OnceLock::new();
            let ptr = *FN.get_or_init(|| load_nccl_symbol(stringify!($name)));
            let func: FnTy = unsafe { std::mem::transmute(ptr) };
            crate::dl::with_real_cuda_dlopen(|| func($($arg),*))
        }
    };
}

fn nccl_error_text(result: ncclResult_t) -> &'static str {
    match result {
        ncclResult_t::ncclSuccess => "no error",
        ncclResult_t::ncclUnhandledCudaError => "unhandled cuda error",
        ncclResult_t::ncclSystemError => "unhandled system error",
        ncclResult_t::ncclInternalError => "internal error",
        ncclResult_t::ncclInvalidArgument => "invalid argument",
        ncclResult_t::ncclInvalidUsage => "invalid usage",
        ncclResult_t::ncclRemoteError => "remote process exited or there was a network error",
        ncclResult_t::ncclInProgress => "NCCL operation in progress",
        ncclResult_t::ncclTimeout => "NCCL operation timed out",
        ncclResult_t::ncclNumResults => "invalid NCCL result",
    }
}

#[no_mangle]
pub extern "C" fn ncclGroupStart() -> ncclResult_t {
    log::debug!(target: "ncclGroupStart", "remoted");

    let result = CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "ncclGroupStart", "[#{}]", client.id);
        let ClientThread {
            channel_sender,
            channel_receiver,
            ..
        } = client;

        3206i32.send(channel_sender).unwrap();
        channel_sender.flush_out().unwrap();

        let mut result = ncclResult_t::ncclSuccess;
        result.recv(channel_receiver).unwrap();
        channel_receiver.recv_ts().unwrap();
        result
    });

    if result == ncclResult_t::ncclSuccess {
        let mut state = nccl_group_state().lock().unwrap();
        state.depth += 1;
    }
    result
}

#[no_mangle]
pub extern "C" fn ncclGroupEnd() -> ncclResult_t {
    log::debug!(target: "ncclGroupEnd", "remoted");

    let (result, handles) = CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "ncclGroupEnd", "[#{}]", client.id);
        let ClientThread {
            channel_sender,
            channel_receiver,
            ..
        } = client;

        3207i32.send(channel_sender).unwrap();
        channel_sender.flush_out().unwrap();

        let mut result = ncclResult_t::ncclSuccess;
        result.recv(channel_receiver).unwrap();

        let mut count = 0usize;
        count.recv(channel_receiver).unwrap();
        let mut handles = Vec::with_capacity(count);
        for _ in 0..count {
            let mut handle = std::mem::MaybeUninit::<ncclWindow_t>::uninit();
            handle.recv(channel_receiver).unwrap();
            handles.push(unsafe { handle.assume_init() });
        }

        channel_receiver.recv_ts().unwrap();
        (result, handles)
    });

    let mut state = nccl_group_state().lock().unwrap();
    if result == ncclResult_t::ncclSuccess {
        if state.depth > 0 {
            state.depth -= 1;
        }
        if state.depth == 0 {
            if handles.len() != state.pending_window_outputs.len() {
                log::error!(
                    target: "ncclGroupEnd",
                    "window handle completion mismatch: client pending {}, server returned {}",
                    state.pending_window_outputs.len(),
                    handles.len(),
                );
                state.pending_window_outputs.clear();
                return ncclResult_t::ncclInternalError;
            }

            for (out_ptr, handle) in state.pending_window_outputs.drain(..).zip(handles) {
                unsafe {
                    *(out_ptr as *mut ncclWindow_t) = handle;
                }
            }
        } else if !handles.is_empty() {
            log::error!(
                target: "ncclGroupEnd",
                "received window handle completions before the outermost group ended",
            );
            return ncclResult_t::ncclInternalError;
        }
    } else {
        state.depth = 0;
        state.pending_window_outputs.clear();
    }

    result
}

#[no_mangle]
pub extern "C" fn ncclCommWindowRegister(
    comm: ncclComm_t,
    buff: *mut c_void,
    size: usize,
    win: *mut ncclWindow_t,
    winFlags: c_int,
) -> ncclResult_t {
    log::debug!(target: "ncclCommWindowRegister", "{comm:p}, {buff:p}, {size}, {win:p}, {winFlags}");
    if win.is_null() {
        return ncclResult_t::ncclInvalidArgument;
    }

    let (result, deferred, handle) = CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "ncclCommWindowRegister", "[#{}]", client.id);
        let ClientThread {
            channel_sender,
            channel_receiver,
            ..
        } = client;

        3239i32.send(channel_sender).unwrap();
        comm.send(channel_sender).unwrap();
        buff.send(channel_sender).unwrap();
        size.send(channel_sender).unwrap();
        winFlags.send(channel_sender).unwrap();
        channel_sender.flush_out().unwrap();

        let mut result = ncclResult_t::ncclSuccess;
        result.recv(channel_receiver).unwrap();
        let mut deferred = false;
        deferred.recv(channel_receiver).unwrap();
        let mut handle = std::mem::MaybeUninit::<ncclWindow_t>::uninit();
        handle.recv(channel_receiver).unwrap();
        channel_receiver.recv_ts().unwrap();

        (result, deferred, unsafe { handle.assume_init() })
    });

    if result == ncclResult_t::ncclSuccess {
        if deferred {
            nccl_group_state()
                .lock()
                .unwrap()
                .pending_window_outputs
                .push(win as usize);
        } else {
            unsafe {
                *win = handle;
            }
        }
    }

    result
}

#[no_mangle]
pub extern "C" fn ncclGetErrorString(result: ncclResult_t) -> *const c_char {
    log::debug!(target: "ncclGetErrorString", "{result:?}");
    static ERROR_TEXTS: OnceLock<Mutex<BTreeMap<c_int, CString>>> = OnceLock::new();
    let code = result as c_int;
    let mut texts = ERROR_TEXTS
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .unwrap();
    let text = texts
        .entry(code)
        .or_insert_with(|| CString::new(nccl_error_text(result)).unwrap());
    text.as_ptr()
}

#[no_mangle]
pub extern "C" fn ncclGetLastError(comm: ncclComm_t) -> *const c_char {
    log::debug!(target: "ncclGetLastError", "{comm:p}");
    static LAST_ERROR_TEXT: OnceLock<Mutex<CString>> = OnceLock::new();

    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "ncclGetLastError", "[#{}]", client.id);
        let ClientThread {
            channel_sender,
            channel_receiver,
            ..
        } = client;

        3238i32.send(channel_sender).unwrap();
        comm.send(channel_sender).unwrap();
        channel_sender.flush_out().unwrap();

        let bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
        channel_receiver.recv_ts().unwrap();
        let text = CString::new(bytes.into_vec()).unwrap_or_else(|_| {
            CString::new("NCCL returned an invalid last-error string").unwrap()
        });
        let mut last_error = LAST_ERROR_TEXT
            .get_or_init(|| Mutex::new(CString::default()))
            .lock()
            .unwrap();
        *last_error = text;
        last_error.as_ptr()
    })
}

forward_nccl!(ncclParamBind(
    out: *mut *mut ncclParamHandle_t,
    key: *const c_char,
) -> ncclResult_t);
forward_nccl!(ncclParamGetI8(
    h: *mut ncclParamHandle_t,
    out: *mut i8,
) -> ncclResult_t);
forward_nccl!(ncclParamGetI16(
    h: *mut ncclParamHandle_t,
    out: *mut i16,
) -> ncclResult_t);
forward_nccl!(ncclParamGetI32(
    h: *mut ncclParamHandle_t,
    out: *mut i32,
) -> ncclResult_t);
forward_nccl!(ncclParamGetI64(
    h: *mut ncclParamHandle_t,
    out: *mut i64,
) -> ncclResult_t);
forward_nccl!(ncclParamGetU8(
    h: *mut ncclParamHandle_t,
    out: *mut u8,
) -> ncclResult_t);
forward_nccl!(ncclParamGetU16(
    h: *mut ncclParamHandle_t,
    out: *mut u16,
) -> ncclResult_t);
forward_nccl!(ncclParamGetU32(
    h: *mut ncclParamHandle_t,
    out: *mut u32,
) -> ncclResult_t);
forward_nccl!(ncclParamGetU64(
    h: *mut ncclParamHandle_t,
    out: *mut u64,
) -> ncclResult_t);
forward_nccl!(ncclParamGetStr(
    h: *mut ncclParamHandle_t,
    out: *mut *const c_char,
) -> ncclResult_t);
forward_nccl!(ncclParamGet(
    h: *mut ncclParamHandle_t,
    out: *mut c_void,
    maxLen: c_int,
    len: *mut c_int,
) -> ncclResult_t);
forward_nccl!(ncclParamGetParameter(
    key: *const c_char,
    value: *mut *const c_char,
    valueLen: *mut c_int,
) -> ncclResult_t);
forward_nccl!(ncclParamGetAllParameterKeys(
    table: *mut *mut *const c_char,
    tableLen: *mut c_int,
) -> ncclResult_t);
forward_nccl!(ncclParamDumpAll() -> ());
