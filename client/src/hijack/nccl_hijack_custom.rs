use cudasys::types::nccl::ncclResult_t;
use std::collections::BTreeMap;
use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::sync::{Mutex, OnceLock};

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
