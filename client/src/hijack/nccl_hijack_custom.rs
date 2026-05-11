use cudasys::types::nccl::*;
use std::collections::BTreeMap;
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::{Mutex, OnceLock};

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
