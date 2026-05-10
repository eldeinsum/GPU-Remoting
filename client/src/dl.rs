use std::cell::Cell;
use std::ffi::*;
use std::sync::OnceLock;
use std::{env, mem};

type DlopenFn = extern "C" fn(*const c_char, c_int) -> *mut c_void;

thread_local! {
    static REAL_CUDA_DLOPEN_DEPTH: Cell<u32> = const { Cell::new(0) };
}

pub(crate) fn with_real_cuda_dlopen<T>(f: impl FnOnce() -> T) -> T {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            REAL_CUDA_DLOPEN_DEPTH.with(|depth| depth.set(depth.get() - 1));
        }
    }

    REAL_CUDA_DLOPEN_DEPTH.with(|depth| depth.set(depth.get() + 1));
    let _guard = Guard;
    f()
}

pub(crate) fn real_cuda_dlopen_active() -> bool {
    REAL_CUDA_DLOPEN_DEPTH.with(|depth| depth.get() > 0)
}

// original dlsym
extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

pub fn dlsym_next(symbol: &CStr) -> *mut c_void {
    const RTLD_NEXT: *mut c_void = usize::MAX as _;
    let result = unsafe { dlsym(RTLD_NEXT, symbol.as_ptr()) };
    if result.is_null() {
        panic!("failed to find symbol {symbol:?}");
    }
    result
}

pub(crate) fn dl_error_string() -> String {
    unsafe {
        let err = libc::dlerror();
        if err.is_null() {
            "unknown dlerror".to_string()
        } else {
            CStr::from_ptr(err).to_string_lossy().into_owned()
        }
    }
}

pub(crate) fn original_dlopen() -> DlopenFn {
    static DLOPEN: OnceLock<DlopenFn> = OnceLock::new();
    *DLOPEN.get_or_init(|| unsafe { mem::transmute(dlsym_next(c"dlopen")) })
}

pub(crate) fn dlsym_handle(handle: *mut c_void, symbol: &str) -> usize {
    let symbol = CString::new(symbol).unwrap();
    let ptr = unsafe { libc::dlsym(handle, symbol.as_ptr()) };
    if ptr.is_null() {
        panic!("failed to resolve {symbol:?}: {}", dl_error_string());
    }
    ptr as usize
}

#[no_mangle]
extern "C" fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void {
    let dlopen_orig = original_dlopen();
    // use the original dlopen to load the library
    if filename.is_null() {
        return dlopen_orig(filename, flags);
    }
    let name = unsafe { CStr::from_ptr(filename) }.to_str().unwrap();
    if name.contains("cpython") {
        return dlopen_orig(filename, flags);
    }
    log::debug!(target: "dlopen", "{name} (flags: {flags:#x})");
    let use_real_cuda = real_cuda_dlopen_active();
    if use_real_cuda && name.contains("libcuda") {
        return dlopen_orig(filename, flags);
    }

    if name.contains("libcuda")
        || name.contains("libnvrtc.so")
        || name.contains("libnvidia-ml")
        || name.contains("libnccl.so")
    {
        if cfg!(feature = "passthrough") {
            assert!(!dlopen_orig(filename, 0x101).is_null());
        }
        // if the library is libcuda, libnvrtc or libnvidia-ml, return a handle to the client
        log::debug!(target: "dlopen", "replacing dlopen call to {} library with a handle to the client", name);
        static SELF_PATH: OnceLock<CString> = OnceLock::new();
        let self_path = SELF_PATH.get_or_init(|| {
            let mut result = env::var("LD_PRELOAD").unwrap().into_bytes();
            if result.last() == Some(&b':') {
                result.pop();
            }
            assert!(!result.contains(&b':'));
            CString::new(result).unwrap()
        });
        let self_handle = dlopen_orig(self_path.as_ptr(), flags);
        if self_handle.is_null() {
            panic!("Failed to load the client handle");
        }
        return self_handle;
    }
    dlopen_orig(filename, flags)
}
