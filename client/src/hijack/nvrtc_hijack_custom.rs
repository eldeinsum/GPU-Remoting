use cudasys::types::nvrtc::*;
use std::ffi::CString;
use std::os::raw::*;
use std::sync::OnceLock;

fn real_nvrtc_handle() -> *mut c_void {
    static HANDLE: OnceLock<usize> = OnceLock::new();
    let handle = *HANDLE.get_or_init(|| {
        let mut candidates = Vec::new();
        if let Ok(path) = std::env::var("GPU_REMOTING_REAL_NVRTC") {
            candidates.push(path);
        }
        candidates.extend([
            "/usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.13".to_string(),
            "/usr/local/cuda/lib64/libnvrtc.so.13".to_string(),
            "/usr/lib/x86_64-linux-gnu/libnvrtc.so.13".to_string(),
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
            "failed to load real libnvrtc; tried: {}",
            candidates.join(", ")
        );
    });
    handle as *mut c_void
}

fn load_nvrtc_symbol(name: &str) -> usize {
    crate::dl::dlsym_handle(real_nvrtc_handle(), name)
}

macro_rules! forward_nvrtc {
    ($name:ident($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty) => {
        #[no_mangle]
        pub extern "C" fn $name($($arg: $ty),*) -> $ret {
            type FnTy = extern "C" fn($($ty),*) -> $ret;
            static FN: OnceLock<usize> = OnceLock::new();
            let ptr = *FN.get_or_init(|| load_nvrtc_symbol(stringify!($name)));
            let func: FnTy = unsafe { std::mem::transmute(ptr) };
            crate::dl::with_real_cuda_dlopen(|| func($($arg),*))
        }
    };
}

forward_nvrtc!(nvrtcAddNameExpression(
    prog: nvrtcProgram,
    name_expression: *const c_char,
) -> nvrtcResult);
forward_nvrtc!(nvrtcCompileProgram(
    prog: nvrtcProgram,
    numOptions: c_int,
    options: *const *const c_char,
) -> nvrtcResult);
forward_nvrtc!(nvrtcCreateProgram(
    prog: *mut nvrtcProgram,
    src: *const c_char,
    name: *const c_char,
    numHeaders: c_int,
    headers: *const *const c_char,
    includeNames: *const *const c_char,
) -> nvrtcResult);
forward_nvrtc!(nvrtcDestroyProgram(prog: *mut nvrtcProgram) -> nvrtcResult);
forward_nvrtc!(nvrtcGetCUBIN(prog: nvrtcProgram, cubin: *mut c_char) -> nvrtcResult);
forward_nvrtc!(nvrtcGetCUBINSize(prog: nvrtcProgram, cubinSizeRet: *mut usize) -> nvrtcResult);
forward_nvrtc!(nvrtcGetErrorString(result: nvrtcResult) -> *const c_char);
forward_nvrtc!(nvrtcGetLTOIR(prog: nvrtcProgram, LTOIR: *mut c_char) -> nvrtcResult);
forward_nvrtc!(nvrtcGetLTOIRSize(prog: nvrtcProgram, LTOIRSizeRet: *mut usize) -> nvrtcResult);
forward_nvrtc!(nvrtcGetLoweredName(
    prog: nvrtcProgram,
    name_expression: *const c_char,
    lowered_name: *mut *const c_char,
) -> nvrtcResult);
forward_nvrtc!(nvrtcGetNumSupportedArchs(numArchs: *mut c_int) -> nvrtcResult);
forward_nvrtc!(nvrtcGetOptiXIR(prog: nvrtcProgram, optixir: *mut c_char) -> nvrtcResult);
forward_nvrtc!(nvrtcGetOptiXIRSize(
    prog: nvrtcProgram,
    optixirSizeRet: *mut usize,
) -> nvrtcResult);
forward_nvrtc!(nvrtcGetPCHCreateStatus(prog: nvrtcProgram) -> nvrtcResult);
forward_nvrtc!(nvrtcGetPCHHeapSize(ret: *mut usize) -> nvrtcResult);
forward_nvrtc!(nvrtcGetPCHHeapSizeRequired(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult);
forward_nvrtc!(nvrtcGetPTX(prog: nvrtcProgram, ptx: *mut c_char) -> nvrtcResult);
forward_nvrtc!(nvrtcGetPTXSize(prog: nvrtcProgram, ptxSizeRet: *mut usize) -> nvrtcResult);
forward_nvrtc!(nvrtcGetProgramLog(prog: nvrtcProgram, log: *mut c_char) -> nvrtcResult);
forward_nvrtc!(nvrtcGetProgramLogSize(prog: nvrtcProgram, logSizeRet: *mut usize) -> nvrtcResult);
forward_nvrtc!(nvrtcGetSupportedArchs(supportedArchs: *mut c_int) -> nvrtcResult);
forward_nvrtc!(nvrtcGetTileIR(prog: nvrtcProgram, TileIR: *mut c_char) -> nvrtcResult);
forward_nvrtc!(nvrtcGetTileIRSize(prog: nvrtcProgram, TileIRSizeRet: *mut usize) -> nvrtcResult);
forward_nvrtc!(nvrtcSetFlowCallback(
    prog: nvrtcProgram,
    callback: Option<unsafe extern "C" fn(arg1: *mut c_void, arg2: *mut c_void) -> c_int>,
    payload: *mut c_void,
) -> nvrtcResult);
forward_nvrtc!(nvrtcSetPCHHeapSize(size: usize) -> nvrtcResult);
