use cudasys::types::cuda::*;
use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::os::raw::*;
use std::sync::{Mutex, OnceLock};

#[no_mangle]
extern "C" fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult {
    super::cuda_hijack::cuModuleLoadDataInternal(module, image.cast(), false)
}

fn write_error_text(error: CUresult, pStr: *mut *const c_char) -> CUresult {
    if pStr.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    static ERROR_TEXTS: OnceLock<Mutex<BTreeMap<c_int, CString>>> = OnceLock::new();
    let error_code = error as c_int;
    let mut texts = ERROR_TEXTS
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .unwrap();
    let text = texts
        .entry(error_code)
        .or_insert_with(|| CString::new(format!("{error:?}")).unwrap());
    unsafe {
        *pStr = text.as_ptr();
    }
    CUresult::CUDA_SUCCESS
}

#[no_mangle]
extern "C" fn cuGetErrorName(error: CUresult, pStr: *mut *const c_char) -> CUresult {
    write_error_text(error, pStr)
}

#[no_mangle]
extern "C" fn cuGetErrorString(error: CUresult, pStr: *mut *const c_char) -> CUresult {
    write_error_text(error, pStr)
}

#[no_mangle]
extern "C" fn cuGetProcAddress_v2(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    _cudaVersion: c_int,
    _flags: cuuint64_t,
    symbolStatus: *mut CUdriverProcAddressQueryResult,
) -> CUresult {
    if symbol.is_null() || pfn.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    let func = unsafe { libc::dlsym(libc::RTLD_DEFAULT, symbol) };
    unsafe {
        if func.is_null() {
            *pfn = std::ptr::null_mut();
            if !symbolStatus.is_null() {
                *symbolStatus =
                    CUdriverProcAddressQueryResult::CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
            }
            log::debug!(
                target: "cuGetProcAddress_v2",
                "symbol not found: {:?}",
                CStr::from_ptr(symbol),
            );
            return CUresult::CUDA_ERROR_NOT_FOUND;
        }

        *pfn = func;
        if !symbolStatus.is_null() {
            *symbolStatus = CUdriverProcAddressQueryResult::CU_GET_PROC_ADDRESS_SUCCESS;
        }
    }

    CUresult::CUDA_SUCCESS
}
