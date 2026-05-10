use cudasys::types::cuda::*;
use std::ffi::CStr;
use std::os::raw::*;

#[no_mangle]
extern "C" fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult {
    super::cuda_hijack::cuModuleLoadDataInternal(module, image.cast(), false)
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
