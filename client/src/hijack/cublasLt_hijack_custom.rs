use cudasys::types::cublasLt::*;
use cudasys::types::cudart::cudaError_t;
use std::collections::BTreeMap;
use std::ffi::CString;
use std::os::raw::*;
use std::sync::{Mutex, OnceLock};

fn status_name(status: cublasStatus_t) -> &'static str {
    match status {
        cublasStatus_t::CUBLAS_STATUS_SUCCESS => "CUBLAS_STATUS_SUCCESS",
        cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => "CUBLAS_STATUS_NOT_INITIALIZED",
        cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED => "CUBLAS_STATUS_ALLOC_FAILED",
        cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE => "CUBLAS_STATUS_INVALID_VALUE",
        cublasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH => "CUBLAS_STATUS_ARCH_MISMATCH",
        cublasStatus_t::CUBLAS_STATUS_MAPPING_ERROR => "CUBLAS_STATUS_MAPPING_ERROR",
        cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => "CUBLAS_STATUS_EXECUTION_FAILED",
        cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR => "CUBLAS_STATUS_INTERNAL_ERROR",
        cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED => "CUBLAS_STATUS_NOT_SUPPORTED",
        cublasStatus_t::CUBLAS_STATUS_LICENSE_ERROR => "CUBLAS_STATUS_LICENSE_ERROR",
    }
}

fn status_description(status: cublasStatus_t) -> &'static str {
    match status {
        cublasStatus_t::CUBLAS_STATUS_SUCCESS => "success",
        cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => "the cuBLASLt library was not initialized",
        cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED => "resource allocation failed",
        cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE => {
            "an unsupported value or parameter was passed"
        }
        cublasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH => "the device architecture is not supported",
        cublasStatus_t::CUBLAS_STATUS_MAPPING_ERROR => "access to GPU memory space failed",
        cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => "the GPU program failed to execute",
        cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR => "an internal cuBLASLt operation failed",
        cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED => "the requested operation is not supported",
        cublasStatus_t::CUBLAS_STATUS_LICENSE_ERROR => "license check failed",
    }
}

fn cached_status_text(status: cublasStatus_t, description: bool) -> *const c_char {
    static STATUS_TEXTS: OnceLock<Mutex<BTreeMap<(c_int, bool), CString>>> = OnceLock::new();
    let code = status as c_int;
    let mut texts = STATUS_TEXTS
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .unwrap();
    let text = texts.entry((code, description)).or_insert_with(|| {
        CString::new(if description {
            status_description(status)
        } else {
            status_name(status)
        })
        .unwrap()
    });
    text.as_ptr()
}

#[no_mangle]
pub extern "C" fn cublasLtGetStatusName(status: cublasStatus_t) -> *const c_char {
    cached_status_text(status, false)
}

#[no_mangle]
pub extern "C" fn cublasLtGetStatusString(status: cublasStatus_t) -> *const c_char {
    cached_status_text(status, true)
}

#[no_mangle]
pub extern "C" fn cublasLtGetVersion() -> usize {
    let mut major = 0;
    let mut minor = 0;
    let mut patch = 0;
    let result =
        super::cublasLt_hijack::cublasLtGetProperty(libraryPropertyType::MAJOR_VERSION, &mut major);
    if result != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return 0;
    }
    let result =
        super::cublasLt_hijack::cublasLtGetProperty(libraryPropertyType::MINOR_VERSION, &mut minor);
    if result != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return 0;
    }
    let result =
        super::cublasLt_hijack::cublasLtGetProperty(libraryPropertyType::PATCH_LEVEL, &mut patch);
    if result != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return 0;
    }

    (major as usize) * 10000 + (minor as usize) * 100 + patch as usize
}

#[no_mangle]
pub extern "C" fn cublasLtGetCudartVersion() -> usize {
    let mut version = 0;
    let result = super::cudart_hijack::cudaRuntimeGetVersion(&mut version);
    if result == cudaError_t::cudaSuccess {
        version as usize
    } else {
        0
    }
}
