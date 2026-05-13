use cudasys::types::cublasLt::*;
use cudasys::types::cudart::cudaError_t;
use std::collections::BTreeMap;
use std::ffi::CString;
use std::os::raw::*;
use std::sync::atomic::{AtomicU32, Ordering};
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

#[no_mangle]
pub extern "C" fn cublasLtDisableCpuInstructionsSetMask(mask: c_uint) -> c_uint {
    static MASK: AtomicU32 = AtomicU32::new(0);
    MASK.swap(mask, Ordering::SeqCst)
}

#[no_mangle]
pub extern "C" fn cublasLtMatrixLayoutInit_internal(
    matLayout: cublasLtMatrixLayout_t,
    size: usize,
    type_: cudaDataType,
    rows: u64,
    cols: u64,
    ld: i64,
) -> cublasStatus_t {
    if matLayout.is_null() {
        return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    }
    if size < std::mem::size_of::<cublasLtMatrixLayoutOpaque_t>() {
        return cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED;
    }

    let mut server_layout = std::ptr::null_mut();
    let result = super::cublasLt_hijack::cublasLtMatrixLayoutCreate(
        &mut server_layout,
        type_,
        rows,
        cols,
        ld,
    );
    if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        unsafe { (matLayout as *mut cublasLtMatrixLayout_t).write(server_layout) };
        crate::cublaslt_bind_matrix_layout(matLayout, server_layout);
    }
    result
}

#[no_mangle]
pub extern "C" fn cublasLtGroupedMatrixLayoutInit_internal(
    matLayout: cublasLtMatrixLayout_t,
    size: usize,
    type_: cudaDataType,
    groupCount: c_int,
    rows_array: *const c_void,
    cols_array: *const c_void,
    ld_array: *const c_void,
) -> cublasStatus_t {
    if matLayout.is_null() {
        return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    }
    if size < std::mem::size_of::<cublasLtMatrixLayoutOpaque_t>() {
        return cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED;
    }

    let mut server_layout = std::ptr::null_mut();
    let result = super::cublasLt_hijack::cublasLtGroupedMatrixLayoutCreate(
        &mut server_layout,
        type_,
        groupCount,
        rows_array,
        cols_array,
        ld_array,
    );
    if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        unsafe { (matLayout as *mut cublasLtMatrixLayout_t).write(server_layout) };
        crate::cublaslt_bind_matrix_layout(matLayout, server_layout);
    }
    result
}

#[no_mangle]
pub extern "C" fn cublasLtMatmulDescInit_internal(
    matmulDesc: cublasLtMatmulDesc_t,
    size: usize,
    computeType: cublasComputeType_t,
    scaleType: cudaDataType_t,
) -> cublasStatus_t {
    if matmulDesc.is_null() {
        return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    }
    if size < std::mem::size_of::<cublasLtMatmulDescOpaque_t>() {
        return cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED;
    }

    let mut server_desc = std::ptr::null_mut();
    let result =
        super::cublasLt_hijack::cublasLtMatmulDescCreate(&mut server_desc, computeType, scaleType);
    if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        unsafe { (matmulDesc as *mut cublasLtMatmulDesc_t).write(server_desc) };
        crate::cublaslt_bind_matmul_desc(matmulDesc, server_desc);
        crate::CUBLAS_CACHE.write().unwrap().lt_matmul_descs.insert(
            matmulDesc,
            crate::CublasLtMatmulDescState {
                pointer_mode: cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST,
                scale_type_size: crate::cublaslt_scale_type_size(scaleType),
            },
        );
    }
    result
}

#[no_mangle]
pub extern "C" fn cublasLtMatrixTransformDescInit_internal(
    transformDesc: cublasLtMatrixTransformDesc_t,
    size: usize,
    scaleType: cudaDataType,
) -> cublasStatus_t {
    if transformDesc.is_null() {
        return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    }
    if size < std::mem::size_of::<cublasLtMatrixTransformDescOpaque_t>() {
        return cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED;
    }

    let mut server_desc = std::ptr::null_mut();
    let result =
        super::cublasLt_hijack::cublasLtMatrixTransformDescCreate(&mut server_desc, scaleType);
    if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        unsafe { (transformDesc as *mut cublasLtMatrixTransformDesc_t).write(server_desc) };
        crate::cublaslt_bind_transform_desc(transformDesc, server_desc);
        crate::CUBLAS_CACHE
            .write()
            .unwrap()
            .lt_transform_descs
            .insert(
                transformDesc,
                crate::CublasLtTransformDescState {
                    pointer_mode: cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST,
                    scale_type_size: crate::cublaslt_scale_type_size(scaleType),
                },
            );
    }
    result
}

#[no_mangle]
pub extern "C" fn cublasLtEmulationDescInit_internal(
    emulationDesc: cublasLtEmulationDesc_t,
    size: usize,
) -> cublasStatus_t {
    if emulationDesc.is_null() {
        return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    }
    if size < std::mem::size_of::<cublasLtEmulationDescOpaque_t>() {
        return cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED;
    }

    let mut server_desc = std::ptr::null_mut();
    let result = super::cublasLt_hijack::cublasLtEmulationDescCreate(&mut server_desc);
    if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        unsafe { (emulationDesc as *mut cublasLtEmulationDesc_t).write(server_desc) };
        crate::cublaslt_bind_emulation_desc(emulationDesc, server_desc);
    }
    result
}

#[no_mangle]
pub extern "C" fn cublasLtMatmulPreferenceInit_internal(
    pref: cublasLtMatmulPreference_t,
    size: usize,
) -> cublasStatus_t {
    if pref.is_null() {
        return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    }
    if size < std::mem::size_of::<cublasLtMatmulPreferenceOpaque_t>() {
        return cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED;
    }

    let mut server_pref = std::ptr::null_mut();
    let result = super::cublasLt_hijack::cublasLtMatmulPreferenceCreate(&mut server_pref);
    if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        unsafe { (pref as *mut cublasLtMatmulPreference_t).write(server_pref) };
        crate::cublaslt_bind_matmul_preference(pref, server_pref);
    }
    result
}

#[no_mangle]
pub extern "C" fn cublasLtLoggerSetCallback(_callback: cublasLtLoggerCallback_t) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub extern "C" fn cublasLtLoggerSetFile(_file: *mut FILE) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_SUCCESS
}
