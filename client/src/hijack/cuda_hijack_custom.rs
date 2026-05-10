use cudasys::types::cuda::*;
use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::os::raw::*;
use std::sync::{Mutex, OnceLock};

#[no_mangle]
extern "C" fn cuModuleLoad(module: *mut CUmodule, fname: *const c_char) -> CUresult {
    if module.is_null() || fname.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    let path = unsafe { CStr::from_ptr(fname) };
    let Ok(mut image) = std::fs::read(path.to_string_lossy().as_ref()) else {
        return CUresult::CUDA_ERROR_FILE_NOT_FOUND;
    };
    if image.last().copied() != Some(0) {
        image.push(0);
    }

    super::cuda_hijack::cuModuleLoadDataInternal(module, image.as_ptr().cast(), false)
}

#[no_mangle]
extern "C" fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult {
    super::cuda_hijack::cuModuleLoadDataInternal(module, image.cast(), false)
}

fn has_unsupported_library_options(numJitOptions: c_uint, numLibraryOptions: c_uint) -> bool {
    numJitOptions != 0 || numLibraryOptions != 0
}

#[no_mangle]
extern "C" fn cuLibraryLoadData(
    library: *mut CUlibrary,
    code: *const c_void,
    _jitOptions: *mut CUjit_option,
    _jitOptionsValues: *mut *mut c_void,
    numJitOptions: c_uint,
    _libraryOptions: *mut CUlibraryOption,
    _libraryOptionValues: *mut *mut c_void,
    numLibraryOptions: c_uint,
) -> CUresult {
    if library.is_null() || code.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }
    if has_unsupported_library_options(numJitOptions, numLibraryOptions) {
        return CUresult::CUDA_ERROR_NOT_SUPPORTED;
    }

    super::cuda_hijack::cuLibraryLoadDataInternal(library, code.cast())
}

#[no_mangle]
extern "C" fn cuLibraryLoadFromFile(
    library: *mut CUlibrary,
    fileName: *const c_char,
    _jitOptions: *mut CUjit_option,
    _jitOptionsValues: *mut *mut c_void,
    numJitOptions: c_uint,
    _libraryOptions: *mut CUlibraryOption,
    _libraryOptionValues: *mut *mut c_void,
    numLibraryOptions: c_uint,
) -> CUresult {
    if library.is_null() || fileName.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }
    if has_unsupported_library_options(numJitOptions, numLibraryOptions) {
        return CUresult::CUDA_ERROR_NOT_SUPPORTED;
    }

    let path = unsafe { CStr::from_ptr(fileName) };
    let Ok(mut image) = std::fs::read(path.to_string_lossy().as_ref()) else {
        return CUresult::CUDA_ERROR_FILE_NOT_FOUND;
    };
    if image.last().copied() != Some(0) {
        image.push(0);
    }

    super::cuda_hijack::cuLibraryLoadDataInternal(library, image.as_ptr().cast())
}

#[no_mangle]
extern "C" fn cuKernelGetName(name: *mut *const c_char, hfunc: CUkernel) -> CUresult {
    if name.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    let driver = crate::DRIVER_CACHE.read().unwrap();
    let Some(kernel_name) = driver.kernel_names.get(&hfunc) else {
        unsafe {
            *name = std::ptr::null();
        }
        return CUresult::CUDA_ERROR_INVALID_HANDLE;
    };
    unsafe {
        *name = kernel_name.as_ptr();
    }
    CUresult::CUDA_SUCCESS
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

fn real_cuda_handle() -> *mut c_void {
    static HANDLE: OnceLock<usize> = OnceLock::new();
    let handle = *HANDLE.get_or_init(|| {
        let mut candidates = Vec::new();
        if let Ok(path) = std::env::var("GPU_REMOTING_REAL_CUDA") {
            candidates.push(path);
        }
        candidates.extend([
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1".to_string(),
            "/usr/local/cuda/compat/libcuda.so.1".to_string(),
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
            "failed to load real libcuda for NVRTC; tried: {}",
            candidates.join(", ")
        );
    });
    handle as *mut c_void
}

#[no_mangle]
extern "C" fn cuGetExportTable(
    ppExportTable: *mut *const c_void,
    pExportTableId: *const CUuuid,
) -> CUresult {
    if !crate::dl::real_cuda_dlopen_active() {
        unimplemented!("cuGetExportTable")
    }
    type FnTy = extern "C" fn(*mut *const c_void, *const CUuuid) -> CUresult;
    static FN: OnceLock<usize> = OnceLock::new();
    let ptr = *FN.get_or_init(|| crate::dl::dlsym_handle(real_cuda_handle(), "cuGetExportTable"));
    let func: FnTy = unsafe { std::mem::transmute(ptr) };
    func(ppExportTable, pExportTableId)
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
