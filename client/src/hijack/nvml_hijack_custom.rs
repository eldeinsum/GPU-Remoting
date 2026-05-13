#![expect(non_snake_case)]

use std::collections::BTreeMap;
use std::ffi::CString;
use std::os::raw::{c_char, c_uint};
use std::sync::{Mutex, OnceLock};

use cudasys::types::nvml::{nvmlDevice_t, nvmlReturn_t};

fn nvml_error_text(result: nvmlReturn_t) -> &'static str {
    match result {
        nvmlReturn_t::NVML_SUCCESS => "Success",
        nvmlReturn_t::NVML_ERROR_UNINITIALIZED => "Uninitialized",
        nvmlReturn_t::NVML_ERROR_INVALID_ARGUMENT => "Invalid Argument",
        nvmlReturn_t::NVML_ERROR_NOT_SUPPORTED => "Not Supported",
        nvmlReturn_t::NVML_ERROR_NO_PERMISSION => "Insufficient Permissions",
        nvmlReturn_t::NVML_ERROR_ALREADY_INITIALIZED => "Already Initialized",
        nvmlReturn_t::NVML_ERROR_NOT_FOUND => "Not Found",
        nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE => "Insufficient Size",
        nvmlReturn_t::NVML_ERROR_INSUFFICIENT_POWER => "Insufficient External Power",
        nvmlReturn_t::NVML_ERROR_DRIVER_NOT_LOADED => "Driver Not Loaded",
        nvmlReturn_t::NVML_ERROR_TIMEOUT => "Timeout",
        nvmlReturn_t::NVML_ERROR_IRQ_ISSUE => "Interrupt Request Issue",
        nvmlReturn_t::NVML_ERROR_LIBRARY_NOT_FOUND => "NVML Shared Library Not Found",
        nvmlReturn_t::NVML_ERROR_FUNCTION_NOT_FOUND => "Function Not Found",
        nvmlReturn_t::NVML_ERROR_CORRUPTED_INFOROM => "Corrupted infoROM",
        nvmlReturn_t::NVML_ERROR_GPU_IS_LOST => "GPU is lost",
        nvmlReturn_t::NVML_ERROR_RESET_REQUIRED => "GPU requires reset",
        nvmlReturn_t::NVML_ERROR_OPERATING_SYSTEM => "Operating System Error",
        nvmlReturn_t::NVML_ERROR_LIB_RM_VERSION_MISMATCH => {
            "RM has detected an NVML/RM version mismatch"
        }
        nvmlReturn_t::NVML_ERROR_IN_USE => "In use",
        nvmlReturn_t::NVML_ERROR_MEMORY => "Insufficient Memory",
        nvmlReturn_t::NVML_ERROR_NO_DATA => "No data",
        nvmlReturn_t::NVML_ERROR_VGPU_ECC_NOT_SUPPORTED => "vGPU ECC Not Supported",
        nvmlReturn_t::NVML_ERROR_INSUFFICIENT_RESOURCES => "Insufficient Resources",
        nvmlReturn_t::NVML_ERROR_FREQ_NOT_SUPPORTED => "Frequency not supported",
        nvmlReturn_t::NVML_ERROR_ARGUMENT_VERSION_MISMATCH => "Argument Version Mismatch",
        nvmlReturn_t::NVML_ERROR_DEPRECATED => "Deprecated",
        nvmlReturn_t::NVML_ERROR_NOT_READY => "Not Ready",
        nvmlReturn_t::NVML_ERROR_GPU_NOT_FOUND => "GPU not found",
        nvmlReturn_t::NVML_ERROR_INVALID_STATE => "Invalid State",
        nvmlReturn_t::NVML_ERROR_RESET_TYPE_NOT_SUPPORTED => "Reset Type Not Supported",
        nvmlReturn_t::NVML_ERROR_UNKNOWN => "Unknown Error",
    }
}

fn cached_nvml_error_text(result: nvmlReturn_t) -> *const c_char {
    static ERROR_TEXTS: OnceLock<Mutex<BTreeMap<i32, CString>>> = OnceLock::new();
    let code = result as i32;
    let mut texts = ERROR_TEXTS
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .unwrap();
    texts
        .entry(code)
        .or_insert_with(|| CString::new(nvml_error_text(result)).unwrap())
        .as_ptr()
}

#[no_mangle]
pub extern "C" fn nvmlErrorString(result: nvmlReturn_t) -> *const c_char {
    log::debug!(target: "nvmlErrorString", "{result:?}");
    cached_nvml_error_text(result)
}

#[no_mangle]
pub extern "C" fn nvmlInit() -> nvmlReturn_t {
    log::debug!(target: "nvmlInit", "");
    super::nvml_hijack::nvmlInit_v2()
}

#[no_mangle]
pub extern "C" fn nvmlDeviceGetCount(deviceCount: *mut c_uint) -> nvmlReturn_t {
    log::debug!(target: "nvmlDeviceGetCount", "");
    super::nvml_hijack::nvmlDeviceGetCount_v2(deviceCount)
}

#[no_mangle]
pub extern "C" fn nvmlDeviceGetHandleByIndex(
    index: c_uint,
    device: *mut nvmlDevice_t,
) -> nvmlReturn_t {
    log::debug!(target: "nvmlDeviceGetHandleByIndex", "");
    super::nvml_hijack::nvmlDeviceGetHandleByIndex_v2(index, device)
}
