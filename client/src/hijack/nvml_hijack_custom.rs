#![expect(non_snake_case)]

use std::os::raw::c_uint;

use cudasys::types::nvml::{nvmlDevice_t, nvmlReturn_t};

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
