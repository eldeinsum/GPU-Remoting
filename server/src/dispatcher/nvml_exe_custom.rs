use cudasys::nvml::{nvmlInitWithFlags, nvmlInit_v2, nvmlReturn_t};
use std::os::raw::c_uint;
use std::sync::Mutex;

#[derive(Clone, Copy)]
struct NvmlState {
    initialized: bool,
    last_result: nvmlReturn_t,
}

static NVML_STATE: Mutex<NvmlState> = Mutex::new(NvmlState {
    initialized: false,
    last_result: nvmlReturn_t::NVML_SUCCESS,
});

pub(super) fn server_nvml_init_v2() -> nvmlReturn_t {
    server_nvml_init(|| unsafe { nvmlInit_v2() })
}

pub(super) fn server_nvml_init_with_flags(flags: c_uint) -> nvmlReturn_t {
    server_nvml_init(|| unsafe { nvmlInitWithFlags(flags) })
}

pub(super) fn server_nvml_shutdown() -> nvmlReturn_t {
    nvmlReturn_t::NVML_SUCCESS
}

fn server_nvml_init(init: impl FnOnce() -> nvmlReturn_t) -> nvmlReturn_t {
    let mut state = NVML_STATE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if state.initialized {
        return nvmlReturn_t::NVML_SUCCESS;
    }

    let result = init();
    state.last_result = result;
    match result {
        nvmlReturn_t::NVML_SUCCESS | nvmlReturn_t::NVML_ERROR_ALREADY_INITIALIZED => {
            state.initialized = true;
            nvmlReturn_t::NVML_SUCCESS
        }
        other => other,
    }
}
