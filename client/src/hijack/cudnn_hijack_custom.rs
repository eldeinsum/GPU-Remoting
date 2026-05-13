use cudasys::types::cudart::cudaError_t;
use cudasys::types::cudnn::*;
use network::{CommChannelInner, Transportable};
use std::collections::BTreeMap;
use std::ffi::CString;
use std::os::raw::*;
use std::sync::{Mutex, OnceLock};

use crate::{ClientThread, CLIENT_THREAD};

fn cached_status_text(status: cudnnStatus_t) -> *const c_char {
    static STATUS_TEXTS: OnceLock<Mutex<BTreeMap<c_int, CString>>> = OnceLock::new();
    let code = status as c_int;
    let mut texts = STATUS_TEXTS
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .unwrap();
    let text = texts
        .entry(code)
        .or_insert_with(|| CString::new(format!("{status:?} ({code})")).unwrap());
    text.as_ptr()
}

#[derive(Copy, Clone)]
struct CudnnCallbackState {
    mask: c_uint,
    udata: usize,
    fptr: cudnnCallback_t,
}

fn cudnn_callback_state() -> &'static Mutex<CudnnCallbackState> {
    static STATE: OnceLock<Mutex<CudnnCallbackState>> = OnceLock::new();
    STATE.get_or_init(|| {
        Mutex::new(CudnnCallbackState {
            mask: 0,
            udata: 0,
            fptr: None,
        })
    })
}

#[no_mangle]
pub extern "C" fn cudnnSetCallback(
    mask: c_uint,
    udata: *mut c_void,
    fptr: cudnnCallback_t,
) -> cudnnStatus_t {
    log::debug!(target: "cudnnSetCallback", "mask = {mask}");
    let default_mask = 1u32 << (cudnnSeverity_t::CUDNN_SEV_ERROR as u32)
        | 1u32 << (cudnnSeverity_t::CUDNN_SEV_WARNING as u32);
    let mask = if fptr.is_none() && mask != 0 {
        mask | default_mask
    } else if fptr.is_none() && udata.is_null() {
        1u32 << (cudnnSeverity_t::CUDNN_SEV_ERROR as u32)
    } else {
        mask
    };
    *cudnn_callback_state().lock().unwrap() = CudnnCallbackState {
        mask,
        udata: udata as usize,
        fptr,
    };
    cudnnStatus_t::CUDNN_STATUS_SUCCESS
}

#[no_mangle]
pub extern "C" fn cudnnGetCallback(
    mask: *mut c_uint,
    udata: *mut *mut c_void,
    fptr: *mut cudnnCallback_t,
) -> cudnnStatus_t {
    log::debug!(target: "cudnnGetCallback", "");
    if mask.is_null() || udata.is_null() || fptr.is_null() {
        return cudnnStatus_t::CUDNN_STATUS_BAD_PARAM;
    }

    let state = *cudnn_callback_state().lock().unwrap();
    unsafe {
        *mask = state.mask;
        *udata = state.udata as *mut c_void;
        *fptr = state.fptr;
    }
    cudnnStatus_t::CUDNN_STATUS_SUCCESS
}

#[no_mangle]
pub extern "C" fn cudnnGetVersion() -> usize {
    let mut major = 0;
    let mut minor = 0;
    let mut patch = 0;
    let result =
        super::cudnn_hijack::cudnnGetProperty(libraryPropertyType::MAJOR_VERSION, &mut major);
    if result != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        return 0;
    }
    let result =
        super::cudnn_hijack::cudnnGetProperty(libraryPropertyType::MINOR_VERSION, &mut minor);
    if result != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        return 0;
    }
    let result =
        super::cudnn_hijack::cudnnGetProperty(libraryPropertyType::PATCH_LEVEL, &mut patch);
    if result != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        return 0;
    }

    (major as usize) * 10000 + (minor as usize) * 100 + patch as usize
}

#[no_mangle]
pub extern "C" fn cudnnGetMaxDeviceVersion() -> usize {
    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        log::debug!(target: "cudnnGetMaxDeviceVersion", "[#{}]", client.id);
        let ClientThread {
            channel_sender,
            channel_receiver,
            ..
        } = client;

        1801i32
            .send(channel_sender)
            .expect("failed to send cudnnGetMaxDeviceVersion proc_id");
        channel_sender
            .flush_out()
            .expect("failed to flush cudnnGetMaxDeviceVersion request");

        let mut version = 0usize;
        version
            .recv(channel_receiver)
            .expect("failed to receive cudnnGetMaxDeviceVersion result");
        channel_receiver
            .recv_ts()
            .expect("failed to receive cudnnGetMaxDeviceVersion timestamp");
        version
    })
}

#[no_mangle]
pub extern "C" fn cudnnGetCudartVersion() -> usize {
    let mut version = 0;
    let result = super::cudart_hijack::cudaRuntimeGetVersion(&mut version);
    if result == cudaError_t::cudaSuccess {
        version as usize
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn cudnnGetErrorString(error_status: cudnnStatus_t) -> *const c_char {
    log::debug!(target: "cudnnGetErrorString", "{error_status:?}");
    cached_status_text(error_status)
}

#[no_mangle]
pub extern "C" fn cudnnGetLastErrorString(message: *mut c_char, max_size: usize) {
    if message.is_null() || max_size == 0 {
        return;
    }

    let text = b"no error";
    let len = text.len().min(max_size.saturating_sub(1));
    unsafe {
        std::ptr::copy_nonoverlapping(text.as_ptr().cast::<c_char>(), message, len);
        *message.add(len) = 0;
    }
}

#[no_mangle]
extern "C" fn cudnnBackendGetAttribute(
    descriptor: cudnnBackendDescriptor_t,
    attributeName: cudnnBackendAttributeName_t,
    attributeType: cudnnBackendAttributeType_t,
    requestedElementCount: i64,
    elementCount: *mut i64,
    arrayOfElements: *mut c_void,
) -> cudnnStatus_t {
    let mut element_count = 0;
    let elementCount = match elementCount.is_null() {
        true => &raw mut element_count,
        false => elementCount,
    };
    if arrayOfElements.is_null() {
        super::cudnn_hijack::cudnnBackendGetAttributeCount(
            descriptor,
            attributeName,
            attributeType,
            requestedElementCount,
            elementCount,
            arrayOfElements,
        )
    } else if let cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR = attributeType {
        super::cudnn_hijack::cudnnBackendGetAttributeDescriptors(
            descriptor,
            attributeName,
            attributeType,
            requestedElementCount,
            elementCount,
            arrayOfElements.cast(),
        )
    } else {
        super::cudnn_hijack::cudnnBackendGetAttributeData(
            descriptor,
            attributeName,
            attributeType,
            requestedElementCount,
            elementCount,
            arrayOfElements.cast(),
        )
    }
}

#[no_mangle]
extern "C" fn cudnnSetFusedOpsConstParamPackAttribute(
    constPack: cudnnFusedOpsConstParamPack_t,
    paramLabel: cudnnFusedOpsConstParamLabel_t,
    param: *const c_void,
) -> cudnnStatus_t {
    if paramLabel.is_descriptor() {
        super::cudnn_hijack::cudnnSetFusedOpsConstParamPackDescriptorAttribute(
            constPack,
            paramLabel,
            param as usize,
        )
    } else {
        super::cudnn_hijack::cudnnSetFusedOpsConstParamPackHostAttribute(
            constPack,
            paramLabel,
            param.cast::<u8>(),
        )
    }
}

#[no_mangle]
extern "C" fn cudnnGetFusedOpsConstParamPackAttribute(
    constPack: cudnnFusedOpsConstParamPack_t,
    paramLabel: cudnnFusedOpsConstParamLabel_t,
    param: *mut c_void,
    isNULL: *mut c_int,
) -> cudnnStatus_t {
    if paramLabel.is_descriptor() {
        super::cudnn_hijack::cudnnGetFusedOpsConstParamPackDescriptorAttribute(
            constPack,
            paramLabel,
            param as usize,
            isNULL,
        )
    } else {
        super::cudnn_hijack::cudnnGetFusedOpsConstParamPackHostAttribute(
            constPack,
            paramLabel,
            param.cast::<u8>(),
            isNULL,
        )
    }
}

#[no_mangle]
extern "C" fn cudnnSetFusedOpsVariantParamPackAttribute(
    varPack: cudnnFusedOpsVariantParamPack_t,
    paramLabel: cudnnFusedOpsVariantParamLabel_t,
    ptr: *mut c_void,
) -> cudnnStatus_t {
    if paramLabel.is_device_pointer() {
        super::cudnn_hijack::cudnnSetFusedOpsVariantParamPackDevicePointerAttribute(
            varPack, paramLabel, ptr,
        )
    } else {
        super::cudnn_hijack::cudnnSetFusedOpsVariantParamPackHostAttribute(
            varPack,
            paramLabel,
            ptr.cast::<u8>(),
        )
    }
}
