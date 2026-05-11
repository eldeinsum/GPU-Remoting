#![expect(non_snake_case)]
use super::*;

use crate::ManagedVariableShadow;
use cudasys::cudart::cudaError_t;
use std::cell::OnceCell;

#[no_mangle]
extern "C" fn __cudaRegisterFatBinary(fatCubin: *const FatBinaryWrapper) -> FatBinaryHandle {
    let fatCubin = unsafe { (*fatCubin).unwrap() };

    #[thread_local]
    static VALIDATE: OnceCell<bool> = OnceCell::new();

    if *VALIDATE.get_or_init(|| std::env::var_os("VALIDATE_FAT_BINARY").is_some()) {
        unsafe { (*fatCubin).validate_code() };
    }

    let mut runtime = RUNTIME_CACHE.write().unwrap();
    runtime.lazy_fatbins.push(fatCubin);
    runtime.lazy_fatbins.len() << 4
}

#[no_mangle]
pub extern "C" fn __cudaUnregisterFatBinary(_fatCubinHandle: MemPtr) {
    // This is called when the client process exits, when the thread local storage is already dropped.
    // Therefore, we implement this at the server side.
}

#[no_mangle]
pub extern "C" fn __cudaRegisterFatBinaryEnd(_fatCubinHandle: MemPtr) {
    // TODO: no actual impact
}

#[no_mangle]
pub extern "C" fn __cudaInitModule(fatCubinHandle: MemPtr) -> ::std::os::raw::c_char {
    if fatCubinHandle == 0 {
        return 0;
    }
    let result = super::cudart_hijack_custom::refresh_managed_variables_from_device();
    (result == cudaError_t::cudaSuccess) as ::std::os::raw::c_char
}

#[no_mangle]
pub extern "C" fn __cudaRegisterFunction(
    fatCubinHandle: MemPtr,
    hostFun: MemPtr,
    _deviceFun: *mut ::std::os::raw::c_char,
    deviceName: *const ::std::os::raw::c_char,
    _thread_limit: ::std::os::raw::c_int,
    _tid: MemPtr,
    _bid: MemPtr,
    _bDim: MemPtr,
    _gDim: MemPtr,
    _wSize: MemPtr,
) {
    if cfg!(debug_assertions) && !std::ptr::eq(deviceName, _deviceFun) {
        log::warn!(
            "deviceName: {:?}, deviceFun: {:?}",
            unsafe { std::ffi::CStr::from_ptr(deviceName) },
            unsafe { std::ffi::CStr::from_ptr(_deviceFun) },
        );
    }

    // Some kernels are registered multiple times from different fatbins
    // e.g. "void cub::EmptyKernel<void>()"
    let mut runtime = RUNTIME_CACHE.write().unwrap();
    runtime
        .lazy_functions
        .entry(hostFun)
        .or_insert((fatCubinHandle, deviceName));
}

#[no_mangle]
pub extern "C" fn __cudaRegisterVar(
    fatCubinHandle: MemPtr,
    hostVar: MemPtr,
    _deviceAddress: *mut ::std::os::raw::c_char,
    deviceName: *const ::std::os::raw::c_char,
    _ext: ::std::os::raw::c_int,
    _size: usize,
    _constant: ::std::os::raw::c_int,
    _global: ::std::os::raw::c_int,
) {
    if cfg!(debug_assertions) && !std::ptr::eq(deviceName, _deviceAddress) {
        log::warn!(
            "deviceName: {:?}, deviceFun: {:?}",
            unsafe { std::ffi::CStr::from_ptr(deviceName) },
            unsafe { std::ffi::CStr::from_ptr(_deviceAddress) },
        );
    }

    let mut runtime = RUNTIME_CACHE.write().unwrap();
    runtime
        .lazy_variables
        .entry(hostVar)
        .or_insert((fatCubinHandle, deviceName));
}

#[no_mangle]
pub extern "C" fn __cudaRegisterManagedVar(
    fatCubinHandle: MemPtr,
    hostVarPtrAddress: *mut *mut ::std::os::raw::c_void,
    _deviceAddress: *mut ::std::os::raw::c_char,
    deviceName: *const ::std::os::raw::c_char,
    _ext: ::std::os::raw::c_int,
    _size: usize,
    _constant: ::std::os::raw::c_int,
    _global: ::std::os::raw::c_int,
) {
    if hostVarPtrAddress.is_null() || deviceName.is_null() {
        return;
    }
    if cfg!(debug_assertions) && !std::ptr::eq(deviceName, _deviceAddress) {
        let device_var = if _deviceAddress.is_null() {
            None
        } else {
            Some(unsafe { std::ffi::CStr::from_ptr(_deviceAddress) })
        };
        log::warn!(
            "deviceName: {:?}, deviceVar: {:?}",
            unsafe { std::ffi::CStr::from_ptr(deviceName) },
            device_var,
        );
    }

    let mut shadow = vec![0u8; _size.max(1)].into_boxed_slice();
    let original_ptr = unsafe { *hostVarPtrAddress };
    if _size > 0 && !original_ptr.is_null() {
        unsafe {
            std::ptr::copy_nonoverlapping(original_ptr.cast::<u8>(), shadow.as_mut_ptr(), _size);
        }
    }
    let shadow_ptr = shadow.as_mut_ptr().cast::<::std::os::raw::c_void>();
    let synced_shadow = shadow.clone();
    unsafe {
        *hostVarPtrAddress = shadow_ptr;
    }

    let mut runtime = RUNTIME_CACHE.write().unwrap();
    runtime
        .lazy_variables
        .insert(hostVarPtrAddress as HostPtr, (fatCubinHandle, deviceName));
    runtime
        .lazy_variables
        .insert(shadow_ptr as HostPtr, (fatCubinHandle, deviceName));
    if let Some(old_shadow) = runtime.managed_variable_shadows.insert(
        hostVarPtrAddress as HostPtr,
        ManagedVariableShadow {
            bytes: shadow,
            synced_bytes: synced_shadow,
            size: _size,
        },
    ) {
        runtime.lazy_variables.remove(&old_shadow.ptr());
    }
}
