#![expect(non_snake_case)]
use super::*;
use cudasys::types::cudart::*;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::ffi::*;
use std::sync::{Mutex, OnceLock};

fn host_allocations() -> &'static Mutex<BTreeSet<usize>> {
    static ALLOCATIONS: OnceLock<Mutex<BTreeSet<usize>>> = OnceLock::new();
    ALLOCATIONS.get_or_init(|| Mutex::new(BTreeSet::new()))
}

fn classify_pointer(ptr: *const c_void) -> Option<cudaMemoryType> {
    if ptr.is_null() {
        return None;
    }

    let mut attributes = std::mem::MaybeUninit::<cudaPointerAttributes>::uninit();
    let result = super::cudart_hijack::cudaPointerGetAttributes(attributes.as_mut_ptr(), ptr);
    if result == cudaError_t::cudaSuccess {
        Some(unsafe { attributes.assume_init() }.type_)
    } else {
        None
    }
}

fn is_device_pointer(ptr: *const c_void) -> bool {
    matches!(
        classify_pointer(ptr),
        Some(cudaMemoryType::cudaMemoryTypeDevice | cudaMemoryType::cudaMemoryTypeManaged)
    )
}

fn resolve_memcpy_kind(
    dst: *mut c_void,
    src: *const c_void,
    kind: cudaMemcpyKind,
) -> cudaMemcpyKind {
    if kind != cudaMemcpyKind::cudaMemcpyDefault {
        return kind;
    }

    match (
        is_device_pointer(dst as *const c_void),
        is_device_pointer(src),
    ) {
        (true, true) => cudaMemcpyKind::cudaMemcpyDeviceToDevice,
        (true, false) => cudaMemcpyKind::cudaMemcpyHostToDevice,
        (false, true) => cudaMemcpyKind::cudaMemcpyDeviceToHost,
        (false, false) => cudaMemcpyKind::cudaMemcpyHostToHost,
    }
}

fn resolve_to_array_kind(src: *const c_void, kind: cudaMemcpyKind) -> cudaMemcpyKind {
    if kind != cudaMemcpyKind::cudaMemcpyDefault {
        return kind;
    }

    if is_device_pointer(src) {
        cudaMemcpyKind::cudaMemcpyDeviceToDevice
    } else {
        cudaMemcpyKind::cudaMemcpyHostToDevice
    }
}

fn resolve_from_array_kind(dst: *mut c_void, kind: cudaMemcpyKind) -> cudaMemcpyKind {
    if kind != cudaMemcpyKind::cudaMemcpyDefault {
        return kind;
    }

    if is_device_pointer(dst.cast_const()) {
        cudaMemcpyKind::cudaMemcpyDeviceToDevice
    } else {
        cudaMemcpyKind::cudaMemcpyDeviceToHost
    }
}

fn allocate_host(ptr_out: *mut *mut c_void, size: usize) -> cudaError_t {
    if ptr_out.is_null() {
        return cudaError_t::cudaErrorInvalidValue;
    }

    if size == 0 {
        unsafe {
            *ptr_out = std::ptr::null_mut();
        }
        return cudaError_t::cudaSuccess;
    }

    let ptr = unsafe { libc::malloc(size) };
    if ptr.is_null() {
        return cudaError_t::cudaErrorMemoryAllocation;
    }

    host_allocations().lock().unwrap().insert(ptr as usize);
    unsafe {
        *ptr_out = ptr;
    }
    cudaError_t::cudaSuccess
}

#[no_mangle]
pub extern "C" fn cudaMemcpy(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    log::debug!(target: "cudaMemcpy", "kind = {kind:?}");
    if count > 0 && (dst.is_null() || src.is_null()) {
        return cudaError_t::cudaErrorInvalidValue;
    }

    let kind = resolve_memcpy_kind(dst, src, kind);
    match kind {
        cudaMemcpyKind::cudaMemcpyHostToHost => unsafe {
            std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, count);
            cudaError_t::cudaSuccess
        },
        cudaMemcpyKind::cudaMemcpyHostToDevice => {
            super::cudart_hijack::cudaMemcpyHtod(dst, src.cast(), count, kind)
        }
        cudaMemcpyKind::cudaMemcpyDeviceToHost => {
            super::cudart_hijack::cudaMemcpyDtoh(dst.cast(), src, count, kind)
        }
        cudaMemcpyKind::cudaMemcpyDeviceToDevice => {
            super::cudart_hijack::cudaMemcpyDtod(dst, src, count, kind)
        }
        cudaMemcpyKind::cudaMemcpyDefault => unreachable!(),
    }
}

#[no_mangle]
pub extern "C" fn cudaMemcpyAsync(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    log::debug!(target: "cudaMemcpyAsync", "kind = {kind:?}");
    if count > 0 && (dst.is_null() || src.is_null()) {
        return cudaError_t::cudaErrorInvalidValue;
    }

    let kind = resolve_memcpy_kind(dst, src, kind);
    match kind {
        cudaMemcpyKind::cudaMemcpyHostToHost => unsafe {
            super::cudart_hijack::cudaStreamSynchronize(stream);
            std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, count);
            cudaError_t::cudaSuccess
        },
        cudaMemcpyKind::cudaMemcpyHostToDevice => {
            super::cudart_hijack::cudaMemcpyAsyncHtod(dst, src.cast(), count, kind, stream)
        }
        cudaMemcpyKind::cudaMemcpyDeviceToHost => {
            super::cudart_hijack::cudaMemcpyAsyncDtoh(dst.cast(), src, count, kind, stream)
        }
        cudaMemcpyKind::cudaMemcpyDeviceToDevice => {
            super::cudart_hijack::cudaMemcpyAsyncDtod(dst, src, count, kind, stream)
        }
        cudaMemcpyKind::cudaMemcpyDefault => unreachable!(),
    }
}

#[no_mangle]
pub extern "C" fn cudaMemcpy2D(
    dst: *mut c_void,
    dpitch: usize,
    src: *const c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    log::debug!(
        target: "cudaMemcpy2D",
        "width = {width}, height = {height}, kind = {kind:?}",
    );

    if height == 0 || width == 0 {
        return cudaError_t::cudaSuccess;
    }

    if dst.is_null() || src.is_null() || width > dpitch || width > spitch {
        return cudaError_t::cudaErrorInvalidValue;
    }

    for row in 0..height {
        let Some(dst_offset) = row.checked_mul(dpitch) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let Some(src_offset) = row.checked_mul(spitch) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let row_dst = unsafe { (dst as *mut u8).add(dst_offset).cast() };
        let row_src = unsafe { (src as *const u8).add(src_offset).cast() };
        let result = cudaMemcpy(row_dst, row_src, width, kind);
        if result != cudaError_t::cudaSuccess {
            return result;
        }
    }

    cudaError_t::cudaSuccess
}

#[no_mangle]
pub extern "C" fn cudaMemcpy2DAsync(
    dst: *mut c_void,
    dpitch: usize,
    src: *const c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    log::debug!(
        target: "cudaMemcpy2DAsync",
        "width = {width}, height = {height}, kind = {kind:?}",
    );

    if height == 0 || width == 0 {
        return cudaError_t::cudaSuccess;
    }

    if dst.is_null() || src.is_null() || width > dpitch || width > spitch {
        return cudaError_t::cudaErrorInvalidValue;
    }

    for row in 0..height {
        let Some(dst_offset) = row.checked_mul(dpitch) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let Some(src_offset) = row.checked_mul(spitch) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let row_dst = unsafe { (dst as *mut u8).add(dst_offset).cast() };
        let row_src = unsafe { (src as *const u8).add(src_offset).cast() };
        let result = cudaMemcpyAsync(row_dst, row_src, width, kind, stream);
        if result != cudaError_t::cudaSuccess {
            return result;
        }
    }

    cudaError_t::cudaSuccess
}

#[no_mangle]
pub extern "C" fn cudaMemcpyToArray(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    log::debug!(target: "cudaMemcpyToArray", "count = {count}, kind = {kind:?}");
    if count > 0 && (dst.is_null() || src.is_null()) {
        return cudaError_t::cudaErrorInvalidValue;
    }

    let kind = resolve_to_array_kind(src, kind);

    match kind {
        cudaMemcpyKind::cudaMemcpyHostToDevice => super::cudart_hijack::cudaMemcpyToArrayHtod(
            dst,
            wOffset,
            hOffset,
            src.cast::<u8>(),
            count,
            kind,
        ),
        cudaMemcpyKind::cudaMemcpyDeviceToDevice => {
            super::cudart_hijack::cudaMemcpyToArrayDtod(dst, wOffset, hOffset, src, count, kind)
        }
        _ => cudaError_t::cudaErrorInvalidValue,
    }
}

#[no_mangle]
pub extern "C" fn cudaMemcpyFromArray(
    dst: *mut c_void,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    log::debug!(target: "cudaMemcpyFromArray", "count = {count}, kind = {kind:?}");
    if count > 0 && (dst.is_null() || src.is_null()) {
        return cudaError_t::cudaErrorInvalidValue;
    }

    let kind = resolve_from_array_kind(dst, kind);

    match kind {
        cudaMemcpyKind::cudaMemcpyDeviceToHost => super::cudart_hijack::cudaMemcpyFromArrayDtoh(
            dst.cast::<u8>(),
            src,
            wOffset,
            hOffset,
            count,
            kind,
        ),
        cudaMemcpyKind::cudaMemcpyDeviceToDevice => {
            super::cudart_hijack::cudaMemcpyFromArrayDtod(dst, src, wOffset, hOffset, count, kind)
        }
        _ => cudaError_t::cudaErrorInvalidValue,
    }
}

#[no_mangle]
pub extern "C" fn cudaMemcpyToArrayAsync(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    log::debug!(target: "cudaMemcpyToArrayAsync", "count = {count}, kind = {kind:?}");
    if count > 0 && (dst.is_null() || src.is_null()) {
        return cudaError_t::cudaErrorInvalidValue;
    }

    let kind = resolve_to_array_kind(src, kind);
    match kind {
        cudaMemcpyKind::cudaMemcpyHostToDevice => super::cudart_hijack::cudaMemcpyToArrayAsyncHtod(
            dst,
            wOffset,
            hOffset,
            src.cast::<u8>(),
            count,
            kind,
            stream,
        ),
        cudaMemcpyKind::cudaMemcpyDeviceToDevice => {
            super::cudart_hijack::cudaMemcpyToArrayAsyncDtod(
                dst, wOffset, hOffset, src, count, kind, stream,
            )
        }
        _ => cudaError_t::cudaErrorInvalidValue,
    }
}

#[no_mangle]
pub extern "C" fn cudaMemcpyFromArrayAsync(
    dst: *mut c_void,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    log::debug!(target: "cudaMemcpyFromArrayAsync", "count = {count}, kind = {kind:?}");
    if count > 0 && (dst.is_null() || src.is_null()) {
        return cudaError_t::cudaErrorInvalidValue;
    }

    let kind = resolve_from_array_kind(dst, kind);
    match kind {
        cudaMemcpyKind::cudaMemcpyDeviceToHost => {
            super::cudart_hijack::cudaMemcpyFromArrayAsyncDtoh(
                dst.cast::<u8>(),
                src,
                wOffset,
                hOffset,
                count,
                kind,
                stream,
            )
        }
        cudaMemcpyKind::cudaMemcpyDeviceToDevice => {
            super::cudart_hijack::cudaMemcpyFromArrayAsyncDtod(
                dst, src, wOffset, hOffset, count, kind, stream,
            )
        }
        _ => cudaError_t::cudaErrorInvalidValue,
    }
}

#[no_mangle]
pub extern "C" fn cudaMemcpy2DToArray(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    src: *const c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    log::debug!(
        target: "cudaMemcpy2DToArray",
        "width = {width}, height = {height}, kind = {kind:?}",
    );

    if height == 0 || width == 0 {
        return cudaError_t::cudaSuccess;
    }
    if dst.is_null() || src.is_null() || width > spitch {
        return cudaError_t::cudaErrorInvalidValue;
    }

    for row in 0..height {
        let Some(src_offset) = row.checked_mul(spitch) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let Some(row_h_offset) = hOffset.checked_add(row) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let row_src = unsafe { (src as *const u8).add(src_offset).cast() };
        let result = cudaMemcpyToArray(dst, wOffset, row_h_offset, row_src, width, kind);
        if result != cudaError_t::cudaSuccess {
            return result;
        }
    }

    cudaError_t::cudaSuccess
}

#[no_mangle]
pub extern "C" fn cudaMemcpy2DFromArray(
    dst: *mut c_void,
    dpitch: usize,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    log::debug!(
        target: "cudaMemcpy2DFromArray",
        "width = {width}, height = {height}, kind = {kind:?}",
    );

    if height == 0 || width == 0 {
        return cudaError_t::cudaSuccess;
    }
    if dst.is_null() || src.is_null() || width > dpitch {
        return cudaError_t::cudaErrorInvalidValue;
    }

    for row in 0..height {
        let Some(dst_offset) = row.checked_mul(dpitch) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let Some(row_h_offset) = hOffset.checked_add(row) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let row_dst = unsafe { (dst as *mut u8).add(dst_offset).cast() };
        let result = cudaMemcpyFromArray(row_dst, src, wOffset, row_h_offset, width, kind);
        if result != cudaError_t::cudaSuccess {
            return result;
        }
    }

    cudaError_t::cudaSuccess
}

#[no_mangle]
pub extern "C" fn cudaMemcpy2DToArrayAsync(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    src: *const c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    log::debug!(
        target: "cudaMemcpy2DToArrayAsync",
        "width = {width}, height = {height}, kind = {kind:?}",
    );

    if height == 0 || width == 0 {
        return cudaError_t::cudaSuccess;
    }
    if dst.is_null() || src.is_null() || width > spitch {
        return cudaError_t::cudaErrorInvalidValue;
    }

    for row in 0..height {
        let Some(src_offset) = row.checked_mul(spitch) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let Some(row_h_offset) = hOffset.checked_add(row) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let row_src = unsafe { (src as *const u8).add(src_offset).cast() };
        let result =
            cudaMemcpyToArrayAsync(dst, wOffset, row_h_offset, row_src, width, kind, stream);
        if result != cudaError_t::cudaSuccess {
            return result;
        }
    }

    cudaError_t::cudaSuccess
}

#[no_mangle]
pub extern "C" fn cudaMemcpy2DFromArrayAsync(
    dst: *mut c_void,
    dpitch: usize,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    log::debug!(
        target: "cudaMemcpy2DFromArrayAsync",
        "width = {width}, height = {height}, kind = {kind:?}",
    );

    if height == 0 || width == 0 {
        return cudaError_t::cudaSuccess;
    }
    if dst.is_null() || src.is_null() || width > dpitch {
        return cudaError_t::cudaErrorInvalidValue;
    }

    for row in 0..height {
        let Some(dst_offset) = row.checked_mul(dpitch) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let Some(row_h_offset) = hOffset.checked_add(row) else {
            return cudaError_t::cudaErrorInvalidValue;
        };
        let row_dst = unsafe { (dst as *mut u8).add(dst_offset).cast() };
        let result =
            cudaMemcpyFromArrayAsync(row_dst, src, wOffset, row_h_offset, width, kind, stream);
        if result != cudaError_t::cudaSuccess {
            return result;
        }
    }

    cudaError_t::cudaSuccess
}

fn get_cufunction(func: HostPtr) -> cudasys::cuda::CUfunction {
    if !CLIENT_THREAD.with_borrow(|client| client.cuda_device_init) {
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#initialization
        assert_eq!(
            super::cudart_hijack::cudaFree(std::ptr::null_mut()),
            Default::default()
        );
        CLIENT_THREAD.with_borrow_mut(|client| client.cuda_device_init = true);
    }

    if let Some(&cufunc) = RUNTIME_CACHE.read().unwrap().loaded_functions.get(&func) {
        return cufunc;
    }

    let runtime = &mut *RUNTIME_CACHE.write().unwrap();

    // TODO: In CUDA 12, use `cuLibrary{LoadData,GetKernel}` to avoid pinning device.
    if let Some(device) = runtime.cuda_device {
        assert_eq!(
            CLIENT_THREAD.with_borrow(|client| client.cuda_device),
            Some(device),
            "current device (left) and registered device (right) mismatch",
        );
    } else {
        log::info!(
            "#fatbins = {}, #functions = {}",
            runtime.lazy_fatbins.len(),
            runtime.lazy_functions.len(),
        );

        let mut device = 0;
        assert_eq!(
            super::cudart_hijack::cudaGetDevice(&mut device),
            Default::default()
        );
        runtime.cuda_device = Some(device);
    }

    let load_module = |fatCubinHandle: &FatBinaryHandle| {
        // See our implementation of `__cudaRegisterFatBinary`
        let index = (*fatCubinHandle >> 4) - 1;
        log::debug!("registering fatbin #{index}");
        let image = runtime.lazy_fatbins[index];
        let mut module = std::ptr::null_mut();
        assert_eq!(
            super::cuda_hijack::cuModuleLoadDataInternal(&raw mut module, image.cast(), true),
            Default::default(),
        );
        module
    };

    let (fatCubinHandle, deviceName) = *runtime.lazy_functions.get(&func).unwrap();
    let module = *runtime
        .loaded_modules
        .entry(fatCubinHandle)
        .or_insert_with_key(load_module);
    log::debug!("registering function {:?}", unsafe {
        CStr::from_ptr(deviceName)
    });
    let mut cufunc = std::ptr::null_mut();
    assert_eq!(
        super::cuda_hijack::cuModuleGetFunction(&raw mut cufunc, module, deviceName),
        Default::default(),
    );
    runtime.loaded_functions.insert(func, cufunc);
    cufunc
}

#[no_mangle]
pub extern "C" fn cudaLaunchKernel(
    func: MemPtr,
    gridDim: dim3,
    blockDim: dim3,
    args: *mut *mut ::std::os::raw::c_void,
    sharedMem: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    log::debug!(target: "cudaLaunchKernel", "");

    let cufunc = get_cufunction(func);

    unsafe {
        std::mem::transmute(super::cuda_hijack::cuLaunchKernel(
            cufunc,
            gridDim.x,
            gridDim.y,
            gridDim.z,
            blockDim.x,
            blockDim.y,
            blockDim.z,
            sharedMem.try_into().unwrap(),
            stream.cast(),
            args,
            std::ptr::null_mut(),
        ))
    }
}

#[no_mangle]
pub extern "C" fn cudaHostAlloc(
    pHost: *mut *mut ::std::os::raw::c_void,
    size: usize,
    flags: c_uint,
) -> cudaError_t {
    log::debug!(target: "cudaHostAlloc", "size = {size}, flags = {flags}");
    allocate_host(pHost, size)
}

#[no_mangle]
pub extern "C" fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> cudaError_t {
    log::debug!(target: "cudaMallocHost", "size = {size}");
    allocate_host(ptr, size)
}

#[no_mangle]
pub extern "C" fn cudaFreeHost(ptr: *mut c_void) -> cudaError_t {
    log::debug!(target: "cudaFreeHost", "");
    if ptr.is_null() {
        return cudaError_t::cudaSuccess;
    }

    if !host_allocations().lock().unwrap().remove(&(ptr as usize)) {
        return cudaError_t::cudaErrorInvalidValue;
    }

    unsafe {
        libc::free(ptr);
    }
    cudaError_t::cudaSuccess
}

#[no_mangle]
pub extern "C" fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: c_uint) -> cudaError_t {
    log::debug!(target: "cudaHostRegister", "size = {size}, flags = {flags}");
    if ptr.is_null() {
        cudaError_t::cudaErrorInvalidValue
    } else {
        cudaError_t::cudaSuccess
    }
}

#[no_mangle]
pub extern "C" fn cudaHostUnregister(ptr: *mut c_void) -> cudaError_t {
    log::debug!(target: "cudaHostUnregister", "");
    if ptr.is_null() {
        cudaError_t::cudaErrorInvalidValue
    } else {
        cudaError_t::cudaSuccess
    }
}

#[no_mangle]
pub extern "C" fn cudaGetErrorString(cudaError: cudaError_t) -> *const ::std::os::raw::c_char {
    log::debug!(target: "cudaGetErrorString", "{cudaError:?}");
    let result = format!("{cudaError:?} ({})", cudaError as u32);
    let result = CString::new(result).unwrap();
    result.into_raw() // leaking the string as the program is about to fail anyway
}

#[no_mangle]
pub extern "C" fn cudaGetErrorName(cudaError: cudaError_t) -> *const ::std::os::raw::c_char {
    log::debug!(target: "cudaGetErrorName", "{cudaError:?}");
    let result = CString::new(format!("{cudaError:?}")).unwrap();
    result.into_raw()
}

#[no_mangle]
pub extern "C" fn cudaCreateChannelDesc(
    x: c_int,
    y: c_int,
    z: c_int,
    w: c_int,
    f: cudaChannelFormatKind,
) -> cudaChannelFormatDesc {
    cudaChannelFormatDesc { x, y, z, w, f }
}

struct CallConfiguration {
    gridDim: dim3,
    blockDim: dim3,
    sharedMem: usize,
    stream: MemPtr,
}

thread_local! {
    static CALL_CONFIGURATIONS: RefCell<Vec<CallConfiguration>> = const {
        RefCell::new(Vec::new())
    };
}

#[no_mangle]
pub extern "C" fn __cudaPushCallConfiguration(
    gridDim: dim3,
    blockDim: dim3,
    sharedMem: usize,
    stream: MemPtr,
) -> cudaError_t {
    CALL_CONFIGURATIONS.with_borrow_mut(|v| {
        v.push(CallConfiguration {
            gridDim,
            blockDim,
            sharedMem,
            stream,
        });
    });
    cudaError_t::cudaSuccess
}

#[no_mangle]
pub extern "C" fn __cudaPopCallConfiguration(
    gridDim: *mut dim3,
    blockDim: *mut dim3,
    sharedMem: *mut usize,
    stream: *mut MemPtr,
) -> cudaError_t {
    if let Some(config) = CALL_CONFIGURATIONS.with_borrow_mut(Vec::pop) {
        unsafe {
            *gridDim = config.gridDim;
            *blockDim = config.blockDim;
            *sharedMem = config.sharedMem;
            *stream = config.stream;
        }
        cudaError_t::cudaSuccess
    } else {
        cudaError_t::cudaErrorMissingConfiguration
    }
}

#[no_mangle]
extern "C" fn cudaFuncGetAttributes(
    attr: *mut cudaFuncAttributes,
    func: *const c_void,
) -> cudaError_t {
    log::debug!(target: "cudaFuncGetAttributes", "");
    let func = get_cufunction(func as HostPtr);
    super::cudart_hijack::cudaFuncGetAttributesInternal(attr, func)
}

#[no_mangle]
extern "C" fn cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    numBlocks: *mut c_int,
    func: *const c_void,
    blockSize: c_int,
    dynamicSMemSize: usize,
    flags: c_uint,
) -> cudaError_t {
    log::debug!(target: "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", "");
    let result = super::cuda_hijack::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks,
        get_cufunction(func as HostPtr),
        blockSize,
        dynamicSMemSize,
        flags,
    );
    unsafe { std::mem::transmute(result) }
}

#[no_mangle]
extern "C" fn cudaFuncSetAttribute(
    func: *const c_void,
    attr: cudaFuncAttribute,
    value: c_int,
) -> cudaError_t {
    log::debug!(target: "cudaFuncSetAttribute", "");
    #[expect(clippy::missing_transmute_annotations)]
    unsafe {
        std::mem::transmute(super::cuda_hijack::cuFuncSetAttribute(
            get_cufunction(func as _),
            std::mem::transmute(attr),
            value,
        ))
    }
}
