#![expect(non_snake_case)]
use super::*;
use cudasys::types::cudart::*;
use std::cell::RefCell;
use std::ffi::*;

#[no_mangle]
#[use_thread_local(client = CLIENT_THREAD.with_borrow_mut)]
pub extern "C" fn cudaMemcpy(
    dst: MemPtr,
    src: MemPtr,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    log::debug!("[{}:{}] cudaMemcpy", std::file!(), std::line!());

    assert_ne!(kind, cudaMemcpyKind::cudaMemcpyDefault, "cudaMemcpyDefault is not supported yet");

    let ClientThread { channel_sender, channel_receiver, .. } = client;

    if cudaMemcpyKind::cudaMemcpyHostToHost == kind {
        unsafe {
            std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, count);
        }
        return cudaError_t::cudaSuccess;
    }

    let proc_id = 278;
    let mut result: cudaError_t = Default::default();
    match proc_id.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send proc_id: {:?}", e),
    }
    match dst.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send dst: {:?}", e),
    }
    match src.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send src: {:?}", e),
    }
    match count.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send count: {:?}", e),
    }
    match kind.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send kind: {:?}", e),
    }

    if cudaMemcpyKind::cudaMemcpyHostToDevice == kind {
        // transport [src; count] to device
        let data = unsafe { std::slice::from_raw_parts(src as *const u8, count) };
        match data.send(channel_sender) {
            Ok(()) => {}
            Err(e) => panic!("failed to send data: {:?}", e),
        }
    }

    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }

    if cudaMemcpyKind::cudaMemcpyDeviceToHost == kind {
        // receive [dst; count] from device
        let data = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, count) };
        match data.recv(channel_receiver) {
            Ok(()) => {}
            Err(e) => panic!("failed to receive data: {:?}", e),
        }
        #[cfg(feature = "async_api")]
        match channel_receiver.recv_ts() {
            Ok(()) => {}
            Err(e) => panic!("failed to receive timestamp: {:?}", e),
        }
    }

    if cfg!(feature = "async_api") {
        return cudaError_t::cudaSuccess;
    } else {
        match result.recv(channel_receiver) {
            Ok(()) => {}
            Err(e) => panic!("failed to receive result: {:?}", e),
        }
        match channel_receiver.recv_ts() {
            Ok(()) => {}
            Err(e) => panic!("failed to receive timestamp: {:?}", e),
        }
        return result;
    }
}

// TODO: maybe we should understand the semantic diff of cudaMemcpyAsync&cudaMemcpy
#[no_mangle]
pub extern "C" fn cudaMemcpyAsync(
    dst: MemPtr,
    src: MemPtr,
    count: usize,
    kind: cudaMemcpyKind,
    _stream: cudaStream_t,
) -> cudaError_t {
    log::debug!("[{}:{}] cudaMemcpyAsync", std::file!(), std::line!());
    cudaMemcpy(dst, src, count, kind)
}

fn get_cufunction(func: HostPtr) -> cudasys::cuda::CUfunction {
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
        assert_eq!(super::cudart_hijack::cudaGetDevice(&mut device), Default::default());
        runtime.cuda_device = Some(device);
    }

    let load_module = |fatCubinHandle: &FatBinaryHandle| {
        // See our implementation of `__cudaRegisterFatBinary`
        let index = (*fatCubinHandle >> 4) - 1;
        log::debug!("registering fatbin #{index}");
        let image = runtime.lazy_fatbins[index];
        CLIENT_THREAD.with_borrow_mut(|client| client.is_cuda_launch_kernel = true);
        let mut module = std::ptr::null_mut();
        assert_eq!(
            super::cuda_hijack::cuModuleLoadData(&raw mut module, image.cast()),
            Default::default(),
        );
        CLIENT_THREAD.with_borrow_mut(|client| client.is_cuda_launch_kernel = false);
        module
    };

    let (fatCubinHandle, deviceName) = *runtime.lazy_functions.get(&func).unwrap();
    let module = *runtime.loaded_modules.entry(fatCubinHandle).or_insert_with_key(load_module);
    log::debug!("registering function {:?}", unsafe { CStr::from_ptr(deviceName) });
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
    assert_eq!(flags, cudaHostAllocDefault);
    // TODO: handle pinned memory at server side in a better way
    // FIXME: some GPU kernels might write to pinned memory directly; currently CUDA will report illegal memory access
    let ptr = Box::into_raw(Box::<[u8]>::new_uninit_slice(size));
    unsafe {
        *pHost = ptr as _;
    }
    cudaError_t::cudaSuccess
}

#[no_mangle]
#[use_thread_local(client = CLIENT_THREAD.with_borrow_mut)]
pub extern "C" fn cudaGetErrorString(
    cudaError: cudaError_t,
) -> *const ::std::os::raw::c_char {
    log::debug!(target: "cudaGetErrorString", "{cudaError:?}");
    let ClientThread { channel_sender, channel_receiver, .. } = client;
    let proc_id = 151;
    let mut result:Vec<u8>  = Default::default();
    match proc_id.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send proc_id: {:?}", e),
    }
    match cudaError.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send cudaError: {:?}", e),
    }
    channel_sender.flush_out().unwrap();
    match result.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => panic!("failed to receive result: {:?}", e),
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    let result = CString::new(result).unwrap();
    result.into_raw() // leaking the string as the program is about to fail anyway
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
    let attr = unsafe {
        attr.write_bytes(0u8, 1);
        &mut *attr
    };
    // HACK: implementation with cuFuncGetAttribute depends on CUDA version
    macro_rules! get_attributes {
        ($func:ident -> $struct:ident $($field:ident: $attr:ident,)+) => {
            $(
                let mut i = 0;
                assert_eq!(
                    super::cuda_hijack::cuFuncGetAttribute(
                        &raw mut i,
                        cudasys::cuda::CUfunction_attribute::$attr,
                        $func,
                    ),
                    Default::default(),
                );
                $struct.$field = i as _;
            )+
        }
    }
    get_attributes! { func -> attr
        sharedSizeBytes: CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
        constSizeBytes: CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
        localSizeBytes: CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
        maxThreadsPerBlock: CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        numRegs: CU_FUNC_ATTRIBUTE_NUM_REGS,
        ptxVersion: CU_FUNC_ATTRIBUTE_PTX_VERSION,
        binaryVersion: CU_FUNC_ATTRIBUTE_BINARY_VERSION,
        cacheModeCA: CU_FUNC_ATTRIBUTE_CACHE_MODE_CA,
        maxDynamicSharedSizeBytes: CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
    }
    cudaError_t::cudaSuccess
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
