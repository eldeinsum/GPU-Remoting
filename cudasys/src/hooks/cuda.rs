use crate::types::cuda::*;
use codegen::{cuda_custom_hook, cuda_hook};
use std::os::raw::*;

#[cuda_hook(proc_id = 900461)]
fn cuArrayCreate_v2(
    pHandle: *mut CUarray,
    #[host(len = 1)] pAllocateArray: *const CUDA_ARRAY_DESCRIPTOR,
) -> CUresult;

#[cuda_hook(proc_id = 900462)]
fn cuArrayGetDescriptor_v2(
    pArrayDescriptor: *mut CUDA_ARRAY_DESCRIPTOR,
    hArray: CUarray,
) -> CUresult;

#[cuda_hook(proc_id = 900490)]
fn cuArray3DCreate_v2(
    pHandle: *mut CUarray,
    #[host(len = 1)] pAllocateArray: *const CUDA_ARRAY3D_DESCRIPTOR,
) -> CUresult;

#[cuda_hook(proc_id = 900491)]
fn cuArray3DGetDescriptor_v2(
    pArrayDescriptor: *mut CUDA_ARRAY3D_DESCRIPTOR,
    hArray: CUarray,
) -> CUresult;

#[cuda_hook(proc_id = 900579)]
fn cuArrayGetPlane(pPlaneArray: *mut CUarray, hArray: CUarray, planeIdx: c_uint) -> CUresult;

#[cuda_hook(proc_id = 900463, async_api = false)]
fn cuArrayDestroy(hArray: CUarray) -> CUresult;

#[cuda_hook(proc_id = 670)]
fn cuDevicePrimaryCtxGetState(dev: CUdevice, flags: *mut c_uint, active: *mut c_int) -> CUresult;

#[cuda_hook(proc_id = 918, async_api)]
fn cuLaunchKernel(
    f: CUfunction,
    gridDimX: c_uint,
    gridDimY: c_uint,
    gridDimZ: c_uint,
    blockDimX: c_uint,
    blockDimY: c_uint,
    blockDimZ: c_uint,
    sharedMemBytes: c_uint,
    hStream: CUstream,
    #[skip] kernelParams: *mut *mut c_void,
    #[skip] extra: *mut *mut c_void,
) -> CUresult {
    'client_before_send: {
        assert!(extra.is_null());
        let (args, arg_offsets) = super::cuda_hijack_utils::pack_kernel_args_with_offsets(
            kernelParams,
            DRIVER_CACHE
                .read()
                .unwrap()
                .function_params
                .get(&f)
                .unwrap(),
        );
    }
    'client_extra_send: {
        send_slice(&arg_offsets, channel_sender).unwrap();
        send_slice(&args, channel_sender).unwrap();
    }
    'server_extra_recv: {
        let arg_offsets = recv_slice::<u32, _>(channel_receiver).unwrap();
        let args = recv_slice::<u8, _>(channel_receiver).unwrap();
    }
    'server_execution: {
        let result = super::cuda_exe_utils::cu_launch_kernel(
            f,
            gridDimX,
            gridDimY,
            gridDimZ,
            blockDimX,
            blockDimY,
            blockDimZ,
            sharedMemBytes,
            hStream,
            &args,
            &arg_offsets,
        );
    }
}

#[cuda_hook(proc_id = 919, async_api)]
fn cuLaunchKernelEx(
    #[host] config: *const CUlaunchConfig,
    f: CUfunction,
    #[skip] kernelParams: *mut *mut c_void,
    #[skip] extra: *mut *mut c_void,
) -> CUresult {
    'client_before_send: {
        assert!(extra.is_null());
        assert!(!config.is_null());
        let config_ref = unsafe { &*config };
        let attrs = if config_ref.numAttrs == 0 {
            &[][..]
        } else {
            assert!(!config_ref.attrs.is_null());
            unsafe { std::slice::from_raw_parts(config_ref.attrs, config_ref.numAttrs as usize) }
        };
        let (args, arg_offsets) = super::cuda_hijack_utils::pack_kernel_args_with_offsets(
            kernelParams,
            DRIVER_CACHE
                .read()
                .unwrap()
                .function_params
                .get(&f)
                .unwrap(),
        );
    }
    'client_extra_send: {
        send_slice(attrs, channel_sender).unwrap();
        send_slice(&arg_offsets, channel_sender).unwrap();
        send_slice(&args, channel_sender).unwrap();
    }
    'server_extra_recv: {
        let _ = config__ptr;
        let attrs = recv_slice::<CUlaunchAttribute, _>(channel_receiver).unwrap();
        let arg_offsets = recv_slice::<u32, _>(channel_receiver).unwrap();
        let args = recv_slice::<u8, _>(channel_receiver).unwrap();
        let mut launch_config = config;
        launch_config.attrs = if attrs.is_empty() {
            std::ptr::null_mut()
        } else {
            attrs.as_ptr().cast_mut()
        };
        launch_config.numAttrs = attrs.len() as _;
    }
    'server_execution: {
        let result =
            super::cuda_exe_utils::cu_launch_kernel_ex(&launch_config, f, &args, &arg_offsets);
    }
}

#[cuda_custom_hook] // calls the internal API below
fn cuModuleLoad(module: *mut CUmodule, fname: *const c_char) -> CUresult;

#[cuda_custom_hook] // calls the internal API below
fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;

#[cuda_hook(proc_id = 701, parent = cuModuleLoadData)]
fn cuModuleLoadDataInternal(
    module: *mut CUmodule,
    #[host(len = len)] image: *const c_void,
    #[skip] is_runtime: bool,
) -> CUresult {
    'client_before_send: {
        let len = crate::elf::module_image_len(image);
    }
    'client_after_recv: {
        let image = if is_runtime {
            std::borrow::Cow::Borrowed(image)
        } else {
            std::borrow::Cow::Owned(image.to_vec())
        };
        assert!(
            DRIVER_CACHE
                .write()
                .unwrap()
                .images
                .insert(*module, image)
                .is_none()
        );
    }
    'server_execution: {
        let result = unsafe { cuModuleLoadData(module__ptr, image__ptr.cast()) };
    }
    'server_after_send: {
        server.modules.push(module);
    }
}

#[cuda_hook(proc_id = 705)]
fn cuModuleGetFunction(hfunc: *mut CUfunction, hmod: CUmodule, name: *const c_char) -> CUresult {
    'client_after_recv: {
        let target_arch = DRIVER_CACHE.read().unwrap().device_arch;
        let mut driver = DRIVER_CACHE.write().unwrap();
        let image = driver.images.get(&hmod).unwrap();
        let params = if FatBinaryHeader::is_fat_binary(image.as_ptr()) {
            let fatbin: &FatBinaryHeader = unsafe { &*image.as_ptr().cast() };
            fatbin.find_kernel_params(name.to_str().unwrap(), target_arch)
        } else {
            crate::elf::find_kernel_params(image, name.to_str().unwrap())
        };
        driver.function_params.insert(*hfunc, params);
    }
}

#[cuda_hook(proc_id = 900706, async_api = false)]
fn cuModuleUnload(hmod: CUmodule) -> CUresult {
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            DRIVER_CACHE.write().unwrap().images.remove(&hmod);
        }
    }
    'server_after_send: {
        if result == CUresult::CUDA_SUCCESS {
            server.modules.retain(|module| *module != hmod);
        }
    }
}

#[cuda_hook(proc_id = 900707)]
fn cuModuleGetLoadingMode(mode: *mut CUmoduleLoadingMode) -> CUresult;

#[cuda_hook(proc_id = 900708)]
fn cuModuleGetGlobal_v2(
    dptr: *mut CUdeviceptr,
    bytes: *mut usize,
    hmod: CUmodule,
    name: *const c_char,
) -> CUresult;

#[cuda_custom_hook] // calls the internal API below
fn cuLibraryLoadData(
    library: *mut CUlibrary,
    code: *const c_void,
    jitOptions: *mut CUjit_option,
    jitOptionsValues: *mut *mut c_void,
    numJitOptions: c_uint,
    libraryOptions: *mut CUlibraryOption,
    libraryOptionValues: *mut *mut c_void,
    numLibraryOptions: c_uint,
) -> CUresult;

#[cuda_custom_hook] // calls the internal API below
fn cuLibraryLoadFromFile(
    library: *mut CUlibrary,
    fileName: *const c_char,
    jitOptions: *mut CUjit_option,
    jitOptionsValues: *mut *mut c_void,
    numJitOptions: c_uint,
    libraryOptions: *mut CUlibraryOption,
    libraryOptionValues: *mut *mut c_void,
    numLibraryOptions: c_uint,
) -> CUresult;

#[cuda_hook(proc_id = 900709, parent = cuLibraryLoadData)]
fn cuLibraryLoadDataInternal(
    library: *mut CUlibrary,
    #[host(len = len)] code: *const c_void,
) -> CUresult {
    'client_before_send: {
        let len = crate::elf::module_image_len(code);
    }
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            assert!(
                DRIVER_CACHE
                    .write()
                    .unwrap()
                    .library_images
                    .insert(*library, std::borrow::Cow::Owned(code.to_vec()))
                    .is_none()
            );
        }
    }
    'server_execution: {
        let result = unsafe {
            cuLibraryLoadData(
                library__ptr,
                code__ptr.cast(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
    }
    'server_after_send: {
        if result == CUresult::CUDA_SUCCESS {
            server.libraries.push(library);
        }
    }
}

#[cuda_hook(proc_id = 900710)]
fn cuLibraryUnload(library: CUlibrary) -> CUresult {
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            let mut driver = DRIVER_CACHE.write().unwrap();
            driver.library_images.remove(&library);
            let kernels = driver
                .kernel_libraries
                .iter()
                .filter_map(|(kernel, owner)| (*owner == library).then_some(*kernel))
                .collect::<Vec<_>>();
            for kernel in kernels {
                driver.kernel_libraries.remove(&kernel);
                driver.kernel_names.remove(&kernel);
                driver.function_params.remove(&kernel.cast::<CUfunc_st>());
            }
        }
    }
    'server_after_send: {
        if result == CUresult::CUDA_SUCCESS {
            server.libraries.retain(|item| *item != library);
        }
    }
}

#[cuda_hook(proc_id = 900711)]
fn cuLibraryGetKernel(pKernel: *mut CUkernel, library: CUlibrary, name: *const c_char) -> CUresult {
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            let target_arch = DRIVER_CACHE.read().unwrap().device_arch;
            let mut driver = DRIVER_CACHE.write().unwrap();
            let image = driver.library_images.get(&library).unwrap();
            let kernel_name = name.to_str().unwrap();
            let params = if FatBinaryHeader::is_fat_binary(image.as_ptr()) {
                let fatbin: &FatBinaryHeader = unsafe { &*image.as_ptr().cast() };
                fatbin.find_kernel_params(kernel_name, target_arch)
            } else {
                crate::elf::find_kernel_params(image, kernel_name)
            };
            driver
                .function_params
                .insert((*pKernel).cast::<CUfunc_st>(), params);
            driver
                .kernel_names
                .insert(*pKernel, std::ffi::CString::new(kernel_name).unwrap());
            driver.kernel_libraries.insert(*pKernel, library);
        }
    }
}

#[cuda_hook(proc_id = 900712)]
fn cuLibraryGetModule(pMod: *mut CUmodule, library: CUlibrary) -> CUresult {
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            let mut driver = DRIVER_CACHE.write().unwrap();
            let image = driver.library_images.get(&library).unwrap().clone();
            driver.images.insert(*pMod, image);
        }
    }
}

#[cuda_hook(proc_id = 900713)]
fn cuKernelGetFunction(pFunc: *mut CUfunction, kernel: CUkernel) -> CUresult {
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            let mut driver = DRIVER_CACHE.write().unwrap();
            if let Some(params) = driver
                .function_params
                .get(&kernel.cast::<CUfunc_st>())
                .cloned()
            {
                driver.function_params.insert(*pFunc, params);
            }
        }
    }
}

#[cuda_hook(proc_id = 900714)]
fn cuKernelGetLibrary(pLib: *mut CUlibrary, kernel: CUkernel) -> CUresult;

#[cuda_hook(proc_id = 900715)]
fn cuLibraryGetGlobal(
    dptr: *mut CUdeviceptr,
    bytes: *mut usize,
    library: CUlibrary,
    name: *const c_char,
) -> CUresult;

#[cuda_hook(proc_id = 900716)]
fn cuKernelGetAttribute(
    pi: *mut c_int,
    attrib: CUfunction_attribute,
    kernel: CUkernel,
    dev: CUdevice,
) -> CUresult;

#[cuda_hook(proc_id = 900717, async_api = false)]
fn cuKernelSetAttribute(
    attrib: CUfunction_attribute,
    val: c_int,
    kernel: CUkernel,
    dev: CUdevice,
) -> CUresult;

#[cuda_hook(proc_id = 900718, async_api = false)]
fn cuKernelSetCacheConfig(kernel: CUkernel, config: CUfunc_cache, dev: CUdevice) -> CUresult;

#[cuda_custom_hook] // local: returns a client-owned C string
fn cuKernelGetName(name: *mut *const c_char, hfunc: CUkernel) -> CUresult;

#[cuda_hook(proc_id = 900719)]
fn cuKernelGetParamInfo(
    kernel: CUkernel,
    paramIndex: usize,
    paramOffset: *mut usize,
    paramSize: *mut usize,
) -> CUresult;

#[cuda_hook(proc_id = 900720)]
fn cuKernelGetParamCount(kernel: CUkernel, paramCount: *mut usize) -> CUresult;

#[cuda_hook(proc_id = 640)]
fn cuDriverGetVersion(driverVersion: *mut c_int) -> CUresult;

#[cuda_custom_hook] // local
fn cuGetErrorName(error: CUresult, pStr: *mut *const c_char) -> CUresult;

#[cuda_custom_hook] // local
fn cuGetErrorString(error: CUresult, pStr: *mut *const c_char) -> CUresult;

#[cuda_custom_hook] // local for NVRTC
fn cuGetExportTable(ppExportTable: *mut *const c_void, pExportTableId: *const CUuuid) -> CUresult;

#[cuda_hook(proc_id = 630, async_api = false)]
fn cuInit(Flags: c_uint) -> CUresult;

#[cuda_hook(proc_id = 684)]
fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult;

#[cuda_hook(proc_id = 900300)]
fn cuCtxGetApiVersion(ctx: CUcontext, version: *mut c_uint) -> CUresult;

#[cuda_hook(proc_id = 900301)]
fn cuCtxGetCacheConfig(pconfig: *mut CUfunc_cache) -> CUresult;

#[cuda_hook(proc_id = 900302)]
fn cuCtxGetDevice(device: *mut CUdevice) -> CUresult;

#[cuda_hook(proc_id = 900303)]
fn cuCtxGetDevice_v2(device: *mut CUdevice, ctx: CUcontext) -> CUresult;

#[cuda_hook(proc_id = 900304)]
fn cuCtxGetFlags(flags: *mut c_uint) -> CUresult;

#[cuda_hook(proc_id = 900449)]
fn cuCtxGetId(ctx: CUcontext, ctxId: *mut c_ulonglong) -> CUresult;

#[cuda_hook(proc_id = 900305)]
fn cuCtxGetLimit(pvalue: *mut usize, limit: CUlimit) -> CUresult;

#[cuda_hook(proc_id = 900450, async_api = false)]
fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;

#[cuda_hook(proc_id = 900451, async_api = false)]
fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult;

#[cuda_hook(proc_id = 900452, async_api = false)]
fn cuCtxPopCurrent_v2(pctx: *mut CUcontext) -> CUresult;

#[cuda_hook(proc_id = 900306)]
fn cuCtxGetSharedMemConfig(pConfig: *mut CUsharedconfig) -> CUresult;

#[cuda_hook(proc_id = 900307)]
fn cuCtxGetStreamPriorityRange(leastPriority: *mut c_int, greatestPriority: *mut c_int)
-> CUresult;

#[cuda_hook(proc_id = 900308, async_api = false)]
fn cuCtxSetCacheConfig(config: CUfunc_cache) -> CUresult;

#[cuda_hook(proc_id = 900309, async_api = false)]
fn cuCtxSetLimit(limit: CUlimit, value: usize) -> CUresult;

#[cuda_hook(proc_id = 900310, async_api = false)]
fn cuCtxSetSharedMemConfig(config: CUsharedconfig) -> CUresult;

#[cuda_hook(proc_id = 900200, async_api = false)]
fn cuCtxSynchronize() -> CUresult;

#[cuda_hook(proc_id = 900311, async_api = false)]
fn cuCtxResetPersistingL2Cache() -> CUresult;

#[cuda_hook(proc_id = 900312)]
fn cuDeviceCanAccessPeer(canAccessPeer: *mut c_int, dev: CUdevice, peerDev: CUdevice) -> CUresult;

#[cuda_hook(proc_id = 900453, async_api = false)]
fn cuCtxEnablePeerAccess(peerContext: CUcontext, Flags: c_uint) -> CUresult;

#[cuda_hook(proc_id = 900454, async_api = false)]
fn cuCtxDisablePeerAccess(peerContext: CUcontext) -> CUresult;

#[cuda_hook(proc_id = 900313)]
fn cuDeviceComputeCapability(major: *mut c_int, minor: *mut c_int, dev: CUdevice) -> CUresult;

#[cuda_hook(proc_id = 650)]
fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;

#[cuda_hook(proc_id = 651)]
fn cuDeviceGetAttribute(pi: *mut c_int, attrib: CUdevice_attribute, dev: CUdevice) -> CUresult {
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            let value = *pi;
            let mut driver = DRIVER_CACHE.write().unwrap();
            match attrib {
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR => {
                    let minor = driver.device_arch.map(|arch| arch % 10).unwrap_or(0);
                    driver.device_arch = Some(value as u32 * 10 + minor);
                }
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR => {
                    let major = driver.device_arch.map(|arch| arch / 10).unwrap_or(0);
                    driver.device_arch = Some(major * 10 + value as u32);
                }
                _ => {}
            }
        }
    }
}

#[cuda_hook(proc_id = 900314)]
fn cuDeviceGetByPCIBusId(dev: *mut CUdevice, pciBusId: *const c_char) -> CUresult;

#[cuda_hook(proc_id = 900315)]
fn cuDeviceGetCount(count: *mut c_int) -> CUresult;

#[cuda_hook(proc_id = 900316)]
fn cuDeviceGetName(
    #[host(output, len = len)] name: *mut c_char,
    len: c_int,
    dev: CUdevice,
) -> CUresult;

#[cuda_hook(proc_id = 900317)]
fn cuDeviceGetPCIBusId(
    #[host(output, len = len)] pciBusId: *mut c_char,
    len: c_int,
    dev: CUdevice,
) -> CUresult;

#[cuda_hook(proc_id = 900318)]
fn cuDeviceGetUuid_v2(uuid: *mut CUuuid, dev: CUdevice) -> CUresult;

#[cuda_hook(proc_id = 900319)]
fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult;

#[cuda_hook(proc_id = 900455)]
fn cuDeviceGetHostAtomicCapabilities(
    #[host(output, len = count as usize)] capabilities: *mut c_uint,
    #[host(len = count as usize)] operations: *const CUatomicOperation,
    count: c_uint,
    dev: CUdevice,
) -> CUresult;

#[cuda_hook(proc_id = 900456)]
fn cuDeviceGetTexture1DLinearMaxWidth(
    maxWidthInElements: *mut usize,
    format: CUarray_format,
    numChannels: c_uint,
    dev: CUdevice,
) -> CUresult;

#[cuda_hook(proc_id = 900322)]
fn cuDeviceGetProperties(prop: *mut CUdevprop, dev: CUdevice) -> CUresult;

#[cuda_hook(proc_id = 900323)]
fn cuDeviceGetP2PAttribute(
    value: *mut c_int,
    attrib: CUdevice_P2PAttribute,
    srcDevice: CUdevice,
    dstDevice: CUdevice,
) -> CUresult;

#[cuda_hook(proc_id = 900324)]
fn cuDeviceGetP2PAtomicCapabilities(
    #[host(output, len = count as usize)] capabilities: *mut c_uint,
    #[host(len = count as usize)] operations: *const CUatomicOperation,
    count: c_uint,
    srcDevice: CUdevice,
    dstDevice: CUdevice,
) -> CUresult;

#[cuda_hook(proc_id = 900320)]
fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult;

#[cuda_hook(proc_id = 900321, async_api = false)]
fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult;

#[cuda_hook(proc_id = 900433)]
fn cuDeviceGetDefaultMemPool(pool_out: *mut CUmemoryPool, dev: CUdevice) -> CUresult;

#[cuda_hook(proc_id = 900434)]
fn cuDeviceGetMemPool(pool: *mut CUmemoryPool, dev: CUdevice) -> CUresult;

#[cuda_hook(proc_id = 900435, async_api = false)]
fn cuDeviceSetMemPool(dev: CUdevice, pool: CUmemoryPool) -> CUresult;

#[cuda_hook(proc_id = 910)]
fn cuFuncGetAttribute(pi: *mut c_int, attrib: CUfunction_attribute, hfunc: CUfunction) -> CUresult;

#[cuda_hook(proc_id = 844)]
fn cuPointerGetAttribute(
    #[host(output, len = attribute.data_size())] data: *mut c_void,
    attribute: CUpointer_attribute,
    ptr: CUdeviceptr,
) -> CUresult;

#[cuda_hook(proc_id = 900201)]
fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;

#[cuda_hook(proc_id = 900492)]
fn cuMemAllocPitch_v2(
    dptr: *mut CUdeviceptr,
    pPitch: *mut usize,
    WidthInBytes: usize,
    Height: usize,
    ElementSizeBytes: c_uint,
) -> CUresult;

#[cuda_hook(proc_id = 900401)]
fn cuMemAllocAsync(dptr: *mut CUdeviceptr, bytesize: usize, hStream: CUstream) -> CUresult;

#[cuda_hook(proc_id = 900202, async_api = false)]
fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;

#[cuda_hook(proc_id = 900402, async_api = false)]
fn cuMemFreeAsync(dptr: CUdeviceptr, hStream: CUstream) -> CUresult;

#[cuda_hook(proc_id = 900403)]
fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;

#[cuda_hook(proc_id = 900436, async_api = false)]
fn cuMemPoolTrimTo(pool: CUmemoryPool, minBytesToKeep: usize) -> CUresult;

#[cuda_hook(proc_id = 900437)]
fn cuMemPoolGetAttribute(
    pool: CUmemoryPool,
    attr: CUmemPool_attribute,
    #[host(output, len = attr.data_size())] value: *mut c_void,
) -> CUresult;

#[cuda_hook(proc_id = 900438)]
fn cuMemPoolSetAttribute(
    pool: CUmemoryPool,
    attr: CUmemPool_attribute,
    #[host(input, len = attr.data_size())] value: *mut c_void,
) -> CUresult;

#[cuda_hook(proc_id = 900439)]
fn cuMemAllocFromPoolAsync(
    dptr: *mut CUdeviceptr,
    bytesize: usize,
    pool: CUmemoryPool,
    hStream: CUstream,
) -> CUresult;

#[cuda_hook(proc_id = 900404, async_api)]
fn cuMemcpy(dst: CUdeviceptr, src: CUdeviceptr, ByteCount: usize) -> CUresult;

#[cuda_hook(proc_id = 900405, async_api)]
fn cuMemcpyAsync(
    dst: CUdeviceptr,
    src: CUdeviceptr,
    ByteCount: usize,
    hStream: CUstream,
) -> CUresult;

#[cuda_hook(proc_id = 900203, async_api)]
fn cuMemcpyHtoD_v2(
    dstDevice: CUdeviceptr,
    #[host(len = ByteCount)] srcHost: *const c_void,
    ByteCount: usize,
) -> CUresult;

#[cuda_hook(proc_id = 900204)]
fn cuMemcpyDtoH_v2(
    #[host(output, len = ByteCount)] dstHost: *mut c_void,
    srcDevice: CUdeviceptr,
    ByteCount: usize,
) -> CUresult;

#[cuda_hook(proc_id = 900205, async_api)]
fn cuMemcpyDtoD_v2(dstDevice: CUdeviceptr, srcDevice: CUdeviceptr, ByteCount: usize) -> CUresult;

#[cuda_hook(proc_id = 900464, async_api)]
fn cuMemcpyDtoA_v2(
    dstArray: CUarray,
    dstOffset: usize,
    srcDevice: CUdeviceptr,
    ByteCount: usize,
) -> CUresult;

#[cuda_hook(proc_id = 900465, async_api)]
fn cuMemcpyAtoD_v2(
    dstDevice: CUdeviceptr,
    srcArray: CUarray,
    srcOffset: usize,
    ByteCount: usize,
) -> CUresult;

#[cuda_hook(proc_id = 900466)]
fn cuMemcpyAtoH_v2(
    #[host(output, len = ByteCount)] dstHost: *mut c_void,
    srcArray: CUarray,
    srcOffset: usize,
    ByteCount: usize,
) -> CUresult;

#[cuda_hook(proc_id = 900475)]
fn cuMemcpyAtoHAsync_v2(
    #[host(output, len = ByteCount)] dstHost: *mut c_void,
    srcArray: CUarray,
    srcOffset: usize,
    ByteCount: usize,
    hStream: CUstream,
) -> CUresult {
    'server_execution: {
        let result = unsafe {
            assert_eq!(cuStreamSynchronize(hStream), Default::default());
            cuMemcpyAtoH_v2(dstHost__ptr.cast(), srcArray, srcOffset, ByteCount)
        };
    }
}

#[cuda_hook(proc_id = 900467, async_api)]
fn cuMemcpyHtoA_v2(
    dstArray: CUarray,
    dstOffset: usize,
    #[host(len = ByteCount)] srcHost: *const c_void,
    ByteCount: usize,
) -> CUresult;

#[cuda_hook(proc_id = 900476)]
fn cuMemcpyHtoAAsync_v2(
    dstArray: CUarray,
    dstOffset: usize,
    #[host(len = ByteCount)] srcHost: *const c_void,
    ByteCount: usize,
    hStream: CUstream,
) -> CUresult {
    'server_execution: {
        let result = unsafe {
            assert_eq!(cuStreamSynchronize(hStream), Default::default());
            cuMemcpyHtoA_v2(dstArray, dstOffset, srcHost__ptr.cast(), ByteCount)
        };
    }
}

#[cuda_hook(proc_id = 900406, async_api)]
fn cuMemcpyDtoDAsync_v2(
    dstDevice: CUdeviceptr,
    srcDevice: CUdeviceptr,
    ByteCount: usize,
    hStream: CUstream,
) -> CUresult;

#[cuda_hook(proc_id = 900407)]
fn cuMemcpyDtoHAsync_v2(
    #[host(output, len = ByteCount)] dstHost: *mut c_void,
    srcDevice: CUdeviceptr,
    ByteCount: usize,
    hStream: CUstream,
) -> CUresult {
    'server_execution: {
        let result = unsafe {
            assert_eq!(cuStreamSynchronize(hStream), Default::default());
            cuMemcpyDtoH_v2(dstHost__ptr.cast(), srcDevice, ByteCount)
        };
    }
}

#[cuda_hook(proc_id = 900408)]
fn cuMemcpyHtoDAsync_v2(
    dstDevice: CUdeviceptr,
    #[host(len = ByteCount)] srcHost: *const c_void,
    ByteCount: usize,
    hStream: CUstream,
) -> CUresult {
    'server_execution: {
        let result = unsafe {
            assert_eq!(cuStreamSynchronize(hStream), Default::default());
            cuMemcpyHtoD_v2(dstDevice, srcHost__ptr.cast(), ByteCount)
        };
    }
}

#[cuda_hook(proc_id = 900206, async_api)]
fn cuMemsetD8_v2(dstDevice: CUdeviceptr, uc: c_uchar, N: usize) -> CUresult;

#[cuda_hook(proc_id = 900207, async_api)]
fn cuMemsetD32_v2(dstDevice: CUdeviceptr, ui: c_uint, N: usize) -> CUresult;

#[cuda_hook(proc_id = 900409, async_api)]
fn cuMemsetD8Async(dstDevice: CUdeviceptr, uc: c_uchar, N: usize, hStream: CUstream) -> CUresult;

#[cuda_hook(proc_id = 900410, async_api)]
fn cuMemsetD32Async(dstDevice: CUdeviceptr, ui: c_uint, N: usize, hStream: CUstream) -> CUresult;

#[cuda_hook(proc_id = 900493, async_api)]
fn cuMemsetD2D8_v2(
    dstDevice: CUdeviceptr,
    dstPitch: usize,
    uc: c_uchar,
    Width: usize,
    Height: usize,
) -> CUresult;

#[cuda_hook(proc_id = 900494, async_api)]
fn cuMemsetD2D16_v2(
    dstDevice: CUdeviceptr,
    dstPitch: usize,
    us: c_ushort,
    Width: usize,
    Height: usize,
) -> CUresult;

#[cuda_hook(proc_id = 900495, async_api)]
fn cuMemsetD2D32_v2(
    dstDevice: CUdeviceptr,
    dstPitch: usize,
    ui: c_uint,
    Width: usize,
    Height: usize,
) -> CUresult;

#[cuda_hook(proc_id = 900496, async_api)]
fn cuMemsetD2D8Async(
    dstDevice: CUdeviceptr,
    dstPitch: usize,
    uc: c_uchar,
    Width: usize,
    Height: usize,
    hStream: CUstream,
) -> CUresult;

#[cuda_hook(proc_id = 900497, async_api)]
fn cuMemsetD2D16Async(
    dstDevice: CUdeviceptr,
    dstPitch: usize,
    us: c_ushort,
    Width: usize,
    Height: usize,
    hStream: CUstream,
) -> CUresult;

#[cuda_hook(proc_id = 900498, async_api)]
fn cuMemsetD2D32Async(
    dstDevice: CUdeviceptr,
    dstPitch: usize,
    ui: c_uint,
    Width: usize,
    Height: usize,
    hStream: CUstream,
) -> CUresult;

#[cuda_hook(proc_id = 900208)]
fn cuStreamCreate(phStream: *mut CUstream, Flags: c_uint) -> CUresult;

#[cuda_hook(proc_id = 900411)]
fn cuStreamCreateWithPriority(phStream: *mut CUstream, flags: c_uint, priority: c_int) -> CUresult;

#[cuda_hook(proc_id = 900209, async_api = false)]
fn cuStreamDestroy_v2(hStream: CUstream) -> CUresult;

#[cuda_hook(proc_id = 900412)]
fn cuStreamGetFlags(hStream: CUstream, flags: *mut c_uint) -> CUresult;

#[cuda_hook(proc_id = 900413)]
fn cuStreamGetPriority(hStream: CUstream, priority: *mut c_int) -> CUresult;

#[cuda_hook(proc_id = 900414, async_api = false)]
fn cuStreamQuery(hStream: CUstream) -> CUresult;

#[cuda_hook(proc_id = 900415)]
fn cuStreamGetDevice(hStream: CUstream, device: *mut CUdevice) -> CUresult;

#[cuda_hook(proc_id = 900416)]
fn cuStreamGetId(hStream: CUstream, streamId: *mut c_ulonglong) -> CUresult;

#[cuda_hook(proc_id = 900417)]
fn cuStreamGetCtx(hStream: CUstream, pctx: *mut CUcontext) -> CUresult;

#[cuda_hook(proc_id = 900418, async_api = false)]
fn cuStreamWaitEvent(hStream: CUstream, hEvent: CUevent, Flags: c_uint) -> CUresult;

#[cuda_hook(proc_id = 900432)]
fn cuStreamCopyAttributes(dst: CUstream, src: CUstream) -> CUresult;

#[cuda_hook(proc_id = 900210, async_api = false)]
fn cuStreamSynchronize(hStream: CUstream) -> CUresult;

#[cuda_hook(proc_id = 900211)]
fn cuEventCreate(phEvent: *mut CUevent, Flags: c_uint) -> CUresult;

#[cuda_hook(proc_id = 900212, async_api)]
fn cuEventRecord(hEvent: CUevent, hStream: CUstream) -> CUresult;

#[cuda_hook(proc_id = 900400, async_api)]
fn cuEventRecordWithFlags(hEvent: CUevent, hStream: CUstream, flags: c_uint) -> CUresult;

#[cuda_hook(proc_id = 900213, async_api = false)]
fn cuEventSynchronize(hEvent: CUevent) -> CUresult;

#[cuda_hook(proc_id = 900419, async_api = false)]
fn cuEventQuery(hEvent: CUevent) -> CUresult;

#[cuda_hook(proc_id = 900214, async_api = false)]
fn cuEventDestroy_v2(hEvent: CUevent) -> CUresult;

#[cuda_hook(proc_id = 900428)]
fn cuEventElapsedTime_v2(pMilliseconds: *mut f32, hStart: CUevent, hEnd: CUevent) -> CUresult;

#[cuda_hook(proc_id = 1002)]
fn cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    numBlocks: *mut c_int,
    func: CUfunction,
    blockSize: c_int,
    dynamicSMemSize: usize,
    flags: c_uint,
) -> CUresult;

#[cuda_hook(proc_id = 912, async_api = false)]
fn cuFuncSetAttribute(hfunc: CUfunction, attrib: CUfunction_attribute, value: c_int) -> CUresult;

#[cuda_hook(proc_id = 900526, async_api = false)]
fn cuFuncSetCacheConfig(hfunc: CUfunction, config: CUfunc_cache) -> CUresult;

#[cuda_hook(proc_id = 900527, async_api = false)]
fn cuFuncSetSharedMemConfig(hfunc: CUfunction, config: CUsharedconfig) -> CUresult;

#[cuda_hook(proc_id = 900528)]
fn cuOccupancyMaxActiveBlocksPerMultiprocessor(
    numBlocks: *mut c_int,
    func: CUfunction,
    blockSize: c_int,
    dynamicSMemSize: usize,
) -> CUresult;

#[cuda_hook(proc_id = 900529)]
fn cuOccupancyAvailableDynamicSMemPerBlock(
    dynamicSmemSize: *mut usize,
    func: CUfunction,
    numBlocks: c_int,
    blockSize: c_int,
) -> CUresult;

#[cuda_hook(proc_id = 900800)]
fn cuGraphCreate(phGraph: *mut CUgraph, flags: c_uint) -> CUresult;

#[cuda_hook(proc_id = 900801, async_api = false)]
fn cuGraphDestroy(hGraph: CUgraph) -> CUresult;

#[cuda_hook(proc_id = 900802)]
fn cuGraphAddEmptyNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    #[device] dependencies: *const CUgraphNode,
    numDependencies: usize,
) -> CUresult {
    'client_before_send: {
        assert!(dependencies.is_null());
        assert_eq!(numDependencies, 0);
    }
}

#[cuda_hook(proc_id = 900803)]
fn cuGraphAddDependencies_v2(
    hGraph: CUgraph,
    #[host(len = numDependencies)] from: *const CUgraphNode,
    #[host(len = numDependencies)] to: *const CUgraphNode,
    #[device] edgeData: *const CUgraphEdgeData,
    numDependencies: usize,
) -> CUresult {
    'client_before_send: {
        assert!(edgeData.is_null());
    }
}

#[cuda_hook(proc_id = 900804)]
fn cuGraphRemoveDependencies_v2(
    hGraph: CUgraph,
    #[host(len = numDependencies)] from: *const CUgraphNode,
    #[host(len = numDependencies)] to: *const CUgraphNode,
    #[device] edgeData: *const CUgraphEdgeData,
    numDependencies: usize,
) -> CUresult {
    'client_before_send: {
        assert!(edgeData.is_null());
    }
}

#[cuda_hook(proc_id = 900805, async_api = false)]
fn cuGraphDestroyNode(hNode: CUgraphNode) -> CUresult;

#[cuda_custom_hook(proc_id = 900806)]
fn cuGraphGetNodes(hGraph: CUgraph, nodes: *mut CUgraphNode, numNodes: *mut usize) -> CUresult;

#[cuda_custom_hook(proc_id = 900807)]
fn cuGraphGetRootNodes(
    hGraph: CUgraph,
    rootNodes: *mut CUgraphNode,
    numRootNodes: *mut usize,
) -> CUresult;

#[cuda_custom_hook(proc_id = 900808)]
fn cuGraphGetEdges_v2(
    hGraph: CUgraph,
    from: *mut CUgraphNode,
    to: *mut CUgraphNode,
    edgeData: *mut CUgraphEdgeData,
    numEdges: *mut usize,
) -> CUresult;

#[cuda_custom_hook(proc_id = 900809)]
fn cuGraphNodeGetDependencies_v2(
    hNode: CUgraphNode,
    dependencies: *mut CUgraphNode,
    edgeData: *mut CUgraphEdgeData,
    numDependencies: *mut usize,
) -> CUresult;

#[cuda_custom_hook(proc_id = 900810)]
fn cuGraphNodeGetDependentNodes_v2(
    hNode: CUgraphNode,
    dependentNodes: *mut CUgraphNode,
    edgeData: *mut CUgraphEdgeData,
    numDependentNodes: *mut usize,
) -> CUresult;

#[cuda_hook(proc_id = 900811)]
fn cuGraphGetId(hGraph: CUgraph, graphId: *mut c_uint) -> CUresult;

#[cuda_hook(proc_id = 900812)]
fn cuGraphClone(phGraphClone: *mut CUgraph, originalGraph: CUgraph) -> CUresult;

#[cuda_hook(proc_id = 900813)]
fn cuGraphNodeFindInClone(
    phNode: *mut CUgraphNode,
    hOriginalNode: CUgraphNode,
    hClonedGraph: CUgraph,
) -> CUresult;

#[cuda_hook(proc_id = 900814)]
fn cuGraphNodeGetType(hNode: CUgraphNode, type_: *mut CUgraphNodeType) -> CUresult;

#[cuda_hook(proc_id = 900815)]
fn cuGraphNodeGetContainingGraph(hNode: CUgraphNode, phGraph: *mut CUgraph) -> CUresult;

#[cuda_hook(proc_id = 900816)]
fn cuGraphNodeGetLocalId(hNode: CUgraphNode, nodeId: *mut c_uint) -> CUresult;

#[cuda_hook(proc_id = 900817)]
fn cuGraphNodeGetToolsId(hNode: CUgraphNode, toolsNodeId: *mut c_ulonglong) -> CUresult;

#[cuda_hook(proc_id = 900818)]
fn cuGraphInstantiateWithFlags(
    phGraphExec: *mut CUgraphExec,
    hGraph: CUgraph,
    flags: c_ulonglong,
) -> CUresult;

#[cuda_hook(proc_id = 900819)]
fn cuGraphInstantiateWithParams(
    phGraphExec: *mut CUgraphExec,
    hGraph: CUgraph,
    #[host(input, len = 1)] instantiateParams: *mut CUDA_GRAPH_INSTANTIATE_PARAMS,
) -> CUresult {
    'client_after_recv: {
        let mut instantiate_params_out: CUDA_GRAPH_INSTANTIATE_PARAMS =
            unsafe { std::mem::zeroed() };
        instantiate_params_out.recv(channel_receiver).unwrap();
        unsafe {
            std::ptr::write(
                instantiateParams.as_ptr().cast_mut(),
                instantiate_params_out,
            );
        }
    }
    'server_after_send: {
        instantiateParams[0].send(channel_sender).unwrap();
        channel_sender.flush_out().unwrap();
    }
}

#[cuda_hook(proc_id = 900820)]
fn cuGraphExecGetFlags(hGraphExec: CUgraphExec, flags: *mut cuuint64_t) -> CUresult;

#[cuda_hook(proc_id = 900821)]
fn cuGraphExecGetId(hGraphExec: CUgraphExec, graphId: *mut c_uint) -> CUresult;

#[cuda_hook(proc_id = 900822)]
fn cuGraphExecUpdate_v2(
    hGraphExec: CUgraphExec,
    hGraph: CUgraph,
    resultInfo: *mut CUgraphExecUpdateResultInfo,
) -> CUresult;

#[cuda_hook(proc_id = 900823, async_api)]
fn cuGraphUpload(hGraphExec: CUgraphExec, hStream: CUstream) -> CUresult;

#[cuda_hook(proc_id = 900824, async_api)]
fn cuGraphLaunch(hGraphExec: CUgraphExec, hStream: CUstream) -> CUresult;

#[cuda_hook(proc_id = 900825, async_api = false)]
fn cuGraphExecDestroy(hGraphExec: CUgraphExec) -> CUresult;

#[cuda_hook(proc_id = 900826)]
fn cuGraphDebugDotPrint(hGraph: CUgraph, path: *const c_char, flags: c_uint) -> CUresult;

#[cuda_custom_hook]
fn cuGetProcAddress_v2(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    cudaVersion: c_int,
    flags: cuuint64_t,
    symbolStatus: *mut CUdriverProcAddressQueryResult,
) -> CUresult;
