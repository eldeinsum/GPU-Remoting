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

#[cuda_hook(proc_id = 900580)]
fn cuArrayGetSparseProperties(
    #[host(output, len = 1)] sparseProperties: *mut CUDA_ARRAY_SPARSE_PROPERTIES,
    array: CUarray,
) -> CUresult;

#[cuda_hook(proc_id = 900581)]
fn cuArrayGetMemoryRequirements(
    #[host(output, len = 1)] memoryRequirements: *mut CUDA_ARRAY_MEMORY_REQUIREMENTS,
    array: CUarray,
    device: CUdevice,
) -> CUresult;

#[cuda_hook(proc_id = 900582)]
fn cuMipmappedArrayCreate(
    pHandle: *mut CUmipmappedArray,
    #[host(len = 1)] pMipmappedArrayDesc: *const CUDA_ARRAY3D_DESCRIPTOR,
    numMipmapLevels: c_uint,
) -> CUresult;

#[cuda_hook(proc_id = 900583, async_api = false)]
fn cuMipmappedArrayDestroy(hMipmappedArray: CUmipmappedArray) -> CUresult;

#[cuda_hook(proc_id = 901116)]
fn cuMemMapArrayAsync(
    #[host(input, len = count as usize)] mapInfoList: *mut CUarrayMapInfo,
    count: c_uint,
    hStream: CUstream,
) -> CUresult;

#[cuda_hook(proc_id = 900584)]
fn cuMipmappedArrayGetLevel(
    pLevelArray: *mut CUarray,
    hMipmappedArray: CUmipmappedArray,
    level: c_uint,
) -> CUresult;

#[cuda_hook(proc_id = 900585)]
fn cuMipmappedArrayGetMemoryRequirements(
    #[host(output, len = 1)] memoryRequirements: *mut CUDA_ARRAY_MEMORY_REQUIREMENTS,
    mipmap: CUmipmappedArray,
    device: CUdevice,
) -> CUresult;

#[cuda_hook(proc_id = 900586)]
fn cuMipmappedArrayGetSparseProperties(
    #[host(output, len = 1)] sparseProperties: *mut CUDA_ARRAY_SPARSE_PROPERTIES,
    mipmap: CUmipmappedArray,
) -> CUresult;

#[cuda_custom_hook(proc_id = 900975)]
fn cuTexObjectCreate(
    pTexObject: *mut CUtexObject,
    pResDesc: *const CUDA_RESOURCE_DESC,
    pTexDesc: *const CUDA_TEXTURE_DESC,
    pResViewDesc: *const CUDA_RESOURCE_VIEW_DESC,
) -> CUresult;

#[cuda_hook(proc_id = 900976, async_api = false)]
fn cuTexObjectDestroy(texObject: CUtexObject) -> CUresult;

#[cuda_custom_hook(proc_id = 900977)]
fn cuTexObjectGetResourceDesc(
    pResDesc: *mut CUDA_RESOURCE_DESC,
    texObject: CUtexObject,
) -> CUresult;

#[cuda_hook(proc_id = 900978)]
fn cuTexObjectGetTextureDesc(
    #[host(output, len = 1)] pTexDesc: *mut CUDA_TEXTURE_DESC,
    texObject: CUtexObject,
) -> CUresult;

#[cuda_hook(proc_id = 900979)]
fn cuTexObjectGetResourceViewDesc(
    #[host(output, len = 1)] pResViewDesc: *mut CUDA_RESOURCE_VIEW_DESC,
    texObject: CUtexObject,
) -> CUresult;

#[cuda_custom_hook(proc_id = 900980)]
fn cuSurfObjectCreate(
    pSurfObject: *mut CUsurfObject,
    pResDesc: *const CUDA_RESOURCE_DESC,
) -> CUresult;

#[cuda_hook(proc_id = 900981, async_api = false)]
fn cuSurfObjectDestroy(surfObject: CUsurfObject) -> CUresult;

#[cuda_custom_hook(proc_id = 900982)]
fn cuSurfObjectGetResourceDesc(
    pResDesc: *mut CUDA_RESOURCE_DESC,
    surfObject: CUsurfObject,
) -> CUresult;

#[cuda_hook(proc_id = 901171)]
fn cuTensorMapEncodeTiled(
    #[host(output, len = 1)] tensorMap: *mut CUtensorMap,
    tensorDataType: CUtensorMapDataType,
    tensorRank: cuuint32_t,
    #[device] globalAddress: *mut c_void,
    #[host(len = tensorRank as usize)] globalDim: *const cuuint64_t,
    #[host(len = tensorRank.saturating_sub(1) as usize)] globalStrides: *const cuuint64_t,
    #[host(len = tensorRank as usize)] boxDim: *const cuuint32_t,
    #[host(len = tensorRank as usize)] elementStrides: *const cuuint32_t,
    interleave: CUtensorMapInterleave,
    swizzle: CUtensorMapSwizzle,
    l2Promotion: CUtensorMapL2promotion,
    oobFill: CUtensorMapFloatOOBfill,
) -> CUresult;

#[cuda_hook(proc_id = 901172)]
fn cuTensorMapEncodeIm2col(
    #[host(output, len = 1)] tensorMap: *mut CUtensorMap,
    tensorDataType: CUtensorMapDataType,
    tensorRank: cuuint32_t,
    #[device] globalAddress: *mut c_void,
    #[host(len = tensorRank as usize)] globalDim: *const cuuint64_t,
    #[host(len = tensorRank.saturating_sub(1) as usize)] globalStrides: *const cuuint64_t,
    #[host(len = tensorRank.saturating_sub(2) as usize)] pixelBoxLowerCorner: *const c_int,
    #[host(len = tensorRank.saturating_sub(2) as usize)] pixelBoxUpperCorner: *const c_int,
    channelsPerPixel: cuuint32_t,
    pixelsPerColumn: cuuint32_t,
    #[host(len = tensorRank as usize)] elementStrides: *const cuuint32_t,
    interleave: CUtensorMapInterleave,
    swizzle: CUtensorMapSwizzle,
    l2Promotion: CUtensorMapL2promotion,
    oobFill: CUtensorMapFloatOOBfill,
) -> CUresult;

#[cuda_hook(proc_id = 901173)]
fn cuTensorMapEncodeIm2colWide(
    #[host(output, len = 1)] tensorMap: *mut CUtensorMap,
    tensorDataType: CUtensorMapDataType,
    tensorRank: cuuint32_t,
    #[device] globalAddress: *mut c_void,
    #[host(len = tensorRank as usize)] globalDim: *const cuuint64_t,
    #[host(len = tensorRank.saturating_sub(1) as usize)] globalStrides: *const cuuint64_t,
    pixelBoxLowerCornerWidth: c_int,
    pixelBoxUpperCornerWidth: c_int,
    channelsPerPixel: cuuint32_t,
    pixelsPerColumn: cuuint32_t,
    #[host(len = tensorRank as usize)] elementStrides: *const cuuint32_t,
    interleave: CUtensorMapInterleave,
    mode: CUtensorMapIm2ColWideMode,
    swizzle: CUtensorMapSwizzle,
    l2Promotion: CUtensorMapL2promotion,
    oobFill: CUtensorMapFloatOOBfill,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901174)]
fn cuTensorMapReplaceAddress(
    tensorMap: *mut CUtensorMap,
    globalAddress: *mut c_void,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901175)]
fn cuMemGetHandleForAddressRange(
    handle: *mut c_void,
    dptr: CUdeviceptr,
    size: usize,
    handleType: CUmemRangeHandleType,
    flags: c_ulonglong,
) -> CUresult;

#[cuda_hook(proc_id = 901053)]
fn cuTexRefCreate(pTexRef: *mut CUtexref) -> CUresult;

#[cuda_hook(proc_id = 901054, async_api = false)]
fn cuTexRefDestroy(hTexRef: CUtexref) -> CUresult;

#[cuda_hook(proc_id = 901055)]
fn cuTexRefSetArray(hTexRef: CUtexref, hArray: CUarray, Flags: c_uint) -> CUresult;

#[cuda_hook(proc_id = 901056)]
fn cuTexRefGetArray(phArray: *mut CUarray, hTexRef: CUtexref) -> CUresult;

#[cuda_hook(proc_id = 901057)]
fn cuTexRefSetMipmappedArray(
    hTexRef: CUtexref,
    hMipmappedArray: CUmipmappedArray,
    Flags: c_uint,
) -> CUresult;

#[cuda_hook(proc_id = 901058)]
fn cuTexRefGetMipmappedArray(
    phMipmappedArray: *mut CUmipmappedArray,
    hTexRef: CUtexref,
) -> CUresult;

#[cuda_hook(proc_id = 901059)]
fn cuTexRefSetAddress_v2(
    ByteOffset: *mut usize,
    hTexRef: CUtexref,
    dptr: CUdeviceptr,
    bytes: usize,
) -> CUresult;

#[cuda_hook(proc_id = 901060)]
fn cuTexRefGetAddress_v2(pdptr: *mut CUdeviceptr, hTexRef: CUtexref) -> CUresult;

#[cuda_hook(proc_id = 901061)]
fn cuTexRefSetAddress2D_v3(
    hTexRef: CUtexref,
    #[host(len = 1)] desc: *const CUDA_ARRAY_DESCRIPTOR,
    dptr: CUdeviceptr,
    Pitch: usize,
) -> CUresult;

#[cuda_hook(proc_id = 901062)]
fn cuTexRefSetFormat(
    hTexRef: CUtexref,
    fmt: CUarray_format,
    NumPackedComponents: c_int,
) -> CUresult;

#[cuda_hook(proc_id = 901063)]
fn cuTexRefGetFormat(
    pFormat: *mut CUarray_format,
    pNumChannels: *mut c_int,
    hTexRef: CUtexref,
) -> CUresult;

#[cuda_hook(proc_id = 901064)]
fn cuTexRefSetAddressMode(hTexRef: CUtexref, dim: c_int, am: CUaddress_mode) -> CUresult;

#[cuda_hook(proc_id = 901065)]
fn cuTexRefGetAddressMode(pam: *mut CUaddress_mode, hTexRef: CUtexref, dim: c_int) -> CUresult;

#[cuda_hook(proc_id = 901066)]
fn cuTexRefSetFilterMode(hTexRef: CUtexref, fm: CUfilter_mode) -> CUresult;

#[cuda_hook(proc_id = 901067)]
fn cuTexRefGetFilterMode(pfm: *mut CUfilter_mode, hTexRef: CUtexref) -> CUresult;

#[cuda_hook(proc_id = 901068)]
fn cuTexRefSetMipmapFilterMode(hTexRef: CUtexref, fm: CUfilter_mode) -> CUresult;

#[cuda_hook(proc_id = 901069)]
fn cuTexRefGetMipmapFilterMode(pfm: *mut CUfilter_mode, hTexRef: CUtexref) -> CUresult;

#[cuda_hook(proc_id = 901070)]
fn cuTexRefSetMipmapLevelBias(hTexRef: CUtexref, bias: f32) -> CUresult;

#[cuda_hook(proc_id = 901071)]
fn cuTexRefGetMipmapLevelBias(pbias: *mut f32, hTexRef: CUtexref) -> CUresult;

#[cuda_hook(proc_id = 901072)]
fn cuTexRefSetMipmapLevelClamp(
    hTexRef: CUtexref,
    minMipmapLevelClamp: f32,
    maxMipmapLevelClamp: f32,
) -> CUresult;

#[cuda_hook(proc_id = 901073)]
fn cuTexRefGetMipmapLevelClamp(
    pminMipmapLevelClamp: *mut f32,
    pmaxMipmapLevelClamp: *mut f32,
    hTexRef: CUtexref,
) -> CUresult;

#[cuda_hook(proc_id = 901074)]
fn cuTexRefSetMaxAnisotropy(hTexRef: CUtexref, maxAniso: c_uint) -> CUresult;

#[cuda_hook(proc_id = 901075)]
fn cuTexRefGetMaxAnisotropy(pmaxAniso: *mut c_int, hTexRef: CUtexref) -> CUresult;

#[cuda_hook(proc_id = 901076)]
fn cuTexRefSetBorderColor(
    hTexRef: CUtexref,
    #[host(input, len = 4)] pBorderColor: *mut f32,
) -> CUresult;

#[cuda_hook(proc_id = 901077)]
fn cuTexRefGetBorderColor(
    #[host(output, len = 4)] pBorderColor: *mut f32,
    hTexRef: CUtexref,
) -> CUresult;

#[cuda_hook(proc_id = 901078)]
fn cuTexRefSetFlags(hTexRef: CUtexref, Flags: c_uint) -> CUresult;

#[cuda_hook(proc_id = 901079)]
fn cuTexRefGetFlags(pFlags: *mut c_uint, hTexRef: CUtexref) -> CUresult;

#[cuda_hook(proc_id = 901082)]
fn cuSurfRefSetArray(hSurfRef: CUsurfref, hArray: CUarray, Flags: c_uint) -> CUresult;

#[cuda_hook(proc_id = 901083)]
fn cuSurfRefGetArray(phArray: *mut CUarray, hSurfRef: CUsurfref) -> CUresult;

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

#[cuda_hook(proc_id = 901034, async_api)]
fn cuLaunchCooperativeKernel(
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
) -> CUresult {
    'client_before_send: {
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
        let result = super::cuda_exe_utils::cu_launch_cooperative_kernel(
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

#[cuda_hook(proc_id = 901037, async_api)]
fn cuLaunchCooperativeKernelMultiDevice(
    #[skip] launchParamsList: *mut CUDA_LAUNCH_PARAMS,
    numDevices: c_uint,
    flags: c_uint,
) -> CUresult {
    'client_before_send: {
        assert!(numDevices == 0 || !launchParamsList.is_null());
        let launch_params =
            unsafe { std::slice::from_raw_parts(launchParamsList, numDevices as usize) };
        let mut packed_launch_params = launch_params.to_vec();
        for params in &mut packed_launch_params {
            params.kernelParams = std::ptr::null_mut();
        }
        let packed_args = {
            let driver = DRIVER_CACHE.read().unwrap();
            launch_params
                .iter()
                .map(|params| {
                    super::cuda_hijack_utils::pack_kernel_args_with_offsets(
                        params.kernelParams,
                        driver.function_params.get(&params.function).unwrap(),
                    )
                })
                .collect::<Vec<_>>()
        };
    }
    'client_extra_send: {
        send_slice(&packed_launch_params, channel_sender).unwrap();
        for (args, arg_offsets) in &packed_args {
            send_slice(arg_offsets, channel_sender).unwrap();
            send_slice(args, channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut launch_params = recv_slice::<CUDA_LAUNCH_PARAMS, _>(channel_receiver).unwrap();
        let mut arg_offsets_list = Vec::with_capacity(launch_params.len());
        let mut arg_buffers = Vec::with_capacity(launch_params.len());
        for _ in 0..launch_params.len() {
            arg_offsets_list.push(recv_slice::<u32, _>(channel_receiver).unwrap());
            arg_buffers.push(recv_slice::<u8, _>(channel_receiver).unwrap());
        }
        let mut kernel_params_list = arg_buffers
            .iter()
            .zip(arg_offsets_list.iter())
            .map(|(args, arg_offsets)| {
                super::cuda_exe_utils::kernel_params_from_packed_args(args, arg_offsets)
            })
            .collect::<Vec<_>>();
        for (params, kernel_params) in launch_params.iter_mut().zip(kernel_params_list.iter_mut()) {
            params.kernelParams = if kernel_params.is_empty() {
                std::ptr::null_mut()
            } else {
                kernel_params.as_mut_ptr()
            };
        }
    }
    'server_execution: {
        let result = unsafe {
            cuLaunchCooperativeKernelMultiDevice(launch_params.as_mut_ptr(), numDevices, flags)
        };
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

#[cuda_custom_hook] // calls the internal API below
fn cuModuleLoadDataEx(
    module: *mut CUmodule,
    image: *const c_void,
    numOptions: c_uint,
    options: *mut CUjit_option,
    optionValues: *mut *mut c_void,
) -> CUresult;

#[cuda_hook(proc_id = 901036)]
fn cuModuleLoadFatBinary(
    module: *mut CUmodule,
    #[host(len = len)] fatCubin: *const c_void,
) -> CUresult {
    'client_before_send: {
        let len = crate::elf::module_image_len(fatCubin);
    }
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            assert!(
                DRIVER_CACHE
                    .write()
                    .unwrap()
                    .images
                    .insert(*module, std::borrow::Cow::Owned(fatCubin.to_vec()))
                    .is_none()
            );
        }
    }
    'server_execution: {
        let result = unsafe { cuModuleLoadFatBinary(module__ptr, fatCubin__ptr.cast()) };
    }
    'server_after_send: {
        if result == CUresult::CUDA_SUCCESS {
            server.modules.push(module);
        }
    }
}

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
        let function_name = name.to_str().unwrap();
        let params = if FatBinaryHeader::is_fat_binary(image.as_ptr()) {
            let fatbin: &FatBinaryHeader = unsafe { &*image.as_ptr().cast() };
            fatbin.find_kernel_params(function_name, target_arch)
        } else {
            crate::elf::find_kernel_params_or_empty(image, function_name)
        };
        driver.function_params.insert(*hfunc, params);
        driver
            .function_names
            .insert(*hfunc, std::ffi::CString::new(function_name).unwrap());
    }
}

#[cuda_hook(proc_id = 901021)]
fn cuModuleGetFunctionCount(count: *mut c_uint, mod_: CUmodule) -> CUresult;

#[cuda_custom_hook(proc_id = 901022)]
fn cuModuleEnumerateFunctions(
    functions: *mut CUfunction,
    numFunctions: c_uint,
    mod_: CUmodule,
) -> CUresult;

#[cuda_hook(proc_id = 901080)]
fn cuModuleGetTexRef(pTexRef: *mut CUtexref, hmod: CUmodule, name: *const c_char) -> CUresult;

#[cuda_hook(proc_id = 901081)]
fn cuModuleGetSurfRef(pSurfRef: *mut CUsurfref, hmod: CUmodule, name: *const c_char) -> CUresult;

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

#[cuda_hook(proc_id = 901085, async_api = false)]
fn cuLinkCreate_v2(
    numOptions: c_uint,
    #[skip] options: *mut CUjit_option,
    #[skip] optionValues: *mut *mut c_void,
    stateOut: *mut CUlinkState,
) -> CUresult {
    'client_before_send: {
        assert_eq!(numOptions, 0);
        assert!(options.is_null());
        assert!(optionValues.is_null());
    }
    'server_execution: {
        let result = unsafe {
            cuLinkCreate_v2(
                numOptions,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                stateOut__ptr,
            )
        };
    }
    'server_after_send: {
        if result == CUresult::CUDA_SUCCESS {
            server.links.push(stateOut);
        }
    }
}

#[cuda_hook(proc_id = 901086)]
fn cuLinkAddData_v2(
    state: CUlinkState,
    type_: CUjitInputType,
    #[host(input, len = size)] data: *mut c_void,
    size: usize,
    name: *const c_char,
    numOptions: c_uint,
    #[skip] options: *mut CUjit_option,
    #[skip] optionValues: *mut *mut c_void,
) -> CUresult {
    'client_before_send: {
        assert!(!data.is_null());
        assert_eq!(numOptions, 0);
        assert!(options.is_null());
        assert!(optionValues.is_null());
    }
    'server_execution: {
        let result = unsafe {
            cuLinkAddData_v2(
                state,
                type_,
                data__ptr.cast(),
                size,
                name__ptr,
                numOptions,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
    }
}

#[cuda_hook(proc_id = 901087)]
fn cuLinkAddFile_v2(
    state: CUlinkState,
    type_: CUjitInputType,
    path: *const c_char,
    numOptions: c_uint,
    #[skip] options: *mut CUjit_option,
    #[skip] optionValues: *mut *mut c_void,
) -> CUresult {
    'client_before_send: {
        assert!(!path.is_null());
        assert_eq!(numOptions, 0);
        assert!(options.is_null());
        assert!(optionValues.is_null());
        let path_str = unsafe { std::ffi::CStr::from_ptr(path) }.to_str().unwrap();
        let file_bytes = std::fs::read(path_str).unwrap();
    }
    'client_extra_send: {
        send_slice(&file_bytes, channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut file_bytes = recv_slice::<u8, _>(channel_receiver).unwrap().into_vec();
        if type_ == CUjitInputType::CU_JIT_INPUT_PTX && !file_bytes.ends_with(&[0]) {
            file_bytes.push(0);
        }
    }
    'server_execution: {
        let result = unsafe {
            cuLinkAddData_v2(
                state,
                type_,
                file_bytes.as_mut_ptr().cast(),
                file_bytes.len(),
                path__ptr,
                numOptions,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
    }
}

#[cuda_hook(proc_id = 901088, async_api = false)]
fn cuLinkComplete(
    state: CUlinkState,
    #[skip] cubinOut: *mut *mut c_void,
    #[skip] sizeOut: *mut usize,
) -> CUresult {
    'client_before_send: {
        assert!(!cubinOut.is_null());
    }
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            let linked_image = recv_slice::<u8, _>(channel_receiver).unwrap();
            let linked_ptr = linked_image.as_ptr().cast_mut().cast::<c_void>();
            let linked_size = linked_image.len();
            DRIVER_CACHE
                .write()
                .unwrap()
                .linked_images
                .insert(state, linked_image);
            unsafe {
                *cubinOut = linked_ptr;
                if !sizeOut.is_null() {
                    *sizeOut = linked_size;
                }
            }
        }
    }
    'server_execution: {
        let mut cubin = std::ptr::null_mut::<c_void>();
        let mut size = 0usize;
        let result = unsafe { cuLinkComplete(state, &raw mut cubin, &raw mut size) };
        let linked_image = if result == CUresult::CUDA_SUCCESS {
            assert!(!cubin.is_null());
            unsafe { std::slice::from_raw_parts(cubin.cast::<u8>(), size).to_vec() }
        } else {
            Vec::new()
        };
    }
    'server_after_send: {
        if result == CUresult::CUDA_SUCCESS {
            send_slice(&linked_image, channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
}

#[cuda_hook(proc_id = 901089, async_api = false)]
fn cuLinkDestroy(state: CUlinkState) -> CUresult {
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            DRIVER_CACHE.write().unwrap().linked_images.remove(&state);
        }
    }
    'server_after_send: {
        if result == CUresult::CUDA_SUCCESS {
            server.links.retain(|item| *item != state);
        }
    }
}

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
                crate::elf::find_kernel_params_or_empty(image, kernel_name)
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

#[cuda_hook(proc_id = 901104)]
fn cuLibraryGetUnifiedFunction(
    fptr: *mut *mut c_void,
    library: CUlibrary,
    symbol: *const c_char,
) -> CUresult {
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            let target_arch = DRIVER_CACHE.read().unwrap().device_arch;
            let mut driver = DRIVER_CACHE.write().unwrap();
            let image = driver.library_images.get(&library).unwrap();
            let function_name = symbol.to_str().unwrap();
            let params = if FatBinaryHeader::is_fat_binary(image.as_ptr()) {
                let fatbin: &FatBinaryHeader = unsafe { &*image.as_ptr().cast() };
                fatbin.find_kernel_params(function_name, target_arch)
            } else {
                crate::elf::find_kernel_params_or_empty(image, function_name)
            };
            let function = (*fptr).cast::<CUfunc_st>();
            driver.function_params.insert(function, params);
            driver
                .function_names
                .insert(function, std::ffi::CString::new(function_name).unwrap());
        }
    }
}

#[cuda_hook(proc_id = 901023)]
fn cuLibraryGetKernelCount(count: *mut c_uint, lib: CUlibrary) -> CUresult;

#[cuda_custom_hook(proc_id = 901024)]
fn cuLibraryEnumerateKernels(
    kernels: *mut CUkernel,
    numKernels: c_uint,
    lib: CUlibrary,
) -> CUresult;

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
            if let Some(name) = driver.kernel_names.get(&kernel).cloned() {
                driver.function_names.insert(*pFunc, name);
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

#[cuda_hook(proc_id = 901035)]
fn cuLibraryGetManaged(
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

#[cuda_hook(proc_id = 900721)]
fn cuFuncGetModule(hmod: *mut CUmodule, hfunc: CUfunction) -> CUresult;

#[cuda_custom_hook]
fn cuFuncGetName(name: *mut *const c_char, hfunc: CUfunction) -> CUresult;

#[cuda_hook(proc_id = 900722)]
fn cuFuncGetParamInfo(
    func: CUfunction,
    paramIndex: usize,
    paramOffset: *mut usize,
    paramSize: *mut usize,
) -> CUresult;

#[cuda_hook(proc_id = 900723)]
fn cuFuncGetParamCount(func: CUfunction, paramCount: *mut usize) -> CUresult;

#[cuda_hook(proc_id = 900724)]
fn cuFuncIsLoaded(state: *mut CUfunctionLoadingState, function: CUfunction) -> CUresult;

#[cuda_hook(proc_id = 900725, async_api = false)]
fn cuFuncLoad(function: CUfunction) -> CUresult;

#[cuda_hook(proc_id = 901025, async_api = false)]
fn cuFuncSetBlockShape(hfunc: CUfunction, x: c_int, y: c_int, z: c_int) -> CUresult;

#[cuda_hook(proc_id = 901026, async_api = false)]
fn cuFuncSetSharedSize(hfunc: CUfunction, bytes: c_uint) -> CUresult;

#[cuda_hook(proc_id = 901027, async_api = false)]
fn cuParamSetSize(hfunc: CUfunction, numbytes: c_uint) -> CUresult;

#[cuda_hook(proc_id = 901028, async_api = false)]
fn cuParamSeti(hfunc: CUfunction, offset: c_int, value: c_uint) -> CUresult;

#[cuda_hook(proc_id = 901029, async_api = false)]
fn cuParamSetf(hfunc: CUfunction, offset: c_int, value: f32) -> CUresult;

#[cuda_hook(proc_id = 901030, async_api = false)]
fn cuParamSetv(
    hfunc: CUfunction,
    offset: c_int,
    #[host(input, len = numbytes as usize)] ptr: *mut c_void,
    numbytes: c_uint,
) -> CUresult;

#[cuda_hook(proc_id = 901084, async_api = false)]
fn cuParamSetTexRef(hfunc: CUfunction, texunit: c_int, hTexRef: CUtexref) -> CUresult;

#[cuda_hook(proc_id = 901031, async_api)]
fn cuLaunch(f: CUfunction) -> CUresult;

#[cuda_hook(proc_id = 901032, async_api)]
fn cuLaunchGrid(f: CUfunction, grid_width: c_int, grid_height: c_int) -> CUresult;

#[cuda_hook(proc_id = 901033, async_api)]
fn cuLaunchGridAsync(
    f: CUfunction,
    grid_width: c_int,
    grid_height: c_int,
    hStream: CUstream,
) -> CUresult;

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

#[cuda_hook(proc_id = 901090, async_api = false)]
fn cuProfilerInitialize(
    configFile: *const c_char,
    outputFile: *const c_char,
    outputMode: CUoutput_mode,
) -> CUresult;

#[cuda_hook(proc_id = 901091, async_api = false)]
fn cuProfilerStart() -> CUresult;

#[cuda_hook(proc_id = 901092, async_api = false)]
fn cuProfilerStop() -> CUresult;

#[cuda_hook(proc_id = 901100, async_api = false)]
fn cuCoredumpGetAttribute(
    attrib: CUcoredumpSettings,
    #[skip] value: *mut c_void,
    #[skip] size: *mut usize,
) -> CUresult {
    'client_before_send: {
        if size.is_null() {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        }
        let value_present = !value.is_null();
        let capacity = if value_present { unsafe { *size } } else { 0 };
    }
    'client_extra_send: {
        value_present.send(channel_sender).unwrap();
        capacity.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut value_present = false;
        value_present.recv(channel_receiver).unwrap();
        let mut capacity = 0usize;
        capacity.recv(channel_receiver).unwrap();
        let mut value_buffer = vec![0u8; capacity];
        let value_ptr = if value_present {
            value_buffer.as_mut_ptr().cast()
        } else {
            std::ptr::null_mut()
        };
        let mut size_value = capacity;
    }
    'server_execution: {
        let result = unsafe { cuCoredumpGetAttribute(attrib, value_ptr, &raw mut size_value) };
    }
    'server_after_send: {
        if result == CUresult::CUDA_SUCCESS {
            size_value.send(channel_sender).unwrap();
            let bytes = if value_present {
                &value_buffer[..size_value.min(capacity)]
            } else {
                &[]
            };
            send_slice(bytes, channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            let mut size_out = 0usize;
            size_out.recv(channel_receiver).unwrap();
            let bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            unsafe {
                *size = size_out;
                if value_present && !bytes.is_empty() {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), value.cast(), bytes.len());
                }
            }
        }
    }
}

#[cuda_hook(proc_id = 901101, async_api = false)]
fn cuCoredumpGetAttributeGlobal(
    attrib: CUcoredumpSettings,
    #[skip] value: *mut c_void,
    #[skip] size: *mut usize,
) -> CUresult {
    'client_before_send: {
        if size.is_null() {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        }
        let value_present = !value.is_null();
        let capacity = if value_present { unsafe { *size } } else { 0 };
    }
    'client_extra_send: {
        value_present.send(channel_sender).unwrap();
        capacity.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut value_present = false;
        value_present.recv(channel_receiver).unwrap();
        let mut capacity = 0usize;
        capacity.recv(channel_receiver).unwrap();
        let mut value_buffer = vec![0u8; capacity];
        let value_ptr = if value_present {
            value_buffer.as_mut_ptr().cast()
        } else {
            std::ptr::null_mut()
        };
        let mut size_value = capacity;
    }
    'server_execution: {
        let result =
            unsafe { cuCoredumpGetAttributeGlobal(attrib, value_ptr, &raw mut size_value) };
    }
    'server_after_send: {
        if result == CUresult::CUDA_SUCCESS {
            size_value.send(channel_sender).unwrap();
            let bytes = if value_present {
                &value_buffer[..size_value.min(capacity)]
            } else {
                &[]
            };
            send_slice(bytes, channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            let mut size_out = 0usize;
            size_out.recv(channel_receiver).unwrap();
            let bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            unsafe {
                *size = size_out;
                if value_present && !bytes.is_empty() {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), value.cast(), bytes.len());
                }
            }
        }
    }
}

#[cuda_hook(proc_id = 901102, async_api = false)]
fn cuCoredumpSetAttribute(
    attrib: CUcoredumpSettings,
    #[skip] value: *mut c_void,
    #[skip] size: *mut usize,
) -> CUresult {
    'client_before_send: {
        if size.is_null() {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        }
        let value_present = !value.is_null();
        let value_size = unsafe { *size };
        let bytes = if value_present && value_size > 0 {
            unsafe { std::slice::from_raw_parts(value.cast::<u8>(), value_size) }
        } else {
            &[]
        };
    }
    'client_extra_send: {
        value_present.send(channel_sender).unwrap();
        value_size.send(channel_sender).unwrap();
        send_slice(bytes, channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut value_present = false;
        value_present.recv(channel_receiver).unwrap();
        let mut size_value = 0usize;
        size_value.recv(channel_receiver).unwrap();
        let mut bytes = recv_slice::<u8, _>(channel_receiver).unwrap().to_vec();
        let value_ptr = if value_present {
            bytes.as_mut_ptr().cast()
        } else {
            std::ptr::null_mut()
        };
    }
    'server_execution: {
        let result = unsafe { cuCoredumpSetAttribute(attrib, value_ptr, &raw mut size_value) };
    }
    'server_after_send: {
        if result == CUresult::CUDA_SUCCESS {
            size_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            let mut size_out = 0usize;
            size_out.recv(channel_receiver).unwrap();
            unsafe {
                *size = size_out;
            }
        }
    }
}

#[cuda_hook(proc_id = 901103, async_api = false)]
fn cuCoredumpSetAttributeGlobal(
    attrib: CUcoredumpSettings,
    #[skip] value: *mut c_void,
    #[skip] size: *mut usize,
) -> CUresult {
    'client_before_send: {
        if size.is_null() {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        }
        let value_present = !value.is_null();
        let value_size = unsafe { *size };
        let bytes = if value_present && value_size > 0 {
            unsafe { std::slice::from_raw_parts(value.cast::<u8>(), value_size) }
        } else {
            &[]
        };
    }
    'client_extra_send: {
        value_present.send(channel_sender).unwrap();
        value_size.send(channel_sender).unwrap();
        send_slice(bytes, channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut value_present = false;
        value_present.recv(channel_receiver).unwrap();
        let mut size_value = 0usize;
        size_value.recv(channel_receiver).unwrap();
        let mut bytes = recv_slice::<u8, _>(channel_receiver).unwrap().to_vec();
        let value_ptr = if value_present {
            bytes.as_mut_ptr().cast()
        } else {
            std::ptr::null_mut()
        };
    }
    'server_execution: {
        let result =
            unsafe { cuCoredumpSetAttributeGlobal(attrib, value_ptr, &raw mut size_value) };
    }
    'server_after_send: {
        if result == CUresult::CUDA_SUCCESS {
            size_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            let mut size_out = 0usize;
            size_out.recv(channel_receiver).unwrap();
            unsafe {
                *size = size_out;
            }
        }
    }
}

#[cuda_hook(proc_id = 901093, async_api = false)]
fn cuLogsCurrent(iterator_out: *mut CUlogIterator, flags: c_uint) -> CUresult;

#[cuda_hook(proc_id = 901094, async_api = false)]
fn cuLogsDumpToFile(
    #[skip] iterator: *mut CUlogIterator,
    pathToFile: *const c_char,
    flags: c_uint,
) -> CUresult {
    'client_before_send: {
        if pathToFile.is_null() {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        }
        let iterator_present = !iterator.is_null();
        let iterator_in = if iterator_present {
            unsafe { *iterator }
        } else {
            0
        };
    }
    'client_extra_send: {
        iterator_present.send(channel_sender).unwrap();
        iterator_in.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut iterator_present = false;
        iterator_present.recv(channel_receiver).unwrap();
        let mut iterator_storage = 0;
        iterator_storage.recv(channel_receiver).unwrap();
        let iterator_ptr = if iterator_present {
            &raw mut iterator_storage
        } else {
            std::ptr::null_mut()
        };
    }
    'server_execution: {
        let result = unsafe { cuLogsDumpToFile(iterator_ptr, pathToFile__ptr, flags) };
    }
    'server_after_send: {
        if result == CUresult::CUDA_SUCCESS {
            iterator_storage.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            let mut iterator_out = 0;
            iterator_out.recv(channel_receiver).unwrap();
            if !iterator.is_null() {
                unsafe {
                    *iterator = iterator_out;
                }
            }
        }
    }
}

#[cuda_hook(proc_id = 901095, async_api = false)]
fn cuLogsDumpToMemory(
    #[skip] iterator: *mut CUlogIterator,
    #[skip] buffer: *mut c_char,
    #[skip] size: *mut usize,
    flags: c_uint,
) -> CUresult {
    'client_before_send: {
        if size.is_null() {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        }
        let capacity = unsafe { *size };
        if capacity > 0 && buffer.is_null() {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        }
        let iterator_present = !iterator.is_null();
        let iterator_in = if iterator_present {
            unsafe { *iterator }
        } else {
            0
        };
    }
    'client_extra_send: {
        iterator_present.send(channel_sender).unwrap();
        iterator_in.send(channel_sender).unwrap();
        capacity.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut iterator_present = false;
        iterator_present.recv(channel_receiver).unwrap();
        let mut iterator_storage = 0;
        iterator_storage.recv(channel_receiver).unwrap();
        let mut capacity = 0usize;
        capacity.recv(channel_receiver).unwrap();
        let iterator_ptr = if iterator_present {
            &raw mut iterator_storage
        } else {
            std::ptr::null_mut()
        };
        let mut log_buffer = vec![0i8; capacity.saturating_add(1)];
        let buffer_ptr = if capacity == 0 {
            std::ptr::null_mut()
        } else {
            log_buffer.as_mut_ptr()
        };
        let mut size_value = capacity;
    }
    'server_execution: {
        let result =
            unsafe { cuLogsDumpToMemory(iterator_ptr, buffer_ptr, &raw mut size_value, flags) };
    }
    'server_after_send: {
        if result == CUresult::CUDA_SUCCESS {
            iterator_storage.send(channel_sender).unwrap();
            size_value.send(channel_sender).unwrap();
            let bytes = log_buffer
                .iter()
                .take(size_value.min(capacity))
                .map(|value| *value as u8)
                .collect::<Vec<_>>();
            send_slice(&bytes, channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            let mut iterator_out = 0;
            iterator_out.recv(channel_receiver).unwrap();
            let mut size_out = 0usize;
            size_out.recv(channel_receiver).unwrap();
            let bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            if !iterator.is_null() {
                unsafe {
                    *iterator = iterator_out;
                }
            }
            unsafe {
                *size = size_out;
                if !buffer.is_null() && !bytes.is_empty() {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr().cast(), buffer, bytes.len());
                }
            }
        }
    }
}

#[cuda_hook(proc_id = 684)]
fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult;

#[cuda_hook(proc_id = 900926)]
fn cuCtxAttach(pctx: *mut CUcontext, flags: c_uint) -> CUresult;

#[cuda_hook(proc_id = 900927, async_api = false)]
fn cuCtxDetach(ctx: CUcontext) -> CUresult;

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

#[cuda_hook(proc_id = 900935)]
fn cuCtxFromGreenCtx(pContext: *mut CUcontext, hCtx: CUgreenCtx) -> CUresult;

#[cuda_hook(proc_id = 900889)]
fn cuCtxGetDevResource(
    hCtx: CUcontext,
    #[host(output, len = 1)] resource: *mut CUdevResource,
    type_: CUdevResourceType,
) -> CUresult;

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

#[cuda_hook(proc_id = 900922, async_api = false)]
fn cuCtxSetFlags(flags: c_uint) -> CUresult;

#[cuda_hook(proc_id = 900200, async_api = false)]
fn cuCtxSynchronize() -> CUresult;

#[cuda_hook(proc_id = 900880, async_api = false)]
fn cuCtxSynchronize_v2(ctx: CUcontext) -> CUresult;

#[cuda_hook(proc_id = 900881, async_api)]
fn cuCtxRecordEvent(hCtx: CUcontext, hEvent: CUevent) -> CUresult;

#[cuda_hook(proc_id = 900882, async_api)]
fn cuCtxWaitEvent(hCtx: CUcontext, hEvent: CUevent) -> CUresult;

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

#[cuda_hook(proc_id = 900919)]
fn cuDeviceGetLuid(
    #[host(output, len = 8)] luid: *mut c_char,
    deviceNodeMask: *mut c_uint,
    dev: CUdevice,
) -> CUresult {
    'server_before_execution: {
        unsafe {
            std::ptr::write_bytes(luid__ptr, 0, 8);
            *deviceNodeMask__ptr = 0;
        }
    }
}

#[cuda_hook(proc_id = 900918)]
fn cuDeviceGetExecAffinitySupport(
    pi: *mut c_int,
    type_: CUexecAffinityType,
    dev: CUdevice,
) -> CUresult;

#[cuda_hook(proc_id = 900920, async_api = false)]
fn cuFlushGPUDirectRDMAWrites(
    target: CUflushGPUDirectRDMAWritesTarget,
    scope: CUflushGPUDirectRDMAWritesScope,
) -> CUresult;

#[cuda_hook(proc_id = 900890)]
fn cuDeviceGetDevResource(
    device: CUdevice,
    #[host(output, len = 1)] resource: *mut CUdevResource,
    type_: CUdevResourceType,
) -> CUresult;

#[cuda_hook(proc_id = 900933)]
fn cuDevResourceGenerateDesc(
    phDesc: *mut CUdevResourceDesc,
    #[host(input, len = nbResources as usize)] resources: *mut CUdevResource,
    nbResources: c_uint,
) -> CUresult {
    'client_before_send: {
        assert!(!resources.is_null());
        let resource_count = nbResources as usize;
        let resource_slice = unsafe { std::slice::from_raw_parts(resources, resource_count) };
        assert!(
            resource_slice
                .iter()
                .all(|resource| resource.nextResource.is_null())
        );
    }
}

#[cuda_hook(proc_id = 900943)]
fn cuDevSmResourceSplitByCount(
    #[skip] result_out: *mut CUdevResource,
    #[skip] nbGroups: *mut c_uint,
    #[skip] input: *const CUdevResource,
    #[skip] remainder: *mut CUdevResource,
    flags: c_uint,
    minCount: c_uint,
) -> CUresult {
    'client_before_send: {
        assert!(!nbGroups.is_null());
        assert!(!input.is_null());
        let requested_groups = unsafe { *nbGroups };
        let has_result = !result_out.is_null();
        let result_capacity = if has_result {
            assert!(requested_groups > 0);
            requested_groups as usize
        } else {
            0
        };
        let input_resource = unsafe { *input };
        assert!(input_resource.nextResource.is_null());
        let has_remainder = !remainder.is_null();
    }
    'client_extra_send: {
        has_result.send(channel_sender).unwrap();
        requested_groups.send(channel_sender).unwrap();
        input_resource.send(channel_sender).unwrap();
        has_remainder.send(channel_sender).unwrap();
    }
    'client_after_recv: {
        let mut returned_groups = 0 as c_uint;
        returned_groups.recv(channel_receiver).unwrap();
        unsafe {
            *nbGroups = returned_groups;
        }
        if result == CUresult::CUDA_SUCCESS {
            if has_result {
                let result_resources = recv_slice::<CUdevResource, _>(channel_receiver).unwrap();
                assert!(result_resources.len() <= result_capacity);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        result_resources.as_ptr(),
                        result_out,
                        result_resources.len(),
                    );
                }
            }
            if has_remainder {
                let mut remainder_resource = std::mem::MaybeUninit::<CUdevResource>::uninit();
                remainder_resource.recv(channel_receiver).unwrap();
                unsafe {
                    std::ptr::write(remainder, remainder_resource.assume_init());
                }
            }
        }
    }
    'server_extra_recv: {
        let mut has_result = false;
        has_result.recv(channel_receiver).unwrap();
        let mut requested_groups = 0 as c_uint;
        requested_groups.recv(channel_receiver).unwrap();
        let mut input_resource = std::mem::MaybeUninit::<CUdevResource>::uninit();
        input_resource.recv(channel_receiver).unwrap();
        let input_resource = unsafe { input_resource.assume_init() };
        let mut has_remainder = false;
        has_remainder.recv(channel_receiver).unwrap();
    }
    'server_execution: {
        let mut returned_groups = requested_groups;
        let mut result_resources = if has_result {
            unsafe { vec![std::mem::zeroed::<CUdevResource>(); requested_groups as usize] }
        } else {
            Vec::new()
        };
        let mut remainder_resource = unsafe { std::mem::zeroed::<CUdevResource>() };
        let result_ptr = if has_result {
            result_resources.as_mut_ptr()
        } else {
            std::ptr::null_mut()
        };
        let remainder_ptr = if has_remainder {
            &raw mut remainder_resource
        } else {
            std::ptr::null_mut()
        };
        let result = unsafe {
            cuDevSmResourceSplitByCount(
                result_ptr,
                &mut returned_groups,
                &raw const input_resource,
                remainder_ptr,
                flags,
                minCount,
            )
        };
        if result == CUresult::CUDA_SUCCESS && has_result {
            assert!(returned_groups as usize <= result_resources.len());
        }
    }
    'server_after_send: {
        returned_groups.send(channel_sender).unwrap();
        if result == CUresult::CUDA_SUCCESS {
            if has_result {
                send_slice(
                    &result_resources[..returned_groups as usize],
                    channel_sender,
                )
                .unwrap();
            }
            if has_remainder {
                remainder_resource.send(channel_sender).unwrap();
            }
        }
        channel_sender.flush_out().unwrap();
    }
}

#[cuda_hook(proc_id = 900944)]
fn cuDevSmResourceSplit(
    #[skip] result_out: *mut CUdevResource,
    nbGroups: c_uint,
    #[skip] input: *const CUdevResource,
    #[skip] remainder: *mut CUdevResource,
    flags: c_uint,
    #[skip] groupParams: *mut CU_DEV_SM_RESOURCE_GROUP_PARAMS,
) -> CUresult {
    'client_before_send: {
        assert!(!input.is_null());
        assert!(!groupParams.is_null());
        let has_result = !result_out.is_null();
        let result_capacity = if has_result { nbGroups as usize } else { 0 };
        let input_resource = unsafe { *input };
        assert!(input_resource.nextResource.is_null());
        let has_remainder = !remainder.is_null();
        let group_params = unsafe { std::slice::from_raw_parts(groupParams, nbGroups as usize) };
    }
    'client_extra_send: {
        has_result.send(channel_sender).unwrap();
        input_resource.send(channel_sender).unwrap();
        has_remainder.send(channel_sender).unwrap();
        send_slice(group_params, channel_sender).unwrap();
    }
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            if has_result {
                let result_resources = recv_slice::<CUdevResource, _>(channel_receiver).unwrap();
                assert!(result_resources.len() <= result_capacity);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        result_resources.as_ptr(),
                        result_out,
                        result_resources.len(),
                    );
                }
            }
            let returned_group_params =
                recv_slice::<CU_DEV_SM_RESOURCE_GROUP_PARAMS, _>(channel_receiver).unwrap();
            assert_eq!(returned_group_params.len(), nbGroups as usize);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    returned_group_params.as_ptr(),
                    groupParams,
                    returned_group_params.len(),
                );
            }
            if has_remainder {
                let mut remainder_resource = std::mem::MaybeUninit::<CUdevResource>::uninit();
                remainder_resource.recv(channel_receiver).unwrap();
                unsafe {
                    std::ptr::write(remainder, remainder_resource.assume_init());
                }
            }
        }
    }
    'server_extra_recv: {
        let mut has_result = false;
        has_result.recv(channel_receiver).unwrap();
        let mut input_resource = std::mem::MaybeUninit::<CUdevResource>::uninit();
        input_resource.recv(channel_receiver).unwrap();
        let input_resource = unsafe { input_resource.assume_init() };
        let mut has_remainder = false;
        has_remainder.recv(channel_receiver).unwrap();
        let mut group_params =
            recv_slice::<CU_DEV_SM_RESOURCE_GROUP_PARAMS, _>(channel_receiver).unwrap();
    }
    'server_execution: {
        let mut result_resources = if has_result {
            unsafe { vec![std::mem::zeroed::<CUdevResource>(); nbGroups as usize] }
        } else {
            Vec::new()
        };
        let mut remainder_resource = unsafe { std::mem::zeroed::<CUdevResource>() };
        let result_ptr = if has_result {
            result_resources.as_mut_ptr()
        } else {
            std::ptr::null_mut()
        };
        let remainder_ptr = if has_remainder {
            &raw mut remainder_resource
        } else {
            std::ptr::null_mut()
        };
        let result = unsafe {
            cuDevSmResourceSplit(
                result_ptr,
                nbGroups,
                &raw const input_resource,
                remainder_ptr,
                flags,
                group_params.as_mut_ptr(),
            )
        };
    }
    'server_after_send: {
        if result == CUresult::CUDA_SUCCESS {
            if has_result {
                send_slice(&result_resources, channel_sender).unwrap();
            }
            send_slice(&group_params, channel_sender).unwrap();
            if has_remainder {
                remainder_resource.send(channel_sender).unwrap();
            }
        }
        channel_sender.flush_out().unwrap();
    }
}

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

#[cuda_hook(proc_id = 900931, async_api = false)]
fn cuDevicePrimaryCtxSetFlags_v2(dev: CUdevice, flags: c_uint) -> CUresult;

#[cuda_hook(proc_id = 900932, async_api = false)]
fn cuDevicePrimaryCtxReset_v2(dev: CUdevice) -> CUresult;

#[cuda_custom_hook(proc_id = 900923)]
fn cuCtxCreate_v4(
    pctx: *mut CUcontext,
    ctxCreateParams: *mut CUctxCreateParams,
    flags: c_uint,
    dev: CUdevice,
) -> CUresult;

#[cuda_hook(proc_id = 900924, async_api = false)]
fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;

#[cuda_hook(proc_id = 900925)]
fn cuCtxGetExecAffinity(
    #[host(output, len = 1)] pExecAffinity: *mut CUexecAffinityParam,
    type_: CUexecAffinityType,
) -> CUresult {
    'server_before_execution: {
        unsafe {
            *pExecAffinity__ptr = std::mem::zeroed();
        }
    }
}

#[cuda_hook(proc_id = 900934)]
fn cuGreenCtxCreate(
    phCtx: *mut CUgreenCtx,
    desc: CUdevResourceDesc,
    dev: CUdevice,
    flags: c_uint,
) -> CUresult;

#[cuda_hook(proc_id = 900936, async_api = false)]
fn cuGreenCtxDestroy(hCtx: CUgreenCtx) -> CUresult;

#[cuda_hook(proc_id = 900937)]
fn cuGreenCtxGetDevResource(
    hCtx: CUgreenCtx,
    #[host(output, len = 1)] resource: *mut CUdevResource,
    type_: CUdevResourceType,
) -> CUresult;

#[cuda_hook(proc_id = 900938)]
fn cuGreenCtxGetId(greenCtx: CUgreenCtx, greenCtxId: *mut c_ulonglong) -> CUresult;

#[cuda_hook(proc_id = 900947, async_api)]
fn cuGreenCtxRecordEvent(hCtx: CUgreenCtx, hEvent: CUevent) -> CUresult;

#[cuda_hook(proc_id = 900939)]
fn cuGreenCtxStreamCreate(
    phStream: *mut CUstream,
    greenCtx: CUgreenCtx,
    flags: c_uint,
    priority: c_int,
) -> CUresult;

#[cuda_hook(proc_id = 900948, async_api)]
fn cuGreenCtxWaitEvent(hCtx: CUgreenCtx, hEvent: CUevent) -> CUresult;

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

#[cuda_hook(proc_id = 901015)]
fn cuPointerSetAttribute(
    #[host(len = attribute.data_size())] value: *const c_void,
    attribute: CUpointer_attribute,
    ptr: CUdeviceptr,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901016)]
fn cuPointerGetAttributes(
    numAttributes: c_uint,
    attributes: *mut CUpointer_attribute,
    data: *mut *mut c_void,
    ptr: CUdeviceptr,
) -> CUresult;

#[cuda_hook(proc_id = 900201)]
fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;

#[cuda_hook(proc_id = 900983)]
fn cuMemAllocManaged(dptr: *mut CUdeviceptr, bytesize: usize, flags: c_uint) -> CUresult;

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

#[cuda_custom_hook] // local
fn cuMemAllocHost_v2(pp: *mut *mut c_void, bytesize: usize) -> CUresult;

#[cuda_custom_hook] // local
fn cuMemHostAlloc(pp: *mut *mut c_void, bytesize: usize, Flags: c_uint) -> CUresult;

#[cuda_custom_hook] // local
fn cuMemFreeHost(p: *mut c_void) -> CUresult;

#[cuda_custom_hook] // local
fn cuMemHostRegister_v2(p: *mut c_void, bytesize: usize, Flags: c_uint) -> CUresult;

#[cuda_custom_hook] // local
fn cuMemHostUnregister(p: *mut c_void) -> CUresult;

#[cuda_custom_hook] // local
fn cuMemHostGetFlags(pFlags: *mut c_uint, p: *mut c_void) -> CUresult;

#[cuda_custom_hook] // unsupported across the remoting boundary
fn cuMemHostGetDevicePointer_v2(
    pdptr: *mut CUdeviceptr,
    p: *mut c_void,
    Flags: c_uint,
) -> CUresult;

#[cuda_hook(proc_id = 900402, async_api = false)]
fn cuMemFreeAsync(dptr: CUdeviceptr, hStream: CUstream) -> CUresult;

#[cuda_hook(proc_id = 900403)]
fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;

#[cuda_hook(proc_id = 901038)]
fn cuMemAddressReserve(
    ptr: *mut CUdeviceptr,
    size: usize,
    alignment: usize,
    addr: CUdeviceptr,
    flags: c_ulonglong,
) -> CUresult;

#[cuda_hook(proc_id = 901039, async_api = false)]
fn cuMemAddressFree(ptr: CUdeviceptr, size: usize) -> CUresult;

#[cuda_hook(proc_id = 901040)]
fn cuMemCreate(
    handle: *mut CUmemGenericAllocationHandle,
    size: usize,
    #[host(len = 1)] prop: *const CUmemAllocationProp,
    flags: c_ulonglong,
) -> CUresult;

#[cuda_hook(proc_id = 901041, async_api = false)]
fn cuMemRelease(handle: CUmemGenericAllocationHandle) -> CUresult;

#[cuda_hook(proc_id = 901042)]
fn cuMemMap(
    ptr: CUdeviceptr,
    size: usize,
    offset: usize,
    handle: CUmemGenericAllocationHandle,
    flags: c_ulonglong,
) -> CUresult;

#[cuda_hook(proc_id = 901043, async_api = false)]
fn cuMemUnmap(ptr: CUdeviceptr, size: usize) -> CUresult;

#[cuda_hook(proc_id = 901044)]
fn cuMemSetAccess(
    ptr: CUdeviceptr,
    size: usize,
    #[host(len = count)] desc: *const CUmemAccessDesc,
    count: usize,
) -> CUresult;

#[cuda_hook(proc_id = 901045)]
fn cuMemGetAccess(
    flags: *mut c_ulonglong,
    #[host(len = 1)] location: *const CUmemLocation,
    ptr: CUdeviceptr,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901106)]
fn cuMemExportToShareableHandle(
    shareableHandle: *mut c_void,
    handle: CUmemGenericAllocationHandle,
    handleType: CUmemAllocationHandleType,
    flags: c_ulonglong,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901107)]
fn cuMemImportFromShareableHandle(
    handle: *mut CUmemGenericAllocationHandle,
    osHandle: *mut c_void,
    shHandleType: CUmemAllocationHandleType,
) -> CUresult;

#[cuda_hook(proc_id = 901046)]
fn cuMemGetAllocationGranularity(
    granularity: *mut usize,
    #[host(len = 1)] prop: *const CUmemAllocationProp,
    option: CUmemAllocationGranularity_flags,
) -> CUresult;

#[cuda_hook(proc_id = 901047)]
fn cuMemGetAllocationPropertiesFromHandle(
    #[host(output, len = 1)] prop: *mut CUmemAllocationProp,
    handle: CUmemGenericAllocationHandle,
) -> CUresult;

#[cuda_hook(proc_id = 901048)]
fn cuMemRetainAllocationHandle(
    handle: *mut CUmemGenericAllocationHandle,
    #[device] addr: *mut c_void,
) -> CUresult;

#[cuda_hook(proc_id = 901163)]
fn cuMulticastCreate(
    mcHandle: *mut CUmemGenericAllocationHandle,
    #[host(len = 1)] prop: *const CUmulticastObjectProp,
) -> CUresult;

#[cuda_hook(proc_id = 901164)]
fn cuMulticastAddDevice(mcHandle: CUmemGenericAllocationHandle, dev: CUdevice) -> CUresult;

#[cuda_hook(proc_id = 901165)]
fn cuMulticastBindMem(
    mcHandle: CUmemGenericAllocationHandle,
    mcOffset: usize,
    memHandle: CUmemGenericAllocationHandle,
    memOffset: usize,
    size: usize,
    flags: c_ulonglong,
) -> CUresult;

#[cuda_hook(proc_id = 901166)]
fn cuMulticastBindMem_v2(
    mcHandle: CUmemGenericAllocationHandle,
    dev: CUdevice,
    mcOffset: usize,
    memHandle: CUmemGenericAllocationHandle,
    memOffset: usize,
    size: usize,
    flags: c_ulonglong,
) -> CUresult;

#[cuda_hook(proc_id = 901167)]
fn cuMulticastBindAddr(
    mcHandle: CUmemGenericAllocationHandle,
    mcOffset: usize,
    memptr: CUdeviceptr,
    size: usize,
    flags: c_ulonglong,
) -> CUresult;

#[cuda_hook(proc_id = 901168)]
fn cuMulticastBindAddr_v2(
    mcHandle: CUmemGenericAllocationHandle,
    dev: CUdevice,
    mcOffset: usize,
    memptr: CUdeviceptr,
    size: usize,
    flags: c_ulonglong,
) -> CUresult;

#[cuda_hook(proc_id = 901169)]
fn cuMulticastUnbind(
    mcHandle: CUmemGenericAllocationHandle,
    dev: CUdevice,
    mcOffset: usize,
    size: usize,
) -> CUresult;

#[cuda_hook(proc_id = 901170)]
fn cuMulticastGetGranularity(
    granularity: *mut usize,
    #[host(len = 1)] prop: *const CUmulticastObjectProp,
    option: CUmulticastGranularity_flags,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901117)]
fn cuImportExternalMemory(
    extMem_out: *mut CUexternalMemory,
    memHandleDesc: *const CUDA_EXTERNAL_MEMORY_HANDLE_DESC,
) -> CUresult;

#[cuda_hook(proc_id = 901118)]
fn cuExternalMemoryGetMappedBuffer(
    devPtr: *mut CUdeviceptr,
    extMem: CUexternalMemory,
    #[host(len = 1)] bufferDesc: *const CUDA_EXTERNAL_MEMORY_BUFFER_DESC,
) -> CUresult;

#[cuda_hook(proc_id = 901119)]
fn cuExternalMemoryGetMappedMipmappedArray(
    mipmap: *mut CUmipmappedArray,
    extMem: CUexternalMemory,
    #[host(len = 1)] mipmapDesc: *const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC,
) -> CUresult;

#[cuda_hook(proc_id = 901120, async_api = false)]
fn cuDestroyExternalMemory(extMem: CUexternalMemory) -> CUresult;

#[cuda_custom_hook(proc_id = 901125)]
fn cuImportExternalSemaphore(
    extSem_out: *mut CUexternalSemaphore,
    semHandleDesc: *const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC,
) -> CUresult;

#[cuda_hook(proc_id = 901126)]
fn cuSignalExternalSemaphoresAsync(
    #[host(len = numExtSems as usize)] extSemArray: *const CUexternalSemaphore,
    #[host(len = numExtSems as usize)] paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS,
    numExtSems: c_uint,
    stream: CUstream,
) -> CUresult;

#[cuda_hook(proc_id = 901127)]
fn cuWaitExternalSemaphoresAsync(
    #[host(len = numExtSems as usize)] extSemArray: *const CUexternalSemaphore,
    #[host(len = numExtSems as usize)] paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS,
    numExtSems: c_uint,
    stream: CUstream,
) -> CUresult;

#[cuda_hook(proc_id = 901128, async_api = false)]
fn cuDestroyExternalSemaphore(extSem: CUexternalSemaphore) -> CUresult;

#[cuda_hook(proc_id = 901049)]
fn cuMemGetAddressRange_v2(
    pbase: *mut CUdeviceptr,
    psize: *mut usize,
    dptr: CUdeviceptr,
) -> CUresult;

#[cuda_hook(proc_id = 901050)]
fn cuIpcGetMemHandle(pHandle: *mut CUipcMemHandle, dptr: CUdeviceptr) -> CUresult;

#[cuda_hook(proc_id = 901051)]
fn cuIpcOpenMemHandle_v2(
    pdptr: *mut CUdeviceptr,
    handle: CUipcMemHandle,
    Flags: c_uint,
) -> CUresult;

#[cuda_hook(proc_id = 901052, async_api = false)]
fn cuIpcCloseMemHandle(dptr: CUdeviceptr) -> CUresult;

#[cuda_custom_hook(proc_id = 900984)]
fn cuMemPrefetchAsync_v2(
    devPtr: CUdeviceptr,
    count: usize,
    location: CUmemLocation,
    flags: c_uint,
    hStream: CUstream,
) -> CUresult;

#[cuda_custom_hook(proc_id = 900985)]
fn cuMemAdvise_v2(
    devPtr: CUdeviceptr,
    count: usize,
    advice: CUmem_advise,
    location: CUmemLocation,
) -> CUresult;

#[cuda_hook(proc_id = 900986)]
fn cuMemRangeGetAttribute(
    #[host(output, len = dataSize)] data: *mut c_void,
    dataSize: usize,
    attribute: CUmem_range_attribute,
    devPtr: CUdeviceptr,
    count: usize,
) -> CUresult;

#[cuda_custom_hook(proc_id = 900987)]
fn cuMemRangeGetAttributes(
    data: *mut *mut c_void,
    dataSizes: *mut usize,
    attributes: *mut CUmem_range_attribute,
    numAttributes: usize,
    devPtr: CUdeviceptr,
    count: usize,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901105)]
fn cuMemBatchDecompressAsync(
    paramsArray: *mut CUmemDecompressParams,
    count: usize,
    flags: c_uint,
    errorIndex: *mut usize,
    stream: CUstream,
) -> CUresult;

#[cuda_hook(proc_id = 901009, async_api)]
fn cuMemPrefetchBatchAsync(
    #[host(input, len = count)] dptrs: *mut CUdeviceptr,
    #[host(input, len = count)] sizes: *mut usize,
    count: usize,
    #[host(input, len = numPrefetchLocs)] prefetchLocs: *mut CUmemLocation,
    #[host(input, len = numPrefetchLocs)] prefetchLocIdxs: *mut usize,
    numPrefetchLocs: usize,
    flags: c_ulonglong,
    hStream: CUstream,
) -> CUresult {
    'client_before_send: {
        assert!(count > 0);
        assert!(!dptrs.is_null());
        assert!(!sizes.is_null());
        assert!(numPrefetchLocs > 0);
        assert!(numPrefetchLocs <= count);
        assert!(!prefetchLocs.is_null());
        assert!(!prefetchLocIdxs.is_null());
        assert_eq!(flags, 0);
        assert!(!hStream.is_null());
        let dptrs_slice = unsafe { std::slice::from_raw_parts(dptrs, count) };
        let sizes_slice = unsafe { std::slice::from_raw_parts(sizes, count) };
        let loc_idxs = unsafe { std::slice::from_raw_parts(prefetchLocIdxs, numPrefetchLocs) };
        assert!(dptrs_slice.iter().all(|dptr| *dptr != 0));
        assert!(sizes_slice.iter().all(|size| *size > 0));
        assert_eq!(loc_idxs[0], 0);
        assert!(loc_idxs.iter().all(|idx| *idx < count));
        assert!(loc_idxs.windows(2).all(|idxs| idxs[0] < idxs[1]));
    }
}

#[cuda_hook(proc_id = 901010, async_api)]
fn cuMemDiscardBatchAsync(
    #[host(input, len = count)] dptrs: *mut CUdeviceptr,
    #[host(input, len = count)] sizes: *mut usize,
    count: usize,
    flags: c_ulonglong,
    hStream: CUstream,
) -> CUresult {
    'client_before_send: {
        assert!(count > 0);
        assert!(!dptrs.is_null());
        assert!(!sizes.is_null());
        assert_eq!(flags, 0);
        assert!(!hStream.is_null());
        let dptrs_slice = unsafe { std::slice::from_raw_parts(dptrs, count) };
        let sizes_slice = unsafe { std::slice::from_raw_parts(sizes, count) };
        assert!(dptrs_slice.iter().all(|dptr| *dptr != 0));
        assert!(sizes_slice.iter().all(|size| *size > 0));
    }
}

#[cuda_hook(proc_id = 901011, async_api)]
fn cuMemDiscardAndPrefetchBatchAsync(
    #[host(input, len = count)] dptrs: *mut CUdeviceptr,
    #[host(input, len = count)] sizes: *mut usize,
    count: usize,
    #[host(input, len = numPrefetchLocs)] prefetchLocs: *mut CUmemLocation,
    #[host(input, len = numPrefetchLocs)] prefetchLocIdxs: *mut usize,
    numPrefetchLocs: usize,
    flags: c_ulonglong,
    hStream: CUstream,
) -> CUresult {
    'client_before_send: {
        assert!(count > 0);
        assert!(!dptrs.is_null());
        assert!(!sizes.is_null());
        assert!(numPrefetchLocs > 0);
        assert!(numPrefetchLocs <= count);
        assert!(!prefetchLocs.is_null());
        assert!(!prefetchLocIdxs.is_null());
        assert_eq!(flags, 0);
        assert!(!hStream.is_null());
        let dptrs_slice = unsafe { std::slice::from_raw_parts(dptrs, count) };
        let sizes_slice = unsafe { std::slice::from_raw_parts(sizes, count) };
        let loc_idxs = unsafe { std::slice::from_raw_parts(prefetchLocIdxs, numPrefetchLocs) };
        assert!(dptrs_slice.iter().all(|dptr| *dptr != 0));
        assert!(sizes_slice.iter().all(|size| *size > 0));
        assert_eq!(loc_idxs[0], 0);
        assert!(loc_idxs.iter().all(|idx| *idx < count));
        assert!(loc_idxs.windows(2).all(|idxs| idxs[0] < idxs[1]));
    }
}

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

#[cuda_custom_hook(proc_id = 901002)]
fn cuMemPoolCreate(pool: *mut CUmemoryPool, poolProps: *const CUmemPoolProps) -> CUresult;

#[cuda_hook(proc_id = 901003, async_api = false)]
fn cuMemPoolDestroy(pool: CUmemoryPool) -> CUresult;

#[cuda_custom_hook(proc_id = 901004)]
fn cuMemGetDefaultMemPool(
    pool_out: *mut CUmemoryPool,
    location: *mut CUmemLocation,
    type_: CUmemAllocationType,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901005)]
fn cuMemGetMemPool(
    pool: *mut CUmemoryPool,
    location: *mut CUmemLocation,
    type_: CUmemAllocationType,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901006)]
fn cuMemSetMemPool(
    location: *mut CUmemLocation,
    type_: CUmemAllocationType,
    pool: CUmemoryPool,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901007)]
fn cuMemPoolSetAccess(pool: CUmemoryPool, map: *const CUmemAccessDesc, count: usize) -> CUresult;

#[cuda_custom_hook(proc_id = 901008)]
fn cuMemPoolGetAccess(
    flags: *mut CUmemAccess_flags,
    memPool: CUmemoryPool,
    location: *mut CUmemLocation,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901108)]
fn cuMemPoolExportToShareableHandle(
    handle_out: *mut c_void,
    pool: CUmemoryPool,
    handleType: CUmemAllocationHandleType,
    flags: c_ulonglong,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901109)]
fn cuMemPoolImportFromShareableHandle(
    pool_out: *mut CUmemoryPool,
    handle: *mut c_void,
    handleType: CUmemAllocationHandleType,
    flags: c_ulonglong,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901110)]
fn cuMemPoolExportPointer(
    shareData_out: *mut CUmemPoolPtrExportData,
    ptr: CUdeviceptr,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901111)]
fn cuMemPoolImportPointer(
    ptr_out: *mut CUdeviceptr,
    pool: CUmemoryPool,
    shareData: *mut CUmemPoolPtrExportData,
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

#[cuda_hook(proc_id = 900949, async_api)]
fn cuMemcpyAtoA_v2(
    dstArray: CUarray,
    dstOffset: usize,
    srcArray: CUarray,
    srcOffset: usize,
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

#[cuda_custom_hook(proc_id = 900950)]
fn cuMemcpy2D_v2(pCopy: *const CUDA_MEMCPY2D) -> CUresult;

#[cuda_custom_hook(proc_id = 900951)]
fn cuMemcpy2DUnaligned_v2(pCopy: *const CUDA_MEMCPY2D) -> CUresult;

#[cuda_custom_hook(proc_id = 900952)]
fn cuMemcpy2DAsync_v2(pCopy: *const CUDA_MEMCPY2D, hStream: CUstream) -> CUresult;

#[cuda_hook(proc_id = 900953, async_api)]
fn cuMemcpy3D_v2(#[host(len = 1)] pCopy: *const CUDA_MEMCPY3D) -> CUresult {
    'client_before_send: {
        assert!(!pCopy.is_null());
        let params = unsafe { &*pCopy };
        assert_eq!(params.srcMemoryType, CUmemorytype::CU_MEMORYTYPE_DEVICE);
        assert_eq!(params.dstMemoryType, CUmemorytype::CU_MEMORYTYPE_DEVICE);
        assert_ne!(params.srcDevice, 0);
        assert_ne!(params.dstDevice, 0);
        assert!(params.srcArray.is_null());
        assert!(params.dstArray.is_null());
        assert!(params.reserved0.is_null());
        assert!(params.reserved1.is_null());
        assert_eq!(params.srcLOD, 0);
        assert_eq!(params.dstLOD, 0);
    }
}

#[cuda_hook(proc_id = 900954, async_api)]
fn cuMemcpy3DAsync_v2(#[host(len = 1)] pCopy: *const CUDA_MEMCPY3D, hStream: CUstream) -> CUresult {
    'client_before_send: {
        assert!(!pCopy.is_null());
        let params = unsafe { &*pCopy };
        assert_eq!(params.srcMemoryType, CUmemorytype::CU_MEMORYTYPE_DEVICE);
        assert_eq!(params.dstMemoryType, CUmemorytype::CU_MEMORYTYPE_DEVICE);
        assert_ne!(params.srcDevice, 0);
        assert_ne!(params.dstDevice, 0);
        assert!(params.srcArray.is_null());
        assert!(params.dstArray.is_null());
        assert!(params.reserved0.is_null());
        assert!(params.reserved1.is_null());
        assert_eq!(params.srcLOD, 0);
        assert_eq!(params.dstLOD, 0);
    }
}

#[cuda_hook(proc_id = 900955, async_api)]
fn cuMemcpyPeer(
    dstDevice: CUdeviceptr,
    dstContext: CUcontext,
    srcDevice: CUdeviceptr,
    srcContext: CUcontext,
    ByteCount: usize,
) -> CUresult;

#[cuda_hook(proc_id = 900956, async_api)]
fn cuMemcpyPeerAsync(
    dstDevice: CUdeviceptr,
    dstContext: CUcontext,
    srcDevice: CUdeviceptr,
    srcContext: CUcontext,
    ByteCount: usize,
    hStream: CUstream,
) -> CUresult;

#[cuda_hook(proc_id = 900957, async_api)]
fn cuMemcpy3DPeer(#[host(len = 1)] pCopy: *const CUDA_MEMCPY3D_PEER) -> CUresult {
    'client_before_send: {
        assert!(!pCopy.is_null());
        let params = unsafe { &*pCopy };
        assert_eq!(params.srcMemoryType, CUmemorytype::CU_MEMORYTYPE_DEVICE);
        assert_eq!(params.dstMemoryType, CUmemorytype::CU_MEMORYTYPE_DEVICE);
        assert_ne!(params.srcDevice, 0);
        assert_ne!(params.dstDevice, 0);
        assert!(!params.srcContext.is_null());
        assert!(!params.dstContext.is_null());
        assert!(params.srcArray.is_null());
        assert!(params.dstArray.is_null());
        assert_eq!(params.srcLOD, 0);
        assert_eq!(params.dstLOD, 0);
    }
}

#[cuda_hook(proc_id = 900958, async_api)]
fn cuMemcpy3DPeerAsync(
    #[host(len = 1)] pCopy: *const CUDA_MEMCPY3D_PEER,
    hStream: CUstream,
) -> CUresult {
    'client_before_send: {
        assert!(!pCopy.is_null());
        let params = unsafe { &*pCopy };
        assert_eq!(params.srcMemoryType, CUmemorytype::CU_MEMORYTYPE_DEVICE);
        assert_eq!(params.dstMemoryType, CUmemorytype::CU_MEMORYTYPE_DEVICE);
        assert_ne!(params.srcDevice, 0);
        assert_ne!(params.dstDevice, 0);
        assert!(!params.srcContext.is_null());
        assert!(!params.dstContext.is_null());
        assert!(params.srcArray.is_null());
        assert!(params.dstArray.is_null());
        assert_eq!(params.srcLOD, 0);
        assert_eq!(params.dstLOD, 0);
    }
}

#[cuda_hook(proc_id = 900961, async_api)]
fn cuMemcpyBatchAsync_v2(
    #[host(input, len = count)] dsts: *mut CUdeviceptr,
    #[host(input, len = count)] srcs: *mut CUdeviceptr,
    #[host(input, len = count)] sizes: *mut usize,
    count: usize,
    #[host(input, len = numAttrs)] attrs: *mut CUmemcpyAttributes,
    #[host(input, len = numAttrs)] attrsIdxs: *mut usize,
    numAttrs: usize,
    hStream: CUstream,
) -> CUresult {
    'client_before_send: {
        assert!(count > 0);
        assert!(!dsts.is_null());
        assert!(!srcs.is_null());
        assert!(!sizes.is_null());
        assert!(numAttrs > 0);
        assert!(numAttrs <= count);
        assert!(!attrs.is_null());
        assert!(!attrsIdxs.is_null());
        let dsts_slice = unsafe { std::slice::from_raw_parts(dsts, count) };
        let srcs_slice = unsafe { std::slice::from_raw_parts(srcs, count) };
        let sizes_slice = unsafe { std::slice::from_raw_parts(sizes, count) };
        assert!(dsts_slice.iter().all(|dst| *dst != 0));
        assert!(srcs_slice.iter().all(|src| *src != 0));
        assert!(sizes_slice.iter().all(|size| *size > 0));
        let attrs_idx_slice = unsafe { std::slice::from_raw_parts(attrsIdxs, numAttrs) };
        assert_eq!(attrs_idx_slice[0], 0);
        assert!(attrs_idx_slice.iter().all(|idx| *idx < count));
        assert!(attrs_idx_slice.windows(2).all(|idxs| idxs[0] < idxs[1]));
    }
}

#[cuda_hook(proc_id = 900962, async_api)]
fn cuMemcpyWithAttributesAsync(
    dst: CUdeviceptr,
    src: CUdeviceptr,
    size: usize,
    #[host(input, len = 1)] attr: *mut CUmemcpyAttributes,
    hStream: CUstream,
) -> CUresult {
    'client_before_send: {
        assert_ne!(dst, 0);
        assert_ne!(src, 0);
        assert!(size > 0);
        assert!(!attr.is_null());
    }
}

#[cuda_hook(proc_id = 900963, async_api)]
fn cuMemcpy3DBatchAsync_v2(
    numOps: usize,
    #[host(input, len = numOps)] opList: *mut CUDA_MEMCPY3D_BATCH_OP,
    flags: c_ulonglong,
    hStream: CUstream,
) -> CUresult {
    'client_before_send: {
        assert!(numOps > 0);
        assert!(!opList.is_null());
        assert_eq!(flags, 0);
        let ops = unsafe { std::slice::from_raw_parts(opList, numOps) };
        for op in ops {
            assert_eq!(
                op.src.type_,
                CUmemcpy3DOperandType::CU_MEMCPY_OPERAND_TYPE_POINTER
            );
            assert_eq!(
                op.dst.type_,
                CUmemcpy3DOperandType::CU_MEMCPY_OPERAND_TYPE_POINTER
            );
            let src = unsafe { op.src.op.ptr };
            let dst = unsafe { op.dst.op.ptr };
            assert_ne!(src.ptr, 0);
            assert_ne!(dst.ptr, 0);
            assert!(op.extent.width > 0);
            assert!(op.extent.height > 0);
            assert!(op.extent.depth > 0);
            assert!(src.rowLength == 0 || src.rowLength >= op.extent.width);
            assert!(dst.rowLength == 0 || dst.rowLength >= op.extent.width);
            assert!(src.layerHeight == 0 || src.layerHeight >= op.extent.height);
            assert!(dst.layerHeight == 0 || dst.layerHeight >= op.extent.height);
        }
    }
}

#[cuda_hook(proc_id = 900964, async_api)]
fn cuMemcpy3DWithAttributesAsync(
    #[host(input, len = 1)] op: *mut CUDA_MEMCPY3D_BATCH_OP,
    flags: c_ulonglong,
    hStream: CUstream,
) -> CUresult {
    'client_before_send: {
        assert!(!op.is_null());
        assert_eq!(flags, 0);
        let op_ref = unsafe { &*op };
        assert_eq!(
            op_ref.src.type_,
            CUmemcpy3DOperandType::CU_MEMCPY_OPERAND_TYPE_POINTER
        );
        assert_eq!(
            op_ref.dst.type_,
            CUmemcpy3DOperandType::CU_MEMCPY_OPERAND_TYPE_POINTER
        );
        let src = unsafe { op_ref.src.op.ptr };
        let dst = unsafe { op_ref.dst.op.ptr };
        assert_ne!(src.ptr, 0);
        assert_ne!(dst.ptr, 0);
        assert!(op_ref.extent.width > 0);
        assert!(op_ref.extent.height > 0);
        assert!(op_ref.extent.depth > 0);
        assert!(src.rowLength == 0 || src.rowLength >= op_ref.extent.width);
        assert!(dst.rowLength == 0 || dst.rowLength >= op_ref.extent.width);
        assert!(src.layerHeight == 0 || src.layerHeight >= op_ref.extent.height);
        assert!(dst.layerHeight == 0 || dst.layerHeight >= op_ref.extent.height);
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

#[cuda_hook(proc_id = 900959, async_api)]
fn cuMemsetD16_v2(dstDevice: CUdeviceptr, us: c_ushort, N: usize) -> CUresult;

#[cuda_hook(proc_id = 900207, async_api)]
fn cuMemsetD32_v2(dstDevice: CUdeviceptr, ui: c_uint, N: usize) -> CUresult;

#[cuda_hook(proc_id = 900409, async_api)]
fn cuMemsetD8Async(dstDevice: CUdeviceptr, uc: c_uchar, N: usize, hStream: CUstream) -> CUresult;

#[cuda_hook(proc_id = 900960, async_api)]
fn cuMemsetD16Async(dstDevice: CUdeviceptr, us: c_ushort, N: usize, hStream: CUstream) -> CUresult;

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

#[cuda_hook(proc_id = 900988, async_api)]
fn cuStreamAttachMemAsync(
    hStream: CUstream,
    dptr: CUdeviceptr,
    length: usize,
    flags: c_uint,
) -> CUresult;

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

#[cuda_hook(proc_id = 900883)]
fn cuStreamGetCtx_v2(
    hStream: CUstream,
    pCtx: *mut CUcontext,
    pGreenCtx: *mut CUgreenCtx,
) -> CUresult;

#[cuda_hook(proc_id = 900891)]
fn cuStreamGetGreenCtx(hStream: CUstream, phCtx: *mut CUgreenCtx) -> CUresult;

#[cuda_hook(proc_id = 900892)]
fn cuStreamGetDevResource(
    hStream: CUstream,
    #[host(output, len = 1)] resource: *mut CUdevResource,
    type_: CUdevResourceType,
) -> CUresult;

#[cuda_hook(proc_id = 900418, async_api = false)]
fn cuStreamWaitEvent(hStream: CUstream, hEvent: CUevent, Flags: c_uint) -> CUresult;

#[cuda_custom_hook]
fn cuStreamAddCallback(
    hStream: CUstream,
    callback: CUstreamCallback,
    userData: *mut c_void,
    flags: c_uint,
) -> CUresult;

#[cuda_custom_hook]
fn cuLaunchHostFunc(hStream: CUstream, fn_: CUhostFn, userData: *mut c_void) -> CUresult;

#[cuda_custom_hook]
fn cuLaunchHostFunc_v2(
    hStream: CUstream,
    fn_: CUhostFn,
    userData: *mut c_void,
    syncMode: c_uint,
) -> CUresult;

#[cuda_hook(proc_id = 900432)]
fn cuStreamCopyAttributes(dst: CUstream, src: CUstream) -> CUresult;

#[cuda_hook(proc_id = 900884)]
fn cuStreamGetAttribute(
    hStream: CUstream,
    attr: CUstreamAttrID,
    #[host(output, len = 1)] value_out: *mut CUstreamAttrValue,
) -> CUresult;

#[cuda_hook(proc_id = 900885)]
fn cuStreamSetAttribute(
    hStream: CUstream,
    attr: CUstreamAttrID,
    #[host(len = 1)] value: *const CUstreamAttrValue,
) -> CUresult;

#[cuda_hook(proc_id = 900876, async_api)]
fn cuStreamWaitValue32_v2(
    stream: CUstream,
    addr: CUdeviceptr,
    value: cuuint32_t,
    flags: c_uint,
) -> CUresult;

#[cuda_hook(proc_id = 900877, async_api)]
fn cuStreamWriteValue32_v2(
    stream: CUstream,
    addr: CUdeviceptr,
    value: cuuint32_t,
    flags: c_uint,
) -> CUresult;

#[cuda_hook(proc_id = 900878, async_api)]
fn cuStreamWaitValue64_v2(
    stream: CUstream,
    addr: CUdeviceptr,
    value: cuuint64_t,
    flags: c_uint,
) -> CUresult;

#[cuda_hook(proc_id = 900879, async_api)]
fn cuStreamWriteValue64_v2(
    stream: CUstream,
    addr: CUdeviceptr,
    value: cuuint64_t,
    flags: c_uint,
) -> CUresult;

#[cuda_hook(proc_id = 901099, async_api)]
fn cuStreamBatchMemOp_v2(
    stream: CUstream,
    count: c_uint,
    #[skip] paramArray: *mut CUstreamBatchMemOpParams,
    flags: c_uint,
) -> CUresult {
    'client_before_send: {
        if count > 0 && paramArray.is_null() {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        }
        let ops = if count == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(paramArray, count as usize) }
        };
    }
    'client_extra_send: {
        send_slice(ops, channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut ops = recv_slice::<CUstreamBatchMemOpParams, _>(channel_receiver).unwrap();
        let ops_ptr = if ops.is_empty() {
            std::ptr::null_mut()
        } else {
            ops.as_mut_ptr()
        };
    }
    'server_execution: {
        let result = unsafe { cuStreamBatchMemOp_v2(stream, count, ops_ptr, flags) };
    }
}

#[cuda_hook(proc_id = 900866, async_api = false)]
fn cuStreamBeginCapture_v2(hStream: CUstream, mode: CUstreamCaptureMode) -> CUresult;

#[cuda_hook(proc_id = 900886)]
fn cuThreadExchangeStreamCaptureMode(mode: *mut CUstreamCaptureMode) -> CUresult {
    'client_extra_send: {
        unsafe { *mode }.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut mode_in = CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL;
        mode_in.recv(channel_receiver).unwrap();
    }
    'server_before_execution: {
        mode.write(mode_in);
    }
}

#[cuda_hook(proc_id = 900867)]
fn cuStreamBeginCaptureToGraph(
    hStream: CUstream,
    hGraph: CUgraph,
    #[host(len = numDependencies)] dependencies: *const CUgraphNode,
    #[device] dependencyData: *const CUgraphEdgeData,
    numDependencies: usize,
    mode: CUstreamCaptureMode,
) -> CUresult {
    'client_before_send: {
        assert!(numDependencies > 0);
        assert!(dependencyData.is_null());
    }
}

#[cuda_hook(proc_id = 900868)]
fn cuStreamEndCapture(hStream: CUstream, phGraph: *mut CUgraph) -> CUresult;

#[cuda_hook(proc_id = 900869)]
fn cuStreamIsCapturing(hStream: CUstream, captureStatus: *mut CUstreamCaptureStatus) -> CUresult;

#[cuda_hook(proc_id = 900870)]
fn cuStreamGetCaptureInfo_v3(
    hStream: CUstream,
    captureStatus_out: *mut CUstreamCaptureStatus,
    id_out: *mut cuuint64_t,
    #[device] graph_out: *mut CUgraph,
    #[device] dependencies_out: *mut *const CUgraphNode,
    #[device] edgeData_out: *mut *const CUgraphEdgeData,
    #[device] numDependencies_out: *mut usize,
) -> CUresult {
    'client_before_send: {
        assert!(graph_out.is_null());
        assert!(dependencies_out.is_null());
        assert!(edgeData_out.is_null());
        assert!(numDependencies_out.is_null());
    }
}

#[cuda_hook(proc_id = 900871)]
fn cuStreamUpdateCaptureDependencies_v2(
    hStream: CUstream,
    #[host(input, len = numDependencies)] dependencies: *mut CUgraphNode,
    #[device] dependencyData: *const CUgraphEdgeData,
    numDependencies: usize,
    flags: c_uint,
) -> CUresult {
    'client_before_send: {
        assert!(numDependencies > 0);
        assert!(dependencyData.is_null());
    }
}

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

#[cuda_custom_hook(proc_id = 900214)]
fn cuEventDestroy_v2(hEvent: CUevent) -> CUresult;

#[cuda_hook(proc_id = 900428)]
fn cuEventElapsedTime_v2(pMilliseconds: *mut f32, hStart: CUevent, hEnd: CUevent) -> CUresult;

#[cuda_custom_hook(proc_id = 900916)]
fn cuIpcGetEventHandle(pHandle: *mut CUipcEventHandle, event: CUevent) -> CUresult;

#[cuda_custom_hook(proc_id = 900917)]
fn cuIpcOpenEventHandle(phEvent: *mut CUevent, handle: CUipcEventHandle) -> CUresult;

#[cuda_hook(proc_id = 1002)]
fn cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    numBlocks: *mut c_int,
    func: CUfunction,
    blockSize: c_int,
    dynamicSMemSize: usize,
    flags: c_uint,
) -> CUresult;

#[cuda_hook(proc_id = 901017)]
fn cuOccupancyMaxPotentialBlockSize(
    minGridSize: *mut c_int,
    blockSize: *mut c_int,
    func: CUfunction,
    #[skip] blockSizeToDynamicSMemSize: CUoccupancyB2DSize,
    dynamicSMemSize: usize,
    blockSizeLimit: c_int,
) -> CUresult {
    'client_before_send: {
        if blockSizeToDynamicSMemSize.is_some() {
            return CUresult::CUDA_ERROR_NOT_SUPPORTED;
        }
    }
    'server_execution: {
        let result = unsafe {
            cuOccupancyMaxPotentialBlockSize(
                minGridSize__ptr,
                blockSize__ptr,
                func,
                None,
                dynamicSMemSize,
                blockSizeLimit,
            )
        };
    }
}

#[cuda_hook(proc_id = 901018)]
fn cuOccupancyMaxPotentialBlockSizeWithFlags(
    minGridSize: *mut c_int,
    blockSize: *mut c_int,
    func: CUfunction,
    #[skip] blockSizeToDynamicSMemSize: CUoccupancyB2DSize,
    dynamicSMemSize: usize,
    blockSizeLimit: c_int,
    flags: c_uint,
) -> CUresult {
    'client_before_send: {
        if blockSizeToDynamicSMemSize.is_some() {
            return CUresult::CUDA_ERROR_NOT_SUPPORTED;
        }
    }
    'server_execution: {
        let result = unsafe {
            cuOccupancyMaxPotentialBlockSizeWithFlags(
                minGridSize__ptr,
                blockSize__ptr,
                func,
                None,
                dynamicSMemSize,
                blockSizeLimit,
                flags,
            )
        };
    }
}

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

#[cuda_hook(proc_id = 901019)]
fn cuOccupancyMaxPotentialClusterSize(
    clusterSize: *mut c_int,
    func: CUfunction,
    #[host] config: *const CUlaunchConfig,
) -> CUresult {
    'client_before_send: {
        if config.is_null() {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        }
        let config_ref = unsafe { &*config };
        let attrs = if config_ref.numAttrs == 0 {
            &[][..]
        } else {
            if config_ref.attrs.is_null() {
                return CUresult::CUDA_ERROR_INVALID_VALUE;
            }
            unsafe { std::slice::from_raw_parts(config_ref.attrs, config_ref.numAttrs as usize) }
        };
    }
    'client_extra_send: {
        send_slice(attrs, channel_sender).unwrap();
    }
    'server_extra_recv: {
        let _ = config__ptr;
        let attrs = recv_slice::<CUlaunchAttribute, _>(channel_receiver).unwrap();
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
            unsafe { cuOccupancyMaxPotentialClusterSize(clusterSize__ptr, func, &launch_config) };
    }
}

#[cuda_hook(proc_id = 901020)]
fn cuOccupancyMaxActiveClusters(
    numClusters: *mut c_int,
    func: CUfunction,
    #[host] config: *const CUlaunchConfig,
) -> CUresult {
    'client_before_send: {
        if config.is_null() {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        }
        let config_ref = unsafe { &*config };
        let attrs = if config_ref.numAttrs == 0 {
            &[][..]
        } else {
            if config_ref.attrs.is_null() {
                return CUresult::CUDA_ERROR_INVALID_VALUE;
            }
            unsafe { std::slice::from_raw_parts(config_ref.attrs, config_ref.numAttrs as usize) }
        };
    }
    'client_extra_send: {
        send_slice(attrs, channel_sender).unwrap();
    }
    'server_extra_recv: {
        let _ = config__ptr;
        let attrs = recv_slice::<CUlaunchAttribute, _>(channel_receiver).unwrap();
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
            unsafe { cuOccupancyMaxActiveClusters(numClusters__ptr, func, &launch_config) };
    }
}

#[cuda_hook(proc_id = 900800)]
fn cuGraphCreate(phGraph: *mut CUgraph, flags: c_uint) -> CUresult;

#[cuda_hook(proc_id = 900801, async_api = false)]
fn cuGraphDestroy(hGraph: CUgraph) -> CUresult {
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            super::user_object::graph_destroy(hGraph);
        }
    }
}

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
fn cuGraphClone(phGraphClone: *mut CUgraph, originalGraph: CUgraph) -> CUresult {
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            super::user_object::graph_clone(originalGraph, *phGraphClone);
        }
    }
}

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

#[cuda_custom_hook(proc_id = 900818)]
fn cuGraphInstantiateWithFlags(
    phGraphExec: *mut CUgraphExec,
    hGraph: CUgraph,
    flags: c_ulonglong,
) -> CUresult;

#[cuda_custom_hook(proc_id = 900819)]
fn cuGraphInstantiateWithParams(
    phGraphExec: *mut CUgraphExec,
    hGraph: CUgraph,
    instantiateParams: *mut CUDA_GRAPH_INSTANTIATE_PARAMS,
) -> CUresult;

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

#[cuda_custom_hook(proc_id = 900824)]
fn cuGraphLaunch(hGraphExec: CUgraphExec, hStream: CUstream) -> CUresult;

#[cuda_hook(proc_id = 900825, async_api = false)]
fn cuGraphExecDestroy(hGraphExec: CUgraphExec) -> CUresult;

#[cuda_hook(proc_id = 900826)]
fn cuGraphDebugDotPrint(hGraph: CUgraph, path: *const c_char, flags: c_uint) -> CUresult;

#[cuda_custom_hook] // local
fn cuUserObjectCreate(
    object_out: *mut CUuserObject,
    ptr: *mut c_void,
    destroy: CUhostFn,
    initialRefcount: c_uint,
    flags: c_uint,
) -> CUresult;

#[cuda_custom_hook] // local
fn cuUserObjectRetain(object: CUuserObject, count: c_uint) -> CUresult;

#[cuda_custom_hook] // local
fn cuUserObjectRelease(object: CUuserObject, count: c_uint) -> CUresult;

#[cuda_custom_hook] // local
fn cuGraphRetainUserObject(
    graph: CUgraph,
    object: CUuserObject,
    count: c_uint,
    flags: c_uint,
) -> CUresult;

#[cuda_custom_hook] // local
fn cuGraphReleaseUserObject(graph: CUgraph, object: CUuserObject, count: c_uint) -> CUresult;

#[cuda_hook(proc_id = 900827, async_api = false)]
fn cuDeviceGraphMemTrim(device: CUdevice) -> CUresult;

#[cuda_hook(proc_id = 900828)]
fn cuDeviceGetGraphMemAttribute(
    device: CUdevice,
    attr: CUgraphMem_attribute,
    #[host(output, len = attr.data_size())] value: *mut c_void,
) -> CUresult;

#[cuda_hook(proc_id = 900829)]
fn cuDeviceSetGraphMemAttribute(
    device: CUdevice,
    attr: CUgraphMem_attribute,
    #[host(input, len = attr.data_size())] value: *mut c_void,
) -> CUresult;

#[cuda_hook(proc_id = 900830)]
fn cuGraphAddChildGraphNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    #[device] dependencies: *const CUgraphNode,
    numDependencies: usize,
    childGraph: CUgraph,
) -> CUresult {
    'client_before_send: {
        assert!(dependencies.is_null());
        assert_eq!(numDependencies, 0);
    }
}

#[cuda_hook(proc_id = 900831)]
fn cuGraphChildGraphNodeGetGraph(hNode: CUgraphNode, phGraph: *mut CUgraph) -> CUresult;

#[cuda_hook(proc_id = 900832)]
fn cuGraphExecChildGraphNodeSetParams(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    childGraph: CUgraph,
) -> CUresult;

#[cuda_hook(proc_id = 900833)]
fn cuGraphAddEventRecordNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    #[device] dependencies: *const CUgraphNode,
    numDependencies: usize,
    event: CUevent,
) -> CUresult {
    'client_before_send: {
        assert!(dependencies.is_null());
        assert_eq!(numDependencies, 0);
    }
}

#[cuda_hook(proc_id = 900834)]
fn cuGraphEventRecordNodeGetEvent(hNode: CUgraphNode, event_out: *mut CUevent) -> CUresult;

#[cuda_hook(proc_id = 900835)]
fn cuGraphEventRecordNodeSetEvent(hNode: CUgraphNode, event: CUevent) -> CUresult;

#[cuda_hook(proc_id = 900836)]
fn cuGraphAddEventWaitNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    #[device] dependencies: *const CUgraphNode,
    numDependencies: usize,
    event: CUevent,
) -> CUresult {
    'client_before_send: {
        assert!(dependencies.is_null());
        assert_eq!(numDependencies, 0);
    }
}

#[cuda_hook(proc_id = 900837)]
fn cuGraphEventWaitNodeGetEvent(hNode: CUgraphNode, event_out: *mut CUevent) -> CUresult;

#[cuda_hook(proc_id = 900838)]
fn cuGraphEventWaitNodeSetEvent(hNode: CUgraphNode, event: CUevent) -> CUresult;

#[cuda_hook(proc_id = 900839)]
fn cuGraphExecEventRecordNodeSetEvent(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    event: CUevent,
) -> CUresult;

#[cuda_hook(proc_id = 900840)]
fn cuGraphExecEventWaitNodeSetEvent(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    event: CUevent,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901149)]
fn cuGraphAddHostNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    dependencies: *const CUgraphNode,
    numDependencies: usize,
    nodeParams: *const CUDA_HOST_NODE_PARAMS,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901150)]
fn cuGraphHostNodeGetParams(
    hNode: CUgraphNode,
    nodeParams: *mut CUDA_HOST_NODE_PARAMS,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901151)]
fn cuGraphHostNodeSetParams(
    hNode: CUgraphNode,
    nodeParams: *const CUDA_HOST_NODE_PARAMS,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901152)]
fn cuGraphExecHostNodeSetParams(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    nodeParams: *const CUDA_HOST_NODE_PARAMS,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901133)]
fn cuGraphAddExternalSemaphoresSignalNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    dependencies: *const CUgraphNode,
    numDependencies: usize,
    nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901134)]
fn cuGraphExternalSemaphoresSignalNodeGetParams(
    hNode: CUgraphNode,
    params_out: *mut CUDA_EXT_SEM_SIGNAL_NODE_PARAMS,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901135)]
fn cuGraphExternalSemaphoresSignalNodeSetParams(
    hNode: CUgraphNode,
    nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901136)]
fn cuGraphExecExternalSemaphoresSignalNodeSetParams(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901137)]
fn cuGraphAddExternalSemaphoresWaitNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    dependencies: *const CUgraphNode,
    numDependencies: usize,
    nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901138)]
fn cuGraphExternalSemaphoresWaitNodeGetParams(
    hNode: CUgraphNode,
    params_out: *mut CUDA_EXT_SEM_WAIT_NODE_PARAMS,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901139)]
fn cuGraphExternalSemaphoresWaitNodeSetParams(
    hNode: CUgraphNode,
    nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901140)]
fn cuGraphExecExternalSemaphoresWaitNodeSetParams(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS,
) -> CUresult;

#[cuda_hook(proc_id = 900841)]
fn cuGraphAddMemsetNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    #[device] dependencies: *const CUgraphNode,
    numDependencies: usize,
    #[host(len = 1)] memsetParams: *const CUDA_MEMSET_NODE_PARAMS,
    ctx: CUcontext,
) -> CUresult {
    'client_before_send: {
        assert!(dependencies.is_null());
        assert_eq!(numDependencies, 0);
    }
}

#[cuda_hook(proc_id = 900842)]
fn cuGraphMemsetNodeGetParams(
    hNode: CUgraphNode,
    #[host(output, len = 1)] nodeParams: *mut CUDA_MEMSET_NODE_PARAMS,
) -> CUresult;

#[cuda_hook(proc_id = 900843)]
fn cuGraphMemsetNodeSetParams(
    hNode: CUgraphNode,
    #[host(len = 1)] nodeParams: *const CUDA_MEMSET_NODE_PARAMS,
) -> CUresult;

#[cuda_hook(proc_id = 900844)]
fn cuGraphExecMemsetNodeSetParams(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    #[host(len = 1)] memsetParams: *const CUDA_MEMSET_NODE_PARAMS,
    ctx: CUcontext,
) -> CUresult;

#[cuda_hook(proc_id = 900845)]
fn cuGraphAddMemcpyNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    #[device] dependencies: *const CUgraphNode,
    numDependencies: usize,
    #[host(len = 1)] copyParams: *const CUDA_MEMCPY3D,
    ctx: CUcontext,
) -> CUresult {
    'client_before_send: {
        assert!(dependencies.is_null());
        assert_eq!(numDependencies, 0);
        let params = unsafe { &*copyParams };
        assert_eq!(params.srcMemoryType, CUmemorytype::CU_MEMORYTYPE_DEVICE);
        assert_eq!(params.dstMemoryType, CUmemorytype::CU_MEMORYTYPE_DEVICE);
        assert!(params.srcArray.is_null());
        assert!(params.dstArray.is_null());
        assert!(params.reserved0.is_null());
        assert!(params.reserved1.is_null());
    }
}

#[cuda_hook(proc_id = 900846)]
fn cuGraphMemcpyNodeGetParams(
    hNode: CUgraphNode,
    #[host(output, len = 1)] nodeParams: *mut CUDA_MEMCPY3D,
) -> CUresult;

#[cuda_hook(proc_id = 900847)]
fn cuGraphMemcpyNodeSetParams(
    hNode: CUgraphNode,
    #[host(len = 1)] nodeParams: *const CUDA_MEMCPY3D,
) -> CUresult {
    'client_before_send: {
        let params = unsafe { &*nodeParams };
        assert_eq!(params.srcMemoryType, CUmemorytype::CU_MEMORYTYPE_DEVICE);
        assert_eq!(params.dstMemoryType, CUmemorytype::CU_MEMORYTYPE_DEVICE);
        assert!(params.srcArray.is_null());
        assert!(params.dstArray.is_null());
        assert!(params.reserved0.is_null());
        assert!(params.reserved1.is_null());
    }
}

#[cuda_hook(proc_id = 900848)]
fn cuGraphExecMemcpyNodeSetParams(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    #[host(len = 1)] copyParams: *const CUDA_MEMCPY3D,
    ctx: CUcontext,
) -> CUresult {
    'client_before_send: {
        let params = unsafe { &*copyParams };
        assert_eq!(params.srcMemoryType, CUmemorytype::CU_MEMORYTYPE_DEVICE);
        assert_eq!(params.dstMemoryType, CUmemorytype::CU_MEMORYTYPE_DEVICE);
        assert!(params.srcArray.is_null());
        assert!(params.dstArray.is_null());
        assert!(params.reserved0.is_null());
        assert!(params.reserved1.is_null());
    }
}

#[cuda_hook(proc_id = 900849)]
fn cuGraphNodeSetEnabled(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    isEnabled: c_uint,
) -> CUresult;

#[cuda_hook(proc_id = 900850)]
fn cuGraphNodeGetEnabled(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    isEnabled: *mut c_uint,
) -> CUresult;

#[cuda_hook(proc_id = 900851)]
fn cuGraphAddMemAllocNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    #[device] dependencies: *const CUgraphNode,
    numDependencies: usize,
    #[host(input, len = 1)] nodeParams: *mut CUDA_MEM_ALLOC_NODE_PARAMS,
) -> CUresult {
    'client_before_send: {
        assert!(dependencies.is_null());
        assert_eq!(numDependencies, 0);
        let params = unsafe { &*nodeParams };
        assert!(params.accessDescs.is_null());
        assert_eq!(params.accessDescCount, 0);
        assert_eq!(
            params.poolProps.allocType,
            CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED
        );
        assert_eq!(
            params.poolProps.handleTypes,
            CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE
        );
        assert_eq!(
            params.poolProps.location.type_,
            CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE
        );
        assert!(params.poolProps.win32SecurityAttributes.is_null());
    }
    'client_after_recv: {
        let mut node_params_out: CUDA_MEM_ALLOC_NODE_PARAMS = unsafe { std::mem::zeroed() };
        node_params_out.recv(channel_receiver).unwrap();
        unsafe {
            std::ptr::write(nodeParams.as_ptr().cast_mut(), node_params_out);
        }
    }
    'server_after_send: {
        nodeParams[0].send(channel_sender).unwrap();
        channel_sender.flush_out().unwrap();
    }
}

#[cuda_hook(proc_id = 900852)]
fn cuGraphMemAllocNodeGetParams(
    hNode: CUgraphNode,
    #[host(output, len = 1)] params_out: *mut CUDA_MEM_ALLOC_NODE_PARAMS,
) -> CUresult;

#[cuda_hook(proc_id = 900853)]
fn cuGraphAddMemFreeNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    #[device] dependencies: *const CUgraphNode,
    numDependencies: usize,
    dptr: CUdeviceptr,
) -> CUresult {
    'client_before_send: {
        assert!(dependencies.is_null());
        assert_eq!(numDependencies, 0);
        assert_ne!(dptr, 0);
    }
}

#[cuda_hook(proc_id = 900854)]
fn cuGraphMemFreeNodeGetParams(hNode: CUgraphNode, dptr_out: *mut CUdeviceptr) -> CUresult;

#[cuda_hook(proc_id = 900855)]
fn cuGraphAddKernelNode_v2(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    #[device] dependencies: *const CUgraphNode,
    numDependencies: usize,
    #[skip] nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
) -> CUresult {
    'client_before_send: {
        assert!(dependencies.is_null());
        assert_eq!(numDependencies, 0);
        assert!(!nodeParams.is_null());
        let node_params = unsafe { &*nodeParams };
        assert!(!node_params.func.is_null());
        assert!(node_params.kern.is_null());
        assert!(node_params.extra.is_null());
        let (args, arg_offsets) = super::cuda_hijack_utils::pack_kernel_args_with_offsets(
            node_params.kernelParams,
            DRIVER_CACHE
                .read()
                .unwrap()
                .function_params
                .get(&node_params.func)
                .unwrap(),
        );
        let mut packed_params = *node_params;
        packed_params.kernelParams = std::ptr::null_mut();
        packed_params.extra = std::ptr::null_mut();
    }
    'client_extra_send: {
        packed_params.send(channel_sender).unwrap();
        send_slice(&arg_offsets, channel_sender).unwrap();
        send_slice(&args, channel_sender).unwrap();
    }
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            DRIVER_CACHE.write().unwrap().graph_kernel_nodes.insert(
                *phGraphNode,
                crate::GraphKernelNodeCache::new(packed_params, args, arg_offsets),
            );
        }
    }
    'server_extra_recv: {
        let mut packed_params = std::mem::MaybeUninit::<CUDA_KERNEL_NODE_PARAMS>::uninit();
        packed_params.recv(channel_receiver).unwrap();
        let mut packed_params = unsafe { packed_params.assume_init() };
        let arg_offsets = recv_slice::<u32, _>(channel_receiver).unwrap();
        let args = recv_slice::<u8, _>(channel_receiver).unwrap();
    }
    'server_execution: {
        let mut kernel_params =
            super::cuda_exe_utils::kernel_params_from_packed_args(&args, &arg_offsets);
        packed_params.kernelParams = if kernel_params.is_empty() {
            std::ptr::null_mut()
        } else {
            kernel_params.as_mut_ptr()
        };
        let result = unsafe {
            cuGraphAddKernelNode_v2(
                phGraphNode__ptr,
                hGraph,
                std::ptr::null(),
                0,
                &raw const packed_params,
            )
        };
    }
}

#[cuda_hook(proc_id = 900856)]
fn cuGraphKernelNodeGetParams_v2(
    hNode: CUgraphNode,
    #[host(output, len = 1)] nodeParams: *mut CUDA_KERNEL_NODE_PARAMS,
) -> CUresult {
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS && !nodeParams.is_empty() {
            if let Some(cache) = DRIVER_CACHE
                .write()
                .unwrap()
                .graph_kernel_nodes
                .get_mut(&hNode)
            {
                nodeParams[0] = cache.params_for_client();
            }
        }
    }
    'server_execution: {
        let result = unsafe { cuGraphKernelNodeGetParams_v2(hNode, nodeParams__ptr) };
        if result == CUresult::CUDA_SUCCESS {
            unsafe {
                (*nodeParams__ptr).kernelParams = std::ptr::null_mut();
                (*nodeParams__ptr).extra = std::ptr::null_mut();
            }
        }
    }
}

#[cuda_hook(proc_id = 900857)]
fn cuGraphKernelNodeSetParams_v2(
    hNode: CUgraphNode,
    #[skip] nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
) -> CUresult {
    'client_before_send: {
        assert!(!nodeParams.is_null());
        let node_params = unsafe { &*nodeParams };
        assert!(!node_params.func.is_null());
        assert!(node_params.kern.is_null());
        assert!(node_params.extra.is_null());
        let (args, arg_offsets) = super::cuda_hijack_utils::pack_kernel_args_with_offsets(
            node_params.kernelParams,
            DRIVER_CACHE
                .read()
                .unwrap()
                .function_params
                .get(&node_params.func)
                .unwrap(),
        );
        let mut packed_params = *node_params;
        packed_params.kernelParams = std::ptr::null_mut();
        packed_params.extra = std::ptr::null_mut();
    }
    'client_extra_send: {
        packed_params.send(channel_sender).unwrap();
        send_slice(&arg_offsets, channel_sender).unwrap();
        send_slice(&args, channel_sender).unwrap();
    }
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            DRIVER_CACHE.write().unwrap().graph_kernel_nodes.insert(
                hNode,
                crate::GraphKernelNodeCache::new(packed_params, args, arg_offsets),
            );
        }
    }
    'server_extra_recv: {
        let mut packed_params = std::mem::MaybeUninit::<CUDA_KERNEL_NODE_PARAMS>::uninit();
        packed_params.recv(channel_receiver).unwrap();
        let mut packed_params = unsafe { packed_params.assume_init() };
        let arg_offsets = recv_slice::<u32, _>(channel_receiver).unwrap();
        let args = recv_slice::<u8, _>(channel_receiver).unwrap();
    }
    'server_execution: {
        let mut kernel_params =
            super::cuda_exe_utils::kernel_params_from_packed_args(&args, &arg_offsets);
        packed_params.kernelParams = if kernel_params.is_empty() {
            std::ptr::null_mut()
        } else {
            kernel_params.as_mut_ptr()
        };
        let result = unsafe { cuGraphKernelNodeSetParams_v2(hNode, &raw const packed_params) };
    }
}

#[cuda_hook(proc_id = 900858)]
fn cuGraphExecKernelNodeSetParams_v2(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    #[skip] nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
) -> CUresult {
    'client_before_send: {
        assert!(!nodeParams.is_null());
        let node_params = unsafe { &*nodeParams };
        assert!(!node_params.func.is_null());
        assert!(node_params.kern.is_null());
        assert!(node_params.extra.is_null());
        let (args, arg_offsets) = super::cuda_hijack_utils::pack_kernel_args_with_offsets(
            node_params.kernelParams,
            DRIVER_CACHE
                .read()
                .unwrap()
                .function_params
                .get(&node_params.func)
                .unwrap(),
        );
        let mut packed_params = *node_params;
        packed_params.kernelParams = std::ptr::null_mut();
        packed_params.extra = std::ptr::null_mut();
    }
    'client_extra_send: {
        packed_params.send(channel_sender).unwrap();
        send_slice(&arg_offsets, channel_sender).unwrap();
        send_slice(&args, channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut packed_params = std::mem::MaybeUninit::<CUDA_KERNEL_NODE_PARAMS>::uninit();
        packed_params.recv(channel_receiver).unwrap();
        let mut packed_params = unsafe { packed_params.assume_init() };
        let arg_offsets = recv_slice::<u32, _>(channel_receiver).unwrap();
        let args = recv_slice::<u8, _>(channel_receiver).unwrap();
    }
    'server_execution: {
        let mut kernel_params =
            super::cuda_exe_utils::kernel_params_from_packed_args(&args, &arg_offsets);
        packed_params.kernelParams = if kernel_params.is_empty() {
            std::ptr::null_mut()
        } else {
            kernel_params.as_mut_ptr()
        };
        let result = unsafe {
            cuGraphExecKernelNodeSetParams_v2(hGraphExec, hNode, &raw const packed_params)
        };
    }
}

#[cuda_hook(proc_id = 900859)]
fn cuGraphKernelNodeCopyAttributes(dst: CUgraphNode, src: CUgraphNode) -> CUresult;

#[cuda_hook(proc_id = 900860)]
fn cuGraphKernelNodeGetAttribute(
    hNode: CUgraphNode,
    attr: CUkernelNodeAttrID,
    #[host(output, len = 1)] value_out: *mut CUkernelNodeAttrValue,
) -> CUresult;

#[cuda_hook(proc_id = 900861)]
fn cuGraphKernelNodeSetAttribute(
    hNode: CUgraphNode,
    attr: CUkernelNodeAttrID,
    #[host(len = 1)] value: *const CUkernelNodeAttrValue,
) -> CUresult;

#[cuda_hook(proc_id = 900872)]
fn cuGraphAddBatchMemOpNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    #[device] dependencies: *const CUgraphNode,
    numDependencies: usize,
    #[skip] nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS,
) -> CUresult {
    'client_before_send: {
        assert!(dependencies.is_null());
        assert_eq!(numDependencies, 0);
        assert!(!nodeParams.is_null());
        let node_params = unsafe { &*nodeParams };
        assert!(node_params.count > 0);
        assert!(!node_params.paramArray.is_null());
        assert_eq!(node_params.flags, 0);
        let ops = unsafe {
            std::slice::from_raw_parts(node_params.paramArray, node_params.count as usize)
        };
        for op in ops {
            match unsafe { op.operation } {
                CUstreamBatchMemOpType::CU_STREAM_MEM_OP_WRITE_VALUE_32 => {
                    let write = unsafe { op.writeValue };
                    assert_ne!(write.address, 0);
                    assert_eq!(
                        write.flags,
                        CUstreamWriteValue_flags::CU_STREAM_WRITE_VALUE_DEFAULT as c_uint
                    );
                    assert_eq!(write.alias, 0);
                }
                CUstreamBatchMemOpType::CU_STREAM_MEM_OP_WRITE_VALUE_64 => {
                    let write = unsafe { op.writeValue };
                    assert_ne!(write.address, 0);
                    assert_eq!(
                        write.flags,
                        CUstreamWriteValue_flags::CU_STREAM_WRITE_VALUE_DEFAULT as c_uint
                    );
                    assert_eq!(write.alias, 0);
                }
                _ => panic!("unsupported graph batch mem-op operation"),
            }
        }
        let mut packed_params = *node_params;
        packed_params.paramArray = std::ptr::null_mut();
    }
    'client_extra_send: {
        packed_params.send(channel_sender).unwrap();
        send_slice(ops, channel_sender).unwrap();
    }
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            DRIVER_CACHE
                .write()
                .unwrap()
                .graph_batch_mem_op_nodes
                .insert(
                    *phGraphNode,
                    crate::GraphBatchMemOpNodeCache::new(
                        packed_params,
                        ops.to_vec().into_boxed_slice(),
                    ),
                );
        }
    }
    'server_extra_recv: {
        let mut packed_params = std::mem::MaybeUninit::<CUDA_BATCH_MEM_OP_NODE_PARAMS>::uninit();
        packed_params.recv(channel_receiver).unwrap();
        let mut packed_params = unsafe { packed_params.assume_init() };
        let mut ops = recv_slice::<CUstreamBatchMemOpParams, _>(channel_receiver).unwrap();
    }
    'server_execution: {
        packed_params.paramArray = ops.as_mut_ptr();
        let result = unsafe {
            cuGraphAddBatchMemOpNode(
                phGraphNode__ptr,
                hGraph,
                std::ptr::null(),
                0,
                &raw const packed_params,
            )
        };
    }
}

#[cuda_hook(proc_id = 900873)]
fn cuGraphBatchMemOpNodeGetParams(
    hNode: CUgraphNode,
    #[skip] nodeParams_out: *mut CUDA_BATCH_MEM_OP_NODE_PARAMS,
) -> CUresult {
    'client_before_send: {
        assert!(!nodeParams_out.is_null());
    }
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            let mut params = std::mem::MaybeUninit::<CUDA_BATCH_MEM_OP_NODE_PARAMS>::uninit();
            params.recv(channel_receiver).unwrap();
            let params = unsafe { params.assume_init() };
            let ops = recv_slice::<CUstreamBatchMemOpParams, _>(channel_receiver).unwrap();
            let mut cache =
                crate::GraphBatchMemOpNodeCache::new(params, ops.to_vec().into_boxed_slice());
            unsafe {
                std::ptr::write(nodeParams_out, cache.params_for_client());
            }
            DRIVER_CACHE
                .write()
                .unwrap()
                .graph_batch_mem_op_nodes
                .insert(hNode, cache);
        }
    }
    'server_execution: {
        let mut params_out: CUDA_BATCH_MEM_OP_NODE_PARAMS = unsafe { std::mem::zeroed() };
        let result = unsafe { cuGraphBatchMemOpNodeGetParams(hNode, &raw mut params_out) };
        let ops: Box<[CUstreamBatchMemOpParams]> = if result == CUresult::CUDA_SUCCESS {
            assert!(params_out.count > 0);
            assert!(!params_out.paramArray.is_null());
            unsafe {
                std::slice::from_raw_parts(params_out.paramArray, params_out.count as usize)
                    .to_vec()
                    .into_boxed_slice()
            }
        } else {
            Box::default()
        };
        params_out.paramArray = std::ptr::null_mut();
    }
    'server_after_send: {
        if result == CUresult::CUDA_SUCCESS {
            params_out.send(channel_sender).unwrap();
            send_slice(&ops, channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
}

#[cuda_hook(proc_id = 900874)]
fn cuGraphBatchMemOpNodeSetParams(
    hNode: CUgraphNode,
    #[skip] nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS,
) -> CUresult {
    'client_before_send: {
        assert!(!nodeParams.is_null());
        let node_params = unsafe { &*nodeParams };
        assert!(node_params.count > 0);
        assert!(!node_params.paramArray.is_null());
        assert_eq!(node_params.flags, 0);
        let ops = unsafe {
            std::slice::from_raw_parts(node_params.paramArray, node_params.count as usize)
        };
        for op in ops {
            match unsafe { op.operation } {
                CUstreamBatchMemOpType::CU_STREAM_MEM_OP_WRITE_VALUE_32 => {
                    let write = unsafe { op.writeValue };
                    assert_ne!(write.address, 0);
                    assert_eq!(
                        write.flags,
                        CUstreamWriteValue_flags::CU_STREAM_WRITE_VALUE_DEFAULT as c_uint
                    );
                    assert_eq!(write.alias, 0);
                }
                CUstreamBatchMemOpType::CU_STREAM_MEM_OP_WRITE_VALUE_64 => {
                    let write = unsafe { op.writeValue };
                    assert_ne!(write.address, 0);
                    assert_eq!(
                        write.flags,
                        CUstreamWriteValue_flags::CU_STREAM_WRITE_VALUE_DEFAULT as c_uint
                    );
                    assert_eq!(write.alias, 0);
                }
                _ => panic!("unsupported graph batch mem-op operation"),
            }
        }
        let mut packed_params = *node_params;
        packed_params.paramArray = std::ptr::null_mut();
    }
    'client_extra_send: {
        packed_params.send(channel_sender).unwrap();
        send_slice(ops, channel_sender).unwrap();
    }
    'client_after_recv: {
        if result == CUresult::CUDA_SUCCESS {
            DRIVER_CACHE
                .write()
                .unwrap()
                .graph_batch_mem_op_nodes
                .insert(
                    hNode,
                    crate::GraphBatchMemOpNodeCache::new(
                        packed_params,
                        ops.to_vec().into_boxed_slice(),
                    ),
                );
        }
    }
    'server_extra_recv: {
        let mut packed_params = std::mem::MaybeUninit::<CUDA_BATCH_MEM_OP_NODE_PARAMS>::uninit();
        packed_params.recv(channel_receiver).unwrap();
        let mut packed_params = unsafe { packed_params.assume_init() };
        let mut ops = recv_slice::<CUstreamBatchMemOpParams, _>(channel_receiver).unwrap();
    }
    'server_execution: {
        packed_params.paramArray = ops.as_mut_ptr();
        let result = unsafe { cuGraphBatchMemOpNodeSetParams(hNode, &raw const packed_params) };
    }
}

#[cuda_hook(proc_id = 900875)]
fn cuGraphExecBatchMemOpNodeSetParams(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    #[skip] nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS,
) -> CUresult {
    'client_before_send: {
        assert!(!nodeParams.is_null());
        let node_params = unsafe { &*nodeParams };
        assert!(node_params.count > 0);
        assert!(!node_params.paramArray.is_null());
        assert_eq!(node_params.flags, 0);
        let ops = unsafe {
            std::slice::from_raw_parts(node_params.paramArray, node_params.count as usize)
        };
        for op in ops {
            match unsafe { op.operation } {
                CUstreamBatchMemOpType::CU_STREAM_MEM_OP_WRITE_VALUE_32 => {
                    let write = unsafe { op.writeValue };
                    assert_ne!(write.address, 0);
                    assert_eq!(
                        write.flags,
                        CUstreamWriteValue_flags::CU_STREAM_WRITE_VALUE_DEFAULT as c_uint
                    );
                    assert_eq!(write.alias, 0);
                }
                CUstreamBatchMemOpType::CU_STREAM_MEM_OP_WRITE_VALUE_64 => {
                    let write = unsafe { op.writeValue };
                    assert_ne!(write.address, 0);
                    assert_eq!(
                        write.flags,
                        CUstreamWriteValue_flags::CU_STREAM_WRITE_VALUE_DEFAULT as c_uint
                    );
                    assert_eq!(write.alias, 0);
                }
                _ => panic!("unsupported graph batch mem-op operation"),
            }
        }
        let mut packed_params = *node_params;
        packed_params.paramArray = std::ptr::null_mut();
    }
    'client_extra_send: {
        packed_params.send(channel_sender).unwrap();
        send_slice(ops, channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut packed_params = std::mem::MaybeUninit::<CUDA_BATCH_MEM_OP_NODE_PARAMS>::uninit();
        packed_params.recv(channel_receiver).unwrap();
        let mut packed_params = unsafe { packed_params.assume_init() };
        let mut ops = recv_slice::<CUstreamBatchMemOpParams, _>(channel_receiver).unwrap();
    }
    'server_execution: {
        packed_params.paramArray = ops.as_mut_ptr();
        let result = unsafe {
            cuGraphExecBatchMemOpNodeSetParams(hGraphExec, hNode, &raw const packed_params)
        };
    }
}

#[cuda_hook(proc_id = 900862)]
fn cuGraphAddNode_v2(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    #[device] dependencies: *const CUgraphNode,
    #[device] dependencyData: *const CUgraphEdgeData,
    numDependencies: usize,
    #[host(input, len = 1)] nodeParams: *mut CUgraphNodeParams,
) -> CUresult {
    'client_before_send: {
        assert!(dependencies.is_null());
        assert!(dependencyData.is_null());
        assert_eq!(numDependencies, 0);
        assert!(!nodeParams.is_null());
        let params = unsafe { &*nodeParams };
        assert_eq!(params.type_, CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET);
        let memset = unsafe { params.__bindgen_anon_1.memset };
        assert_eq!(memset.elementSize, 1);
        assert_eq!(memset.height, 1);
    }
}

#[cuda_hook(proc_id = 900863)]
fn cuGraphNodeGetParams(
    hNode: CUgraphNode,
    #[host(output, len = 1)] nodeParams: *mut CUgraphNodeParams,
) -> CUresult {
    'server_execution: {
        let result = unsafe { cuGraphNodeGetParams(hNode, nodeParams__ptr) };
        if result == CUresult::CUDA_SUCCESS {
            unsafe {
                assert_eq!(
                    (*nodeParams__ptr).type_,
                    CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET
                );
            }
        }
    }
}

#[cuda_hook(proc_id = 900864)]
fn cuGraphNodeSetParams(
    hNode: CUgraphNode,
    #[host(input, len = 1)] nodeParams: *mut CUgraphNodeParams,
) -> CUresult {
    'client_before_send: {
        assert!(!nodeParams.is_null());
        let params = unsafe { &*nodeParams };
        assert_eq!(params.type_, CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET);
        let memset = unsafe { params.__bindgen_anon_1.memset };
        assert_eq!(memset.elementSize, 1);
        assert_eq!(memset.height, 1);
    }
}

#[cuda_hook(proc_id = 900865)]
fn cuGraphExecNodeSetParams(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    #[host(input, len = 1)] nodeParams: *mut CUgraphNodeParams,
) -> CUresult {
    'client_before_send: {
        assert!(!nodeParams.is_null());
        let params = unsafe { &*nodeParams };
        assert_eq!(params.type_, CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET);
        let memset = unsafe { params.__bindgen_anon_1.memset };
        assert_eq!(memset.elementSize, 1);
        assert_eq!(memset.height, 1);
    }
}

#[cuda_hook(proc_id = 900928)]
fn cuGraphConditionalHandleCreate(
    pHandle_out: *mut CUgraphConditionalHandle,
    hGraph: CUgraph,
    ctx: CUcontext,
    defaultLaunchValue: c_uint,
    flags: c_uint,
) -> CUresult;

#[cuda_custom_hook(proc_id = 901157)]
fn cuCheckpointProcessGetRestoreThreadId(pid: c_int, tid: *mut c_int) -> CUresult;

#[cuda_custom_hook(proc_id = 901158)]
fn cuCheckpointProcessGetState(pid: c_int, state: *mut CUprocessState) -> CUresult;

#[cuda_custom_hook(proc_id = 901159)]
fn cuCheckpointProcessLock(pid: c_int, args: *mut CUcheckpointLockArgs) -> CUresult;

#[cuda_custom_hook(proc_id = 901160)]
fn cuCheckpointProcessCheckpoint(pid: c_int, args: *mut CUcheckpointCheckpointArgs) -> CUresult;

#[cuda_custom_hook(proc_id = 901161)]
fn cuCheckpointProcessRestore(pid: c_int, args: *mut CUcheckpointRestoreArgs) -> CUresult;

#[cuda_custom_hook(proc_id = 901162)]
fn cuCheckpointProcessUnlock(pid: c_int, args: *mut CUcheckpointUnlockArgs) -> CUresult;

#[cuda_custom_hook]
fn cuGetProcAddress_v2(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    cudaVersion: c_int,
    flags: cuuint64_t,
    symbolStatus: *mut CUdriverProcAddressQueryResult,
) -> CUresult;
