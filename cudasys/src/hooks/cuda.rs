use crate::types::cuda::*;
use codegen::{cuda_custom_hook, cuda_hook};
use std::os::raw::*;

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
        let args = super::cuda_hijack_utils::pack_kernel_args(
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
        send_slice(&args, channel_sender).unwrap();
    }
    'server_extra_recv: {
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
        );
    }
}

#[cuda_custom_hook] // calls the internal API below
fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;

#[cuda_hook(proc_id = 701, parent = cuModuleLoadData)]
fn cuModuleLoadDataInternal(
    module: *mut CUmodule,
    #[host(len = len)] image: *const c_void,
    #[skip] is_runtime: bool,
) -> CUresult {
    'client_before_send: {
        let len = if FatBinaryHeader::is_fat_binary(image) {
            let header: &FatBinaryHeader = unsafe { &*image.cast() };
            header.entire_len()
        } else {
            crate::elf::elf_len(image)
        };
    }
    'client_after_recv: {
        let image = if is_runtime {
            std::borrow::Cow::Borrowed(image)
        } else {
            std::borrow::Cow::Owned(image.to_vec())
        };
        assert!(DRIVER_CACHE
            .write()
            .unwrap()
            .images
            .insert(*module, image)
            .is_none());
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
        let mut driver = DRIVER_CACHE.write().unwrap();
        let image = driver.images.get(&hmod).unwrap();
        let params = if FatBinaryHeader::is_fat_binary(image.as_ptr()) {
            let fatbin: &FatBinaryHeader = unsafe { &*image.as_ptr().cast() };
            fatbin.find_kernel_params(name.to_str().unwrap())
        } else {
            crate::elf::find_kernel_params(image, name.to_str().unwrap())
        };
        assert!(driver.function_params.insert(*hfunc, params).is_none());
    }
}

#[cuda_hook(proc_id = 640)]
fn cuDriverGetVersion(driverVersion: *mut c_int) -> CUresult;

#[cuda_custom_hook] // local
fn cuGetErrorName(error: CUresult, pStr: *mut *const c_char) -> CUresult;

#[cuda_custom_hook] // local
fn cuGetErrorString(error: CUresult, pStr: *mut *const c_char) -> CUresult;

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

#[cuda_hook(proc_id = 900305)]
fn cuCtxGetLimit(pvalue: *mut usize, limit: CUlimit) -> CUresult;

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

#[cuda_hook(proc_id = 900313)]
fn cuDeviceComputeCapability(major: *mut c_int, minor: *mut c_int, dev: CUdevice) -> CUresult;

#[cuda_hook(proc_id = 650)]
fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;

#[cuda_hook(proc_id = 651)]
fn cuDeviceGetAttribute(pi: *mut c_int, attrib: CUdevice_attribute, dev: CUdevice) -> CUresult;

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

#[cuda_hook(proc_id = 900320)]
fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult;

#[cuda_hook(proc_id = 900321, async_api = false)]
fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult;

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

#[cuda_hook(proc_id = 900401)]
fn cuMemAllocAsync(dptr: *mut CUdeviceptr, bytesize: usize, hStream: CUstream) -> CUresult;

#[cuda_hook(proc_id = 900202, async_api = false)]
fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;

#[cuda_hook(proc_id = 900402, async_api = false)]
fn cuMemFreeAsync(dptr: CUdeviceptr, hStream: CUstream) -> CUresult;

#[cuda_hook(proc_id = 900403)]
fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;

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

#[cuda_hook(proc_id = 900214, async_api = false)]
fn cuEventDestroy_v2(hEvent: CUevent) -> CUresult;

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

#[cuda_custom_hook]
fn cuGetProcAddress_v2(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    cudaVersion: c_int,
    flags: cuuint64_t,
    symbolStatus: *mut CUdriverProcAddressQueryResult,
) -> CUresult;
