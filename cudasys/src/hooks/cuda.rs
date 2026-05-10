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

#[cuda_hook(proc_id = 900200, async_api = false)]
fn cuCtxSynchronize() -> CUresult;

#[cuda_hook(proc_id = 650)]
fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;

#[cuda_hook(proc_id = 651)]
fn cuDeviceGetAttribute(pi: *mut c_int, attrib: CUdevice_attribute, dev: CUdevice) -> CUresult;

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

#[cuda_hook(proc_id = 900202, async_api = false)]
fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;

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

#[cuda_hook(proc_id = 900206, async_api)]
fn cuMemsetD8_v2(dstDevice: CUdeviceptr, uc: c_uchar, N: usize) -> CUresult;

#[cuda_hook(proc_id = 900207, async_api)]
fn cuMemsetD32_v2(dstDevice: CUdeviceptr, ui: c_uint, N: usize) -> CUresult;

#[cuda_hook(proc_id = 900208)]
fn cuStreamCreate(phStream: *mut CUstream, Flags: c_uint) -> CUresult;

#[cuda_hook(proc_id = 900209, async_api = false)]
fn cuStreamDestroy_v2(hStream: CUstream) -> CUresult;

#[cuda_hook(proc_id = 900210, async_api = false)]
fn cuStreamSynchronize(hStream: CUstream) -> CUresult;

#[cuda_hook(proc_id = 900211)]
fn cuEventCreate(phEvent: *mut CUevent, Flags: c_uint) -> CUresult;

#[cuda_hook(proc_id = 900212, async_api)]
fn cuEventRecord(hEvent: CUevent, hStream: CUstream) -> CUresult;

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
