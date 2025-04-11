use crate::types::cuda::*;
use codegen::cuda_hook;
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
            DRIVER_CACHE.read().unwrap().function_params.get(&f).unwrap(),
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
            #[cfg(feature = "phos")]
            server.pos_cuda_ws,
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

#[cuda_hook(proc_id = 701)]
fn cuModuleLoadData(module: *mut CUmodule, #[host(len = len)] image: *const c_void) -> CUresult {
    'client_before_send: {
        let len = if client.is_cuda_launch_kernel {
            let header: &FatBinaryHeader = unsafe { &*image.cast() };
            header.entire_len()
        } else {
            todo!()
        };
    }
    'client_after_recv: {
        let image = if client.is_cuda_launch_kernel {
            std::borrow::Cow::Borrowed(image)
        } else {
            std::borrow::Cow::Owned(image.to_vec())
        };
        assert!(DRIVER_CACHE.write().unwrap().images.insert(*module, image).is_none());
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
        let params = if let std::borrow::Cow::Borrowed(image) = image {
            let fatbin: &FatBinaryHeader = unsafe { &*image.as_ptr().cast() };
            fatbin.find_kernel_params(name.to_str().unwrap())
        } else {
            todo!()
        };
        assert!(driver.function_params.insert(*hfunc, params).is_none());
    }
}

#[cuda_hook(proc_id = 640)]
fn cuDriverGetVersion(driverVersion: *mut c_int) -> CUresult;

#[cuda_hook(proc_id = 630)]
fn cuInit(Flags: c_uint) -> CUresult;

#[cuda_hook(proc_id = 684)]
fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult;
