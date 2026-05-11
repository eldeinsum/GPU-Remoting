use std::os::raw::*;

use cudasys::cuda::*;

#[allow(clippy::too_many_arguments)]
pub fn cu_launch_kernel(
    f: CUfunction,
    gridDimX: c_uint,
    gridDimY: c_uint,
    gridDimZ: c_uint,
    blockDimX: c_uint,
    blockDimY: c_uint,
    blockDimZ: c_uint,
    sharedMemBytes: c_uint,
    hStream: CUstream,
    args: &[u8],
    arg_offsets: &[u32],
) -> CUresult {
    unsafe {
        let mut kernel_params = kernel_params_from_packed_args(args, arg_offsets);
        cuLaunchKernel(
            f,
            gridDimX,
            gridDimY,
            gridDimZ,
            blockDimX,
            blockDimY,
            blockDimZ,
            sharedMemBytes,
            hStream,
            if kernel_params.is_empty() {
                std::ptr::null_mut()
            } else {
                kernel_params.as_mut_ptr()
            },
            std::ptr::null_mut(),
        )
    }
}

pub fn cu_launch_kernel_ex(
    config: &CUlaunchConfig,
    f: CUfunction,
    args: &[u8],
    arg_offsets: &[u32],
) -> CUresult {
    unsafe {
        let mut kernel_params = kernel_params_from_packed_args(args, arg_offsets);
        cuLaunchKernelEx(
            config as *const _,
            f,
            if kernel_params.is_empty() {
                std::ptr::null_mut()
            } else {
                kernel_params.as_mut_ptr()
            },
            std::ptr::null_mut(),
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn cu_launch_cooperative_kernel(
    f: CUfunction,
    gridDimX: c_uint,
    gridDimY: c_uint,
    gridDimZ: c_uint,
    blockDimX: c_uint,
    blockDimY: c_uint,
    blockDimZ: c_uint,
    sharedMemBytes: c_uint,
    hStream: CUstream,
    args: &[u8],
    arg_offsets: &[u32],
) -> CUresult {
    unsafe {
        let mut kernel_params = kernel_params_from_packed_args(args, arg_offsets);
        cuLaunchCooperativeKernel(
            f,
            gridDimX,
            gridDimY,
            gridDimZ,
            blockDimX,
            blockDimY,
            blockDimZ,
            sharedMemBytes,
            hStream,
            if kernel_params.is_empty() {
                std::ptr::null_mut()
            } else {
                kernel_params.as_mut_ptr()
            },
        )
    }
}

pub fn kernel_params_from_packed_args(args: &[u8], arg_offsets: &[u32]) -> Vec<*mut c_void> {
    arg_offsets
        .iter()
        .map(|offset| unsafe {
            args.as_ptr()
                .add(*offset as usize)
                .cast_mut()
                .cast::<c_void>()
        })
        .collect::<Vec<_>>()
}

pub fn cu_func_get_attributes(
    attr: *mut cudasys::cudart::cudaFuncAttributes,
    func: CUfunction,
) -> CUresult {
    let attr = unsafe { &mut *attr };
    // HACK: implementation with cuFuncGetAttribute depends on CUDA version
    macro_rules! get_attributes {
        ($func:ident -> $struct:ident $($field:ident: $attr:ident,)+) => {
            $(
                let mut i = 0;
                let result =
                    unsafe { cuFuncGetAttribute(&raw mut i, CUfunction_attribute::$attr, $func) };
                if result != Default::default() {
                    return result;
                }
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
    CUresult::CUDA_SUCCESS
}
