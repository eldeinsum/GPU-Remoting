#![expect(non_snake_case)]

use codegen::cuda_hook_exe;
use log::error;
use network::type_impl::{recv_slice, send_slice};
use network::{CommChannel, Transportable};

use crate::ServerWorker;

mod cublasLt_exe;
mod cublas_exe;
mod cublas_exe_custom;
mod cuda_exe;
mod cuda_exe_custom;
mod cuda_exe_utils;
mod cudart_exe;
mod cudart_exe_custom;
mod cudnn_exe;
mod cudnn_exe_custom;
mod nccl_exe;
mod nccl_exe_custom;
mod nvml_exe;
mod nvml_exe_custom;
mod nvrtc_exe;

include!("mod_exe.rs");

fn dispatch_target<C: CommChannel>(proc_id: i32) -> Option<fn(&mut ServerWorker<C>)> {
    #[deny(unreachable_patterns)]
    let func = dispatcher_match! { proc_id,
        other => {
            error!(
                "[{}:{}] invalid proc_id: {}",
                std::file!(),
                std::line!(),
                other
            );
            return None;
        }
    };
    Some(func)
}

pub fn dispatch<C: CommChannel>(proc_id: i32, server: &mut ServerWorker<C>) -> bool {
    // let start = network::NsTimestamp::now();
    let Some(func) = dispatch_target(proc_id) else {
        return false;
    };
    func(server);
    // let end = network::NsTimestamp::now();
    // let elapsed = (end.sec_timestamp - start.sec_timestamp) as f64 * 1000000000.0
    //             + (end.ns_timestamp as i32 - start.ns_timestamp as i32) as f64;
    // info!("exe: {}", elapsed / 1000.0);
    true
}

#[repr(align(16))]
struct CudnnScalarArg([u8; 16]);

impl CudnnScalarArg {
    fn zeroed() -> Self {
        Self([0; 16])
    }

    fn as_ptr(&self) -> *const std::ffi::c_void {
        self.0.as_ptr().cast()
    }

    fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        self.0.as_mut_ptr().cast()
    }

    fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

fn cudnn_recv_scalar_arg<C: CommChannel>(channel_receiver: &C) -> CudnnScalarArg {
    let bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
    assert!(bytes.len() <= 16);
    let mut arg = CudnnScalarArg([0; 16]);
    arg.0[..bytes.len()].copy_from_slice(&bytes);
    arg
}

fn cudnn_data_type_scalar_size(data_type: cudasys::types::cudnn::cudnnDataType_t) -> Option<usize> {
    use cudasys::types::cudnn::cudnnDataType_t;

    Some(match data_type {
        cudnnDataType_t::CUDNN_DATA_DOUBLE | cudnnDataType_t::CUDNN_DATA_INT64 => 8,
        cudnnDataType_t::CUDNN_DATA_FLOAT
        | cudnnDataType_t::CUDNN_DATA_INT32
        | cudnnDataType_t::CUDNN_DATA_UINT32
        | cudnnDataType_t::CUDNN_DATA_FAST_FLOAT_FOR_FP8 => 4,
        cudnnDataType_t::CUDNN_DATA_HALF | cudnnDataType_t::CUDNN_DATA_BFLOAT16 => 2,
        cudnnDataType_t::CUDNN_DATA_INT8
        | cudnnDataType_t::CUDNN_DATA_UINT8
        | cudnnDataType_t::CUDNN_DATA_BOOLEAN
        | cudnnDataType_t::CUDNN_DATA_FP8_E4M3
        | cudnnDataType_t::CUDNN_DATA_FP8_E5M2
        | cudnnDataType_t::CUDNN_DATA_FP8_E8M0 => 1,
        cudnnDataType_t::CUDNN_DATA_COMPLEX_FP32 => 8,
        cudnnDataType_t::CUDNN_DATA_COMPLEX_FP64 => 16,
        cudnnDataType_t::CUDNN_DATA_INT8x4
        | cudnnDataType_t::CUDNN_DATA_UINT8x4
        | cudnnDataType_t::CUDNN_DATA_INT8x32
        | cudnnDataType_t::CUDNN_DATA_FP4_E2M1
        | cudnnDataType_t::CUDNN_DATA_INT4
        | cudnnDataType_t::CUDNN_DATA_UINT4 => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_proc_id_has_no_dispatch_target() {
        assert!(dispatch_target::<network::Channel>(i32::MIN).is_none());
    }
}
