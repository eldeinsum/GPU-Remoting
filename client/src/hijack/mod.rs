#![expect(non_snake_case)]

mod cublasLt_hijack;
mod cublasLt_hijack_custom;
mod cublasLt_unimplement;
mod cublas_hijack;
mod cublas_unimplement;
mod cuda_hijack;
mod cuda_hijack_custom;
mod cuda_hijack_utils;
mod cuda_unimplement;
mod cudart_hijack;
mod cudart_hijack_custom;
mod cudart_register;
mod cudart_unimplement;
mod cudnn_hijack;
mod cudnn_hijack_custom;
mod cudnn_unimplement;
mod nccl_hijack;
mod nccl_unimplement;
mod nvml_hijack;
mod nvml_unimplement;
mod nvrtc_hijack;
mod nvrtc_hijack_custom;
mod nvrtc_unimplement;

#[expect(unused_imports)]
use codegen::{cuda_hook_hijack, use_thread_local};
use network::type_impl::{MemPtr, recv_slice_to, send_slice};
use network::{CommChannel, Transportable};

use crate::elf::{FatBinaryHeader, FatBinaryWrapper};
use crate::{
    CLIENT_THREAD, CUBLAS_CACHE, ClientThread, CublasLtMatmulDescState, CublasLtTransformDescState,
    DRIVER_CACHE, FatBinaryHandle, HostPtr, RUNTIME_CACHE, RuntimeCache,
    cublaslt_pointer_mode_from_u32, cublaslt_scale_type_from_u32, cublaslt_scale_type_size,
};
