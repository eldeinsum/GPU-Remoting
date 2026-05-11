#![expect(non_snake_case)]

mod cublasLt_hijack;
mod cublasLt_hijack_custom;
mod cublasLt_unimplement;
mod cublas_hijack;
mod cublas_hijack_custom;
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
mod host_memory;
mod nccl_hijack;
mod nccl_hijack_custom;
mod nccl_unimplement;
mod nvml_hijack;
mod nvml_hijack_custom;
mod nvml_unimplement;
mod nvrtc_hijack;
mod nvrtc_hijack_custom;
mod nvrtc_unimplement;
mod user_object;

#[expect(unused_imports)]
use codegen::{cuda_hook_hijack, use_thread_local};
use network::type_impl::{recv_slice, recv_slice_to, send_slice, MemPtr};
use network::{CommChannel, Transportable};

use crate::elf::{FatBinaryHeader, FatBinaryWrapper};
use crate::{
    cublaslt_pointer_mode_from_u32, cublaslt_scale_type_from_u32, cublaslt_scale_type_size,
    ClientThread, CublasLtMatmulDescState, CublasLtTransformDescState, FatBinaryHandle, HostPtr,
    RuntimeCache, CLIENT_THREAD, CUBLAS_CACHE, DRIVER_CACHE, RUNTIME_CACHE,
};
