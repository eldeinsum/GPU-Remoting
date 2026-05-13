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
use network::type_impl::{MemPtr, recv_slice, recv_slice_to, send_slice};
use network::{CommChannel, Transportable};

use crate::elf::{FatBinaryHeader, FatBinaryWrapper};
use crate::{
    CLIENT_THREAD, CUBLAS_CACHE, ClientThread, CublasLtMatmulDescState, CublasLtTransformDescState,
    DRIVER_CACHE, FatBinaryHandle, HostPtr, RUNTIME_CACHE, RuntimeCache,
    cublaslt_pointer_mode_from_u32, cublaslt_resolve_emulation_desc, cublaslt_resolve_matmul_desc,
    cublaslt_resolve_matmul_preference, cublaslt_resolve_matrix_layout,
    cublaslt_resolve_transform_desc, cublaslt_scale_type_from_u32, cublaslt_scale_type_size,
    cublaslt_unbind_emulation_desc, cublaslt_unbind_matmul_desc, cublaslt_unbind_matmul_preference,
    cublaslt_unbind_matrix_layout, cublaslt_unbind_transform_desc, cudnn_ctc_batch_size,
    cudnn_ctc_label_count, cudnn_data_type_scalar_size, cudnn_filter_desc_scalar_size,
    cudnn_record_filter_desc_type, cudnn_record_seq_data_desc, cudnn_record_tensor_desc,
    cudnn_remove_filter_desc, cudnn_remove_seq_data_desc, cudnn_remove_tensor_desc,
    cudnn_seq_data_time_length, cudnn_tensor_desc_scalar_size,
};
