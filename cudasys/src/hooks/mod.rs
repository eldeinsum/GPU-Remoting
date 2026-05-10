#![allow(dead_code)] // doesn't work well with expect
#![expect(unused_variables)]
#![expect(clippy::too_many_arguments)]

mod cublas;
mod cublasLt;
mod cuda;
mod cudart;
mod cudnn;
mod nccl;
mod nvml;
mod nvrtc;
