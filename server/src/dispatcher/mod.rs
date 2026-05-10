#![expect(non_snake_case)]

use codegen::cuda_hook_exe;
use log::error;
use network::type_impl::{recv_slice, send_slice};
use network::{CommChannel, Transportable};

use crate::ServerWorker;

mod cublasLt_exe;
mod cublas_exe;
mod cuda_exe;
mod cuda_exe_utils;
mod cudart_exe;
mod cudart_exe_custom;
mod cudnn_exe;
mod nccl_exe;
mod nvml_exe;
mod nvrtc_exe;

include!("mod_exe.rs");

pub fn dispatch<C: CommChannel>(proc_id: i32, server: &mut ServerWorker<C>) {
    // let start = network::NsTimestamp::now();
    #[deny(unreachable_patterns)]
    let func = dispatcher_match! { proc_id,
        other => {
            error!(
                "[{}:{}] invalid proc_id: {}",
                std::file!(),
                std::line!(),
                other
            );
            panic!();
        }
    };
    func(server);
    // let end = network::NsTimestamp::now();
    // let elapsed = (end.sec_timestamp - start.sec_timestamp) as f64 * 1000000000.0
    //             + (end.ns_timestamp as i32 - start.ns_timestamp as i32) as f64;
    // info!("exe: {}", elapsed / 1000.0);
}
