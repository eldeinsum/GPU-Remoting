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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_proc_id_has_no_dispatch_target() {
        assert!(dispatch_target::<network::Channel>(i32::MIN).is_none());
    }
}
