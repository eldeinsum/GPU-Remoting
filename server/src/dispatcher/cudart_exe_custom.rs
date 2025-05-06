use super::*;
use cudasys::cudart::*;

pub fn cudaGetErrorStringExe<C: CommChannel>(#[cfg(feature = "phos")] proc_id: i32, server: &mut ServerWorker<C>) {
    let ServerWorker { channel_sender, channel_receiver, .. } = server;
    log::debug!(target: "cudaGetErrorString", "");
    let mut error: cudaError_t = Default::default();
    error.recv(channel_receiver).unwrap();
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    #[cfg(not(feature = "phos"))]
    let result = unsafe { cudaGetErrorString(error) };
    #[cfg(feature = "phos")]
    panic!("PhOS returns an i32 which cannot be converted to a pointer");
    let result = unsafe { std::ffi::CStr::from_ptr(result).to_bytes().to_vec() };
    if let Err(e) = result.send(channel_sender) {
        error!("Error sending result: {:?}", e);
    }
    channel_sender.flush_out().unwrap();
}
