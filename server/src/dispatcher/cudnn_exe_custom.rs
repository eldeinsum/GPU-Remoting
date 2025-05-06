use super::*;
use cudasys::cudnn::*;

pub fn cudnnGetErrorStringExe<C: CommChannel>(
    #[cfg(feature = "phos")] proc_id: i32,
    server: &mut ServerWorker<C>,
) {
    let ServerWorker { channel_sender, channel_receiver, .. } = server;
    log::debug!(target: "cudnnGetErrorString", "");
    let mut status: cudnnStatus_t = Default::default();
    if let Err(e) = status.recv(channel_receiver) {
        error!("Error receiving status: {:?}", e);
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    #[cfg(not(feature = "phos"))]
    let result = unsafe { cudnnGetErrorString(status) };
    #[cfg(feature = "phos")]
    panic!("PhOS returns an i32 which cannot be converted to a pointer");
    // transfer to Vec<u8>
    let result = unsafe { std::ffi::CStr::from_ptr(result).to_bytes().to_vec() };
    if let Err(e) = result.send(channel_sender) {
        error!("Error sending result: {:?}", e);
    }
    channel_sender.flush_out().unwrap();
}
