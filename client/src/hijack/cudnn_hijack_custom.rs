use super::*;
use cudasys::types::cudnn::*;
use std::os::raw::*;

#[no_mangle]
#[use_thread_local(client = CLIENT_THREAD.with_borrow_mut)]
pub extern "C" fn cudnnGetErrorString(
    error_status: cudnnStatus_t,
) -> *const c_char {
    log::debug!(target: "cudnnGetErrorString", "{error_status:?}");
    let ClientThread { channel_sender, channel_receiver, .. } = client;
    let proc_id = 1834;
    let mut result: Vec<u8> = Default::default();
    if let Err(e) = proc_id.send(channel_sender) {
        error!("Error sending proc_id: {:?}", e);
    }
    if let Err(e) = error_status.send(channel_sender) {
        error!("Error sending error_string: {:?}", e);
    }
    if let Err(e) = channel_sender.flush_out() {
        panic!("failed to send: {:?}", e);
    }
    if let Err(e) = result.recv(channel_receiver) {
        error!("Error receiving result: {:?}", e);
    }
    if let Err(e) = channel_receiver.recv_ts() {
        error!("Error receiving timestamp: {:?}", e);
    }
    let c_str = std::ffi::CString::new(result).unwrap();
    c_str.into_raw()
}