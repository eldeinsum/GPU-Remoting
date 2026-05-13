#![expect(non_snake_case)]

use cudasys::cudnn::cudnnGetMaxDeviceVersion;
use network::{CommChannel, Transportable};

use crate::ServerWorker;

pub fn cudnnGetMaxDeviceVersionExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    log::debug!(target: "cudnnGetMaxDeviceVersion", "[#{}]", server.id);
    server
        .channel_receiver
        .recv_ts()
        .expect("failed to receive cudnnGetMaxDeviceVersion timestamp");

    let result = unsafe { cudnnGetMaxDeviceVersion() };
    result
        .send(&server.channel_sender)
        .expect("failed to send cudnnGetMaxDeviceVersion result");
    server
        .channel_sender
        .flush_out()
        .expect("failed to flush cudnnGetMaxDeviceVersion response");
}
