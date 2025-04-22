use super::*;
use cudasys::cudart::*;
use std::alloc::{alloc, dealloc, Layout};

pub fn cudaMemcpyExe<C: CommChannel>(#[cfg(feature = "phos")] _proc_id: i32, server: &mut ServerWorker<C>) {
    let ServerWorker { channel_sender, channel_receiver, .. } = server;
    log::debug!("[{}:{}] cudaMemcpy", std::file!(), std::line!());

    let mut dst: MemPtr = Default::default();
    dst.recv(channel_receiver).unwrap();
    let mut src: MemPtr = Default::default();
    src.recv(channel_receiver).unwrap();
    let mut count: usize = Default::default();
    count.recv(channel_receiver).unwrap();
    let mut kind: cudaMemcpyKind = cudaMemcpyKind::cudaMemcpyHostToHost;
    kind.recv(channel_receiver).unwrap();

    let mut data_buf = 0 as *mut u8;

    if cudaMemcpyKind::cudaMemcpyHostToDevice == kind {
        data_buf = unsafe { alloc(Layout::from_size_align(count, 1).unwrap()) };
        if data_buf.is_null() {
            panic!("failed to allocate data_buf");
        }
        let data = unsafe { std::slice::from_raw_parts_mut(data_buf, count) };
        data.recv(channel_receiver).unwrap();
        src = data_buf as MemPtr;
    } else if cudaMemcpyKind::cudaMemcpyDeviceToHost == kind {
        data_buf = unsafe { alloc(Layout::from_size_align(count, 1).unwrap()) };
        if data_buf.is_null() {
            panic!("failed to allocate data_buf");
        }
        dst = data_buf as MemPtr;
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }

    #[cfg(not(feature = "phos"))]
    let result = unsafe {
        cudaMemcpy(
            dst as *mut std::os::raw::c_void,
            src as *const std::os::raw::c_void,
            count as usize,
            kind,
        )
    };
    #[cfg(feature = "phos")]
    let result = cudaError_t::from_i32(
        match kind {
            cudaMemcpyKind::cudaMemcpyHostToDevice => call_pos_process(
                server.pos_cuda_ws,
                320, // cudaMemcpyHtod
                0u64,
                &[
                    &raw const dst as usize, size_of_val(&dst),
                    src as usize, count as usize,
                ],
            ),
            cudaMemcpyKind::cudaMemcpyDeviceToHost => call_pos_process(
                server.pos_cuda_ws,
                321, // cudaMemcpyDtoh
                0u64,
                &[
                    &raw const src as usize, size_of_val(&src),
                    &raw const count as usize, size_of_val(&count),
                ],
            ),
            cudaMemcpyKind::cudaMemcpyDeviceToDevice => call_pos_process(
                server.pos_cuda_ws,
                322, // cudaMemcpyDtod
                0u64,
                &[
                    &raw const dst as usize, size_of_val(&dst),
                    &raw const src as usize, size_of_val(&src),
                    &raw const count as usize, size_of_val(&count),
                ],
            ),
            _ => panic!("Illegal cudaMemcpy kind"),
        }
    ).expect("Illegal result ID");


    if cudaMemcpyKind::cudaMemcpyHostToDevice == kind {
        unsafe { dealloc(data_buf, Layout::from_size_align(count, 1).unwrap()) };
    }
    if cudaMemcpyKind::cudaMemcpyDeviceToHost == kind {
        let data = unsafe { std::slice::from_raw_parts(data_buf as *const u8, count) };
        data.send(channel_sender).unwrap();
        if cfg!(feature = "async_api") {
            channel_sender.flush_out().unwrap();
        }
    }
    if cfg!(not(feature = "async_api")) {
        result.send(channel_sender).unwrap();
        channel_sender.flush_out().unwrap();
    }
    if cudaMemcpyKind::cudaMemcpyDeviceToHost == kind {
        unsafe { dealloc(data_buf, Layout::from_size_align(count, 1).unwrap()) };
    }
}

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
    let result = call_pos_process(
        server.pos_cuda_ws,
        proc_id,
        0u64,
        &[
            &raw const error as usize, size_of_val(&error),
        ],
    ) as *const i8;
    let result = unsafe { std::ffi::CStr::from_ptr(result).to_bytes().to_vec() };
    if let Err(e) = result.send(channel_sender) {
        error!("Error sending result: {:?}", e);
    }
    channel_sender.flush_out().unwrap();
}
