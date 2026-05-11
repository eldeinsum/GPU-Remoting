mod dispatcher;

use cudasys::{
    cuda::{CUlibrary, CUlinkState, CUmodule},
    cudart::{cudaError_t, cudaGetDeviceCount},
};
use dispatcher::dispatch;

#[cfg(feature = "rdma")]
use network::ringbufferchannel::RDMAChannel;

use network::ringbufferchannel::{EmulatorChannel, SHMChannel};
use network::{Channel, CommChannel, CommChannelError, NetworkConfig, Transportable, tcp};

use log::{error, info};

use std::collections::BTreeMap;
use std::io;
use std::sync::{Arc, Barrier};

struct ServerWorker<C> {
    pub id: i32,
    pub channel_sender: C,
    pub channel_receiver: C,
    pub modules: Vec<CUmodule>,
    pub libraries: Vec<CUlibrary>,
    pub links: Vec<CUlinkState>,
    pub resources: BTreeMap<usize, usize>,
    opt_async_api: bool,
    opt_shadow_desc: bool,
}

impl<C> Drop for ServerWorker<C> {
    fn drop(&mut self) {
        for module in &self.modules {
            unsafe {
                cudasys::cuda::cuModuleUnload(*module);
            }
        }
        for library in &self.libraries {
            unsafe {
                cudasys::cuda::cuLibraryUnload(*library);
            }
        }
        for link in &self.links {
            unsafe {
                cudasys::cuda::cuLinkDestroy(*link);
            }
        }
    }
}

fn create_buffer(
    config: &NetworkConfig,
    id: i32,
    barrier: Option<Arc<Barrier>>,
) -> io::Result<(Channel, Channel)> {
    // Use features when compiling to decide what arm(s) will be supported.
    // In the server side, the sender's name is stoc_channel_name,
    // receiver's name is ctos_channel_name.
    match config.comm_type.as_str() {
        "shm" => {
            let (receiver, sender) = SHMChannel::new_server_with_id(config, id)?;
            let barrier = barrier.ok_or_else(|| missing_barrier("shm"))?;
            barrier.wait();
            if config.emulator {
                return Ok((
                    Channel::new(Box::new(EmulatorChannel::new(sender, config))),
                    Channel::new(Box::new(EmulatorChannel::new(receiver, config))),
                ));
            }
            Ok((
                Channel::new(Box::new(sender)),
                Channel::new(Box::new(receiver)),
            ))
        }
        "tcp" => {
            let barrier = barrier.ok_or_else(|| missing_barrier("tcp"))?;
            let (receiver, sender) = tcp::new_server(config, id, &barrier)?;
            Ok((
                Channel::new(Box::new(sender)),
                Channel::new(Box::new(receiver)),
            ))
        }
        #[cfg(feature = "rdma")]
        "rdma" => {
            if barrier.is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "rdma server channel should not use a startup barrier",
                ));
            }
            let (receiver, sender) = RDMAChannel::new_server(config, id);
            Ok((
                Channel::new(Box::new(sender)),
                Channel::new(Box::new(receiver)),
            ))
        }
        other => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unsupported communication type in config: {other}"),
        )),
    }
}

fn receive_request<T: CommChannel>(channel_receiver: &mut T) -> Result<i32, CommChannelError> {
    let mut proc_id = 0;
    if let Ok(()) = proc_id.recv(channel_receiver) {
        Ok(proc_id)
    } else {
        Err(CommChannelError::IoError)
    }
}

pub fn launch_server(
    config: &NetworkConfig,
    id: i32,
    barrier: Option<Arc<Barrier>>,
    is_main_thread: bool,
) {
    let barrier_on_error = barrier.clone();
    let (channel_sender, channel_receiver) = match create_buffer(config, id, barrier) {
        Ok(channels) => channels,
        Err(err) => {
            error!(
                "[{}:{}] failed to create {} buffer: {}",
                std::file!(),
                std::line!(),
                config.comm_type,
                err
            );
            if let Some(barrier) = barrier_on_error {
                barrier.wait();
            }
            return;
        }
    };
    info!(
        "[{}:{}] {} buffer created",
        std::file!(),
        std::line!(),
        config.comm_type
    );
    let mut max_devices = 0;
    if let cudaError_t::cudaSuccess =
        unsafe { cudaGetDeviceCount(&mut max_devices as *mut ::std::os::raw::c_int) }
    {
        info!(
            "[{}:{}] found {} cuda devices",
            std::file!(),
            std::line!(),
            max_devices
        );
    } else {
        error!(
            "[{}:{}] failed to find cuda devices",
            std::file!(),
            std::line!()
        );
        return;
    }

    let mut server = ServerWorker {
        id,
        channel_sender,
        channel_receiver,
        modules: Default::default(),
        libraries: Default::default(),
        links: Default::default(),
        resources: Default::default(),
        opt_async_api: config.opt_async_api,
        opt_shadow_desc: config.opt_shadow_desc,
    };

    loop {
        if let Ok(proc_id) = receive_request(&mut server.channel_receiver) {
            if proc_id == -1 {
                info!("[#{}] received client shutdown", server.id);
                break;
            }
            log::debug!("[#{}] dispatching proc_id {}", server.id, proc_id);
            if !dispatch(proc_id, &mut server) {
                error!(
                    "[#{}] stopping after invalid proc_id {}",
                    server.id, proc_id
                );
                break;
            }
            log::debug!("[#{}] completed proc_id {}", server.id, proc_id);
        } else {
            error!(
                "[{}:{}] failed to receive request",
                std::file!(),
                std::line!()
            );
            break;
        }
    }

    info!(
        "[{}:{}] server #{} terminated",
        std::file!(),
        std::line!(),
        server.id
    );

    if is_main_thread {
        std::process::exit(0);
    }
}

fn missing_barrier(comm_type: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("{comm_type} server channel requires a startup barrier"),
    )
}
