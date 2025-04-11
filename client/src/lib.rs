#![cfg_attr(feature = "passthrough", expect(dead_code))]
#![feature(thread_local)]

#[cfg(feature = "rdma")]
use network::ringbufferchannel::RDMAChannel;

use network::{
    ringbufferchannel::{EmulatorChannel, SHMChannel},
    Channel, Transportable,
};

#[cfg(not(feature = "passthrough"))]
mod hijack;
#[cfg(feature = "passthrough")]
mod passthrough;

mod elf;
use elf::{FatBinaryHeader, KernelParamInfo};

mod dl;

#[cfg(feature = "phos")]
mod phos;

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::c_char;
use std::io::Read as _;
use std::sync::RwLock;

use cudasys::types::cuda::{CUfunction, CUmodule};
type FatBinaryHandle = usize;
type HostPtr = usize;

struct ClientThread {
    id: i32,
    channel_sender: Channel,
    channel_receiver: Channel,
    resource_idx: usize,
    /// Used in `cuModuleLoadData` to judge if the image is a static fatbin.
    is_cuda_launch_kernel: bool,
    cuda_device: Option<std::ffi::c_int>,
    #[cfg(feature = "phos")]
    phos_agent: *mut std::ffi::c_void,
}

impl ClientThread {
    // Use features when compiling to decide what arm(s) will be supported.
    // In the client side, the sender's name is ctos_channel_name,
    // receiver's name is stoc_channel_name.
    fn new() -> Self {
        let config = &*network::CONFIG;
        let (id, channel_sender, channel_receiver) = match config.comm_type.as_str() {
            "shm" => {
                let id = {
                    let mut stream = std::net::TcpStream::connect(&config.daemon_socket).unwrap();
                    let mut buf = [0u8; 4];
                    stream.read_exact(&mut buf).unwrap();
                    i32::from_be_bytes(buf)
                };
                log::info!("Client id: {id}");
                let sender =
                    SHMChannel::new_client_with_id(&config.ctos_channel_name, id, config.buf_size)
                        .unwrap();
                let receiver =
                    SHMChannel::new_client_with_id(&config.stoc_channel_name, id, config.buf_size)
                        .unwrap();
                if cfg!(feature = "emulator") {
                    (
                        id,
                        Channel::new(Box::new(EmulatorChannel::new(Box::new(sender)))),
                        Channel::new(Box::new(EmulatorChannel::new(Box::new(receiver)))),
                    )
                } else {
                    (id, Channel::new(Box::new(sender)), Channel::new(Box::new(receiver)))
                }
            }
            #[cfg(feature = "rdma")]
            "rdma" => {
                // client side sender should connect to server's receiver socket.
                let sender = RDMAChannel::new_client(
                    &config.ctos_channel_name,
                    config.buf_size,
                    config.receiver_socket.parse().unwrap(),
                    1,
                )
                .unwrap();
                // client side receiver should connect to server's sender socket.
                let receiver = RDMAChannel::new_client(
                    &config.stoc_channel_name,
                    config.buf_size,
                    config.sender_socket.parse().unwrap(),
                    1,
                )
                .unwrap();
                (0, Channel::new(Box::new(sender)), Channel::new(Box::new(receiver)))
            }
            &_ => panic!("Unsupported communication type in config"),
        };

        Self {
            id,
            channel_sender,
            channel_receiver,
            resource_idx: 0,
            is_cuda_launch_kernel: false,
            cuda_device: None,
            #[cfg(feature = "phos")]
            phos_agent: unsafe { phos::pos_create_agent() },
        }
    }
}

impl Drop for ClientThread {
    fn drop(&mut self) {
        #[cfg(feature = "phos")]
        unsafe {
            phos::pos_destory_agent(self.phos_agent);
        }

        let proc_id = -1;
        proc_id.send(&self.channel_sender).unwrap();
    }
}

thread_local! {
    static CLIENT_THREAD: RefCell<ClientThread> = RefCell::new(ClientThread::new());
}

static DRIVER_CACHE: RwLock<DriverCache> = RwLock::new(DriverCache::new());
static RUNTIME_CACHE: RwLock<RuntimeCache> = RwLock::new(RuntimeCache::new());

struct DriverCache {
    /// Used in `cuModuleGetFunction`, populated by `cuModuleLoadData`.
    images: BTreeMap<CUmodule, Cow<'static, [u8]>>,
    /// Used in `cuLaunchKernel`, populated by `cuModuleGetFunction`.
    function_params: BTreeMap<CUfunction, Box<[KernelParamInfo]>>,
}

// The pointers are server-side.
unsafe impl Send for DriverCache {}
unsafe impl Sync for DriverCache {}

impl DriverCache {
    const fn new() -> Self {
        Self {
            images: BTreeMap::new(),
            function_params: BTreeMap::new(),
        }
    }
}

struct RuntimeCache {
    cuda_device: Option<std::ffi::c_int>,
    /// Populated by `__cudaRegisterFatBinary`.
    lazy_fatbins: Vec<*const FatBinaryHeader>,
    /// Populated by `__cudaRegisterFunction`.
    lazy_functions: BTreeMap<HostPtr, (FatBinaryHandle, *const c_char)>,
    /// Populated by `__cudaRegisterVar`.
    lazy_variables: BTreeMap<HostPtr, (FatBinaryHandle, *const c_char)>,
    /// Result of `cuModuleLoadData` calls.
    loaded_modules: BTreeMap<FatBinaryHandle, CUmodule>,
    /// Used in `cudaLaunchKernel`. Cache of `cuModuleGetFunction` calls.
    loaded_functions: BTreeMap<HostPtr, CUfunction>,
}

// The pointers are either static or server-side.
unsafe impl Send for RuntimeCache {}
unsafe impl Sync for RuntimeCache {}

impl RuntimeCache {
    const fn new() -> Self {
        Self {
            cuda_device: None,
            lazy_fatbins: Vec::new(),
            lazy_functions: BTreeMap::new(),
            lazy_variables: BTreeMap::new(),
            loaded_modules: BTreeMap::new(),
            loaded_functions: BTreeMap::new(),
        }
    }
}

#[small_ctor::ctor]
unsafe fn init() {
//     core_affinity::set_for_current(1);
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("debug"));
    log::info!("[{}:{}] client init", std::file!(), std::line!());
    for (i, arg) in std::env::args().enumerate() {
        log::info!("arg[{i}]: {arg}");
    }
    for (key, value) in std::env::vars() {
        if key.starts_with("LD_") || key.starts_with("RUST_") {
            log::info!("{key}: {value}");
        }
    }
}
