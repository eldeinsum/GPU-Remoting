#![cfg_attr(feature = "passthrough", expect(dead_code))]
#![feature(thread_local)]

#[cfg(feature = "rdma")]
use network::ringbufferchannel::RDMAChannel;

use network::ringbufferchannel::{EmulatorChannel, SHMChannel};
use network::{Channel, CommChannel, Transportable, tcp};

#[cfg(not(feature = "passthrough"))]
mod hijack;
#[cfg(feature = "passthrough")]
mod passthrough;

mod elf;
use elf::{FatBinaryHeader, KernelParamInfo};

mod dl;

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::{CString, c_char, c_void};
use std::io::{Read as _, Write as _};
use std::sync::RwLock;

use cudasys::types::cublas::{cublasHandle_t, cublasPointerMode_t};
use cudasys::types::cublasLt::{
    cublasLtMatmulDesc_t, cublasLtMatrixTransformDesc_t, cublasLtPointerMode_t,
    cudaDataType_t as CublasLtCudaDataType,
};
use cudasys::types::cuda::{
    CUDA_BATCH_MEM_OP_NODE_PARAMS, CUDA_KERNEL_NODE_PARAMS, CUfunction, CUgraphNode, CUkernel,
    CUlibrary, CUmodule, CUstreamBatchMemOpParams,
};
use cudasys::types::cudart::{cudaGraphNode_t, cudaKernelNodeParams};
type FatBinaryHandle = usize;
type HostPtr = usize;

struct ClientThread {
    pid: u32,
    id: i32,
    channel_sender: Channel,
    channel_receiver: Channel,
    resource_idx: usize,
    cuda_device: Option<std::ffi::c_int>,
    cuda_device_init: bool,
    opt_async_api: bool,
    opt_shadow_desc: bool,
    opt_local: bool,
}

impl ClientThread {
    // Use features when compiling to decide what arm(s) will be supported.
    // In the client side, the sender's name is ctos_channel_name,
    // receiver's name is stoc_channel_name.
    fn new() -> Self {
        log::info!("[{}:{}] client init", std::file!(), std::line!());
        for (i, arg) in std::env::args().enumerate() {
            log::info!("arg[{i}]: {arg}");
        }
        for (key, value) in std::env::vars() {
            if key.starts_with("LD_") || key.starts_with("RUST_") {
                log::info!("{key}: {value}");
            }
        }
        let config = network::NetworkConfig::read_from_file();
        let id = {
            let mut stream = std::net::TcpStream::connect(&config.daemon_socket).unwrap();
            stream.write_all(&std::process::id().to_be_bytes()).unwrap();
            let mut buf = [0u8; 4];
            stream.read_exact(&mut buf).unwrap();
            i32::from_be_bytes(buf)
        };
        log::info!(
            "[#{id}] PID = {}, {:?}",
            std::process::id(),
            std::thread::current().id()
        );
        let (channel_sender, channel_receiver) = match config.comm_type.as_str() {
            "shm" => {
                let (sender, receiver) = SHMChannel::new_client_with_id(&config, id).unwrap();
                if config.emulator {
                    (
                        Channel::new(Box::new(EmulatorChannel::new(sender, &config))),
                        Channel::new(Box::new(EmulatorChannel::new(receiver, &config))),
                    )
                } else {
                    (
                        Channel::new(Box::new(sender)),
                        Channel::new(Box::new(receiver)),
                    )
                }
            }
            "tcp" => {
                let (sender, receiver) = tcp::new_client(&config, id).unwrap();
                (
                    Channel::new(Box::new(sender)),
                    Channel::new(Box::new(receiver)),
                )
            }
            #[cfg(feature = "rdma")]
            "rdma" => {
                let (sender, receiver) = RDMAChannel::new_client(&config, id);
                (
                    Channel::new(Box::new(sender)),
                    Channel::new(Box::new(receiver)),
                )
            }
            &_ => panic!("Unsupported communication type in config"),
        };

        unsafe {
            // HACK: should just send something to the daemon socket
            fn atsignal(_info: &libc::siginfo_t) {
                std::process::exit(0);
            }
            signal_hook_registry::register_sigaction(libc::SIGQUIT, atsignal).unwrap();
            signal_hook_registry::register_sigaction(libc::SIGTERM, atsignal).unwrap();
        }

        Self {
            pid: std::process::id(),
            id,
            channel_sender,
            channel_receiver,
            resource_idx: 0,
            cuda_device: None,
            cuda_device_init: false,
            opt_async_api: config.opt_async_api,
            opt_shadow_desc: config.opt_shadow_desc,
            opt_local: config.opt_local,
        }
    }

    fn ensure_current_process(&mut self) {
        if self.pid == std::process::id() {
            return;
        }

        DRIVER_CACHE.write().unwrap().reset_after_fork();
        RUNTIME_CACHE.write().unwrap().reset_after_fork();
        CUBLAS_CACHE.write().unwrap().reset_after_fork();
        let stale = std::mem::replace(self, ClientThread::new());
        std::mem::forget(stale);
    }
}

impl Drop for ClientThread {
    fn drop(&mut self) {
        if self.pid != std::process::id() {
            return;
        }
        let proc_id = -1;
        proc_id.send(&self.channel_sender).unwrap();
        self.channel_sender.flush_out().unwrap();
    }
}

thread_local! {
    static CLIENT_THREAD: RefCell<ClientThread> = RefCell::new(ClientThread::new());
}

static DRIVER_CACHE: RwLock<DriverCache> = RwLock::new(DriverCache::new());
static RUNTIME_CACHE: RwLock<RuntimeCache> = RwLock::new(RuntimeCache::new());
static CUBLAS_CACHE: RwLock<CublasCache> = RwLock::new(CublasCache::new());

struct DriverCache {
    /// Used in `cuModuleGetFunction`, populated by `cuModuleLoadData`.
    images: BTreeMap<CUmodule, Cow<'static, [u8]>>,
    /// Used in `cuLibraryGetKernel`, populated by `cuLibraryLoadData`.
    library_images: BTreeMap<CUlibrary, Cow<'static, [u8]>>,
    /// Used in `cuLaunchKernel`, populated by `cuModuleGetFunction`.
    function_params: BTreeMap<CUfunction, Box<[KernelParamInfo]>>,
    /// Client-side copies of kernel node parameters whose raw pointers are process-local.
    graph_kernel_nodes: BTreeMap<CUgraphNode, GraphKernelNodeCache>,
    /// Client-side copies of batch mem-op node arrays whose raw pointers are process-local.
    graph_batch_mem_op_nodes: BTreeMap<CUgraphNode, GraphBatchMemOpNodeCache>,
    /// Used in `cuKernelGetName`, populated by `cuLibraryGetKernel`.
    kernel_names: BTreeMap<CUkernel, CString>,
    /// Used to remove library-owned kernel metadata on unload.
    kernel_libraries: BTreeMap<CUkernel, CUlibrary>,
    device_arch: Option<u32>,
}

// The pointers are server-side.
unsafe impl Send for DriverCache {}
unsafe impl Sync for DriverCache {}

impl DriverCache {
    const fn new() -> Self {
        Self {
            images: BTreeMap::new(),
            library_images: BTreeMap::new(),
            function_params: BTreeMap::new(),
            graph_kernel_nodes: BTreeMap::new(),
            graph_batch_mem_op_nodes: BTreeMap::new(),
            kernel_names: BTreeMap::new(),
            kernel_libraries: BTreeMap::new(),
            device_arch: None,
        }
    }

    fn reset_after_fork(&mut self) {
        self.images.clear();
        self.library_images.clear();
        self.function_params.clear();
        self.graph_kernel_nodes.clear();
        self.graph_batch_mem_op_nodes.clear();
        self.kernel_names.clear();
        self.kernel_libraries.clear();
        self.device_arch = None;
    }
}

struct GraphKernelNodeCache {
    params: CUDA_KERNEL_NODE_PARAMS,
    args: Box<[u8]>,
    arg_offsets: Box<[u32]>,
    kernel_param_ptrs: Box<[*mut c_void]>,
}

impl GraphKernelNodeCache {
    fn new(mut params: CUDA_KERNEL_NODE_PARAMS, args: Box<[u8]>, arg_offsets: Box<[u32]>) -> Self {
        params.kernelParams = std::ptr::null_mut();
        params.extra = std::ptr::null_mut();
        let mut cache = Self {
            params,
            args,
            arg_offsets,
            kernel_param_ptrs: Box::default(),
        };
        cache.refresh_kernel_param_ptrs();
        cache
    }

    fn refresh_kernel_param_ptrs(&mut self) {
        self.kernel_param_ptrs = self
            .arg_offsets
            .iter()
            .map(|offset| unsafe {
                self.args
                    .as_mut_ptr()
                    .add(*offset as usize)
                    .cast::<c_void>()
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
    }

    fn params_for_client(&mut self) -> CUDA_KERNEL_NODE_PARAMS {
        let mut params = self.params;
        params.kernelParams = if self.kernel_param_ptrs.is_empty() {
            std::ptr::null_mut()
        } else {
            self.kernel_param_ptrs.as_mut_ptr()
        };
        params.extra = std::ptr::null_mut();
        params
    }
}

struct GraphBatchMemOpNodeCache {
    params: CUDA_BATCH_MEM_OP_NODE_PARAMS,
    ops: Box<[CUstreamBatchMemOpParams]>,
}

impl GraphBatchMemOpNodeCache {
    fn new(
        mut params: CUDA_BATCH_MEM_OP_NODE_PARAMS,
        ops: Box<[CUstreamBatchMemOpParams]>,
    ) -> Self {
        params.paramArray = std::ptr::null_mut();
        Self { params, ops }
    }

    fn params_for_client(&mut self) -> CUDA_BATCH_MEM_OP_NODE_PARAMS {
        let mut params = self.params;
        params.paramArray = if self.ops.is_empty() {
            std::ptr::null_mut()
        } else {
            self.ops.as_mut_ptr()
        };
        params
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
    /// Client-side copies of runtime graph kernel node parameters.
    graph_kernel_nodes: BTreeMap<cudaGraphNode_t, RuntimeGraphKernelNodeCache>,
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
            graph_kernel_nodes: BTreeMap::new(),
        }
    }

    fn reset_after_fork(&mut self) {
        self.cuda_device = None;
        self.loaded_modules.clear();
        self.loaded_functions.clear();
        self.graph_kernel_nodes.clear();
    }
}

struct RuntimeGraphKernelNodeCache {
    params: cudaKernelNodeParams,
    args: Box<[u8]>,
    arg_offsets: Box<[u32]>,
    kernel_param_ptrs: Box<[*mut c_void]>,
}

impl RuntimeGraphKernelNodeCache {
    fn new(mut params: cudaKernelNodeParams, args: Box<[u8]>, arg_offsets: Box<[u32]>) -> Self {
        params.kernelParams = std::ptr::null_mut();
        params.extra = std::ptr::null_mut();
        let mut cache = Self {
            params,
            args,
            arg_offsets,
            kernel_param_ptrs: Box::default(),
        };
        cache.refresh_kernel_param_ptrs();
        cache
    }

    fn refresh_kernel_param_ptrs(&mut self) {
        self.kernel_param_ptrs = self
            .arg_offsets
            .iter()
            .map(|offset| unsafe {
                self.args
                    .as_mut_ptr()
                    .add(*offset as usize)
                    .cast::<c_void>()
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
    }

    fn params_for_client(&mut self) -> cudaKernelNodeParams {
        let mut params = self.params;
        params.kernelParams = if self.kernel_param_ptrs.is_empty() {
            std::ptr::null_mut()
        } else {
            self.kernel_param_ptrs.as_mut_ptr()
        };
        params.extra = std::ptr::null_mut();
        params
    }
}

struct CublasCache {
    pointer_modes: BTreeMap<cublasHandle_t, cublasPointerMode_t>,
    lt_matmul_descs: BTreeMap<cublasLtMatmulDesc_t, CublasLtMatmulDescState>,
    lt_transform_descs: BTreeMap<cublasLtMatrixTransformDesc_t, CublasLtTransformDescState>,
}

// The handles are server-side.
unsafe impl Send for CublasCache {}
unsafe impl Sync for CublasCache {}

impl CublasCache {
    const fn new() -> Self {
        Self {
            pointer_modes: BTreeMap::new(),
            lt_matmul_descs: BTreeMap::new(),
            lt_transform_descs: BTreeMap::new(),
        }
    }

    fn reset_after_fork(&mut self) {
        self.pointer_modes.clear();
        self.lt_matmul_descs.clear();
        self.lt_transform_descs.clear();
    }
}

#[derive(Copy, Clone)]
struct CublasLtMatmulDescState {
    pointer_mode: cublasLtPointerMode_t,
    scale_type_size: Option<usize>,
}

#[derive(Copy, Clone)]
struct CublasLtTransformDescState {
    pointer_mode: cublasLtPointerMode_t,
    scale_type_size: Option<usize>,
}

fn cublaslt_pointer_mode_from_u32(value: u32) -> Option<cublasLtPointerMode_t> {
    Some(match value {
        0 => cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST,
        1 => cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_DEVICE,
        2 => cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_DEVICE_VECTOR,
        3 => cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO,
        4 => cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST,
        _ => return None,
    })
}

fn cublaslt_scale_type_from_u32(value: u32) -> Option<CublasLtCudaDataType> {
    Some(match value {
        0 => CublasLtCudaDataType::CUDA_R_32F,
        1 => CublasLtCudaDataType::CUDA_R_64F,
        2 => CublasLtCudaDataType::CUDA_R_16F,
        3 => CublasLtCudaDataType::CUDA_R_8I,
        4 => CublasLtCudaDataType::CUDA_C_32F,
        5 => CublasLtCudaDataType::CUDA_C_64F,
        6 => CublasLtCudaDataType::CUDA_C_16F,
        7 => CublasLtCudaDataType::CUDA_C_8I,
        8 => CublasLtCudaDataType::CUDA_R_8U,
        9 => CublasLtCudaDataType::CUDA_C_8U,
        10 => CublasLtCudaDataType::CUDA_R_32I,
        11 => CublasLtCudaDataType::CUDA_C_32I,
        12 => CublasLtCudaDataType::CUDA_R_32U,
        13 => CublasLtCudaDataType::CUDA_C_32U,
        14 => CublasLtCudaDataType::CUDA_R_16BF,
        15 => CublasLtCudaDataType::CUDA_C_16BF,
        16 => CublasLtCudaDataType::CUDA_R_4I,
        17 => CublasLtCudaDataType::CUDA_C_4I,
        18 => CublasLtCudaDataType::CUDA_R_4U,
        19 => CublasLtCudaDataType::CUDA_C_4U,
        20 => CublasLtCudaDataType::CUDA_R_16I,
        21 => CublasLtCudaDataType::CUDA_C_16I,
        22 => CublasLtCudaDataType::CUDA_R_16U,
        23 => CublasLtCudaDataType::CUDA_C_16U,
        24 => CublasLtCudaDataType::CUDA_R_64I,
        25 => CublasLtCudaDataType::CUDA_C_64I,
        26 => CublasLtCudaDataType::CUDA_R_64U,
        27 => CublasLtCudaDataType::CUDA_C_64U,
        28 => CublasLtCudaDataType::CUDA_R_8F_E4M3,
        29 => CublasLtCudaDataType::CUDA_R_8F_E5M2,
        30 => CublasLtCudaDataType::CUDA_R_8F_UE8M0,
        31 => CublasLtCudaDataType::CUDA_R_6F_E2M3,
        32 => CublasLtCudaDataType::CUDA_R_6F_E3M2,
        33 => CublasLtCudaDataType::CUDA_R_4F_E2M1,
        _ => return None,
    })
}

fn cublaslt_scale_type_size(scale_type: CublasLtCudaDataType) -> Option<usize> {
    Some(match scale_type {
        CublasLtCudaDataType::CUDA_R_8I
        | CublasLtCudaDataType::CUDA_R_8U
        | CublasLtCudaDataType::CUDA_R_8F_E4M3
        | CublasLtCudaDataType::CUDA_R_8F_E5M2
        | CublasLtCudaDataType::CUDA_R_8F_UE8M0 => 1,
        CublasLtCudaDataType::CUDA_R_16F
        | CublasLtCudaDataType::CUDA_R_16BF
        | CublasLtCudaDataType::CUDA_R_16I
        | CublasLtCudaDataType::CUDA_R_16U
        | CublasLtCudaDataType::CUDA_C_8I
        | CublasLtCudaDataType::CUDA_C_8U => 2,
        CublasLtCudaDataType::CUDA_R_32F
        | CublasLtCudaDataType::CUDA_R_32I
        | CublasLtCudaDataType::CUDA_R_32U
        | CublasLtCudaDataType::CUDA_C_16F
        | CublasLtCudaDataType::CUDA_C_16BF
        | CublasLtCudaDataType::CUDA_C_16I
        | CublasLtCudaDataType::CUDA_C_16U => 4,
        CublasLtCudaDataType::CUDA_R_64F
        | CublasLtCudaDataType::CUDA_R_64I
        | CublasLtCudaDataType::CUDA_R_64U
        | CublasLtCudaDataType::CUDA_C_32F
        | CublasLtCudaDataType::CUDA_C_32I
        | CublasLtCudaDataType::CUDA_C_32U => 8,
        CublasLtCudaDataType::CUDA_C_64F
        | CublasLtCudaDataType::CUDA_C_64I
        | CublasLtCudaDataType::CUDA_C_64U => 16,
        CublasLtCudaDataType::CUDA_R_4I
        | CublasLtCudaDataType::CUDA_C_4I
        | CublasLtCudaDataType::CUDA_R_4U
        | CublasLtCudaDataType::CUDA_C_4U
        | CublasLtCudaDataType::CUDA_R_6F_E2M3
        | CublasLtCudaDataType::CUDA_R_6F_E3M2
        | CublasLtCudaDataType::CUDA_R_4F_E2M1 => return None,
    })
}

#[small_ctor::ctor]
unsafe fn init() {
    //     core_affinity::set_for_current(1);
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
}
