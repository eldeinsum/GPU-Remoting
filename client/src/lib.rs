#![cfg_attr(feature = "passthrough", expect(dead_code))]
#![feature(thread_local)]

#[cfg(feature = "rdma")]
use network::ringbufferchannel::RDMAChannel;

use network::ringbufferchannel::{EmulatorChannel, SHMChannel};
use network::{tcp, Channel, CommChannel, Transportable};

#[cfg(not(feature = "passthrough"))]
mod hijack;
#[cfg(feature = "passthrough")]
mod passthrough;

mod elf;
use elf::{FatBinaryHeader, KernelParamInfo};

mod dl;

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::ffi::{c_char, c_int, c_void, CString};
use std::io::{Read as _, Write as _};
use std::net::Shutdown;
use std::sync::{Mutex, OnceLock, RwLock};

use cudasys::types::cublas::{cublasHandle_t, cublasPointerMode_t};
use cudasys::types::cublasLt::{
    cublasLtEmulationDesc_t, cublasLtMatmulDesc_t, cublasLtMatmulPreference_t,
    cublasLtMatrixLayout_t, cublasLtMatrixTransformDesc_t, cublasLtPointerMode_t,
    cudaDataType_t as CublasLtCudaDataType,
};
use cudasys::types::cuda::{
    CUfunction, CUgraphNode, CUkernel, CUlibrary, CUlinkState, CUmodule, CUstreamBatchMemOpParams,
    CUDA_BATCH_MEM_OP_NODE_PARAMS, CUDA_KERNEL_NODE_PARAMS,
};
use cudasys::types::cudart::{cudaGraphNode_t, cudaKernelNodeParams};
use cudasys::types::cudnn::{cudnnDataType_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t};
type FatBinaryHandle = usize;
type HostPtr = usize;

struct ClientThread {
    pid: u32,
    id: i32,
    daemon_stream: std::net::TcpStream,
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
        let (id, daemon_stream) = {
            let mut stream = std::net::TcpStream::connect(&config.daemon_socket).unwrap();
            stream.write_all(&std::process::id().to_be_bytes()).unwrap();
            let mut buf = [0u8; 4];
            stream.read_exact(&mut buf).unwrap();
            (i32::from_be_bytes(buf), stream)
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
            daemon_stream,
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
        CUDNN_CACHE.write().unwrap().reset_after_fork();
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
        let _ = self.daemon_stream.shutdown(Shutdown::Both);
    }
}

struct ClientThreadState {
    inner: OnceLock<Mutex<Option<ClientThread>>>,
}

impl ClientThreadState {
    const fn new() -> Self {
        Self {
            inner: OnceLock::new(),
        }
    }

    fn with_borrow_mut<R>(&self, f: impl FnOnce(&mut ClientThread) -> R) -> R {
        let mutex = self.inner.get_or_init(|| {
            unsafe {
                libc::atexit(shutdown_client_thread_state);
            }
            Mutex::new(Some(ClientThread::new()))
        });
        let mut guard = mutex
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let client = guard.get_or_insert_with(ClientThread::new);
        client.ensure_current_process();
        f(client)
    }

    fn shutdown(&self) {
        let Some(mutex) = self.inner.get() else {
            return;
        };
        if let Ok(mut guard) = mutex.lock() {
            drop(guard.take());
        }
    }
}

extern "C" fn shutdown_client_thread_state() {
    CLIENT_THREAD.shutdown();
}

static CLIENT_THREAD: ClientThreadState = ClientThreadState::new();

static DRIVER_CACHE: RwLock<DriverCache> = RwLock::new(DriverCache::new());
static RUNTIME_CACHE: RwLock<RuntimeCache> = RwLock::new(RuntimeCache::new());
static CUBLAS_CACHE: RwLock<CublasCache> = RwLock::new(CublasCache::new());
static CUDNN_CACHE: RwLock<CudnnCache> = RwLock::new(CudnnCache::new());

struct DriverCache {
    /// Used in `cuModuleGetFunction`, populated by `cuModuleLoadData`.
    images: BTreeMap<CUmodule, Cow<'static, [u8]>>,
    /// Used in `cuLibraryGetKernel`, populated by `cuLibraryLoadData`.
    library_images: BTreeMap<CUlibrary, Cow<'static, [u8]>>,
    /// Client-owned outputs from `cuLinkComplete`, valid until `cuLinkDestroy`.
    linked_images: BTreeMap<CUlinkState, Box<[u8]>>,
    /// Used in `cuLaunchKernel`, populated by `cuModuleGetFunction`.
    function_params: BTreeMap<CUfunction, Box<[KernelParamInfo]>>,
    /// Used in `cuFuncGetName`, populated by function lookup APIs.
    function_names: BTreeMap<CUfunction, CString>,
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
            linked_images: BTreeMap::new(),
            function_params: BTreeMap::new(),
            function_names: BTreeMap::new(),
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
        self.linked_images.clear();
        self.function_params.clear();
        self.function_names.clear();
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
    /// Keeps client-side placeholders for managed variable host pointers alive.
    managed_variable_shadows: BTreeMap<HostPtr, ManagedVariableShadow>,
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
            managed_variable_shadows: BTreeMap::new(),
            loaded_modules: BTreeMap::new(),
            loaded_functions: BTreeMap::new(),
            graph_kernel_nodes: BTreeMap::new(),
        }
    }

    fn reset_after_fork(&mut self) {
        self.cuda_device = None;
        self.clear_context_bound_state();
    }

    fn set_cuda_device(&mut self, device: std::ffi::c_int) {
        if self.cuda_device == Some(device) {
            return;
        }
        self.cuda_device = Some(device);
        self.clear_context_bound_state();
    }

    fn clear_context_bound_state(&mut self) {
        self.loaded_modules.clear();
        self.loaded_functions.clear();
        self.graph_kernel_nodes.clear();
    }
}

struct ManagedVariableShadow {
    bytes: Box<[u8]>,
    synced_bytes: Box<[u8]>,
    pending_device_write: bool,
    size: usize,
}

impl ManagedVariableShadow {
    fn ptr(&self) -> HostPtr {
        self.bytes.as_ptr() as HostPtr
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
    lt_matmul_desc_handles: BTreeMap<cublasLtMatmulDesc_t, cublasLtMatmulDesc_t>,
    lt_matmul_preference_handles: BTreeMap<cublasLtMatmulPreference_t, cublasLtMatmulPreference_t>,
    lt_matrix_layout_handles: BTreeMap<cublasLtMatrixLayout_t, cublasLtMatrixLayout_t>,
    lt_emulation_desc_handles: BTreeMap<cublasLtEmulationDesc_t, cublasLtEmulationDesc_t>,
    lt_transform_descs: BTreeMap<cublasLtMatrixTransformDesc_t, CublasLtTransformDescState>,
    lt_transform_desc_handles:
        BTreeMap<cublasLtMatrixTransformDesc_t, cublasLtMatrixTransformDesc_t>,
}

// The handles are server-side.
unsafe impl Send for CublasCache {}
unsafe impl Sync for CublasCache {}

impl CublasCache {
    const fn new() -> Self {
        Self {
            pointer_modes: BTreeMap::new(),
            lt_matmul_descs: BTreeMap::new(),
            lt_matmul_desc_handles: BTreeMap::new(),
            lt_matmul_preference_handles: BTreeMap::new(),
            lt_matrix_layout_handles: BTreeMap::new(),
            lt_emulation_desc_handles: BTreeMap::new(),
            lt_transform_descs: BTreeMap::new(),
            lt_transform_desc_handles: BTreeMap::new(),
        }
    }

    fn reset_after_fork(&mut self) {
        self.pointer_modes.clear();
        self.lt_matmul_descs.clear();
        self.lt_matmul_desc_handles.clear();
        self.lt_matmul_preference_handles.clear();
        self.lt_matrix_layout_handles.clear();
        self.lt_emulation_desc_handles.clear();
        self.lt_transform_descs.clear();
        self.lt_transform_desc_handles.clear();
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

struct CudnnTensorDescState {
    data_type: cudnnDataType_t,
    dims: Vec<c_int>,
}

struct CudnnCache {
    tensor_descs: BTreeMap<cudnnTensorDescriptor_t, CudnnTensorDescState>,
    filter_desc_types: BTreeMap<cudnnFilterDescriptor_t, cudnnDataType_t>,
}

unsafe impl Send for CudnnCache {}
unsafe impl Sync for CudnnCache {}

impl CudnnCache {
    const fn new() -> Self {
        Self {
            tensor_descs: BTreeMap::new(),
            filter_desc_types: BTreeMap::new(),
        }
    }

    fn reset_after_fork(&mut self) {
        self.tensor_descs.clear();
        self.filter_desc_types.clear();
    }
}

fn cudnn_data_type_scalar_size(data_type: cudnnDataType_t) -> Option<usize> {
    Some(match data_type {
        cudnnDataType_t::CUDNN_DATA_DOUBLE | cudnnDataType_t::CUDNN_DATA_INT64 => 8,
        cudnnDataType_t::CUDNN_DATA_FLOAT
        | cudnnDataType_t::CUDNN_DATA_INT32
        | cudnnDataType_t::CUDNN_DATA_UINT32
        | cudnnDataType_t::CUDNN_DATA_FAST_FLOAT_FOR_FP8 => 4,
        cudnnDataType_t::CUDNN_DATA_HALF | cudnnDataType_t::CUDNN_DATA_BFLOAT16 => 2,
        cudnnDataType_t::CUDNN_DATA_INT8
        | cudnnDataType_t::CUDNN_DATA_UINT8
        | cudnnDataType_t::CUDNN_DATA_BOOLEAN
        | cudnnDataType_t::CUDNN_DATA_FP8_E4M3
        | cudnnDataType_t::CUDNN_DATA_FP8_E5M2
        | cudnnDataType_t::CUDNN_DATA_FP8_E8M0 => 1,
        cudnnDataType_t::CUDNN_DATA_COMPLEX_FP32 => 8,
        cudnnDataType_t::CUDNN_DATA_COMPLEX_FP64 => 16,
        cudnnDataType_t::CUDNN_DATA_INT8x4
        | cudnnDataType_t::CUDNN_DATA_UINT8x4
        | cudnnDataType_t::CUDNN_DATA_INT8x32
        | cudnnDataType_t::CUDNN_DATA_FP4_E2M1
        | cudnnDataType_t::CUDNN_DATA_INT4
        | cudnnDataType_t::CUDNN_DATA_UINT4 => return None,
    })
}

fn cudnn_record_tensor_desc(
    desc: cudnnTensorDescriptor_t,
    data_type: cudnnDataType_t,
    dims: &[c_int],
) {
    CUDNN_CACHE.write().unwrap().tensor_descs.insert(
        desc,
        CudnnTensorDescState {
            data_type,
            dims: dims.to_vec(),
        },
    );
}

fn cudnn_remove_tensor_desc(desc: cudnnTensorDescriptor_t) {
    CUDNN_CACHE.write().unwrap().tensor_descs.remove(&desc);
}

fn cudnn_tensor_desc_scalar_size(desc: cudnnTensorDescriptor_t) -> usize {
    CUDNN_CACHE
        .read()
        .unwrap()
        .tensor_descs
        .get(&desc)
        .and_then(|state| cudnn_data_type_scalar_size(state.data_type))
        .unwrap_or(std::mem::size_of::<f32>())
}

fn cudnn_tensor_desc_dim(desc: cudnnTensorDescriptor_t, index: usize) -> Option<c_int> {
    CUDNN_CACHE
        .read()
        .unwrap()
        .tensor_descs
        .get(&desc)
        .and_then(|state| state.dims.get(index).copied())
}

fn cudnn_ctc_batch_size(probs_desc: cudnnTensorDescriptor_t) -> usize {
    cudnn_tensor_desc_dim(probs_desc, 1)
        .and_then(|dim| usize::try_from(dim).ok())
        .filter(|dim| *dim > 0)
        .unwrap_or(0)
}

fn cudnn_ctc_label_count(
    probs_desc: cudnnTensorDescriptor_t,
    label_lengths: *const c_int,
) -> usize {
    let batch_size = cudnn_ctc_batch_size(probs_desc);
    if batch_size == 0 || label_lengths.is_null() {
        return 0;
    }
    let label_lengths = unsafe { std::slice::from_raw_parts(label_lengths, batch_size) };
    label_lengths
        .iter()
        .filter_map(|len| usize::try_from(*len).ok())
        .sum()
}

fn cudnn_record_filter_desc_type(desc: cudnnFilterDescriptor_t, data_type: cudnnDataType_t) {
    CUDNN_CACHE
        .write()
        .unwrap()
        .filter_desc_types
        .insert(desc, data_type);
}

fn cudnn_remove_filter_desc(desc: cudnnFilterDescriptor_t) {
    CUDNN_CACHE.write().unwrap().filter_desc_types.remove(&desc);
}

fn cudnn_filter_desc_scalar_size(desc: cudnnFilterDescriptor_t) -> usize {
    CUDNN_CACHE
        .read()
        .unwrap()
        .filter_desc_types
        .get(&desc)
        .and_then(|data_type| cudnn_data_type_scalar_size(*data_type))
        .unwrap_or(std::mem::size_of::<f32>())
}

fn cublaslt_bind_matmul_desc(client: cublasLtMatmulDesc_t, server: cublasLtMatmulDesc_t) {
    CUBLAS_CACHE
        .write()
        .unwrap()
        .lt_matmul_desc_handles
        .insert(client, server);
}

fn cublaslt_resolve_matmul_desc(desc: cublasLtMatmulDesc_t) -> cublasLtMatmulDesc_t {
    CUBLAS_CACHE
        .read()
        .unwrap()
        .lt_matmul_desc_handles
        .get(&desc)
        .copied()
        .unwrap_or(desc)
}

fn cublaslt_unbind_matmul_desc(desc: cublasLtMatmulDesc_t) {
    let mut cache = CUBLAS_CACHE.write().unwrap();
    cache.lt_matmul_desc_handles.remove(&desc);
    cache.lt_matmul_descs.remove(&desc);
}

fn cublaslt_bind_matmul_preference(
    client: cublasLtMatmulPreference_t,
    server: cublasLtMatmulPreference_t,
) {
    CUBLAS_CACHE
        .write()
        .unwrap()
        .lt_matmul_preference_handles
        .insert(client, server);
}

fn cublaslt_resolve_matmul_preference(
    pref: cublasLtMatmulPreference_t,
) -> cublasLtMatmulPreference_t {
    CUBLAS_CACHE
        .read()
        .unwrap()
        .lt_matmul_preference_handles
        .get(&pref)
        .copied()
        .unwrap_or(pref)
}

fn cublaslt_unbind_matmul_preference(pref: cublasLtMatmulPreference_t) {
    CUBLAS_CACHE
        .write()
        .unwrap()
        .lt_matmul_preference_handles
        .remove(&pref);
}

fn cublaslt_bind_matrix_layout(client: cublasLtMatrixLayout_t, server: cublasLtMatrixLayout_t) {
    CUBLAS_CACHE
        .write()
        .unwrap()
        .lt_matrix_layout_handles
        .insert(client, server);
}

fn cublaslt_resolve_matrix_layout(layout: cublasLtMatrixLayout_t) -> cublasLtMatrixLayout_t {
    CUBLAS_CACHE
        .read()
        .unwrap()
        .lt_matrix_layout_handles
        .get(&layout)
        .copied()
        .unwrap_or(layout)
}

fn cublaslt_unbind_matrix_layout(layout: cublasLtMatrixLayout_t) {
    CUBLAS_CACHE
        .write()
        .unwrap()
        .lt_matrix_layout_handles
        .remove(&layout);
}

fn cublaslt_bind_emulation_desc(client: cublasLtEmulationDesc_t, server: cublasLtEmulationDesc_t) {
    CUBLAS_CACHE
        .write()
        .unwrap()
        .lt_emulation_desc_handles
        .insert(client, server);
}

fn cublaslt_resolve_emulation_desc(desc: cublasLtEmulationDesc_t) -> cublasLtEmulationDesc_t {
    CUBLAS_CACHE
        .read()
        .unwrap()
        .lt_emulation_desc_handles
        .get(&desc)
        .copied()
        .unwrap_or(desc)
}

fn cublaslt_unbind_emulation_desc(desc: cublasLtEmulationDesc_t) {
    CUBLAS_CACHE
        .write()
        .unwrap()
        .lt_emulation_desc_handles
        .remove(&desc);
}

fn cublaslt_bind_transform_desc(
    client: cublasLtMatrixTransformDesc_t,
    server: cublasLtMatrixTransformDesc_t,
) {
    CUBLAS_CACHE
        .write()
        .unwrap()
        .lt_transform_desc_handles
        .insert(client, server);
}

fn cublaslt_resolve_transform_desc(
    desc: cublasLtMatrixTransformDesc_t,
) -> cublasLtMatrixTransformDesc_t {
    CUBLAS_CACHE
        .read()
        .unwrap()
        .lt_transform_desc_handles
        .get(&desc)
        .copied()
        .unwrap_or(desc)
}

fn cublaslt_unbind_transform_desc(desc: cublasLtMatrixTransformDesc_t) {
    let mut cache = CUBLAS_CACHE.write().unwrap();
    cache.lt_transform_desc_handles.remove(&desc);
    cache.lt_transform_descs.remove(&desc);
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
