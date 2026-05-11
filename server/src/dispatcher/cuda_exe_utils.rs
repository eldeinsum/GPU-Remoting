use std::os::raw::*;
use std::fs;
use std::io;
use std::os::unix::io::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::atomic::{AtomicU64, Ordering};

use cudasys::cuda::*;
use network::type_impl::send_slice;
use network::CommChannel;

use crate::ServerWorker;

#[allow(clippy::too_many_arguments)]
pub fn cu_launch_kernel(
    f: CUfunction,
    gridDimX: c_uint,
    gridDimY: c_uint,
    gridDimZ: c_uint,
    blockDimX: c_uint,
    blockDimY: c_uint,
    blockDimZ: c_uint,
    sharedMemBytes: c_uint,
    hStream: CUstream,
    args: &[u8],
    arg_offsets: &[u32],
) -> CUresult {
    unsafe {
        let mut kernel_params = kernel_params_from_packed_args(args, arg_offsets);
        cuLaunchKernel(
            f,
            gridDimX,
            gridDimY,
            gridDimZ,
            blockDimX,
            blockDimY,
            blockDimZ,
            sharedMemBytes,
            hStream,
            if kernel_params.is_empty() {
                std::ptr::null_mut()
            } else {
                kernel_params.as_mut_ptr()
            },
            std::ptr::null_mut(),
        )
    }
}

pub fn cu_launch_kernel_ex(
    config: &CUlaunchConfig,
    f: CUfunction,
    args: &[u8],
    arg_offsets: &[u32],
) -> CUresult {
    unsafe {
        let mut kernel_params = kernel_params_from_packed_args(args, arg_offsets);
        cuLaunchKernelEx(
            config as *const _,
            f,
            if kernel_params.is_empty() {
                std::ptr::null_mut()
            } else {
                kernel_params.as_mut_ptr()
            },
            std::ptr::null_mut(),
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn cu_launch_cooperative_kernel(
    f: CUfunction,
    gridDimX: c_uint,
    gridDimY: c_uint,
    gridDimZ: c_uint,
    blockDimX: c_uint,
    blockDimY: c_uint,
    blockDimZ: c_uint,
    sharedMemBytes: c_uint,
    hStream: CUstream,
    args: &[u8],
    arg_offsets: &[u32],
) -> CUresult {
    unsafe {
        let mut kernel_params = kernel_params_from_packed_args(args, arg_offsets);
        cuLaunchCooperativeKernel(
            f,
            gridDimX,
            gridDimY,
            gridDimZ,
            blockDimX,
            blockDimY,
            blockDimZ,
            sharedMemBytes,
            hStream,
            if kernel_params.is_empty() {
                std::ptr::null_mut()
            } else {
                kernel_params.as_mut_ptr()
            },
        )
    }
}

pub fn kernel_params_from_packed_args(args: &[u8], arg_offsets: &[u32]) -> Vec<*mut c_void> {
    arg_offsets
        .iter()
        .map(|offset| unsafe {
            args.as_ptr()
                .add(*offset as usize)
                .cast_mut()
                .cast::<c_void>()
        })
        .collect::<Vec<_>>()
}

pub fn cu_func_get_attributes(
    attr: *mut cudasys::cudart::cudaFuncAttributes,
    func: CUfunction,
) -> CUresult {
    let attr = unsafe { &mut *attr };
    // HACK: implementation with cuFuncGetAttribute depends on CUDA version
    macro_rules! get_attributes {
        ($func:ident -> $struct:ident $($field:ident: $attr:ident,)+) => {
            $(
                let mut i = 0;
                let result =
                    unsafe { cuFuncGetAttribute(&raw mut i, CUfunction_attribute::$attr, $func) };
                if result != Default::default() {
                    return result;
                }
                $struct.$field = i as _;
            )+
        }
    }
    get_attributes! { func -> attr
        sharedSizeBytes: CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
        constSizeBytes: CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
        localSizeBytes: CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
        maxThreadsPerBlock: CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        numRegs: CU_FUNC_ATTRIBUTE_NUM_REGS,
        ptxVersion: CU_FUNC_ATTRIBUTE_PTX_VERSION,
        binaryVersion: CU_FUNC_ATTRIBUTE_BINARY_VERSION,
        cacheModeCA: CU_FUNC_ATTRIBUTE_CACHE_MODE_CA,
        maxDynamicSharedSizeBytes: CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
    }
    CUresult::CUDA_SUCCESS
}

fn recv_fd(socket: &UnixStream) -> io::Result<c_int> {
    let mut byte = [0u8; 1];
    let mut iov = libc::iovec {
        iov_base: byte.as_mut_ptr().cast(),
        iov_len: byte.len(),
    };
    let mut control = vec![0u8; unsafe {
        libc::CMSG_SPACE(std::mem::size_of::<c_int>() as _) as usize
    }];
    let mut msg = unsafe { std::mem::zeroed::<libc::msghdr>() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.as_mut_ptr().cast();
    msg.msg_controllen = control.len();

    unsafe {
        let received = libc::recvmsg(socket.as_raw_fd(), &mut msg, 0);
        if received <= 0 {
            return if received == 0 {
                Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "socket closed before file descriptor was received",
                ))
            } else {
                Err(io::Error::last_os_error())
            };
        }
        let cmsg = libc::CMSG_FIRSTHDR(&msg);
        if cmsg.is_null()
            || (*cmsg).cmsg_level != libc::SOL_SOCKET
            || (*cmsg).cmsg_type != libc::SCM_RIGHTS
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "missing SCM_RIGHTS file descriptor",
            ));
        }
        Ok(*libc::CMSG_DATA(cmsg).cast::<c_int>())
    }
}

pub fn receive_client_fd<C, E>(
    server: &mut ServerWorker<C>,
    target: &'static str,
    error_result: E,
) -> Result<c_int, E>
where
    C: CommChannel,
    E: Copy,
{
    static FD_SOCKET_COUNTER: AtomicU64 = AtomicU64::new(0);
    let counter = FD_SOCKET_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = format!(
        "/tmp/gpu-remoting-fd-{}-{}-{counter}.sock",
        std::process::id(),
        server.id
    );
    let _ = fs::remove_file(&path);
    let listener = match UnixListener::bind(&path) {
        Ok(listener) => listener,
        Err(error) => {
            log::error!(target: target, "failed to bind fd socket {path}: {error}");
            send_slice::<u8, _>(&[], &server.channel_sender).unwrap();
            server.channel_sender.flush_out().unwrap();
            return Err(error_result);
        }
    };
    send_slice(path.as_bytes(), &server.channel_sender).unwrap();
    server.channel_sender.flush_out().unwrap();

    let result = match listener.accept() {
        Ok((socket, _)) => recv_fd(&socket).map_err(|error| {
            log::error!(target: target, "failed to receive file descriptor: {error}");
            error_result
        }),
        Err(error) => {
            log::error!(target: target, "failed to accept fd socket: {error}");
            Err(error_result)
        }
    };
    let _ = fs::remove_file(&path);
    result
}

pub fn send_fd(socket_path: &[u8], fd: c_int) -> io::Result<()> {
    let path = std::str::from_utf8(socket_path)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let stream = UnixStream::connect(path)?;
    let mut byte = [0u8; 1];
    let mut iov = libc::iovec {
        iov_base: byte.as_mut_ptr().cast(),
        iov_len: byte.len(),
    };
    let mut control = vec![0u8; unsafe {
        libc::CMSG_SPACE(std::mem::size_of::<c_int>() as _) as usize
    }];
    let mut msg = unsafe { std::mem::zeroed::<libc::msghdr>() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.as_mut_ptr().cast();
    msg.msg_controllen = control.len();

    unsafe {
        let cmsg = libc::CMSG_FIRSTHDR(&msg);
        if cmsg.is_null() {
            return Err(io::Error::other("missing control message header"));
        }
        (*cmsg).cmsg_level = libc::SOL_SOCKET;
        (*cmsg).cmsg_type = libc::SCM_RIGHTS;
        (*cmsg).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<c_int>() as _) as _;
        *libc::CMSG_DATA(cmsg).cast::<c_int>() = fd;
        let sent = libc::sendmsg(stream.as_raw_fd(), &msg, 0);
        if sent == byte.len() as isize {
            Ok(())
        } else if sent < 0 {
            Err(io::Error::last_os_error())
        } else {
            Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "short sendmsg while sending file descriptor",
            ))
        }
    }
}
