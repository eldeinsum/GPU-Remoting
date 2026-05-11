use std::io;
use std::os::raw::c_int;
use std::os::unix::io::AsRawFd;
use std::os::unix::net::UnixStream;

pub fn pack_kernel_args(
    arg_ptrs: *mut *mut std::ffi::c_void,
    info: &[crate::elf::KernelParamInfo],
) -> Box<[u8]> {
    let Some(last) = info.last() else {
        return Default::default();
    };
    let mut result = vec![0u8; (last.offset + last.size()) as usize];
    for (param, arg_ptr) in info
        .iter()
        .zip(unsafe { std::slice::from_raw_parts(arg_ptrs, info.len()) })
    {
        unsafe {
            std::ptr::copy_nonoverlapping(
                arg_ptr.cast(),
                result.as_mut_ptr().wrapping_add(param.offset as usize),
                param.size() as usize,
            );
        }
        match param.size() {
            8 if arg_ptr.cast::<u64>().is_aligned() => {
                let arg = unsafe { *arg_ptr.cast::<u64>() };
                log::trace!(target: "cuLaunchKernel", "arg = {arg:#x}");
            }
            4 if arg_ptr.cast::<i32>().is_aligned() => {
                let arg = unsafe { *arg_ptr.cast::<i32>() };
                log::trace!(target: "cuLaunchKernel", "arg = {arg}");
            }
            size => log::trace!(target: "cuLaunchKernel", "arg<{size}> = {:?}", unsafe {
                std::slice::from_raw_parts(arg_ptr.cast::<u8>(), param.size() as usize)
            }),
        }
    }
    result.into_boxed_slice()
}

pub fn pack_kernel_args_with_offsets(
    arg_ptrs: *mut *mut std::ffi::c_void,
    info: &[crate::elf::KernelParamInfo],
) -> (Box<[u8]>, Box<[u32]>) {
    let args = pack_kernel_args(arg_ptrs, info);
    let offsets = info
        .iter()
        .map(|param| u32::from(param.offset))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    (args, offsets)
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
