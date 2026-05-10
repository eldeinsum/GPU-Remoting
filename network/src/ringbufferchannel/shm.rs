use crate::ringbufferchannel::{BufferManager, RingBufferChannel, RingBufferManager};
use crate::{CommChannelError, CommChannelInner, NetworkConfig};

use log::error;
use std::ffi::CString;
use std::io::Result as IOResult;
use std::os::unix::io::RawFd;

/// A shared memory channel buffer manager
pub struct SHMChannel {
    shm_name: String,
    shm_ptr: *mut u8,
    shm_len: usize,
}

unsafe impl Send for SHMChannel {}

impl SHMChannel {
    /// Create a new shared memory channel buffer manager for the server
    /// The name server is more consistent with the remoting library
    pub fn new_server(shm_name: &str, shm_len: usize) -> IOResult<Self> {
        Self::new_inner(
            shm_name,
            shm_len,
            libc::O_CREAT | libc::O_TRUNC | libc::O_RDWR,
            (libc::S_IRUSR | libc::S_IWUSR) as _,
        )
    }

    pub fn new_server_with_id(config: &NetworkConfig, id: i32) -> IOResult<(Self, Self)> {
        Ok((
            Self::new_server(
                &format!("{}_{}", config.ctos_channel_name, id),
                config.buf_size,
            )?,
            Self::new_server(
                &format!("{}_{}", config.stoc_channel_name, id),
                config.buf_size,
            )?,
        ))
    }

    pub fn new_client(shm_name: &str, shm_len: usize) -> IOResult<Self> {
        Self::new_inner(
            shm_name,
            shm_len,
            libc::O_RDWR,
            (libc::S_IRUSR | libc::S_IWUSR) as _,
        )
    }

    pub fn new_client_with_id(config: &NetworkConfig, id: i32) -> IOResult<(Self, Self)> {
        Ok((
            Self::new_client(
                &format!("{}_{}", config.ctos_channel_name, id),
                config.buf_size,
            )?,
            Self::new_client(
                &format!("{}_{}", config.stoc_channel_name, id),
                config.buf_size,
            )?,
        ))
    }

    fn new_inner(shm_name: &str, shm_len: usize, oflag: i32, sflag: i32) -> IOResult<Self> {
        let shm_name_c_str = CString::new(shm_name).unwrap();
        let fd: RawFd = unsafe { libc::shm_open(shm_name_c_str.as_ptr(), oflag, sflag as _) };

        if fd == -1 {
            error!("Error on shm_open for new_host");
            return Err(std::io::Error::last_os_error());
        }

        if unsafe { libc::ftruncate(fd, shm_len as libc::off_t) } == -1 {
            error!("Error on ftruncate");
            unsafe { libc::shm_unlink(shm_name.as_ptr() as _) };
            return Err(std::io::Error::last_os_error());
        }

        // map the shared memory to the process's address space
        let shm_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                shm_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };

        if std::ptr::eq(shm_ptr, libc::MAP_FAILED) {
            error!("Error on mmap the SHM pointer");
            unsafe { libc::shm_unlink(shm_name.as_ptr() as _) };
            return Err(std::io::Error::last_os_error());
        }

        Ok(Self {
            shm_name: String::from(shm_name),
            shm_len,
            shm_ptr: shm_ptr as *mut u8,
        })
    }
}

impl Drop for SHMChannel {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.shm_ptr as *mut libc::c_void, self.shm_len);
            let shm_name_ = CString::new(self.shm_name.clone()).unwrap();
            libc::shm_unlink(shm_name_.as_ptr());
        }
    }
}

impl BufferManager for SHMChannel {
    fn get_ptr(&self) -> *mut u8 {
        self.shm_ptr
    }

    fn get_len(&self) -> usize {
        self.shm_len
    }
}

impl RingBufferManager for SHMChannel {}

impl RingBufferChannel for SHMChannel {}

impl CommChannelInner for SHMChannel {
    fn flush_out(&self) -> Result<(), CommChannelError> {
        while self.is_full() {
            // Busy-waiting
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shm_channel_buffer_manager() {
        let shm_name = "/stoc";
        let shm_len = 64;
        let manager = SHMChannel::new_server(shm_name, shm_len).unwrap();
        assert_eq!(manager.shm_len, shm_len);
    }
}
