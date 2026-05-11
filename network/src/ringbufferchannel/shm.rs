use crate::ringbufferchannel::{BufferManager, RingBufferChannel, RingBufferManager};
use crate::{CommChannelError, CommChannelInner, NetworkConfig};

use log::error;
use std::ffi::{CStr, CString};
use std::io::{self, Result as IOResult};
use std::os::unix::io::RawFd;

/// A shared memory channel buffer manager
pub struct SHMChannel {
    shm_name: String,
    shm_ptr: *mut u8,
    shm_len: usize,
    unlink_on_drop: bool,
}

unsafe impl Send for SHMChannel {}

impl SHMChannel {
    /// Create a new shared memory channel buffer manager for the server
    /// The name server is more consistent with the remoting library
    pub fn new_server(shm_name: &str, shm_len: usize) -> IOResult<Self> {
        Self::new_inner(shm_name, shm_len, ShmRole::Server)
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
        Self::new_inner(shm_name, shm_len, ShmRole::Client)
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

    fn new_inner(shm_name: &str, shm_len: usize, role: ShmRole) -> IOResult<Self> {
        let shm_name_c_str = CString::new(shm_name).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "shared memory name contains an interior NUL byte",
            )
        })?;
        let fd: RawFd = unsafe {
            libc::shm_open(
                shm_name_c_str.as_ptr(),
                role.oflag(),
                (libc::S_IRUSR | libc::S_IWUSR) as _,
            )
        };

        if fd == -1 {
            error!("Error on shm_open for shared memory channel");
            return Err(std::io::Error::last_os_error());
        }

        if role.truncate() && unsafe { libc::ftruncate(fd, shm_len as libc::off_t) } == -1 {
            error!("Error on ftruncate");
            let err = std::io::Error::last_os_error();
            close_fd(fd);
            let _ = unlink_shm_name(&shm_name_c_str);
            return Err(err);
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
            let err = std::io::Error::last_os_error();
            close_fd(fd);
            if role.unlink_on_drop() {
                let _ = unlink_shm_name(&shm_name_c_str);
            }
            return Err(err);
        }

        close_fd(fd);
        if role.unlink_after_map() {
            let _ = unlink_shm_name(&shm_name_c_str);
        }

        Ok(Self {
            shm_name: String::from(shm_name),
            shm_len,
            shm_ptr: shm_ptr as *mut u8,
            unlink_on_drop: role.unlink_on_drop(),
        })
    }
}

impl Drop for SHMChannel {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.shm_ptr as *mut libc::c_void, self.shm_len);
        }
        if self.unlink_on_drop {
            let shm_name = CString::new(self.shm_name.as_str()).unwrap();
            let _ = unlink_shm_name(&shm_name);
        }
    }
}

#[derive(Clone, Copy)]
enum ShmRole {
    Server,
    Client,
}

impl ShmRole {
    fn oflag(self) -> i32 {
        match self {
            Self::Server => libc::O_CREAT | libc::O_TRUNC | libc::O_RDWR,
            Self::Client => libc::O_RDWR,
        }
    }

    fn truncate(self) -> bool {
        matches!(self, Self::Server)
    }

    fn unlink_after_map(self) -> bool {
        matches!(self, Self::Client)
    }

    fn unlink_on_drop(self) -> bool {
        matches!(self, Self::Server)
    }
}

fn close_fd(fd: RawFd) {
    if unsafe { libc::close(fd) } == -1 {
        error!(
            "Error on closing SHM file descriptor: {}",
            io::Error::last_os_error()
        );
    }
}

fn unlink_shm_name(shm_name: &CStr) -> io::Result<()> {
    if unsafe { libc::shm_unlink(shm_name.as_ptr()) } == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
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
    use std::path::PathBuf;

    #[test]
    fn shm_channel_buffer_manager() {
        let shm_name = unique_name("buffer_manager");
        cleanup_name(&shm_name);
        let shm_len = 64;
        let manager = SHMChannel::new_server(&shm_name, shm_len).unwrap();
        assert_eq!(manager.shm_len, shm_len);
    }

    #[test]
    fn server_drop_unlinks_name() {
        let shm_name = unique_name("server_drop_unlinks");
        cleanup_name(&shm_name);
        let shm_path = shm_path(&shm_name);

        {
            let _server = SHMChannel::new_server(&shm_name, 4096).unwrap();
            assert!(shm_path.exists());
        }

        assert!(!shm_path.exists());
    }

    #[test]
    fn client_unlinks_name_after_mapping() {
        let shm_name = unique_name("client_unlinks_name_after_mapping");
        cleanup_name(&shm_name);
        let shm_path = shm_path(&shm_name);

        let server = SHMChannel::new_server(&shm_name, 4096).unwrap();
        assert!(shm_path.exists());

        let client = SHMChannel::new_client(&shm_name, 4096).unwrap();
        assert!(!shm_path.exists());

        drop(client);
        drop(server);
        cleanup_name(&shm_name);
    }

    fn unique_name(test_name: &str) -> String {
        format!(
            "/gpu_remoting_shm_test_{}_{}",
            std::process::id(),
            test_name
        )
    }

    fn shm_path(shm_name: &str) -> PathBuf {
        PathBuf::from("/dev/shm").join(shm_name.trim_start_matches('/'))
    }

    fn cleanup_name(shm_name: &str) {
        let Ok(shm_name) = CString::new(shm_name) else {
            return;
        };
        let _ = unlink_shm_name(&shm_name);
    }
}
