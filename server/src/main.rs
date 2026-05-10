use std::collections::btree_map::{BTreeMap, Entry};
use std::io::{self, Read as _, Write as _};
use std::net::TcpListener;
use std::sync::{Arc, Barrier};
use std::time::Duration;
use std::{ptr, thread};

use network::NetworkConfig;
use server::*;

fn main() {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    if let Err(err) = run() {
        log::error!("{err}");
        std::process::exit(1);
    }
}

fn run() -> io::Result<()> {
    // core_affinity::set_for_current(0);
    let config = NetworkConfig::read_from_file();
    match config.comm_type.as_str() {
        "shm" => {
            log::info!("Using shared memory channel")
        }
        "tcp" => log::info!("Using TCP channel"),
        #[cfg(feature = "rdma")]
        "rdma" => log::info!("Using RDMA channel"),
        other => return Err(unsupported_comm_type(other)),
    }

    let (rx, tx) = daemon(&config)?; // fork return
    server_process(config, rx, tx)
}

fn daemon(config: &NetworkConfig) -> io::Result<(io::PipeReader, io::PipeWriter)> {
    let mut children = BTreeMap::new();

    let listener = TcpListener::bind(&config.daemon_socket)?;
    let listener = socket2::Socket::from(listener);
    listener.set_read_timeout(Some(Duration::from_secs(1)))?;
    let mut id: i32 = 0;
    loop {
        let finished = reap_children();
        if !finished.is_empty() {
            children.retain(|_, (server_pid, _, _)| !finished.contains(server_pid));
        }
        drop(finished);

        let mut stream = match listener.accept() {
            Ok((stream, _)) => stream,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => continue,
            Err(e) => return Err(e),
        };
        stream.set_read_timeout(None)?;
        let mut buf = [0u8; 4];
        if let Err(err) = stream.read_exact(&mut buf) {
            log::warn!("failed to read client handshake: {err}");
            continue;
        }
        let client_pid = u32::from_be_bytes(buf);
        log::info!("[#{id}] Client PID = {client_pid}");

        let (_, daemon_rx, daemon_tx) = match children.entry(client_pid) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let (child_rx, daemon_tx) = io::pipe()?;
                let (daemon_rx, child_tx) = io::pipe()?;
                let server_pid = unsafe { libc::fork() };
                if server_pid < 0 {
                    return Err(io::Error::last_os_error());
                }
                if server_pid == 0 {
                    return Ok((child_rx, child_tx));
                }
                entry.insert((server_pid, daemon_rx, daemon_tx))
            }
        };
        daemon_tx.write_all(&id.to_ne_bytes())?;
        daemon_rx.read_exact(&mut buf)?;
        if id.to_ne_bytes() != buf {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "server child returned the wrong channel id",
            ));
        }
        if let Err(err) = stream.write_all(&id.to_be_bytes()) {
            log::warn!("failed to write daemon handshake: {err}");
        }
        id = id.checked_add(1).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "daemon channel id overflowed")
        })?;
    }
}

fn server_process(
    config: NetworkConfig,
    mut rx: io::PipeReader,
    mut tx: io::PipeWriter,
) -> io::Result<()> {
    let config = Arc::new(config);
    let mut is_main_thread = true;
    loop {
        let mut buf = [0u8; 4];
        match rx.read_exact(&mut buf) {
            Ok(()) => {}
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(err) => return Err(err),
        }
        let id = i32::from_ne_bytes(buf);
        let child_config = Arc::clone(&config);
        let (barrier, child_barrier) = match config.comm_type.as_str() {
            "shm" | "tcp" => {
                let barrier = Arc::new(Barrier::new(2));
                let child_barrier = Arc::clone(&barrier);
                (Some(barrier), Some(child_barrier))
            }
            #[cfg(feature = "rdma")]
            "rdma" => (None, None),
            other => return Err(unsupported_comm_type(other)),
        };
        thread::spawn(move || {
            launch_server(&child_config, id, child_barrier, is_main_thread);
        });
        barrier.map(|barrier| barrier.wait());
        tx.write_all(&buf)?;
        is_main_thread = false;
    }
}

fn reap_children() -> Vec<libc::pid_t> {
    let mut result = Vec::new();
    loop {
        let pid = unsafe { libc::waitpid(-1, ptr::null_mut(), libc::WNOHANG) };
        if pid > 0 {
            result.push(pid);
        } else {
            return result;
        }
    }
}

fn unsupported_comm_type(comm_type: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("unsupported communication type in config: {comm_type}"),
    )
}
