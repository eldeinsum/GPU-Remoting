use std::cell::RefCell;
use std::io::{BufReader, BufWriter, Error, Read as _, Write as _};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::slice;
use std::sync::Barrier;

use crate::{
    CommChannel, CommChannelError, CommChannelInnerIO, NetworkConfig, RawMemory, RawMemoryMut,
};

pub fn new_server(
    config: &NetworkConfig,
    id: i32,
    barrier: &Barrier,
) -> Result<(TcpReceiver, TcpSender), Error> {
    let mut addr: SocketAddr = config.receiver_socket.parse().unwrap();
    addr.set_port(addr.port() + id as u16);
    let listener = TcpListener::bind(addr)?;
    barrier.wait();
    let (stream, _) = listener.accept()?;
    Ok((
        TcpReceiver(RefCell::new(BufReader::new(stream.try_clone()?))),
        TcpSender(RefCell::new(BufWriter::new(stream))),
    ))
}

pub fn new_client(config: &NetworkConfig, id: i32) -> Result<(TcpSender, TcpReceiver), Error> {
    let mut addr: SocketAddr = config.receiver_socket.parse().unwrap();
    addr.set_port(addr.port() + id as u16);
    let stream = TcpStream::connect(addr)?;
    Ok((
        TcpSender(RefCell::new(BufWriter::new(stream.try_clone()?))),
        TcpReceiver(RefCell::new(BufReader::new(stream))),
    ))
}

pub struct TcpSender(RefCell<BufWriter<TcpStream>>);
pub struct TcpReceiver(RefCell<BufReader<TcpStream>>);

fn invalid_direction() -> Result<usize, CommChannelError> {
    Err(CommChannelError::InvalidOperation)
}

impl CommChannel for TcpSender {
    fn flush_out(&self) -> Result<(), CommChannelError> {
        match self.0.borrow_mut().flush() {
            Ok(()) => Ok(()),
            Err(e) => {
                log::error!("flush failed: {e}");
                Err(CommChannelError::IoError)
            }
        }
    }
}

impl CommChannelInnerIO for TcpSender {
    fn put_bytes(&self, src: &RawMemory) -> Result<usize, CommChannelError> {
        let buf = unsafe { slice::from_raw_parts(src.ptr, src.len) };
        match self.0.borrow_mut().write_all(buf) {
            Ok(()) => Ok(src.len),
            Err(e) => {
                log::error!("write failed: {e}");
                Err(CommChannelError::IoError)
            }
        }
    }

    fn try_put_bytes(&self, src: &RawMemory) -> Result<usize, CommChannelError> {
        self.put_bytes(src)
    }

    fn get_bytes(&self, _dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        invalid_direction()
    }

    fn try_get_bytes(&self, _dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        invalid_direction()
    }

    fn safe_try_get_bytes(&self, _dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        invalid_direction()
    }
}

impl CommChannel for TcpReceiver {
    fn flush_out(&self) -> Result<(), CommChannelError> {
        Ok(())
    }
}

impl CommChannelInnerIO for TcpReceiver {
    fn put_bytes(&self, _src: &RawMemory) -> Result<usize, CommChannelError> {
        invalid_direction()
    }

    fn try_put_bytes(&self, _src: &RawMemory) -> Result<usize, CommChannelError> {
        invalid_direction()
    }

    fn get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        let buf = unsafe { slice::from_raw_parts_mut(dst.ptr, dst.len) };
        match self.0.borrow_mut().read_exact(buf) {
            Ok(()) => Ok(dst.len),
            Err(e) => {
                log::error!("read failed: {e}");
                Err(CommChannelError::IoError)
            }
        }
    }

    fn try_get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        self.get_bytes(dst)
    }

    fn safe_try_get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        self.get_bytes(dst)
    }
}
