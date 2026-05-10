use crate::{
    CommChannel, CommChannelError, RawMemory, RawMemoryMut, Transportable, TransportableMarker,
};

impl<T: TransportableMarker> Transportable for T {
    fn send<C: CommChannel>(&self, channel: &C) -> Result<(), CommChannelError> {
        if size_of::<Self>() == 0 {
            return Ok(());
        }
        let memory = RawMemory::new(self, size_of::<Self>());
        match channel.put_bytes(&memory)? == size_of::<Self>() {
            true => Ok(()),
            false => Err(CommChannelError::IoError),
        }
    }

    fn recv<C: CommChannel>(&mut self, channel: &C) -> Result<(), CommChannelError> {
        if size_of::<Self>() == 0 {
            return Ok(());
        }
        let mut memory = RawMemoryMut::new(self, size_of::<Self>());
        match channel.get_bytes(&mut memory)? == size_of::<Self>() {
            true => Ok(()),
            false => Err(CommChannelError::IoError),
        }
    }
}

macro_rules! impl_transportable {
    ($($t:ty),*) => {
        $(
            impl TransportableMarker for $t {}
        )*
    };
}

impl_transportable!(
    u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize, f32, f64, bool, char
);

impl Transportable for () {
    fn send<C: CommChannel>(&self, _channel: &C) -> Result<(), CommChannelError> {
        Ok(())
    }

    fn recv<C: CommChannel>(&mut self, _channel: &C) -> Result<(), CommChannelError> {
        Ok(())
    }
}

impl<T> TransportableMarker for *const T {}
impl<T> TransportableMarker for *mut T {}
impl<T: Copy> TransportableMarker for std::mem::MaybeUninit<T> {}

/// a pointer type, we just need to use usize to represent it
/// the raw type `*mut void` is hard to handle:(.
///
/// IMPORTANT on replacing `*mut *mut` like parameters in memory operations.
pub type MemPtr = usize;

impl<T: TransportableMarker> Transportable for [T] {
    fn send<C: CommChannel>(&self, channel: &C) -> Result<(), CommChannelError> {
        send_slice(self, channel)
    }

    fn recv<C: CommChannel>(&mut self, channel: &C) -> Result<(), CommChannelError> {
        recv_slice_to(self, channel)
    }
}

impl<T: TransportableMarker> Transportable for Vec<T> {
    fn send<C: CommChannel>(&self, channel: &C) -> Result<(), CommChannelError> {
        send_slice(self, channel)
    }

    fn recv<C: CommChannel>(&mut self, channel: &C) -> Result<(), CommChannelError> {
        recv_slice(channel).map(|slice| *self = slice.into_vec())
    }
}

pub fn send_slice<T: TransportableMarker, C: CommChannel>(
    data: &[T],
    channel: &C,
) -> Result<(), CommChannelError> {
    let len = data.len();
    len.send(channel)?;
    let bytes = size_of_val(data);
    let memory = RawMemory::from_ptr(data.as_ptr() as *const u8, bytes);
    match channel.put_bytes(&memory)? == bytes {
        true => Ok(()),
        false => Err(CommChannelError::IoError),
    }
}

pub fn recv_slice<T: TransportableMarker, C: CommChannel>(
    channel: &C,
) -> Result<Box<[T]>, CommChannelError> {
    let mut len = 0;
    len.recv(channel)?;
    let bytes = checked_byte_len::<T>(len)?;
    let mut data = Box::<[T]>::new_uninit_slice(len);
    let mut memory = RawMemoryMut::from_ptr(data.as_mut_ptr() as *mut u8, bytes);
    match channel.get_bytes(&mut memory)? == bytes {
        true => Ok(unsafe { data.assume_init() }),
        false => Err(CommChannelError::IoError),
    }
}

pub fn recv_slice_to<T: TransportableMarker, C: CommChannel>(
    data: &mut [T],
    channel: &C,
) -> Result<(), CommChannelError> {
    let mut len = 0;
    len.recv(channel)?;
    if len != data.len() {
        return Err(CommChannelError::InvalidOperation);
    }
    let bytes = checked_byte_len::<T>(len)?;
    let mut memory = RawMemoryMut::from_ptr(data.as_mut_ptr() as *mut u8, bytes);
    match channel.get_bytes(&mut memory)? == bytes {
        true => Ok(()),
        false => Err(CommChannelError::IoError),
    }
}

fn checked_byte_len<T>(len: usize) -> Result<usize, CommChannelError> {
    len.checked_mul(size_of::<T>())
        .ok_or(CommChannelError::InvalidOperation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ringbufferchannel::{LocalChannel, META_AREA},
        Channel,
    };

    /// Test bool Transportable impl
    #[test]
    fn test_bool_io() {
        let channel = Channel::new(Box::new(LocalChannel::new(10 + META_AREA)));
        let a = true;
        let mut b = false;
        a.send(&channel).unwrap();
        b.recv(&channel).unwrap();
        assert_eq!(a, b);
    }

    /// Test i32 Transportable impl
    #[test]
    fn test_i32_io() {
        let channel = Channel::new(Box::new(LocalChannel::new(10 + META_AREA)));
        let a = 123;
        let mut b = 0;
        a.send(&channel).unwrap();
        b.recv(&channel).unwrap();
        assert_eq!(a, b);
    }

    /// Test [u8] Transportable impl
    #[test]
    fn test_u8_array_io() {
        let channel = Channel::new(Box::new(LocalChannel::new(50 + META_AREA)));
        let a = [1u8, 2, 3, 4, 5];
        let mut b = [0u8; 5];
        a.send(&channel).unwrap();
        b.recv(&channel).unwrap();
        assert_eq!(a, b);
    }

    /// Test [i32] Transportable impl
    #[test]
    fn test_i32_array_io() {
        let channel = Channel::new(Box::new(LocalChannel::new(50 + META_AREA)));
        let a = [1i32, 2, 3, 4, 5];
        let mut b = [0i32; 5];
        a.send(&channel).unwrap();
        b.recv(&channel).unwrap();
        assert_eq!(a, b);
    }

    /// Test Vec<i32> Transportable impl
    #[test]
    fn test_vec_io() {
        let channel = Channel::new(Box::new(LocalChannel::new(50 + META_AREA)));
        let a = vec![1, 2, 3, 4, 5];
        let mut b = vec![0; 5];
        a.send(&channel).unwrap();
        b.recv(&channel).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn rejects_oversized_slice_length() {
        let channel = Channel::new(Box::new(LocalChannel::new(16 + META_AREA)));
        usize::MAX.send(&channel).unwrap();
        assert!(matches!(
            recv_slice::<u16, _>(&channel),
            Err(CommChannelError::InvalidOperation)
        ));
    }

    #[test]
    fn recv_slice_to_rejects_length_mismatch() {
        let channel = Channel::new(Box::new(LocalChannel::new(16 + META_AREA)));
        2usize.send(&channel).unwrap();
        let mut dst = [0u8; 3];
        assert!(matches!(
            recv_slice_to(&mut dst, &channel),
            Err(CommChannelError::InvalidOperation)
        ));
    }
}
