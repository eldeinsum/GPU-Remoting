#[cfg(test)]
mod tests {
    use super::super::LocalChannel;
    use super::super::META_AREA;
    use crate::{CommChannelInnerIO, RawMemory, RawMemoryMut};

    #[test]
    fn basic_send_receive() {
        let channel = LocalChannel::new(10 + META_AREA);
        let data_to_send: [u8; 5] = [1, 2, 3, 4, 5];
        let mut receive_buffer = [0u8; 5];

        let send_memory = RawMemory::new(&data_to_send, data_to_send.len());
        channel.put_bytes(&send_memory).unwrap();

        let len = receive_buffer.len();
        let mut receive_memory = RawMemoryMut::new(&mut receive_buffer, len);
        channel.get_bytes(&mut receive_memory).unwrap();

        assert_eq!(receive_buffer, data_to_send);
    }

    #[test]
    fn partial_receive() {
        let channel = LocalChannel::new(10 + META_AREA);
        let data_to_send: [u8; 5] = [1, 2, 3, 4, 5];
        let mut receive_buffer = [0u8; 3];

        let send_memory = RawMemory::new(&data_to_send, data_to_send.len());
        channel.put_bytes(&send_memory).unwrap();

        let len = receive_buffer.len();
        let mut receive_memory = RawMemoryMut::new(&mut receive_buffer, len);
        channel.get_bytes(&mut receive_memory).unwrap();

        assert_eq!(receive_buffer, [1, 2, 3]);
    }

    #[test]
    fn wrap_around() {
        println!("Wrap around test");
        let channel = LocalChannel::new(10 + META_AREA);

        let first_send: [u8; 3] = [1, 2, 3];
        let second_send: [u8; 3] = [4, 5, 6];
        let mut first_recv = [0u8; 3];
        let mut second_recv = [0u8; 3];

        let send_memory = RawMemory::new(&first_send, first_send.len());
        channel.put_bytes(&send_memory).unwrap();

        let len = first_recv.len();
        let mut receive_memory = RawMemoryMut::new(&mut first_recv, len);
        channel.get_bytes(&mut receive_memory).unwrap(); // Create a gap for wrap-around

        let send_memory = RawMemory::new(&second_send, second_send.len());
        channel.put_bytes(&send_memory).unwrap();

        let len = second_recv.len();
        let mut receive_memory = RawMemoryMut::new(&mut second_recv, len);
        channel.get_bytes(&mut receive_memory).unwrap();

        assert_eq!(first_recv, [1, 2, 3]);
        assert_eq!(second_recv, [4, 5, 6]);
    }

    // TBD
}
