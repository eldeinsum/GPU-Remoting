use network::ringbufferchannel::RDMAChannel;
use network::{Channel, CommChannel, CommChannelInnerIO, NetworkConfig, RawMemory, RawMemoryMut};

const BUF_SIZE: usize = 1024 + network::ringbufferchannel::META_AREA;
const PORT: u8 = 1;

#[test]
fn rdma_channel_buffer_manager() {
    let config = NetworkConfig {
        receiver_socket: "127.0.0.1:8001".to_owned(),
        device_name: "mlx5_1".to_owned(),
        device_port: PORT,
        stoc_channel_name: "/stoc".to_owned(),
        ctos_channel_name: "/ctos".to_owned(),
        buf_size: BUF_SIZE,
        ..Default::default()
    };

    // First, new a RDMA server to listen at a socket address (s_sender_addr).
    // Then new a client with server's socket address to handshake with it.
    // The client side will use the server name (s_sender_name) to get its
    // remote info like raddr and rkey.

    let (r, s) = std::thread::scope(|scope| {
        let s_handler = scope.spawn(
            || match RDMAChannel::new_server(&config, 0) {
                (server, _) => {
                    println!("Server created successfully");
                    server
                }
            },
        );

        let c_handler = scope.spawn(
            || match RDMAChannel::new_client(&config, 0) {
                (client, _) => {
                    println!("Client created successfully");
                    client
                }
            },
        );

        (s_handler.join().unwrap(), c_handler.join().unwrap())
    });
    let recver = Channel::new(Box::new(r));
    let sender = Channel::new(Box::new(s));

    const SZ: usize = 256;
    let data = [48_u8; SZ];
    let send_memory = RawMemory::new(&data, SZ);
    sender.put_bytes(&send_memory).unwrap();

    let data2 = [97_u8; SZ];
    let send_memory = RawMemory::new(&data2, SZ);
    sender.put_bytes(&send_memory).unwrap();

    let _ = sender.flush_out();

    let mut buffer = [0u8; 2 * SZ];
    let mut recv_memory = RawMemoryMut::new(&mut buffer, 2 * SZ);
    match recver.get_bytes(&mut recv_memory) {
        Ok(size) => {
            for i in 0..SZ {
                assert_eq!(buffer[i], 48);
            }
            for i in SZ..size {
                assert_eq!(buffer[i], 97);
            }
        }
        Err(_) => todo!(),
    }
}
