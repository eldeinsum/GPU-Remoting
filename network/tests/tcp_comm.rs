use std::net::TcpListener;
use std::sync::{Arc, Barrier};

use network::{tcp, CommChannel, NetworkConfig, Transportable};

fn local_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

#[test]
fn tcp_channel_round_trip() {
    let config = NetworkConfig {
        comm_type: "tcp".to_string(),
        receiver_socket: format!("127.0.0.1:{}", local_port()),
        ..NetworkConfig::default()
    };
    let barrier = Arc::new(Barrier::new(2));

    std::thread::scope(|scope| {
        let server_barrier = Arc::clone(&barrier);
        let config_ref = &config;
        let server = scope.spawn(move || {
            let (receiver, sender) = tcp::new_server(config_ref, 0, &server_barrier).unwrap();
            let mut payload = [0u8; 5];
            payload.recv(&receiver).unwrap();
            assert_eq!(payload, [1, 2, 3, 4, 5]);

            0x1234_u32.send(&sender).unwrap();
            sender.flush_out().unwrap();
        });

        barrier.wait();
        let (sender, receiver) = tcp::new_client(&config, 0).unwrap();
        [1u8, 2, 3, 4, 5].send(&sender).unwrap();
        sender.flush_out().unwrap();

        let mut ack = 0u32;
        ack.recv(&receiver).unwrap();
        assert_eq!(ack, 0x1234);

        server.join().unwrap();
    });
}
