use network::{ringbufferchannel::LocalChannel, CommChannelInnerIO, RawMemory, RawMemoryMut};

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

#[test]
fn test_ring_buffer_producer_consumer() {
    let channel = Arc::new(LocalChannel::new(
        1024 + network::ringbufferchannel::META_AREA,
    ));
    let producer_channel = Arc::clone(&channel);
    let consumer_channel = Arc::clone(&channel);

    let barrier = Arc::new(Barrier::new(2)); // Set up a barrier for 2 threads
    let producer_barrier = barrier.clone();
    let consumer_barrier = barrier.clone();

    let test_iters = 1000;

    // Producer thread
    let producer = thread::spawn(move || {
        producer_barrier.wait(); // Wait for both threads to be ready

        for i in 0..test_iters {
            let data = [(i % 256) as u8; 10]; // Simplified data to send
            let send_memory = RawMemory::new(&data, data.len());
            producer_channel.put_bytes(&send_memory).unwrap();
        }

        println!("Producer done");
    });

    // Consumer thread
    let consumer = thread::spawn(move || {
        consumer_barrier.wait(); // Wait for both threads to be ready

        let mut received = 0;
        let mut buffer = [0u8; 10];

        while received < test_iters {
            let len = buffer.len();
            let mut recv_memory = RawMemoryMut::new(&mut buffer, len);
            match consumer_channel.get_bytes(&mut recv_memory) {
                Ok(size) => {
                    for byte in buffer.iter().take(size) {
                        assert_eq!(*byte, (received % 256) as u8);
                    }

                    received += 1;
                }
                Err(_) => thread::sleep(Duration::from_millis(10)), // Wait if buffer is empty
            }
        }
    });

    // Note: producer must be joined later, since the consumer will reuse the buffer
    consumer.join().unwrap();
    producer.join().unwrap();
}
