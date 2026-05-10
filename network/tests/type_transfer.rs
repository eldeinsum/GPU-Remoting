use cudasys::{cudart::cudaError_t, FromPrimitive};
use network::{ringbufferchannel::SHMChannel, Channel, CommChannel, Transportable};

use std::sync::{Arc, Barrier};
use std::thread;

#[test]
fn test_cudaerror() {
    let shm_name = "/stoc";
    let shm_len = 1024;

    let consumer_channel =
        Channel::new(Box::new(SHMChannel::new_server(shm_name, shm_len).unwrap()));
    let producer_channel =
        Channel::new(Box::new(SHMChannel::new_client(shm_name, shm_len).unwrap()));

    let barrier = Arc::new(Barrier::new(2)); // Set up a barrier for 2 threads
    let producer_barrier = barrier.clone();
    let consumer_barrier = barrier.clone();

    let test_iters = 1000;

    // Producer thread
    let producer = thread::spawn(move || {
        producer_barrier.wait(); // Wait for both threads to be ready

        for i in 0..test_iters {
            let var = match cudaError_t::from_u32(i % 10) {
                Some(v) => v,
                None => panic!("failed to convert from u32"),
            };
            var.send(&producer_channel).unwrap();
            producer_channel.flush_out().unwrap();
        }

        println!("Producer done");
    });

    // Consumer thread
    let consumer = thread::spawn(move || {
        consumer_barrier.wait(); // Wait for both threads to be ready

        let mut received = 0;

        while received < test_iters {
            let test = match cudaError_t::from_u32(received % 10) {
                Some(v) => v,
                None => panic!("failed to convert from u32"),
            };
            let mut var = cudaError_t::cudaSuccess;
            var.recv(&consumer_channel).unwrap();
            assert_eq!(var, test);
            received += 1;
        }
    });

    // Note: producer must be joined later, since the consumer will reuse the buffer
    consumer.join().unwrap();
    producer.join().unwrap();
}
