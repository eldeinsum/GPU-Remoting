fn main() {
    #[cfg(target_os = "linux")]
    {
        use network::{ringbufferchannel::SHMChannel, Channel, Transportable};
        use std::boxed::Box;

        let shm_name = "/stoc";
        let shm_len = 1024;
        let channel = Channel::new(Box::new(SHMChannel::new_server(shm_name, shm_len).unwrap()));

        let mut dst = [0u8; 5];
        let res = dst.recv(&channel);
        match res {
            Ok(()) => println!("Received {:?}", dst),
            Err(e) => {
                println!("Error {}", e);
                panic!("failed to receive from channel: {e}");
            }
        }
    }

    println!("SHM server only works on linux");
}
