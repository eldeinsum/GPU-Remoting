# GPU-Remoting

GPU-Remoting is a CUDA API remoting runtime. Applications load a client shared library with `LD_PRELOAD`; CUDA calls are marshalled through the network layer and executed by a GPU-side server process.

This fork removes the PhoenixOS/libpos integration path and targets GPU-CR for NVIDIA checkpoint and restore. The default execution path uses CUDA directly on the remoting server, and GPU-CR is applied by running the server under the GPU-CR NVIDIA runtime.

## Components

- `server`: GPU-side CUDA API execution server.
- `client`: preload library used by applications.
- `network`: shared memory, TCP, and optional RDMA transports.
- `codegen` and `cudasys`: CUDA binding and hook generation.

## Requirements

- Linux
- CUDA Toolkit and NVIDIA driver
- Rust nightly, pinned by `rust-toolchain.toml`
- GPU-CR built with the NVIDIA runtime

GPU-CR expected layout:

```bash
cd ~/Projects/GPU-CR
cmake -S . -B build-nvidia -DGPU_VENDOR=NVIDIA -DGPUCR_BUILD_CPP=OFF -DGPUCR_BUILD_RUST=ON -DGPUCR_RUST_RELEASE=ON
cmake --build build-nvidia -j$(nproc)
```

## Build

```bash
cd ~/Projects/GPU-Remoting
cargo build --workspace
```

For release builds:

```bash
cargo build --workspace --release
```

## Config

Copy `config.example.toml` to `config.toml` to customize the default settings. Set `NETWORK_CONFIG` to override the config path.

The default transport is shared memory:

```toml
comm_type = "shm"
```

## Running

Start the GPU-side server under GPU-CR:

```bash
GPUCR_HOME=~/Projects/GPU-CR \
scripts/server --release
```

Run a CUDA application through the client preload library:

```bash
scripts/client --release ./path/to/cuda_app
```

Checkpoint and restore the remoting server process:

```bash
scripts/checkpoint checkpoint <server-pid>
scripts/checkpoint restore <server-pid>
```

`GPUCR_RUNTIME` and `GPUCR_CLIENT` can be set to override the default GPU-CR runtime and client paths.

## Tests

Build the CUDA API tests:

```bash
cmake -S tests/cuda_api -B tests/cuda_api/build
cmake --build tests/cuda_api/build -j$(nproc)
```

Run the server in one terminal:

```bash
scripts/server
```

Run tests through the client in another terminal:

```bash
for test in tests/cuda_api/build/test_*; do
    scripts/client --release "$test"
done
```

## Development

New CUDA APIs are declared in `cudasys/src/hooks/*.rs`. The build scripts regenerate client hijacks and server dispatchers from those declarations.

See [docs/implement_new_apis.md](docs/implement_new_apis.md).

## Attribution

GPU-Remoting is based on PhoenixOS-Remoting by SJTU-IPADS: https://github.com/SJTU-IPADS/PhoenixOS-Remoting

GPU-CR integration targets the NVIDIA runtime in this fork: https://github.com/eldeinsum/GPU-CR
