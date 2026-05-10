#!/bin/bash
# LD_LIBRARY_PATH=/opt/conda/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so:../../target/release:$LD_LIBRARY_PATH \
LD_LIBRARY_PATH=../../target/release/cuda-symlinks:../../target/release:$LD_LIBRARY_PATH \
LD_PRELOAD=../../target/release/libgpu_remoting_client.so:$LD_PRELOAD \
python3 "$@"
