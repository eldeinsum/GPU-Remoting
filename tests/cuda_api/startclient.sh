#!/bin/bash

LD_LIBRARY_PATH=../../target/release/cuda-symlinks:../../target/release:$LD_LIBRARY_PATH \
LD_PRELOAD=../../target/release/libgpu_remoting_client.so:$LD_PRELOAD \
"$@"
