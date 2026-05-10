#!/bin/bash

LD_LIBRARY_PATH=../../target/release/cuda-symlinks:../../target/release:$LD_LIBRARY_PATH \
LD_PRELOAD=../../target/release/libclient.so:$LD_PRELOAD \
"$@"
