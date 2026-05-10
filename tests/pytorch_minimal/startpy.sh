#!/bin/bash

LD_LIBRARY_PATH=../../target/debug/cuda-symlinks:../../target/debug:$LD_LIBRARY_PATH \
LD_PRELOAD=../../target/debug/libclient.so:$LD_PRELOAD \
python3 "$@"
