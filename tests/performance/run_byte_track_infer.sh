#!/bin/bash

set -e

OPT_FLAG="--features async_api,shadow_desc,local,log_rperf"

cd ${BASH_SOURCE[0]%/*}
cd ../.. || {
    echo "Failed to change directory to root path"
    exit 1
}

# "BERT" "gpt2" "ResNet18_Cifar10_95.46" "STABLEDIFFUSION-v1-4"
models=("BERT")

declare -A model_params
model_params["BERT"]="100 64"
model_params["ResNet18_Cifar10_95.46"]="100 64"
model_params["STABLEDIFFUSION-v1-4"]="10 1"
model_params["gpt2"]="100 512"
cargo build --release ${OPT_FLAG}

for model in "${models[@]}"; do
    params=${model_params[$model]}
    server_file="server_${model}.log"
    client_file="client_${model}.log"

    echo "Stopping old server instance if any..."
    pkill -f gpu-remoting-server || true

    echo "Running server"
    numactl --cpunodebind=0 cargo run --release ${OPT_FLAG} --bin gpu-remoting-server >"log/${server_file}" 2>&1 &

    sleep 3

    echo "Running: run.sh infer/${model}/inference.py ${params}"
    cd tests/apps || {
        echo "Failed to change directory to tests/apps"
        exit 1
    }
    NETWORK_CONFIG=../../config.toml numactl --cpunodebind=0 ./run.sh infer/${model}/inference.py ${params} >"../../log/${client_file}" 2>&1
    cd ../..

    echo "extract"

    python3 log/extract_server.py "log/${server_file}" "log/out_${server_file}"
    python3 log/extract_client.py "log/${client_file}" "log/out_${client_file}"

    # echo "merge"

    # python3 log/merge.py "log/out_${client_file}" "log/out_${model}.log"

    echo "done---"

done

echo "All operations completed."
