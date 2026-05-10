#!/bin/bash

default_config_file="default_train.json"
if [ $# -eq 2 ]; then
  config_file="$2"
else
  config_file="$default_config_file"
fi

readarray -t rtt_values < <(jq -r '.rtt_values[]' "$config_file")
readarray -t bandwidth_values < <(jq -r '.bandwidth_values[]' "$config_file")
output_dir=$(jq -r '.output_dir' "$config_file")
readarray -t models < <(jq -r '.models[]' "$config_file")

set -e

cd ${BASH_SOURCE[0]%/*}
cd ../.. || {
  echo "Failed to change directory to root path"
  exit 1
}

if [ $# -eq 0 ]; then
  echo "usage: $0 [opt|raw] ([config_file])"
  exit 1
fi

if [ "$1" == "opt" ]; then
  OPT_FLAG="--features async_api,shadow_desc,local"
elif [ "$1" == "raw" ]; then
  OPT_FLAG=""
else
  echo "invalid param: $1"
  echo "usage: $0 [opt|raw]"
  exit 1
fi

declare -A model_params
model_params["BERT-pytorch"]="1 64"
model_params["ResNet18_Cifar10_95.46"]="1 64"
model_params["naifu-diffusion"]="1 1"

cargo build --release ${OPT_FLAG}
for rtt in "${rtt_values[@]}"; do
  for bandwidth in "${bandwidth_values[@]}"; do
    echo "Setting RTT to $rtt and Bandwidth to $bandwidth in config.toml"

    sed -i "s/^rtt = .*/rtt = $rtt/" ${config_path}
    sed -i "s/^bandwidth = .*/bandwidth = $bandwidth/" ${config_path}
    for model in "${models[@]}"; do
      params=${model_params[$model]}
      echo "Stopping old server instance if any..."
      pkill -f gpu-remoting-server || true
      echo "Running: RUST_LOG=warn cargo run --bin gpu-remoting-server"
      RUST_LOG=warn cargo run --release ${OPT_FLAG} --bin gpu-remoting-server >/dev/null 2>&1 &

      sleep 2

      echo "Running: RUST_LOG=warn run.sh train/${model}/train.py"
      cd tests/apps || {
        echo "Failed to change directory to tests/apps"
        exit 1
      }
      if [ ! -d "../../${output_dir}" ]; then
        mkdir -p "../../${output_dir}"
      fi
      NETWORK_CONFIG=../../config.toml RUST_LOG=warn ./run.sh train/${model}/train.py ${params} >"../../${output_dir}/${model}_train_($1)_${rtt}_${bandwidth}.log" 2>&1
      cd ../..

      echo "done ---"
    done

  done
done

echo "All operations completed."
