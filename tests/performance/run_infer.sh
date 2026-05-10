#!/bin/bash
default_config_file="default_infer.json"
if [ $# -eq 2 ]; then
  config_file="$2"
else
  config_file="$default_config_file"
fi

readarray -t rtt_values < <(jq -r '.rtt_values[]' "$config_file")
readarray -t bandwidth_values < <(jq -r '.bandwidth_values[]' "$config_file")
output_dir=$(jq -r '.output_dir' "$config_file")
readarray -t models < <(jq -r '.models[]' "$config_file")

bert_model_path=$(jq -r '.bert_model_path' "$config_file")
sd_model_path=$(jq -r '.sd_model_path' "$config_file")
gpt_model_path=$(jq -r '.gpt_model_path' "$config_file")
config_path=$(jq -r '.config_path' "$config_file")


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
model_params["BERT"]="1 64 ${bert_model_path}"
model_params["ResNet18_Cifar10_95.46"]="1 64"
model_params["STABLEDIFFUSION-v1-4"]="1 1 ${sd_model_path}"
model_params["gpt2"]="1 512 ${gpt_model_path}" 
cargo build --release ${OPT_FLAG}

for rtt in "${rtt_values[@]}"; do
  for bandwidth in "${bandwidth_values[@]}"; do
    echo "Setting RTT to $rtt and Bandwidth to $bandwidth in config.toml"

    sed -i "s/^rtt = .*/rtt = $rtt/" ${config_path}
    sed -i "s/^bandwidth = .*/bandwidth = $bandwidth/" ${config_path}
    for model in "${models[@]}"; do
      params=${model_params[$model]}

      echo "Stopping old server instance if any..."
      pkill -f server || true

      echo "Running: RUST_LOG=warn cargo run --bin server"
      RUST_LOG=warn cargo run --release ${OPT_FLAG} --bin server >/dev/null 2>&1 &

      sleep 2

      echo "Running: RUST_LOG=warn run.sh infer/${model}/inference.py"
      cd tests/apps || {
        echo "Failed to change directory to tests/apps"
        exit 1
      }
      if [ ! -d "../../${output_dir}" ]; then
            mkdir -p "../../${output_dir}"
      fi
      NETWORK_CONFIG=../../${config_path} RUST_LOG=warn ./run.sh infer/${model}/inference.py ${params} >"../../${output_dir}/${model}_infer_($1)_${batch_size}_${rtt}_${bandwidth}.log" 2>&1
      cd ../..

      echo "done ---"

    done

  done
done
pkill -f server
echo "All operations completed."