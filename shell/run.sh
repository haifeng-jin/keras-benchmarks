#!/bin/bash

# Check if environment name is provided as an argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <env_name>"
    exit 1
fi

env_name=$1
venv_path=~/.venv
output_file=results.txt

if [ -e "$output_file" ]; then
    rm -f "$output_file"
fi

export LD_LIBRARY_PATH=

models=(
    "bert"
    "sam"
    "stable_diffusion"
    "gemma"
    "mistral"
)

# Activate the virtual environment
source $venv_path/$env_name/bin/activate
export CUDNN_PATH=$(dirname $(python3.10 -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=${CUDNN_PATH}/lib

if [[ $env_name == torch ]]; then
    file_name=torch
else
    file_name=keras
fi

if [[ $env_name == tensorflow ]]; then
    export KERAS_HOME=configs/tensorflow
fi

if [[ $env_name == keras* ]]; then
    export KERAS_HOME=configs/${env_name#keras-}
fi

printf "# Benchmarking $env_name\n\n" | tee -a $output_file
printf "compiled\n\n"
if [[ $env_name == torch ]]; then
    export TORCH_COMPILE="1"
    for model_name in "${models[@]}"; do
        printf "$model_name:\n" | tee -a $output_file
        printf "fit:\n" | tee -a $output_file
        python benchmark/$model_name/$file_name/fit.py $output_file
        printf "predict:\n" | tee -a $output_file
        python benchmark/$model_name/$file_name/predict.py $output_file
        printf "\n\n" | tee -a $output_file
    done
    export TORCH_COMPILE="0"
    printf "not compiled\n\n"
fi

for model_name in "${models[@]}"; do
    printf "$model_name:\n" | tee -a $output_file
    printf "fit:\n" | tee -a $output_file
    python benchmark/$model_name/$file_name/fit.py $output_file
    printf "predict:\n" | tee -a $output_file
    python benchmark/$model_name/$file_name/predict.py $output_file
    printf "\n\n" | tee -a $output_file
done

# Deactivate the virtual environment
deactivate
