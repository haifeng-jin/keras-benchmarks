#!/bin/bash

venv_path=~/.venv
venvs=(
    "tensorflow"
    "keras-tensorflow"
    "keras-jax"
    "keras-torch"
)
output_file=results.txt

if [ -e "$output_file" ]; then
    rm -f "$output_file"
fi

export LD_LIBRARY_PATH=
export NVIDIA_TF32_OVERRIDE=0

models=(
    "bert"
    "sam"
    "stable_diffusion"
    "gemma"
    "mistral"
)

for venv_name in "${venvs[@]}"; do
    printf "# Benchmarking $venv_name\n\n" | tee -a $output_file
    source $venv_path/$venv_name/bin/activate

    if [[ $venv_name == tensorflow ]]; then
        export KERAS_HOME=configs/tensorflow
    fi

    if [[ $venv_name == keras* ]]; then
        export KERAS_HOME=configs/${venv_name#keras-}
    fi

    for model_name in "${models[@]}"; do
        printf "$model_name:\n" | tee -a $output_file
        printf "fit:\n" | tee -a $output_file
        python benchmark/$model_name/fit.py $output_file
        printf "predict:\n" | tee -a $output_file
        python benchmark/$model_name/predict.py $output_file
        printf "\n\n" | tee -a $output_file
    done

    deactivate
done
