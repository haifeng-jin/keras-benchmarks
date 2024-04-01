#!/bin/bash

venv_path=~/.venv
venvs=(
    "torch"
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
    if [[ $venv_name == torch ]]; then
        file_name=torch
    else
        file_name=keras
    fi

    if [[ $venv_name == tensorflow ]]; then
        export KERAS_HOME=configs/tensorflow
    fi

    if [[ $venv_name == keras* ]]; then
        export KERAS_HOME=configs/${venv_name#keras-}
    fi

    printf "compiled\n\n"
    if [[ $venv_name == torch ]]; then
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

    deactivate
done
