#!/bin/bash

# Check if environment name is provided as an argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <env_name>"
    exit 1
fi

env_name=$1
venv_path=~/.venv
output_file=results.txt
events_file=events.csv

if [ -e "$output_file" ]; then
    rm -f "$output_file"
fi

if [ -e "$events_file" ]; then
    rm -f "$events_file"
fi

get_timestamp() {
  date +"%s" # current time
}

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


printf "label,time\n" >> $events_file
printf "INIT,$(get_timestamp)\n" >> $events_file


if [[ $env_name == torch ]]; then
    export TORCH_COMPILE="1"
    for model_name in "${models[@]}"; do
        printf "$model_name:\n" | tee -a $output_file

        printf "fit:\n" | tee -a $output_file
        python metrics/nvidia_metrics.py $env_name "${model_name}_compiled" fit &
        PID_PYTHON=$!
        python benchmark/$model_name/$file_name/fit.py $output_file
        printf "${model_name}_FIT,$(get_timestamp)\n" >> $events_file
        kill $PID_PYTHON
        wait $PID_PYTHON

        printf "predict:\n" | tee -a $output_file
        python metrics/nvidia_metrics.py $env_name "${model_name}_compiled" predict &
        PID_PYTHON=$!
        python benchmark/$model_name/$file_name/predict.py $output_file
        printf "${model_name}_PREDICT,$(get_timestamp)\n" >> $events_file
        kill $PID_PYTHON
        wait $PID_PYTHON

        printf "\n\n" | tee -a $output_file
    done
    export TORCH_COMPILE="0"
    printf "not compiled\n\n"
fi


for model_name in "${models[@]}"; do
    printf "$model_name:\n" | tee -a $output_file
    
    printf "fit:\n" | tee -a $output_file
    python metrics/nvidia_metrics.py $env_name $model_name fit &
    PID_PYTHON=$!
    python benchmark/$model_name/$file_name/fit.py $output_file
    printf "${model_name}_FIT,$(get_timestamp)\n" >> $events_file
    kill $PID_PYTHON
    wait $PID_PYTHON

    printf "predict:\n" | tee -a $output_file
    python metrics/nvidia_metrics.py $env_name $model_name predict &
    PID_PYTHON=$!
    python benchmark/$model_name/$file_name/predict.py $output_file
    printf "${model_name}_PREDICT,$(get_timestamp)\n" >> $events_file
    kill $PID_PYTHON
    wait $PID_PYTHON

    printf "\n\n" | tee -a $output_file
done

# Deactivate the virtual environment
deactivate
