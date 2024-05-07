# keras-benchmarking
Scripts for Keras 3 benchmarking.

## Hardware
* UNC Nabu1
* Host RAM: 64GB
* GPUs: 1 x NVIDIA A30
* GPU memory: 25GB

## Software
* Python 3.10
Refer to the text files under `requirements` directory for more detailed on
Python package versions for each framework.

## Permission setups

### HuggingFace setup

On the [HuggingFace Gemma model page](https://huggingface.co/google/gemma-7b),
make sure you have accepted the license near the top of the page.

```shell
pip install --upgrade huggingface_hub
```

```shell
huggingface-cli login
```

It may require you to input a token.
[More information about tokens.](https://huggingface.co/docs/hub/en/security-tokens)


### Kaggle setup

On the [Kaggle Gemma model page](https://www.kaggle.com/models/keras/gemma),
make sure you have accepted the license near the top of the page.

Sign in to Kaggle and go to `Settings > API > Create New Token`. After clicking,
it will download a `kaggle.json` file.

In the file, you will find your username and key. Append the following lines to
your `~/.bashrc` file. Make sure you replace the `<your_username>` and
`<your_key>` with the ones you found in `kaggle.json`.

```shell
export KAGGLE_USERNAME=<your_username>
export KAGGLE_KEY=<your_key>
```

## Running the benchmarks

First, build the needed images with 

```bash
bash shell/build.sh
```

That will create dedicated docker images with their python environments to run.

Then start docker compose with the metrics stats for monitoring.

```bash
docker-compose -f metrics/stack/docker-compose.yml up -d
``` 

Then you can run each docker image individually, as they need exclusivity on the GPU resources.

For keras-torch
```bash
docker run -v /users/jpmarshall/repos/keras-benchmarks/cached_models:/root/.cache/kagglehub/models/keras \
  --name keras-torch \
  -v /users/jpmarshall/repos/keras-benchmarks/benchmark/:/benchmark \
  -v /users/jpmarshall/repos/keras-benchmarks/configs/:/configs \
  -v /users/jpmarshall/repos/keras-benchmarks/shell/:/shell \
  -d --gpus device=GPU-0d58720e-34f6-3fd5-510d-e6d5249693f4 keras-torch
```

For keras-jax
```bash
docker run -v /users/jpmarshall/repos/keras-benchmarks/cached_models:/root/.cache/kagglehub/models/keras \
  --name keras-jax \
   -v /users/jpmarshall/repos/keras-benchmarks/benchmark/:/benchmark \
   -v /users/jpmarshall/repos/keras-benchmarks/configs/:/configs \
   -v /users/jpmarshall/repos/keras-benchmarks/shell/:/shell \
   -d --gpus device=GPU-0d58720e-34f6-3fd5-510d-e6d5249693f4 keras-jax 
```

For keras-tensorflow
```bash
docker run -v /users/jpmarshall/repos/keras-benchmarks/cached_models:/root/.cache/kagglehub/models/keras \
  --name keras-tensorflow \
  -v /users/jpmarshall/repos/keras-benchmarks/benchmark/:/benchmark \
  -v /users/jpmarshall/repos/keras-benchmarks/configs/:/configs \
  -v /users/jpmarshall/repos/keras-benchmarks/shell/:/shell \
  -d --gpus device=GPU-0d58720e-34f6-3fd5-510d-e6d5249693f4 keras-tensorflow 
```

For tensorflow
```bash
docker run -v /users/jpmarshall/repos/keras-benchmarks/cached_models:/root/.cache/kagglehub/models/keras \
  --name tensorflow \
  -v /users/jpmarshall/repos/keras-benchmarks/benchmark/:/benchmark \
  -v /users/jpmarshall/repos/keras-benchmarks/configs/:/configs \
  -v /users/jpmarshall/repos/keras-benchmarks/shell/:/shell \
  -d --gpus device=GPU-0d58720e-34f6-3fd5-510d-e6d5249693f4 tensorflow 
```

For torch
```bash
docker run -v /users/jpmarshall/repos/keras-benchmarks/cached_models:/root/.cache/kagglehub/models/keras \
  --name torch \
  -v /users/jpmarshall/repos/keras-benchmarks/benchmark/:/benchmark \
  -v /users/jpmarshall/repos/keras-benchmarks/configs/:/configs \
  -v /users/jpmarshall/repos/keras-benchmarks/shell/:/shell \
  -d --gpus device=GPU-0d58720e-34f6-3fd5-510d-e6d5249693f4 torch 
```

Any of those commands will deploy the image in a container with access to the GPU, in an idle state.
The volumes keep benchmark, configs and shell script folders updated with local. 
The cached models are kept in that volume, so you only download them once from the internet

From there, you need to exec to the container and run the benchmark.

Keras Torch
```bash
docker exec -it keras-torch /bin/bash
bash shell/run.sh kears-torch
```

Keras Jax
```bash
docker exec -it keras-jax /bin/bash
bash shell/run.sh keras-jax
```

Keras Tensorflow
```bash
docker exec -it keras-tensorflow /bin/bash
bash shell/run.sh keras-tensorflow
```

Tensorflow
```bash
docker exec -it tensorflow /bin/bash
bash shell/run.sh tensorflow
```

Torch
```bash
docker exec -it torch /bin/bash
bash shell/run.sh torch
```



## Directories

* `benchmark` contains the Python code for benchmarking each model. It is
  structured as a Python package. It needs `pip install -e .` before using. Most
  of the settings are in `benchmark/__init__.py`. You can run a single benchmark
  by calling each script, for example,
  `python benchmark/gemma/keras/predict.py results.txt`
* `shell` contains all the shell scripts for benchmarking.
* `requirements` contains the version requirements for the PyPI packages in the
  dependencies.
* `configs` contains the Keras config files for each backend.
