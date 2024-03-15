# keras-benchmarking
Scripts for Keras 3 benchmarking.

## Hardware
* Google Cloud Platform
* Compute Engine
* Machine type: a2-highgpu-1g
* Host RAM: 85GB
* GPUs: 1 x NVIDIA A100
* GPU memory: 40GB

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

First, change directory to the root directory of the repository.

```bash
cd keras-benchmarking/
```

Then, create Python vritual environments for all the frameworks under
`~/.venv/`. Make sure you have `pip` and `venv` installed before running the
script.

```bash
bash shell/install.sh
```

To run the benchmarks, you can run the following script.

```bash
bash shell/run.sh
```

If you want to remove all the virtual environments afterwards or if you
encounter an error want to clean up the half-way installed dependencies, you can
run `shell/cleanup.sh`.

## Directories

* `benchmark` contains the Python code for benchmarking each model. It is
  structured as a Python package. I needs `pip install -e .` before using. Most
  of the settings are in `benchmark/__init__.py`. You can run a single benchmark
  by calling each script, for example,
  `python benchmark/gemma/keras/predict.py results.txt`
* `shell` contains all the shell scripts for benchmarking.
* `requirements` contains the version requirements for the PyPI packages in the
  dependencies.
* `configs` contains the Keras config files for each backend.
