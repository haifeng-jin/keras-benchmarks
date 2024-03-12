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
