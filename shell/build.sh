#!/bin/bash

image_already_exists() {
    local image_name="$1"
    docker image inspect "$image_name" &> /dev/null
}

# Build Base cuda with python 3.10 image
if ! image_already_exists "cuda-python310:latest"; then
    docker build -t cuda-python310 -f Dockerfile .
fi

# Build Tensorflow standalone image
if ! image_already_exists "tensorflow"; then
    docker build -t tensorflow -f Dockerfile.tensorflow .
fi

# Build Torch standalone image
if ! image_already_exists "torch"; then
    docker build -t torch -f Dockerfile.torch .
fi

# Build Keras with torch backend image
if ! image_already_exists "keras-torch"; then
    docker build -t keras-torch -f Dockerfile.keras-torch .
fi

# Build Keras with jax backend image
if ! image_already_exists "keras-jax"; then
    docker build -t keras-jax -f Dockerfile.keras-jax .
fi

# Build Keras with tensorflow backend image
if ! image_already_exists "keras-tensorflow"; then
    docker build -t keras-tensorflow -f Dockerfile.keras-tensorflow .
fi
