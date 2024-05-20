import keras
import keras_cv
import numpy as np
import tensorflow as tf

import benchmark
from benchmark import utils


def get_train_dataset(batch_size):
    if keras.backend.image_data_format() == "channels_last":
        images = np.random.rand(1, 512, 512, 3)
        features = np.random.rand(1, 64, 64, 8)
    else:
        images = np.random.rand(1, 3, 512, 512)
        features = np.random.rand(1, 8, 64, 64)

    return (
        tf.data.Dataset.from_tensor_slices((images, features))
        .repeat((benchmark.NUM_STEPS + 1) * batch_size)
        .batch(batch_size)
    )


def run(batch_size=benchmark.SD_FIT_BATCH_SIZE):
    train_dataset = get_train_dataset(batch_size=batch_size)
    model = keras_cv.models.StableDiffusion(jit_compile=utils.use_jit())
    backbone = keras.Model(
        model.image_encoder.inputs, model.image_encoder.layers[-3].output
    )
    backbone.compile(loss="mse", optimizer="adam")
    return utils.fit(backbone, train_dataset)


if __name__ == "__main__":
    benchmark.benchmark(run)
