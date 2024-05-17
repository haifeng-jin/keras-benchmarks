import keras
import keras_cv
import numpy as np
import tensorflow as tf

import benchmark
from benchmark import keras_utils


def get_train_dataset(batch_size):
    if keras.backend.image_data_format() == "channels_last":
        images = np.random.rand(1, 1024, 1024, 3)
        features = np.random.rand(1, 64, 64, 256)
    else:
        images = np.random.rand(1, 3, 1024, 1024)
        features = np.random.rand(1, 256, 64, 64)

    return (
        tf.data.Dataset.from_tensor_slices((images, features))
        .repeat((benchmark.NUM_STEPS + 1) * batch_size)
        .batch(batch_size)
    )


def run(batch_size=benchmark.SAM_FIT_BATCH_SIZE):
    train_dataset = get_train_dataset(batch_size)
    model = keras_cv.models.SegmentAnythingModel.from_preset("sam_base_sa1b")
    backbone = model.backbone
    backbone.compile(
        loss="mse", optimizer="adam", jit_compile=keras_utils.use_jit()
    )
    return keras_utils.fit(backbone, train_dataset)


if __name__ == "__main__":
    benchmark.benchmark(run)
