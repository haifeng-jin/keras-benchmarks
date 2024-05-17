import keras
import keras_cv
import numpy as np
import tensorflow as tf

import benchmark
from benchmark import keras_utils


def get_dataset(batch_size):
    if keras.backend.image_data_format() == "channels_last":
        images = np.random.rand(1, 1024, 1024, 3)
    else:
        images = np.random.rand(1, 3, 1024, 1024)

    data = {
        "images": images,
        "points": np.array([[[500, 375], [250, 375]]]),
        "labels": np.array([[1, 2]]),
    }
    return (
        tf.data.Dataset.from_tensor_slices(data)
        .repeat((benchmark.NUM_STEPS + 1) * batch_size)
        .batch(batch_size)
    )


def run(batch_size=benchmark.SAM_BATCH_SIZE):
    dataset = get_dataset(batch_size)
    model = keras_cv.models.SegmentAnythingModel.from_preset("sam_base_sa1b")
    backbone = model.backbone
    backbone.compile(jit_compile=keras_utils.use_jit())
    return keras_utils.predict(model, dataset)


if __name__ == "__main__":
    benchmark.benchmark(run)
