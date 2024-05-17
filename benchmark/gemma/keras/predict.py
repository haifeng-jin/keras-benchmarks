import keras
import keras_nlp

import benchmark
from benchmark import keras_utils


def run(batch_size=benchmark.GEMMA_BATCH_SIZE):
    if hasattr(keras, "config"):
        keras.config.set_dtype_policy(benchmark.FLOAT_A100)
    else:
        keras.mixed_precision.set_global_policy(benchmark.FLOAT_A100)
    model = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
    model.compile(sampler="greedy")
    return keras_utils.generate(
        model=model,
        batch_size=batch_size,
        max_length=benchmark.GEMMA_MAX_LENGTH,
    )


if __name__ == "__main__":
    benchmark.benchmark(run)
