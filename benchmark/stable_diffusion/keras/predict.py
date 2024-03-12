import time

import keras_cv

import benchmark
from benchmark import keras_utils


def run(batch_size=benchmark.SD_BATCH_SIZE):
    model = keras_cv.models.StableDiffusion(jit_compile=keras_utils.use_jit())
    prompts = "a photograph of an astronaut riding a horse"

    # Build the model by running.
    model.text_to_image(prompts, batch_size=batch_size, num_steps=2)

    # Run another time to get the time of first step and python overhead.
    start_time = time.time()
    model.text_to_image(prompts, batch_size=batch_size, num_steps=2)
    end_time = time.time()
    overhead_time = end_time - start_time

    # Benchmark the running time
    start_time = time.time()
    model.text_to_image(
        prompts,
        batch_size=batch_size,
        num_steps=benchmark.NUM_STEPS + 2,
    )
    end_time = time.time()
    total_time = end_time - start_time

    return (total_time - overhead_time) / benchmark.NUM_STEPS * 1000


if __name__ == "__main__":
    benchmark.benchmark(run)
