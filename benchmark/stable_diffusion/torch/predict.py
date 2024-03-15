import time

import torch
from diffusers import StableDiffusionPipeline

import benchmark
from benchmark import torch_utils


@torch.no_grad
def inference(model, batch_size):
    prompts = ["a photograph of an astronaut riding a horse"] * batch_size

    # Generate once to build the model.
    model(prompts, height=512, width=512, num_inference_steps=1)

    start_time = time.time()
    model(
        prompts,
        height=512,
        width=512,
        num_inference_steps=benchmark.NUM_STEPS + 1,
    )
    end_time = time.time()
    total_time = end_time - start_time

    start_time = time.time()
    model(prompts, height=512, width=512, num_inference_steps=1)
    end_time = time.time()
    total_time -= end_time - start_time

    return total_time / benchmark.NUM_STEPS * 1000


def run(batch_size=benchmark.SD_BATCH_SIZE):
    model = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4"
    ).to("cuda")
    model = torch.compile(model, mode=torch_utils.COMPILE_MODE)
    return inference(model, batch_size=batch_size)


if __name__ == "__main__":
    benchmark.benchmark(run)
