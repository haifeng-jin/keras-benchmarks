import time

import torch
from diffusers import StableDiffusionPipeline

import benchmark
from benchmark import torch_utils


def train(model, input_image, y_true):
    optimizer = torch.optim.Adam(model.parameters())

    def train_fn(model, input_image, y_true):
        optimizer.zero_grad()
        y_pred = model(input_image)
        loss = torch.nn.MSELoss()(y_pred, y_true)
        loss.backward()
        optimizer.step()

    compiled_train_fn = torch.compile(train_fn, mode=torch_utils.COMPILE_MODE)

    compiled_train_fn(model, input_image, y_true)

    start_time = time.time()
    for _ in range(benchmark.NUM_STEPS):
        compiled_train_fn(model, input_image, y_true)
    end_time = time.time()

    return (end_time - start_time) / benchmark.NUM_STEPS * 1000


def run(batch_size=benchmark.SD_FIT_BATCH_SIZE):
    model = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4"
    ).to("cuda")
    return train(
        model.vae.encoder,
        torch.rand(batch_size, 3, 512, 512).to("cuda"),
        torch.rand(batch_size, 8, 64, 64).to("cuda"),
    )


if __name__ == "__main__":
    benchmark.benchmark(run)
