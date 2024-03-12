import time

import torch
from diffusers import StableDiffusionPipeline

import benchmark


def train(model, input_image, y_true):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    optimizer.zero_grad()
    y_pred = model(input_image)
    loss = loss_fn(y_pred, y_true)
    loss.backward()
    optimizer.step()

    start_time = time.time()
    for _ in range(benchmark.NUM_STEPS):
        optimizer.zero_grad()
        y_pred = model(input_image)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()
    end_time = time.time()

    return (end_time - start_time) / benchmark.NUM_STEPS * 1000


def run(batch_size=benchmark.SD_FIT_BATCH_SIZE):
    model = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4"
    ).to("cuda")
    model = torch.compile(model)
    return train(
        model.vae.encoder,
        torch.rand(batch_size, 3, 512, 512).to("cuda"),
        torch.rand(batch_size, 8, 64, 64).to("cuda"),
    )


if __name__ == "__main__":
    benchmark.benchmark(run)
