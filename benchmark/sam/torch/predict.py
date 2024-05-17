import time

import segment_anything
import torch

import benchmark
from benchmark import torch_utils

HUGE_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
)
HUGE_BUILD = segment_anything.build_sam_vit_h
HUGE_LOCAL = "/tmp/sam_h.pth"
LARGE_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
)
LARGE_BUILD = segment_anything.build_sam_vit_l
LARGE_LOCAL = "/tmp/sam_l.pth"
BASE_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
)
BASE_BUILD = segment_anything.build_sam_vit_b
BASE_LOCAL = "/tmp/sam_b.pth"

URL = BASE_URL
LOCAL = BASE_LOCAL
build_sam = BASE_BUILD


def get_dataset(batch_size):
    input_image = torch.Tensor(batch_size, 3, 1024, 1024).cuda()
    input_point = torch.Tensor([[[500, 375], [250, 375]]]).cuda()
    input_label = torch.Tensor([[1, 2]]).cuda()
    return input_image, input_point, input_label


@torch.no_grad
def inference(model, input_image, input_point, input_label):
    features = model.image_encoder(input_image)
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=(input_point, input_label), boxes=None, masks=None
    )
    return model.mask_decoder(
        image_embeddings=features,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
    )


def run(batch_size=benchmark.SAM_BATCH_SIZE):
    benchmark.download_file(URL, LOCAL)
    model = build_sam(checkpoint=LOCAL).cuda()
    input_image, input_point, input_label = get_dataset(batch_size)
    inference_fn = inference
    if torch_utils.use_compile():
        inference_fn = torch.compile(
            inference_fn, mode=torch_utils.COMPILE_MODE
        )

    # Inference twice to build the model
    inference_fn(model, input_image, input_point, input_label)
    inference_fn(model, input_image, input_point, input_label)

    start_time = time.time()
    for i in range(benchmark.NUM_STEPS + 1):
        inference_fn(model, input_image, input_point, input_label)
    end_time = time.time()
    total_time = end_time - start_time

    start_time = time.time()
    inference_fn(model, input_image, input_point, input_label)
    end_time = time.time()
    total_time -= end_time - start_time

    inference_time = total_time / benchmark.NUM_STEPS * 1000
    return inference_time


if __name__ == "__main__":
    benchmark.benchmark(run)
