import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import benchmark
from benchmark import torch_utils


def run(batch_size=benchmark.GEMMA_BATCH_SIZE):
    preset = "google/gemma-2b"
    model = AutoModelForCausalLM.from_pretrained(
        preset, torch_dtype=torch_utils.get_torch_dtype(benchmark.FLOAT_A100)
    ).cuda()
    if torch_utils.use_compile():
        model = torch.compile(model, mode=torch_utils.COMPILE_MODE)
    tokenizer = AutoTokenizer.from_pretrained(preset)
    tokenizer.pad_token = tokenizer.eos_token

    return torch_utils.generate(
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=benchmark.GEMMA_MAX_LENGTH,
    )


if __name__ == "__main__":
    benchmark.benchmark(run)
