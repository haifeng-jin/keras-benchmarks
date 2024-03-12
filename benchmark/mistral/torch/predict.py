import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import benchmark
from benchmark import torch_utils


def run(batch_size=benchmark.MISTRAL_BATCH_SIZE):
    preset = "mistralai/Mistral-7B-v0.1"
    model = AutoModelForCausalLM.from_pretrained(
        preset, torch_dtype=torch_utils.get_torch_dtype(benchmark.FLOAT_A100)
    ).cuda()
    model = torch.compile(model)
    tokenizer = AutoTokenizer.from_pretrained(preset)
    tokenizer.pad_token = tokenizer.eos_token

    return torch_utils.generate(
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=benchmark.MISTRAL_MAX_LENGTH,
    )


if __name__ == "__main__":
    benchmark.benchmark(run)
