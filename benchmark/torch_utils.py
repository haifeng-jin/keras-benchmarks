import os
import random
import time

import torch
from datasets import Dataset
from transformers import TrainerCallback

import benchmark

TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

COMPILE_MODE = "reduce-overhead"


class TimingCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.end_time = None

    def on_step_begin(self, args, state, control, **kwargs):
        # Record start time only once at the beginning of the second step
        # Steps are [0, 101].
        if state.global_step == 2 and self.start_time is None:
            self.start_time = time.time()
        super().on_step_begin(args, state, control, **kwargs)

    def on_step_end(self, args, state, control, **kwargs):
        super().on_step_end(args, state, control, **kwargs)
        # Record end time at the end of the last step
        # Steps are [0, 101].
        if state.global_step == benchmark.NUM_STEPS + 1:
            self.end_time = time.time()


def generate(
    model,
    tokenizer,
    batch_size,
    max_length,
):
    inputs = benchmark.get_prompts(batch_size, benchmark.NUM_WORDS)
    num_input_tokens = benchmark.NUM_WORDS

    def generate_once():
        tokenized_inputs = tokenizer(
            inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        outputs = model.generate(
            **tokenized_inputs,
            max_new_tokens=max_length - num_input_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
        tokenizer.decode(outputs[0])

    # Generate twice to build the model.
    generate_once()
    generate_once()

    start_time = time.time()
    for _ in range(benchmark.NUM_STEPS + 1):
        generate_once()
    end_time = time.time()
    total_time = end_time - start_time

    start_time = time.time()
    generate_once()
    end_time = time.time()
    total_time -= end_time - start_time

    return total_time / benchmark.NUM_STEPS * 1000


def get_torch_dtype(dtype):
    return TORCH_DTYPES[dtype]


def _get_text_and_label(num_prompts, num_words):
    def gen():
        for prompt in benchmark.get_prompts(
            num_prompts=num_prompts,
            num_words=num_words,
        ):
            yield {"text": prompt, "label": random.randint(0, 1)}

    return Dataset.from_generator(gen)


def get_train_dataset_for_text_classification(tokenizer, batch_size, seq_len):
    dataset = _get_text_and_label(
        num_prompts=batch_size * (benchmark.NUM_STEPS + 1),
        num_words=seq_len,
    )

    tokenized_datasets = dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            padding="max_length",
            max_length=seq_len,
            truncation=True,
        ),
        batched=True,
    )

    return tokenized_datasets


def get_train_dataset_for_text_gen(tokenizer, batch_size, seq_len):
    dataset = _get_text_and_label(
        num_prompts=batch_size * (benchmark.NUM_STEPS + 1),
        num_words=seq_len,
    )

    # Tokenize the dataset
    def tokenize_batch(batch):
        batch = tokenizer(
            batch["text"],
            padding="max_length",
            max_length=seq_len,
            truncation=True,
        )
        batch["labels"] = batch["input_ids"].copy()
        return batch

    tokenized_dataset = dataset.map(tokenize_batch, batched=True)
    tokenized_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    return tokenized_dataset


def use_compile():
    return os.environ.get("TORCH_COMPILE", "0") == "1"
