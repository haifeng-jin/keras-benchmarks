from peft import LoraConfig
from peft import get_peft_model
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments

import benchmark
from benchmark import torch_utils


def run(batch_size=benchmark.GEMMA_FIT_BATCH_SIZE):
    preset = "google/gemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(preset)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = torch_utils.get_train_dataset_for_text_gen(
        tokenizer, batch_size, seq_len=benchmark.GEMMA_SEQ_LENGTH
    )
    model = AutoModelForCausalLM.from_pretrained(
        preset, torch_dtype=torch_utils.get_torch_dtype(benchmark.FLOAT_A100)
    ).cuda()
    config = LoraConfig(r=4)
    model = get_peft_model(model, config)

    training_args = TrainingArguments(
        output_dir="test_trainer",
        per_device_train_batch_size=batch_size,
        num_train_epochs=1.0,
        torch_compile=torch_utils.use_compile(),
        torch_compile_mode=(
            torch_utils.COMPILE_MODE if torch_utils.use_compile() else None
        ),
        max_steps=benchmark.NUM_STEPS + 2,
    )

    timing_callback = torch_utils.TimingCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[timing_callback],
    )

    trainer.train()

    # Calculate overall training time
    overall_training_time = (
        timing_callback.end_time - timing_callback.start_time
    )
    training_per_step = overall_training_time / benchmark.NUM_STEPS * 1000

    return training_per_step


if __name__ == "__main__":
    benchmark.benchmark(run)
