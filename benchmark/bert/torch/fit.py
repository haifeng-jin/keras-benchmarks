from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments

import benchmark
from benchmark import torch_utils


def run(batch_size=benchmark.BERT_FIT_BATCH_SIZE):
    dataset = torch_utils.get_train_dataset_for_text_classification(
        AutoTokenizer.from_pretrained("bert-base-cased"),
        batch_size=batch_size,
        seq_len=benchmark.BERT_SEQ_LENGTH,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased",
        num_labels=2,
    )

    training_args = TrainingArguments(
        output_dir="test_trainer",
        per_device_train_batch_size=batch_size,
        num_train_epochs=1.0,
        max_steps=benchmark.NUM_STEPS + 1,
        torch_compile=True,
        torch_compile_mode=torch_utils.COMPILE_MODE,
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
