import time

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments

import benchmark
from benchmark import torch_utils


def run(batch_size=benchmark.BERT_BATCH_SIZE):
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
        per_device_eval_batch_size=batch_size,
    )

    trainer = Trainer(model=model, args=training_args)

    # Predict once to build the model.
    trainer.predict(dataset.select(list(range(batch_size))))

    start_time = time.time()
    trainer.predict(
        dataset.select(list(range((benchmark.NUM_STEPS + 1) * batch_size)))
    )
    end_time = time.time()
    total_time = end_time - start_time

    start_time = time.time()
    trainer.predict(dataset.select(list(range(batch_size))))
    end_time = time.time()
    total_time -= end_time - start_time

    inferencing_per_step = total_time / benchmark.NUM_STEPS * 1000
    return inferencing_per_step


if __name__ == "__main__":
    benchmark.benchmark(run)
