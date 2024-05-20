import keras
import keras_nlp

import benchmark
from benchmark import utils


def run(batch_size=benchmark.BERT_FIT_BATCH_SIZE):
    preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
        "bert_base_en", sequence_length=benchmark.BERT_SEQ_LENGTH
    )
    dataset = utils.get_train_dataset_for_text_classification(
        preprocessor=preprocessor,
        batch_size=batch_size,
        seq_len=benchmark.BERT_SEQ_LENGTH,
    )
    model = keras_nlp.models.BertClassifier.from_preset(
        "bert_base_en",
        num_classes=2,
        preprocessor=None,
    )
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.AdamW(),
        jit_compile=utils.use_jit(),
    )

    return utils.fit(model, dataset)


if __name__ == "__main__":
    benchmark.benchmark(run)
