import keras_nlp

import benchmark
from benchmark import keras_utils


def run(batch_size=benchmark.BERT_BATCH_SIZE):
    preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
        "bert_base_en", sequence_length=benchmark.BERT_SEQ_LENGTH
    )
    dataset = keras_utils.get_train_dataset_for_text_classification(
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
        jit_compile=keras_utils.use_jit(),
    )

    return keras_utils.predict(model, dataset)


if __name__ == "__main__":
    benchmark.benchmark(run)
