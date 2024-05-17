import keras
import keras_nlp

import benchmark
from benchmark import keras_utils


def get_model():
    model = keras_nlp.models.GemmaCausalLM.from_preset(
        "gemma_2b_en",
        preprocessor=None,
    )
    model.backbone.enable_lora(rank=4)
    return model


def run(batch_size=benchmark.GEMMA_FIT_BATCH_SIZE):
    if hasattr(keras, "config"):
        keras.config.set_dtype_policy(benchmark.FLOAT_A100)
    else:
        keras.mixed_precision.set_global_policy(benchmark.FLOAT_A100)
    preprocessor = keras_nlp.models.GemmaCausalLMPreprocessor.from_preset(
        "gemma_7b_en", sequence_length=benchmark.GEMMA_SEQ_LENGTH
    )
    dataset = keras_utils.get_train_dataset_for_text_gen(
        preprocessor, batch_size, seq_len=benchmark.GEMMA_SEQ_LENGTH
    )
    model = get_model()

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.AdamW(),
        jit_compile=keras_utils.use_jit(),
    )
    return keras_utils.fit(model, dataset)


if __name__ == "__main__":
    benchmark.benchmark(run)
