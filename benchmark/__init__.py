import itertools
import os
import sys

import requests

NUM_STEPS = 100
NUM_WORDS = 5
FLOAT_A100 = "bfloat16"

BERT_FIT_BATCH_SIZE = 32
BERT_BATCH_SIZE = 256
BERT_SEQ_LENGTH = 512

GEMMA_FIT_BATCH_SIZE = 8
GEMMA_BATCH_SIZE = 32
GEMMA_MAX_LENGTH = 50
GEMMA_SEQ_LENGTH = 128

MISTRAL_FIT_BATCH_SIZE = 8
MISTRAL_BATCH_SIZE = 32
MISTRAL_MAX_LENGTH = 50
MISTRAL_SEQ_LENGTH = 128

SAM_FIT_BATCH_SIZE = 1
SAM_BATCH_SIZE = 7

SD_FIT_BATCH_SIZE = 8
SD_BATCH_SIZE = 13


def append_to_file(file_path, content):
    try:
        with open(file_path, "a") as file:
            file.write(content + "\n")
        print(f"Content appended to {file_path}")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")


def benchmark(run):
    if len(sys.argv) not in (2, 3):
        print("Usage: python bert/keras/fit.py <file_path> [batch_size]")
    else:
        if len(sys.argv) == 3:
            batch_size = int(sys.argv[2])
            per_step = run(batch_size=batch_size)
            content = f"{1000*batch_size/per_step} examples/s\n"
        else:
            per_step = run()
            content = f"{per_step} ms/step\n"
        print(content)
        file_path = sys.argv[1]
        append_to_file(file_path, content)


def get_prompts(num_prompts, num_words):
    dictionary = (
        "I you get take it of for from can not do use would yes".split()
    )
    iter = itertools.product(dictionary, repeat=num_words)
    return [" ".join(next(iter)) for i in range(num_prompts)]


def download_file(url, local_filename):
    if not os.path.exists(local_filename):
        os.makedirs(os.path.dirname(local_filename), exist_ok=True)
        print(f"Downloading {url}...")
        response = requests.get(url)
        with open(local_filename, "wb") as file:
            file.write(response.content)
        print(f"Download complete: {local_filename}")
    else:
        print(f"File already exists: {local_filename}")
