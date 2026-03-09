import time
import torch

from model import MiniGPTCached
from tokenizer import DummyTokenizer
from generate import generate_fast


def run_benchmark():

    tokenizer = DummyTokenizer()

    vocab_size = 256
    dim = 128

    model = MiniGPTCached(vocab_size, dim=dim)

    start_prompt = "AI is"

    start = time.time()

    output = generate_fast(model, start_prompt, tokenizer)

    elapsed = time.time() - start

    print("Generated text:")
    print(output)
    print()
    print("Inference time with KV cache:", elapsed)


if __name__ == "__main__":
    run_benchmark()
