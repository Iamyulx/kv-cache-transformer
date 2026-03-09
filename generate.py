import torch


def generate_fast(model, start_text, tokenizer, steps=100):
    """
    Generate text using KV cache for fast autoregressive inference.
    """

    tokens = tokenizer.encode(start_text)

    x = torch.tensor(tokens).unsqueeze(0)

    cache = None

    for _ in range(steps):

        logits, cache = model(x[:, -1:], cache)

        next_token = torch.argmax(logits[:, -1, :], dim=-1)

        x = torch.cat([x, next_token.unsqueeze(0)], dim=1)

    return tokenizer.decode(x[0].tolist())
