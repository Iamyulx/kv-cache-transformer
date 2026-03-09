class DummyTokenizer:
    """
    Simple ASCII tokenizer used for demonstration.
    """

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return ''.join([chr(t) for t in tokens])
