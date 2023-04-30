import numpy as np
import torch.nn as nn

import torchtext.transforms as T

from src.data.sampling import sample_sequences


class PreProcess(nn.Module):
    def __init__(self, tokenizer, vocab, seq_length) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.vocab = T.VocabTransform(vocab)
        self.seq_length = seq_length

        self.add_bos = T.AddToken(token=0, begin=True)
        self.add_eos = T.AddToken(token=2, begin=False)

    def forward(self, input):
        # Tokenize text.
        tokens = self.tokenizer(input["text"])
        tokens = self.vocab(tokens)
        tokens = np.array(tokens, dtype=object)

        # Sample sequences with the target equal to the source shifted by one.
        src, tgt = sample_sequences(tokens, self.seq_length)

        # Add tokens to mark start and end of each sequence.
        src, tgt = self.add_bos(src), self.add_bos(tgt)
        src, tgt = self.add_eos(src), self.add_eos(tgt)

        input["source"] = src
        input["target"] = tgt

        return input
