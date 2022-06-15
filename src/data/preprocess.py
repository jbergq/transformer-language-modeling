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
        tokens = self.tokenizer(input["text"])
        tokens = self.vocab(tokens)
        tokens = np.array(tokens)

        src, tgt = sample_sequences(tokens, self.seq_length)

        src, tgt = self.add_bos(src), self.add_bos(tgt)
        src, tgt = self.add_eos(src), self.add_eos(tgt)

        input["source"] = src
        input["target"] = tgt

        return input
