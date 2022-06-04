import torch.nn as nn

from .encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, embedding_size, hidden_size):
        super().__init__()

        self.encoder = Encoder(vocab_size, max_seq_len, embedding_size, hidden_size)

    def forward(self, src, tgt):
        encoded = self.encoder(src, tgt)
