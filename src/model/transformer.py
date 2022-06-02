import torch.nn as nn

from .encoder import Encoder

class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()

        self.encoder = Encoder(vocab_size, embedding_size)

    def forward(self, src, tgt):
        encoded = self.encoder(src, tgt) 