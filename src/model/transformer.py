import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self, vocab_size, max_seq_len, embedding_size, hidden_size, ff_hidden_size
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_size, max_seq_len, embedding_size, hidden_size, ff_hidden_size
        )
        self.decoder = Decoder(
            vocab_size, max_seq_len, embedding_size, hidden_size, ff_hidden_size
        )

    def forward(self, src, tgt):
        src_enc = self.encoder(src)
