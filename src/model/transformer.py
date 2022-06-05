import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, hidden_size, ff_hidden_size):
        super().__init__()

        self.encoder = Encoder(vocab_size, max_seq_len, hidden_size, ff_hidden_size)
        self.decoder = Decoder(vocab_size, max_seq_len, hidden_size, ff_hidden_size)

    def forward(self, src, tgt):
        tgt_mask = self.create_lookahead_mask(tgt.shape[1])

        src_enc = self.encoder(src)
        out = self.decoder(tgt, src_enc, tgt_mask)

        return out

    def create_lookahead_mask(self, tgt_seq_len):
        ones = torch.ones(tgt_seq_len, tgt_seq_len, dtype=torch.uint8)

        return torch.triu(ones, diagonal=1).unsqueeze(0)
