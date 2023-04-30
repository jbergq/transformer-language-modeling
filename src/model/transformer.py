import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    def create_lookahead_mask(self, tgt_seq_len):
        ones = torch.ones(tgt_seq_len, tgt_seq_len, dtype=torch.uint8)

        return torch.tril(ones, diagonal=0).unsqueeze(0)


class TransformerEncoderDecoder(Transformer):
    """Transformer encoder-decoder, using cross-attention with encoded input sequence when decoding output."""

    def __init__(self, vocab_size, max_seq_len, hidden_size, ff_hidden_size):
        super().__init__()

        self.encoder = Encoder(vocab_size, max_seq_len, hidden_size, ff_hidden_size)
        self.decoder = Decoder(vocab_size, max_seq_len, hidden_size, ff_hidden_size)

    def forward(self, src, tgt):
        tgt_mask = self.create_lookahead_mask(tgt.shape[1])

        src_enc = self.encoder(src)
        out = self.decoder(tgt, src_enc, tgt_mask)

        return out


class TransformerDecoder(Transformer):
    """Transformer decoder, using auto-regressive decoder blocks for language modeling."""

    def __init__(self, vocab_size, max_seq_len, hidden_size, ff_hidden_size):
        super().__init__()

        self.decoder = Decoder(vocab_size, max_seq_len, hidden_size, ff_hidden_size)

    def forward(self, tgt):
        tgt_mask = self.create_lookahead_mask(tgt.shape[1])

        out = self.decoder(tgt, None, tgt_mask)

        return out

    def create_lookahead_mask(self, tgt_seq_len):
        ones = torch.ones(tgt_seq_len, tgt_seq_len, dtype=torch.uint8)

        return torch.tril(ones, diagonal=0).unsqueeze(0)
