import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, context_size):
        super().__init__()

        self.context_size = context_size
        self.register_buffer("tri", torch.tril(torch.ones(context_size, context_size)))

    def create_lookahead_mask(self, tgt_seq_len):
        return self.tri[:tgt_seq_len, :tgt_seq_len].unsqueeze(0)


class TransformerEncoderDecoder(Transformer):
    """Transformer encoder-decoder, using cross-attention with encoded source domain sequence when decoding output."""

    def __init__(
        self,
        vocab_size,
        context_size,
        hidden_size,
        ff_hidden_size,
        num_blocks=5,
        num_heads=8,
    ):
        super().__init__(context_size)

        self.encoder = Encoder(
            vocab_size,
            context_size,
            hidden_size,
            ff_hidden_size,
            num_blocks,
            num_heads,
        )
        self.decoder = Decoder(
            vocab_size,
            context_size,
            hidden_size,
            ff_hidden_size,
            num_blocks,
            num_heads,
            use_cross_attn=True,
        )

    def forward(self, x, src):
        decoder_lookahead_mask = self.create_lookahead_mask(x.shape[1])

        src_enc = self.encoder(src)
        out = self.decoder(x, src_enc, decoder_lookahead_mask)

        return out


class TransformerDecoder(Transformer):
    """Transformer decoder, using auto-regressive decoder blocks for language modeling."""

    def __init__(
        self,
        vocab_size,
        context_size,
        hidden_size,
        ff_hidden_size,
        num_blocks=5,
        num_heads=8,
    ):
        super().__init__(context_size)

        self.decoder = Decoder(
            vocab_size,
            context_size,
            hidden_size,
            ff_hidden_size,
            num_blocks,
            num_heads,
            use_cross_attn=False,
        )

    def forward(self, x):
        lookahead_mask = self.create_lookahead_mask(x.shape[1])

        out = self.decoder(x, None, lookahead_mask)

        return out
