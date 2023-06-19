import torch.nn as nn

from .embedding.transformer_embedding import TransformerEmbedding
from .layers.decoder_block import DecoderBlock


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_size,
        hidden_size,
        ff_hidden_size,
        num_blocks=5,
        num_heads=8,
        use_cross_attn=False,
    ):
        super().__init__()

        self.embedding = TransformerEmbedding(vocab_size, context_size, hidden_size)

        self.decoder = []
        for _ in range(num_blocks):
            self.decoder.append(
                DecoderBlock(
                    hidden_size,
                    ff_hidden_size,
                    num_heads,
                    use_cross_attn,
                )
            )

        self.decoder = nn.ModuleList(self.decoder)

        self.ln_final = nn.LayerNorm(hidden_size)
        self.lin_final = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, src_enc=None, lookahead_mask=None):
        x = self.embedding(x)

        for block in self.decoder:
            x = block(x, src_enc, lookahead_mask)

        out = self.lin_final(self.ln_final(x))

        return out
