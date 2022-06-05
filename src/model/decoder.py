import torch.nn as nn

from .embedding.transformer_embedding import TransformerEmbedding
from .layers.decoder_block import DecoderBlock


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        hidden_size,
        ff_hidden_size,
        num_blocks=5,
        num_heads=8,
    ):
        super().__init__()

        self.embedding = TransformerEmbedding(vocab_size, max_seq_len, hidden_size)

        self.decoder = []
        for _ in range(num_blocks):
            self.decoder.append(DecoderBlock(hidden_size, ff_hidden_size, num_heads))

        self.decoder = nn.ModuleList(self.decoder)

        self.lin_final = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt, src_enc, tgt_mask):
        tgt = self.embedding(tgt)

        for block in self.decoder:
            tgt = block(tgt, src_enc, tgt_mask)

        out = self.lin_final(tgt)

        return out
