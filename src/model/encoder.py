import torch.nn as nn

from .embedding.transformer_embedding import TransformerEmbedding
from .layers.encoder_block import EncoderBlock


class Encoder(nn.Module):
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

        self.encoder = []
        for _ in range(num_blocks):
            self.encoder.append(EncoderBlock(hidden_size, ff_hidden_size, num_heads))

        self.encoder = nn.ModuleList(self.encoder)

    def forward(self, src):
        src = self.embedding(src)

        for block in self.encoder:
            src = block(src)

        return src
