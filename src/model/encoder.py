import torch.nn as nn

from .embedding.transformer_embedding import TransformerEmbedding
from .layers.encoder_layer import EncoderBlock


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        embedding_size,
        hidden_size,
        num_blocks=5,
        num_heads=8,
    ):
        super().__init__()

        self.embedding = TransformerEmbedding(vocab_size, max_seq_len, embedding_size)

        self.encoder = []

        for _ in range(num_blocks):
            self.encoder.append(EncoderBlock(embedding_size, hidden_size, num_heads))

        self.encoder = nn.Sequential(*self.encoder)

    def forward(self, src, tgt):
        x = self.embedding(src)
        src_enc = self.encoder(x)

        assert True
