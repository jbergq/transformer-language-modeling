import torch.nn as nn

from .embedding.transformer_embedding import TransformerEmbedding
from .layers.decoder_block import DecoderBlock


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        embedding_size,
        hidden_size,
        ff_hidden_size,
        num_blocks=5,
        num_heads=8,
    ):
        super().__init__()

        self.embedding = TransformerEmbedding(vocab_size, max_seq_len, embedding_size)

        self.decoder = []

        for _ in range(num_blocks):
            self.decoder.append(
                DecoderBlock(
                    in_size=embedding_size,
                    hidden_size=hidden_size,
                    ff_hidden_size=ff_hidden_size,
                    num_heads=num_heads,
                )
            )

        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        pass
