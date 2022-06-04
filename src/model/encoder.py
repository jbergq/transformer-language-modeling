import torch.nn as nn

from .embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):
    def __init__(self, vocab_size, max_seq_length, embedding_size):
        super().__init__()

        self.embedding = TransformerEmbedding(
            vocab_size, max_seq_length, embedding_size
        )

    def forward(self, src, tgt):
        src_emb = self.embedding(src)
