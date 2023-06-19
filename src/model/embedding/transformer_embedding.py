import torch.nn as nn

from .token_embedding import TokenEmbedding
from .positional_encoding import PositionalEncodingLearned as PositionalEncoding


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, context_size, embedding_size):
        super().__init__()

        self.token_embedding = TokenEmbedding(vocab_size, embedding_size)
        self.pos_embedding = PositionalEncoding(context_size, embedding_size)

    def forward(self, tokens):
        return self.token_embedding(tokens) + self.pos_embedding(tokens)
