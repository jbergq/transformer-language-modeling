import torch.nn as nn

from .token_embedding import TokenEmbedding

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()

        self.token_embedding = TokenEmbedding(vocab_size, embedding_size)

    def forward(self, tokens):
        return self.token_embedding(tokens)