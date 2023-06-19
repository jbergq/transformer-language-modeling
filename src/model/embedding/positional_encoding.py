from numpy import gradient
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, context_size, embedding_size, n=10000):
        super().__init__()

        i = torch.arange(embedding_size // 2)
        k = torch.arange(context_size).unsqueeze(dim=1)

        pos_embeddings = torch.zeros(context_size, embedding_size, requires_grad=False)
        pos_embeddings[:, 0::2] = torch.sin(k / (n ** (2 * i / embedding_size)))
        pos_embeddings[:, 1::2] = torch.cos(k / (n ** (2 * i / embedding_size)))

        self.register_buffer("pos_embeddings", pos_embeddings)

    def forward(self, x):
        return self.pos_embeddings[: x.shape[1], :]


class PositionalEncodingLearned(nn.Module):
    def __init__(self, context_size, embedding_size) -> None:
        super().__init__()

        self.pos_emb = nn.Embedding(context_size, embedding_size)

    def forward(self, x):
        return self.pos_emb(torch.arange(x.shape[1], device=x.device))
