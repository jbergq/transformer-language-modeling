from turtle import forward
import torch.nn as nn

from .multi_head_attention import MultiHeadAttention


class DecoderBlock(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_heads):
        super().__init__()

        self.attention1 = MultiHeadAttention(embedding_size, hidden_size, num_heads)

        self.attention2 = MultiHeadAttention(embedding_size, hidden_size, num_heads)

    def forward(self, x):
        x_a = self.attention1(x)
