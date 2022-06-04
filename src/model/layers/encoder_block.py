import torch.nn as nn

from .multi_head_attention import MultiHeadAttention


class EncoderBlock(nn.Module):
    def __init__(self, in_size, hidden_size, num_heads):
        super().__init__()

        self.attention = MultiHeadAttention(in_size, hidden_size, num_heads)

    def forward(self, x):
        x = self.attention(q=x, k=x, v=x)

        return x
