import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .feedforward_block import FeedForwardBlock


class EncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        ff_hidden_size,
        num_heads,
        dropout_prob=0.1,
        layer_norm_eps=1e-5,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(hidden_size, hidden_size, num_heads)

        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.ff_block = FeedForwardBlock(hidden_size, ff_hidden_size, dropout_prob)

        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout2 = nn.Dropout(dropout_prob)

    def forward(self, x):
        x_a = self.attention(q=x, k=x, v=x)

        x = self.norm1(x + x_a)
        x = self.dropout1(x)

        x_f = self.ff_block(x)

        x = self.norm2(x + x_f)
        x = self.dropout2(x)

        return x
