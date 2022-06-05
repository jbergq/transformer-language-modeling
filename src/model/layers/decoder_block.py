from turtle import forward
import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .feedforward_block import FeedForwardBlock


class DecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        ff_hidden_size,
        num_heads,
        dropout_prob=0.1,
        layer_norm_eps=1e-5,
    ):
        super().__init__()

        self.attention1 = MultiHeadAttention(hidden_size, hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.enc_dec_attention = MultiHeadAttention(hidden_size, hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout2 = nn.Dropout(dropout_prob)

        self.ff_block = FeedForwardBlock(hidden_size, ff_hidden_size, dropout_prob)

        self.norm3 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout3 = nn.Dropout(dropout_prob)

    def forward(self, tgt, src, tgt_mask):
        x_a = self.attention1(q=tgt, k=tgt, v=tgt, mask=tgt_mask)

        x = self.norm1(tgt + x_a)
        x = self.dropout1(x)

        if src is not None:
            x_a = self.enc_dec_attention(q=x, k=src, v=src)

            x = self.norm2(x + x_a)
            x = self.dropout2(x)

        x_f = self.ff_block(x)

        x = self.norm3(x + x_f)
        x = self.dropout3(x)

        return x
