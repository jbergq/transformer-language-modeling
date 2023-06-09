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
        use_cross_attn,
        dropout_prob=0.1,
        layer_norm_eps=1e-5,
    ):
        super().__init__()

        self.attention1 = MultiHeadAttention(hidden_size, hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout_prob)

        if use_cross_attn:
            self.enc_dec_attention = MultiHeadAttention(
                hidden_size, hidden_size, num_heads
            )
            self.enc_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
            self.dropout2 = nn.Dropout(dropout_prob)

        self.ff_block = FeedForwardBlock(hidden_size, ff_hidden_size, dropout_prob)

        self.norm3 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout3 = nn.Dropout(dropout_prob)

    def forward(self, x, src=None, lookahead_mask=None):
        x_n = self.norm1(x)
        x_a = self.attention1(q=x_n, k=x_n, v=x_n, mask=lookahead_mask)

        x = self.dropout1(x + x_a)

        if src is not None:
            # Cross-attention with source domain.
            x_n = self.norm2(x)
            src_n = self.enc_norm(src)
            x_a = self.enc_dec_attention(q=x_n, k=src_n, v=src_n)

            x = self.dropout2(x + x_a)

        x_f = self.ff_block(self.norm3(x))

        x = self.dropout3(x + x_f)

        return x
