import torch.nn as nn

from .scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_heads=8):
        super().__init__()

        self.num_heads = num_heads

        self.attention = ScaleDotProductAttention()

        self.lin_q = nn.Linear(embedding_size, hidden_size)
        self.lin_k = nn.Linear(embedding_size, hidden_size)
        self.lin_v = nn.Linear(embedding_size, hidden_size)

        self.lin_concat = nn.Linear(hidden_size, embedding_size)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.lin_q(q), self.lin_k(k), self.lin_v(v)

        q, k, v = self.split(q), self.split(k), self.split(v)

        out = self.attention(q, k, v, mask)

        out = self.concat(out)
        out = self.lin_concat(out)

        return out

    def split(self, x):
        batch_size, seq_len, hidden_size = x.shape

        per_head_size = hidden_size // self.num_heads

        return x.view(batch_size, seq_len, self.num_heads, per_head_size).transpose(
            1, 2
        )

    def concat(self, x):
        batch_size, num_heads, seq_len, head_size = x.shape
        hidden_size = num_heads * head_size

        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
