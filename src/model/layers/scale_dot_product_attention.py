import math
import torch.nn as nn


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch_size, head, seq_length, head_size = k.shape

        score = (q @ k.transpose(2, 3)) / math.sqrt(head_size)

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        score = self.softmax(score)

        return score @ v
