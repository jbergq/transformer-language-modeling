import math
import torch.nn as nn


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        batch_size, head, seq_length, head_size = k.shape

        w = q @ k.transpose(2, 3)
        w = w / math.sqrt(head_size)

        a = self.softmax(w)

        return a @ v
