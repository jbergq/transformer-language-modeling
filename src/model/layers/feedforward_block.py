import torch.nn as nn


class FeedForwardBlock(nn.Module):
    def __init__(self, in_size, hidden_size, dropout_prob=0.1):
        super().__init__()

        self.lin1 = nn.Linear(in_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.lin2 = nn.Linear(hidden_size, in_size)

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)

        return x
