import numpy as np
import torch.nn as nn

from src.data.sampling import sample_sequences


class PreProcess(nn.Module):
    def __init__(self, tokenizer, seq_length) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def forward(self, input):
        # Tokenize text.
        tokens = self.tokenizer(input["text"])
        tokens = np.array(tokens["input_ids"], dtype=object)

        # Sample sequences with the target equal to the source shifted by one.
        src, tgt = sample_sequences(tokens, self.seq_length)

        input["source"] = src
        input["target"] = tgt

        return input
