import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, block_size):
        super().__init__()

        self.block_size = block_size
        self.register_buffer("tri", torch.tril(torch.ones(block_size, block_size)))

    def create_lookahead_mask(self, tgt_seq_len):
        return self.tri[:tgt_seq_len, :tgt_seq_len].unsqueeze(0)

    def generate(self, inp_seq, eos_token_id=2, max_output_len=100):
        B, T = inp_seq.shape
        device = inp_seq.device

        seq = inp_seq

        # Create mask for checking which generated sequences have encountered end-of-sequence (EOS) tokens.
        eos_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)

        for _ in range(max_output_len):
            out = self.forward(seq[..., -self.block_size :])  # Truncate input sequence.
            probs = F.softmax(out[:, -1, :], dim=1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            # Check for EOS and update mask.
            has_eos = next_tokens == eos_token_id
            eos_mask = eos_mask | has_eos

            # Set the next tokens to EOS tokens for sequences that have already encountered an EOS.
            next_tokens = torch.where(
                eos_mask, torch.tensor(eos_token_id, device=device), next_tokens
            )

            # Append the next tokens to the generated sequences.
            seq = torch.cat((seq, next_tokens), dim=-1)

            # Break if all sequences have encountered an EOS token.
            if eos_mask.all():
                break

        return seq


class TransformerEncoderDecoder(Transformer):
    """Transformer encoder-decoder, using cross-attention with encoded input sequence when decoding output."""

    def __init__(
        self, vocab_size, max_seq_len, hidden_size, ff_hidden_size, block_size
    ):
        super().__init__(block_size)

        self.encoder = Encoder(vocab_size, max_seq_len, hidden_size, ff_hidden_size)
        self.decoder = Decoder(vocab_size, max_seq_len, hidden_size, ff_hidden_size)

    def forward(self, src, tgt):
        tgt_mask = self.create_lookahead_mask(tgt.shape[1])

        src_enc = self.encoder(src)
        out = self.decoder(tgt, src_enc, tgt_mask)

        return out


class TransformerDecoder(Transformer):
    """Transformer decoder, using auto-regressive decoder blocks for language modeling."""

    def __init__(
        self, vocab_size, max_seq_len, hidden_size, ff_hidden_size, block_size
    ):
        super().__init__(block_size)

        self.decoder = Decoder(vocab_size, max_seq_len, hidden_size, ff_hidden_size)

    def forward(self, tgt):
        tgt_mask = self.create_lookahead_mask(tgt.shape[1])

        out = self.decoder(tgt, None, tgt_mask)

        return out
