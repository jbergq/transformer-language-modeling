import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, max_seq_len):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.register_buffer("tri", torch.tril(torch.ones(max_seq_len, max_seq_len)))

    def create_lookahead_mask(self, tgt_seq_len):
        return self.tri[:tgt_seq_len, :tgt_seq_len].unsqueeze(0)

    def generate(self, inp_seq, eos_token_id=2, max_output_len=100):
        B, T = inp_seq.shape
        device = inp_seq.device

        seq = inp_seq

        # Create mask for checking which generated sequences have encountered end-of-sequence (EOS) tokens.
        eos_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)

        for _ in range(max_output_len):
            out = self.forward(
                seq[..., -self.max_seq_len :]  # Truncate input sequence to max length.
            )
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
    """Transformer encoder-decoder, using cross-attention with encoded source domain sequence when decoding output."""

    def __init__(self, vocab_size, max_seq_len, hidden_size, ff_hidden_size):
        super().__init__(max_seq_len)

        self.encoder = Encoder(vocab_size, max_seq_len, hidden_size, ff_hidden_size)
        self.decoder = Decoder(vocab_size, max_seq_len, hidden_size, ff_hidden_size)

    def forward(self, x, src):
        decoder_lookahead_mask = self.create_lookahead_mask(x.shape[1])

        src_enc = self.encoder(src)
        out = self.decoder(x, src_enc, decoder_lookahead_mask)

        return out


class TransformerDecoder(Transformer):
    """Transformer decoder, using auto-regressive decoder blocks for language modeling."""

    def __init__(self, vocab_size, max_seq_len, hidden_size, ff_hidden_size):
        super().__init__(max_seq_len)

        self.decoder = Decoder(vocab_size, max_seq_len, hidden_size, ff_hidden_size)

    def forward(self, x):
        lookahead_mask = self.create_lookahead_mask(x.shape[1])

        out = self.decoder(x, None, lookahead_mask)

        return out
