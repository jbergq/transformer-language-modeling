import torch


def sample_sequences(line, length, batch_size):
    starts = torch.randint(size=(batch_size,), low=0, high=line.size(0) - length - 1)

    seqs_inputs = [line[start : start + length] for start in starts]
    seqs_target = [line[start + 1 : start + length + 1] for start in starts]

    inputs = torch.cat([s[None, :] for s in seqs_inputs], dim=0).to(torch.long)
    targets = torch.cat([s[None, :] for s in seqs_target], dim=0).to(torch.long)

    return inputs, targets
