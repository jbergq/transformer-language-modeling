import torch


def sample_sequences(lines, length):
    def get_start(line):
        high = max(1, len(line) - length - 1)
        return torch.randint(size=(1,), low=0, high=high)

    starts = [get_start(line) for line in lines]
    src_seqs = [l[s : s + length] for s, l in zip(starts, lines)]
    tgt_seqs = [l[s + 1 : s + length + 1] for s, l in zip(starts, lines)]

    return src_seqs, tgt_seqs
