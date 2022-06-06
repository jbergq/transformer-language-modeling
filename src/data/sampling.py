import torch


def sample_sequences(lines, length):
    def get_start(line):
        return torch.randint(size=(1,), low=0, high=len(line) - length - 1)

    starts = [get_start(line) for line in lines]
    src_seqs = [line[start : start + length] for start, line in zip(starts, lines)]
    tgt_seqs = [
        line[start + 1 : start + length + 1] for start, line in zip(starts, lines)
    ]

    return src_seqs, tgt_seqs
