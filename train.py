from functools import partial

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.hub import load_state_dict_from_url
from torch.nn.utils.rnn import pad_sequence

import torchtext.transforms as T
import torchtext.functional as F
from torchtext.datasets import IMDB

from src.model.transformer import Transformer
from src.data.preprocess import PreProcess
from src.utils.utils import print_metrics

NUM_EPOCHS = 10
SEQUENCE_LENGTH = 10
BATCH_SIZE = 8
LR = 1e-5
WEIGHT_DECAY = 5e-4


def train():
    train_dp = IMDB(split="train")

    encoder_json_path = "https://download.pytorch.org/models/text/gpt2_bpe_encoder.json"
    vocab_bpe_path = "https://download.pytorch.org/models/text/gpt2_bpe_vocab.bpe"

    tokenizer = T.GPT2BPETokenizer(encoder_json_path, vocab_bpe_path)
    vocab_path = "https://download.pytorch.org/models/text/roberta.vocab.pt"
    vocab = load_state_dict_from_url(vocab_path)

    transform = PreProcess(tokenizer, vocab, SEQUENCE_LENGTH)

    train_dp = train_dp.batch(BATCH_SIZE).rows2columnar(["label", "text"])
    train_dp = train_dp.map(transform)
    train_dp = train_dp.map(partial(F.to_tensor, padding_value=1), input_col="source")
    train_dp = train_dp.map(partial(F.to_tensor, padding_value=1), input_col="target")

    train_dataloader = DataLoader(train_dp, batch_size=None)

    model = Transformer(len(vocab), 100, 128, 512)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=5e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=1)

    for epoch in range(NUM_EPOCHS):
        for i, batch in enumerate(train_dataloader):
            src, tgt = batch["source"], batch["target"]

            out = model(src, tgt)

            out_reshape = out.contiguous().view(-1, out.shape[-1])
            tgt_reshape = tgt.contiguous().view(-1)

            loss = criterion(out_reshape, tgt_reshape)
            loss.backward()
            optimizer.step()

            print_metrics(epoch, i, loss.item())


if __name__ == "__main__":
    train()
