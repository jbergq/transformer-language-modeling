import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from src.model.transformer import Transformer
from src.data.sampling import sample_sequences


SEQUENCE_LENGTH = 10
BATCH_SIZE = 8

tokenizer = get_tokenizer("basic_english")
train_iter = IMDB(split="train")


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
vocab_size = len(vocab)

model = Transformer(vocab_size, 100, 512, 512)


for i, (label, line) in enumerate(train_iter):
    tokens = tokenizer(line)
    token_ids = torch.tensor(vocab(tokens))

    src, tgt = sample_sequences(token_ids, SEQUENCE_LENGTH, BATCH_SIZE)
    test = model(src, tgt)
