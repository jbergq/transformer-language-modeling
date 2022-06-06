from functools import partial

from torch.utils.data import DataLoader
from torch.hub import load_state_dict_from_url

import torchtext.transforms as T
import torchtext.functional as F
from torchtext.datasets import IMDB

from src.model.transformer import Transformer
from src.data.preprocess import PreProcess


SEQUENCE_LENGTH = 10
BATCH_SIZE = 8


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


for i, batch in enumerate(train_dataloader):
    src, tgt = batch["source"], batch["target"]

    test = model(src, tgt)
