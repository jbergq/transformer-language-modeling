from functools import partial

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import torchtext.functional as F
from torchtext.vocab import vocab as tt_vocab
from torchtext.datasets import IMDB
from transformers import GPT2Tokenizer

from src.model.transformer import TransformerDecoder
from src.data.preprocess import PreProcess
from src.utils import ExperimentLogger, iter_print, epoch_print, load_config


def train(model, dataloader, optimizer, criterion, epoch):
    train_losses = []

    model.train()

    for i, batch in enumerate(dataloader):
        tgt = batch["target"]

        out = model(tgt)

        out_reshape = out.contiguous().view(-1, out.shape[-1])
        tgt_reshape = tgt.contiguous().view(-1)

        loss = criterion(out_reshape, tgt_reshape)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        train_losses.append(loss_val)
        iter_print(epoch, i, loss_val)

        if i == 10:
            break

    return torch.tensor(train_losses)


def validate(model, dataloader, criterion, epoch):
    val_losses = []

    model.eval()

    for i, batch in enumerate(dataloader):
        tgt = batch["target"]

        out = model(tgt)

        out_reshape = out.contiguous().view(-1, out.shape[-1])
        tgt_reshape = tgt.contiguous().view(-1)

        loss = criterion(out_reshape, tgt_reshape)

        loss_val = loss.item()
        val_losses.append(loss_val)
        iter_print(epoch, i, loss_val)

        pred = out.softmax(dim=2).argmax(dim=2)

    return torch.tensor(val_losses)


def run():
    cfg = load_config("config.yaml")

    train_dp, val_dp = IMDB(split=("train", "test"))

    # Tokenizer used by GPT-2.
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    vocab = tt_vocab(tokenizer.get_vocab())
    transform = PreProcess(tokenizer, cfg.sequence_length)

    # Create datapipes for training and validation.
    # Each datapipe samples a batch and applies preprocessing.
    train_dp = train_dp.batch(cfg.batch_size).rows2columnar(["label", "text"])
    train_dp = train_dp.map(transform)
    train_dp = train_dp.map(partial(F.to_tensor, padding_value=1), input_col="source")
    train_dp = train_dp.map(partial(F.to_tensor, padding_value=1), input_col="target")

    val_dp = val_dp.batch(cfg.batch_size).rows2columnar(["label", "text"])
    val_dp = val_dp.map(transform)
    val_dp = val_dp.map(partial(F.to_tensor, padding_value=1), input_col="source")
    val_dp = val_dp.map(partial(F.to_tensor, padding_value=1), input_col="target")

    train_dataloader = DataLoader(train_dp, batch_size=None)
    val_dataloader = DataLoader(val_dp, batch_size=None)

    model = TransformerDecoder(len(vocab), 100, 128, 512)
    optimizer = Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, eps=5e-9
    )
    criterion = nn.CrossEntropyLoss(ignore_index=1)

    logger = ExperimentLogger(log_dir="./experiments")

    if cfg.print_example:
        batch = next(iter(train_dataloader))
        out = model.generate(torch.zeros((1, 1), dtype=torch.long))

        print("Example sequence: ", tokenizer.decode(batch["target"][0].numpy()))
        print("Model output: ", tokenizer.decode(out[0].numpy()))

    for epoch in range(cfg.num_epochs):
        train_losses = train(model, train_dataloader, optimizer, criterion, epoch)
        val_losses = validate(model, val_dataloader, criterion, epoch)

        logger.log_params(
            {
                "train_loss": train_losses.mean().item(),
                "val_loss": val_losses.mean().item(),
            },
            step=epoch,
        )
        epoch_print(epoch, val_losses)

        out = model.generate(torch.zeros((1, 1), dtype=torch.long))
        print(tokenizer.decode(out[0].numpy()))


if __name__ == "__main__":
    run()
