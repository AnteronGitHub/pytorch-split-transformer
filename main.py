import math
import os
from tempfile import TemporaryDirectory

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from models.transformer import TransformerModel

from datasets.wiki_text2 import prepare_wikitext_dataset, get_batch

import copy
import time

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def train(model: nn.Module, train_data, criterion, optimizer, scheduler, bptt) -> None:
    r"""Trains the model with the given criterion

    Args:
        model (nn.Module): Model to be trained.
        train_data: Training data
        criterion: Loss function used in training.
        optimizer (Optimizer): Wrapped optimizer.
        scheduler: scheduler.
    """
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 10
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, bptt, i)
        seq_len = data.size(0)
        if seq_len != bptt:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate(model: nn.Module, eval_data: Tensor, criterion, bptt) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, bptt, i)
            seq_len = data.size(0)
            if seq_len != bptt:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading the WikiText2 dataset")
    vocab, train_data, val_data, test_data = prepare_wikitext_dataset()

    ntokens = len(vocab)
    emsize = 200
    d_hid = 200
    nlayers = 2
    nhead = 2
    dropout = 0.2

    print("Loading the model")
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

    best_val_loss = float('inf')
    epochs = 1
    bptt = 35
    criterion = nn.CrossEntropyLoss()
    lr = 2.5
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        print("Starting the training")
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            print('-' * 89)
            train(model, train_data.to(device), criterion, optimizer, scheduler, bptt)
            print('-' * 89)
            print("Evaluating the model")
            val_loss = evaluate(model, val_data.to(device), bptt)
            val_ppl = math.exp(val_loss)
            elapsed = time.time() - epoch_start_time
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
            print('-' * 89)

            if val_loss < best_val_loss:
                print("Saving the trained model")
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_params_path)

            scheduler.step()
        model.load_state_dict(torch.load(best_model_params_path)) # load best model states

    test_loss = evaluate(model, test_data.to(device), bptt)
    test_ppl = math.exp(test_loss)
    print('=' * 89)
    print(f'| End of training | test loss {test_loss:5.2f} | '
          f'test ppl {test_ppl:8.2f}')
    print('=' * 89)
