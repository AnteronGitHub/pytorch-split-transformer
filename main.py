import os
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from datasets.wiki_text2 import prepare_wikitext_dataset
from models.transformer import TransformerModel
from pipelines.training import train_epoch, evaluate

import time

def run_suite():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading the WikiText2 dataset")
    vocab, train_data, val_data, test_data = prepare_wikitext_dataset()

    print("Loading the model")
    ntokens = len(vocab)
    emsize = 200
    d_hid = 200
    nlayers = 2
    nhead = 2
    dropout = 0.2
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

    print("Starting the training")
    best_val_loss = float('inf')
    epochs = 1
    bptt = 35
    criterion = nn.CrossEntropyLoss()
    lr = 2.5
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    param_dir = '.params'
    os.makedirs(param_dir, exist_ok=True)
    best_model_params_path = os.path.join(param_dir, "best_model_params.pt")
    try:
        model.load_state_dict(torch.load(best_model_params_path))
        print("Loaded model parameters")
    except Exception as e:
        print(e)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        print('-' * 89)
        try:
            train_epoch(model, train_data, criterion, optimizer, scheduler, bptt, ntokens, device)
        except KeyboardInterrupt:
            print("Training interrupted. Saving the model...")
            torch.save(model.state_dict(), best_model_params_path)
            return
        print('-' * 89)

        print("Evaluating the model")
        val_loss, _ = evaluate(model, val_data, criterion, bptt, ntokens, device)
        print('-' * 89)
        print('-' * 89)
        elapsed = time.time() - epoch_start_time
        print(f'Epoch compute time {elapsed} s')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)
            print("Saved the model parameters")

        scheduler.step()

    test_loss, test_ppl = evaluate(model, test_data, criterion, bptt, ntokens, device)

if __name__ == '__main__':
    run_suite()
