import os
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from datasets.wiki_text2 import prepare_wikitext_dataset
from models.transformer import TransformerModel
from pipelines.training import train, evaluate

import time

if __name__ == '__main__':
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
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            print('-' * 89)
            train(model, train_data, criterion, optimizer, scheduler, bptt, ntokens, device)
            print('-' * 89)

            print("Evaluating the model")
            val_loss = evaluate(model, val_data, bptt, ntokens, device)
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
