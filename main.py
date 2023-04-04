import os
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from datasets.wiki_text2 import prepare_wikitext_dataset
from models.transformer import TransformerModel
from pipelines.training import train_epoch, evaluate

import time

class Session():
    def __init__(self):
        self.param_dir = '.params'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_dataset(self):
        self.vocab, self.train_data, self.val_data, self.test_data = prepare_wikitext_dataset()
        self.ntokens = len(self.vocab)

    def init_model(self):
        emsize = 200
        d_hid = 200
        nlayers = 2
        nhead = 2
        dropout = 0.2
        self.model = TransformerModel(self.ntokens, emsize, nhead, d_hid, nlayers, dropout).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def save_model(self):
        os.makedirs(self.param_dir, exist_ok=True)
        best_model_params_path = os.path.join(self.param_dir, "best_model_params.pt")
        torch.save(self.model.state_dict(), best_model_params_path)

    def load_model(self):
        os.makedirs(self.param_dir, exist_ok=True)
        best_model_params_path = os.path.join(self.param_dir, "best_model_params.pt")
        try:
            self.model.load_state_dict(torch.load(best_model_params_path))
            print("Loaded model parameters")
        except Exception as e:
            print(e)


    def train_model(self, bptt = 35):
        best_val_loss = float('inf')
        epochs = 1
        lr = 2.5
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            print('-' * 89)

            try:
                train_epoch(self.model, self.train_data, self.criterion, optimizer, scheduler, bptt, self.ntokens, self.device)
            except KeyboardInterrupt:
                print("Training interrupted.")
                return
            print('-' * 89)

            print("Evaluating the model")
            val_loss, val_ppl = self.evaluate_model(bptt)
            print(f'| valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
            print('-' * 89)
            print('-' * 89)
            elapsed = time.time() - epoch_start_time
            print(f'Epoch compute time {elapsed} s')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()
                print("Saved the model parameters")

            scheduler.step()

    def evaluate_model(self, bptt = 35):
        return evaluate(self.model, self.test_data, self.criterion, bptt, self.ntokens, self.device)

class UI():
    def run(self):
        session = Session()

        print("Loading the WikiText2 dataset")
        session.prepare_dataset()

        print("Initializing the transformer model")
        session.init_model()

        print("Trying to load model parameters")
        session.load_model()

        print("Starting the training")
        session.train_model()

        print("Evaluating model")
        val_loss, val_ppl = session.evaluate_model()
        print(f'| valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')

        print("Saving the updated model parameters")
        session.save_model()

if __name__ == '__main__':
    UI().run()
