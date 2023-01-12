from models.transformer import Transformer
from datasets.wiki_text2 import get_batches
from pipelines.training import train

def prepare_model():
    ntokens = 10
    emsize = 200
    d_hid = 200
    nlayers = 2
    nhead = 2
    dropout = 0.2
    return Transformer(ntokens, emsize, nhead, d_hid, nlayers, dropout).to("cpu")

if __name__ == '__main__':
    model = prepare_model()

    num_parameters = 0
    for param in model.parameters():
        num_parameters += param.nelement()

    train_data, val_data, test_data = get_batches()
    train(model, train_data)

    print("Model has {} parameters".format(num_parameters))
