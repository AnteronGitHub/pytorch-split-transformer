from typing import Tuple

import torch
from torch import nn, Tensor
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import dataset

def build_vocabulary():
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab, tokenizer

def data_process(vocab, tokenizer, raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def batchify(data: Tensor, bsz: int, device : str) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

def get_batch(source: Tensor, bptt : int, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

# train_iter was "consumed" by the process of building the vocab,
# so we have to create it again

def prepare_wikitext_dataset(train_batch_size = 20, eval_batch_size = 10, device = 'cpu'):
    print("Loading the WikiText2 dataset")
    vocab, tokenizer = build_vocabulary()
    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(vocab, tokenizer, train_iter)
    val_data = data_process(vocab, tokenizer, val_iter)
    test_data = data_process(vocab, tokenizer, test_iter)
    return vocab, \
        batchify(train_data, train_batch_size, device), \
        batchify(val_data, eval_batch_size, device), \
        batchify(test_data, eval_batch_size, device)

