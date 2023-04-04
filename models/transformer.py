import math
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, ntoken: int, d_model: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.input_embedding = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

    def init_weights(self, initrange = 0.1) -> None:
        self.input_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        return self.pos_encoder(src)

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.positional_embedding = PositionalEmbedding(ntoken, d_model, dropout)
        self.transformer_encoder = TransformerEncoder(TransformerEncoderLayer(d_model, nhead, d_hid, dropout), nlayers)
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self, initrange = 0.1) -> None:
        self.positional_embedding.init_weights()
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.positional_embedding(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
