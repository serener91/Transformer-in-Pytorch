import numpy as np
import math
import torch
import torch.nn as nn
from torch import Tensor


def generate_positional_encoding_vector(max_length: int = 1, d_model: int = 512, n: int = 10000):
    # generate an empty matrix for the positional encodings (pe)
    pe = np.zeros(max_length * d_model).reshape(max_length, d_model)

    # for each position
    for k in np.arange(max_length):

        # for each dimension
        for i in np.arange(d_model // 2):
            # calculate the internal value for sin and cos
            theta = k / (n ** ((2 * i) / d_model))

            # even dims: sin
            pe[k, 2 * i] = math.sin(theta)

            # odd dims: cos
            pe[k, 2 * i + 1] = math.cos(theta)

    return pe


def generate_positional_encoding_torch(max_length: int = 1, d_model: int = 512, n: int = 10000):
    # calculate all the divisors needed
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(n) / d_model))

    # generate the positions into a column matrix
    k = torch.arange(0, max_length).unsqueeze(1)

    # generate an empty tensor
    pe = torch.zeros(max_length, d_model)

    # set the even values (every row, even column)
    pe[:, 0::2] = torch.sin(k * div_term)

    # set the odd values (every row, odd column)
    pe[:, 1::2] = torch.cos(k * div_term)

    # add a dimension for broadcasting across sequences: optional
    pe = pe.unsqueeze(0)

    # the output has a shape of (1, max_length, d_model)
    return pe


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):
        """
        Args:
          d_model:      dimension of embeddings
          dropout:      randomly zeroes-out some of the input
          max_length:   max sequence length
        """
        # inherit from Module
        super().__init__()

        # initialize dropout
        self.dropout = nn.Dropout(p=dropout)

        # create tensor of 0s
        pe = torch.zeros(max_length, d_model)

        # create position column
        k = torch.arange(0, max_length).unsqueeze(1)

        # calc divisor for positional encoding
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        # calc sine on even indices
        pe[:, 0::2] = torch.sin(k * div_term)

        # calc cosine on odd indices
        pe[:, 1::2] = torch.cos(k * div_term)

        # add dimension
        pe = pe.unsqueeze(0)

        # buffers are saved in state_dict but not trained by the optimizer
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor):
        """
        Args:
          x:        embeddings (batch_size, seq_length, d_model)

        Returns:
                    embeddings + positional encodings (batch_size, seq_length, d_model)
        """
        # add positional encoding to the embeddings
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)

        # perform dropout
        return self.dropout(x)