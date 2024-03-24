import numpy as np
import math
import torch


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


