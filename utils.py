import torch
import torch.nn as nn


def check_available_device():

    if torch.backends.mps.is_available():
        return "mps"

    elif torch.cuda_is_available():
        return "cuda"

    else:
        return "cpu"


def apply_softmax_function(logtis):
    return nn.Softmax(dim=1)(logtis)



