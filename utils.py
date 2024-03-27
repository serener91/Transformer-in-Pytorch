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


def tokenize(sequence):
    # remove punctuation
    for punc in ["!", ".", "?"]:
        sequence = sequence.replace(punc, "")

    # split the sequence on spaces and lowercase each token
    return [token.lower() for token in sequence.split(" ")]


def build_vocab(data):
    # tokenize the data and remove duplicates
    vocab = list(set(tokenize(data)))

    # sort the vocabulary
    vocab.sort()

    # assign an integer to each word
    look_up_table = {word: i for i, word in enumerate(vocab)}

    return look_up_table




