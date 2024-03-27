import math
import torch.nn as nn
import torch
from utils import tokenize, build_vocab


class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.look_up_table = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        # Paper mentioned to multiply by sqrt(d_model) to scale the embeddings
        return self.look_up_table(x) * math.sqrt(self.d_model)


if __name__ == '__main__':

    # list of sequences (3,)
    sequences = ["I wonder what will come next!",
                 "This is a basic example paragraph.",
                 "Hello, what is a basic split?"]

    # tokenize the sequences -> (batch_size, seq_length) -> (3,6)
    tokenized_sequences = [tokenize(seq) for seq in sequences]
    print(tokenized_sequences)

    # concatenate the sequences
    sequence_cluster = " ".join(sequences)
    print(sequence_cluster)

    # build the vocabulary (corpus)
    word_table = build_vocab(sequence_cluster)
    print(word_table)

    # Integer encoding
    encoded_sequence = [word_table[word] for word in tokenize(sequence_cluster)]
    print(encoded_sequence)

    vocab_size = len(word_table)
    print(vocab_size)

    # set model embedding dimensions
    d_model = 3

    # create the initial embedding layer
    # lookuptable = torch.rand(vocab_size, d_model)  # matrix of size (14, 3)
    lookuptable = nn.Embedding(vocab_size, d_model)
    print(lookuptable.state_dict()['weight'])

    # apply embedding
    indices = torch.Tensor(encoded_sequence).long()  # (batch_size, seq_length, d_model)
    # embeddings = embeddings[encoded_sequence]
    embeddings = lookuptable(indices)
    print(embeddings)


