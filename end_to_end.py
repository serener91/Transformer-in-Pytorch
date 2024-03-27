import torch.nn as nn
import torch
from utils import tokenize, build_vocab
from embedding_layer import Embeddings
from positional_encoding import PositionalEncoding
from multi_head_attention import MultiHeadAttention


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
encoded_sequence = [[word_table[word] for word in seq] for seq in tokenized_sequences]
print(encoded_sequence)

vocab_size = len(word_table)
print(vocab_size)

# set model embedding dimensions
d_model = 8

# create the initial embedding layer
# lookuptable = torch.rand(vocab_size, d_model)  # matrix of size (14, 3)
# lookuptable = nn.Embedding(vocab_size, d_model)
# print(lookuptable.state_dict()['weight'])

lookuptable = Embeddings(vocab_size, d_model)

# apply embedding
indices = torch.Tensor(encoded_sequence).long()  # (batch_size, seq_length, d_model)
# embeddings = embeddings[encoded_sequence]
embedded_input = lookuptable(indices)
print(embedded_input)
print(embedded_input.size())

pe = PositionalEncoding(d_model=d_model, dropout=0.1, max_length=10)

X = pe(embedded_input)

# set the n_heads
n_heads = 4

# create the attention layer
attention = MultiHeadAttention(d_model, n_heads, dropout=0.1)

# pass X through the attention layer three times to create Q, K, and V
output, attn_probs = attention(X, X, X, mask=None)

print("attention_value", output)

