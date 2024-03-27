import math
import torch.nn as nn
import torch


class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.look_up_table = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        # Paper mentioned to multiply by sqrt(d_model) to scale the embeddings
        return self.look_up_table(x) * math.sqrt(self.d_model)



