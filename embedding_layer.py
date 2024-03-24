import torch.nn as nn
import math

class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # Paper mentioned to multiply by sqrt(d_model) to scale the embeddings
        return self.embedding(x) * math.sqrt(self.d_model)


