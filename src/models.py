import torch
import numpy as np
import torch.nn as nn
from torch.nn.modules import batchnorm
import torch.nn.functional as F

class RNNModel(nn.Module):
    def __init__(self, n_tokens, embedding_size):
        super(RNNModel, self).__init__()
        self.n_tokens = n_tokens
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(self.n_tokens, self.embedding_size)
        self.rnn = nn.RNN(self.embedding_size, self.embedding_size*4, batch_first=True)
        self.linear = nn.Linear(self.embedding_size*4, self.n_tokens)

    def forward(self, x):
        x = self.embedding(x)
        output, hidden = self.rnn(x)
        decoded = self.linear(output)
        decoded = decoded.view(-1, self.n_tokens)
        return F.log_softmax(decoded, dim=1)


# tensor = torch.Tensor(np.arange(10).reshape(2, 5)).long()
# model = RNNModel(10, 512)
# model(tensor)