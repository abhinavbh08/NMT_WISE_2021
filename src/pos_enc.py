import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):

    def __init__(self, dimension, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos = torch.zeros((1, max_len, dimension))
        x = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(
        0, dimension, 2, dtype=torch.float32) / dimension)
        self.pos[:, :, 0::2] = torch.sin(x)
        self.pos[:, :, 1::2] = torch.cos(x)

    def forward(self, x):
        # Add positional embeddings to the word embeddings obtained as input.
        x = x + self.pos[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)


# import matplotlib.pyplot as plt
# encoding_dim, num_steps = 32, 60
# pos_encoding = PositionalEncoding(encoding_dim, 0)
# pos_encoding.eval()
# X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
# P = pos_encoding.pos[:, :X.shape[1], :]
# print(P.shape)
# plt.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
        #  figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])
