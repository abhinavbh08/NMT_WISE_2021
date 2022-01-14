import torch
import torch.nn as nn

class FFNs(nn.Module):

    def __init__(self, n_input, n_hidden, n_output, **kwargs):
        super(FFNs, self).__init__(**kwargs)
        self.first_ffl = nn.Linear(n_input, n_hidden)
        self.relu = nn.ReLU()
        self.second_ffl = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        return self.second_ffl(self.relu(self.first_ffl(x)))


class LNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(LNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, x, y):
        return self.layer_norm(x + self.dropout(y))

# add_norm = LNorm([3, 4], 0.5)
# add_norm.eval()
# print(add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape)