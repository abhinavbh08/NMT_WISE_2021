import torch
from torch.autograd.grad_mode import F
import torch.nn as nn
import math
from loss import sequence_mask

def masked_softmax(X, valid_lens):
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

# print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))

class AdditiveAttention(nn.Module):

    def __init__(self, query_dimension, key_dimension, hidden_size, dropout):
        super(AdditiveAttention, self).__init__()
        self.w_k = nn.Linear(key_dimension, hidden_size, bias=False)
        self.w_q = nn.Linear(query_dimension, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, 1, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries = self.w_q(queries)
        keys = self.w_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


# AdditiveAttention()
# queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# values = torch.arange(40, dtype=torch.float32).reshape(1, 10,
#                                                        4).repeat(2, 1, 1)
# valid_lens = torch.tensor([2, 6])
# att = AdditiveAttention(20, 2, 8, 0.1)
# att.eval()
# att(queries, keys, values, valid_lens)

class DotProductAttention(nn.Module):

    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.attention_weights, values)

# queries, keys, values = torch.normal(0, 1, (2, 6, 10)), torch.ones((2, 6, 10)), torch.ones((2, 6, 10))
# valid_lens = torch.tensor([2, 3, 4, 5, 6])
# dpa = DotProductAttention(0.1)
# dpa(queries, keys, values, valid_lens)

class MultiHeadAttention(nn.Module):

    def __init__(self, query_size, key_size, value_size, dimensionality, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.wq = nn.Linear(query_size, dimensionality, bias=False)
        self.wk = nn.Linear(key_size, dimensionality, bias=False)
        self.wv = nn.Linear(value_size, dimensionality, bias=False)
        self.wo = nn.Linear(dimensionality, dimensionality, bias=False)
        self.attention = DotProductAttention(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries = self.wq(queries)
        keys = self.wk(keys)
        values = self.wv(values)
        queries = self.transpose_input(queries)
        keys = self.transpose_input(keys)
        values = self.transpose_input(values)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)

        output = self.transpose_output(output)
        output = self.wo(output)
        return output


    def transpose_input(self, x):
        x = x.reshape(x.shape[0], x.shape[1], self.num_heads, -1)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, x.shape[2], x.shape[3])
        return x

    def transpose_output(self, x):
        x = x.reshape(-1, self.num_heads, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        return x        

# num_hiddens, num_heads = 100, 5
# attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.1)
# attention.eval()

# batch_size = 2
# valid_lens = torch.tensor([3, 4])
# q = torch.ones((batch_size, 10, 100))
# attention(q, q, q, valid_lens)