import torch
import numpy as np
import torch.nn as nn
from torch.nn.modules import batchnorm
import torch.nn.functional as F
from attention_scoring import AdditiveAttention, DotProductAttention


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

class S2SEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers):
        super(S2SEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        output, state = self.rnn(x)
        return output, state


class S2SDecoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers):
        super(S2SDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(embedding_size + hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, x, enc_state):
        x = self.embedding(x)
        con = enc_state[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        concatenated_input = torch.cat((x, con), 2)
        output, state = self.rnn(concatenated_input, enc_state)
        output = self.linear(output)
        return output, state


class S2SAttentionDecoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers):
        super(S2SAttentionDecoder, self).__init__()
        # self.attention = DotProductAttention(0.0)
        self.attention = AdditiveAttention(hidden_size, hidden_size, hidden_size, dropout=0)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(embedding_size + hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def init_state(self, enc_outputs, x_len, *args):
        outputs, hidden_state = enc_outputs
        return (outputs, hidden_state, x_len)

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_lengths = state
        X = self.embedding(X).permute(1, 0, 2)
        outputs = []
        for x in X:
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lengths)
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            output, hidden_state = self.rnn(x, hidden_state)
            outputs.append(output)
        
        outputs = self.linear(torch.cat(outputs, dim=1))
        return outputs, [enc_outputs, hidden_state, enc_valid_lengths]


class S2SEncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder):
        super(S2SEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_x, dec_x, *args):
        enc_outputs = self.encoder(enc_x)
        enc_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_x, enc_state)

# class AttentionDecoder()
# x = torch.arange(12).reshape(3, 4)
# s2s = S2SEncoder(12, 100, 200, 1)
# output = s2s(x)
# print(op.shape, enc_state.shape)
# m = torch.arange(600).reshape(3, 200)
# b = m.unsqueeze(1).repeat(1, 4, 1)
# print(torch.cat((b, b), 2).shape)
# dec = S2SDecoder(12, 100, 200, 1)
# dec(x, enc_state)
# dec = S2SAttentionDecoder(12, 100, 200, 1)
# state = dec.init_state(output, torch.tensor([1, 2, 1]))
# dec(x, state)
# print(torch.Tensor([-1, 2, -2, 3]).sign())