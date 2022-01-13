import torch
import numpy as np
from torch._C import dtype
from torch.functional import norm
import torch.nn as nn
from torch.nn.modules import batchnorm
import torch.nn.functional as F
from attention_scoring import AdditiveAttention, DotProductAttention, MultiHeadAttention
from ffns_layer_norm import FFNs, LNorm
from pos_enc import PositionalEncoding
import math

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

    def forward(self, x, *args):
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

    def __init__(self, encoder, decoder, **kwargs):
        super(S2SEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_x, dec_x, *args):
        enc_outputs = self.encoder(enc_x, *args)
        enc_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_x, enc_state)


class TransformerEncoderBlock(nn.Module):

    def __init__(self, query, key, value, hidden_size, num_head, dropout, norm_shape, ffn_input, ffn_hidden, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(query, key, value, hidden_size, num_head, dropout)
        self.l_norm1 = LNorm(norm_shape, dropout)
        # self.fully_connected = FFNs(ffn_input, ffn_hidden, hidden_size)
        self.first_ffl = nn.Linear(ffn_input, ffn_hidden)
        self.relu = nn.ReLU()
        self.second_ffl = nn.Linear(ffn_hidden, hidden_size)
        self.l_norm2 = LNorm(norm_shape, dropout)

    def forward(self, x, valid_lens):
        attn_op = self.attention(x, x, x, valid_lens)
        x = self.l_norm1(x, attn_op)
        ffn_op = self.second_ffl(self.relu(self.first_ffl(x)))
        return self.l_norm2(x, ffn_op)
        # x = self.l_norm1(x, self.attention(x, x, x, valid_lens))
        # return self.l_norm2(x, self.fully_connected(x))


class TransformerEncoder(nn.Module):

    def __init__(self, query, key, value, hidden_size, num_head, dropout, norm_shape, ffn_input, ffn_hidden, vocab_size, num_layers, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, dropout)
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            self.blocks.add_module(str(i), TransformerEncoderBlock(query, key, value, hidden_size, num_head, dropout, norm_shape, ffn_input, ffn_hidden))

    def forward(self, x, valid_lens, *args):
        x = self.pos_encoding(self.embedding(x) * math.sqrt(self.hidden_size))
        for i, block in enumerate(self.blocks):
            x = block(x, valid_lens)
        return x


class Transformerdecoderblock(nn.Module):

    def __init__(self, query, key, value, hidden_size, num_head, dropout, norm_shape, ffn_input, ffn_hidden, i, **kwargs):
        super(Transformerdecoderblock, self).__init__(**kwargs)
        self.i = i
        self.att1 = MultiHeadAttention(query, key, value, hidden_size, num_head, dropout)
        self.l_norm1 = LNorm(norm_shape, dropout)
        self.att2 = MultiHeadAttention(query, key, value, hidden_size, num_head, dropout)
        self.l_norm2 = LNorm(norm_shape, dropout)
        self.first_ffl = nn.Linear(ffn_input, ffn_hidden)
        self.relu = nn.ReLU()
        self.second_ffl = nn.Linear(ffn_hidden, hidden_size)
        self.l_norm3 = LNorm(norm_shape, dropout)

    def forward(self, x, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            keys = x
        else:
            keys = torch.cat((state[2][self.i], x), axis=1)
        state[2][self.i] = keys
        if self.training:
            bs, num_steps, dim = x.shape
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=x.device).repeat(bs, 1)
        else:
            dec_valid_lens = None

        op_att1 = self.att1(x, keys, keys, dec_valid_lens)
        y = self.l_norm1(x, op_att1)
        op_att2 = self.att2(y, enc_outputs, enc_outputs, enc_valid_lens)
        z = self.l_norm2(y, op_att2)
        z_ffn = self.second_ffl(self.relu(self.first_ffl(z)))
        return self.l_norm3(z, z_ffn), state

class TransformerDecoder(nn.Module):

    def __init__(self, query, key, value, hidden_size, num_head, dropout, norm_shape, ffn_input, ffn_hidden, vocab_size, num_layers, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, dropout)
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            self.blocks.add_module(
                str(i), Transformerdecoderblock(
                    query, key, value, hidden_size, num_head, dropout, norm_shape, ffn_input, ffn_hidden, i
                ))
        self.linear = nn.Linear(hidden_size, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, x, state):
        x = self.pos_encoding(self.embedding(x) * math.sqrt(self.hidden_size))
        for i, block in enumerate(self.blocks):
            x, state = block(x, state)
        return self.linear(x), state

# x = torch.ones((2, 100, 24))
# valid_lens = torch.tensor([3, 2])
# encoder_block = TransformerEncoderBlock(query=24, key=24, value=24, hidden_size=24, num_head=8, dropout=0.1, norm_shape=[100, 24], ffn_input=24, ffn_hidden=48)
# encoder_block.eval()
# # print(encoder_block(x, valid_lens).shape)
# # print("abc")
# dec_blck = Transformerdecoderblock(query=24, key=24, value=24, hidden_size=24, num_head=8, dropout=0.1, norm_shape=[100, 24], ffn_input=24, ffn_hidden=48, i=0)
# dec_blck.eval()
# state = [encoder_block(x, valid_lens), valid_lens, [None]]
# ans = dec_blck(x, state)
# print("abc")


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