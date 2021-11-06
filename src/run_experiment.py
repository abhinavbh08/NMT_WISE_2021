import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from models import RNNModel
from corpus import Corpus

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../data')
parser.add_argument('--embedding_size', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--save', type=str, default='model.pt')

args = parser.parse_args()
torch.manual_seed(args.seed)
corpus = Corpus(path="./data")
n_tokens = len(corpus.vocab)
model = RNNModel(n_tokens, args.embedding_size).to(device)
criterion = nn.NLLLoss()
batch_size = args.batch_size
train_data_in = corpus.train_in.to(device)
train_data_out = corpus.train_out.to(device)

epochs = 500
lr = args.lr

def train(train_data_in, train_data_out, batch_size, epoch):
    model.train()
    i = 0
    epoch_loss = 0
    while i+batch_size <= train_data_in.size(0):
        data = train_data_in[i:i+batch_size, :]
        targets = train_data_out[i: i+batch_size, :].view(-1)
        model.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        epoch_loss += loss.item()
        loss.backward()
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)
        
        i+=batch_size
    print(epoch_loss, epoch)

for epoch in range(epochs):
    train(train_data_in, train_data_out, batch_size, epoch)

# Corpus(path="./data")
line = "Just last December, fellow economists Martin Feldstein and Nouriel Roubini each penned op-eds bravely questioning bullish market sentiment, sensibly pointing out gold’s risks."
words = line.split()
ids = []
idss = []
for word in words:
    ids.append(corpus.vocab.word2idx[word])
if len(ids) < corpus.max_len:
    while len(ids) < corpus.max_len:
        ids.append(corpus.vocab.word2idx['<pad>'])
else:
    ids = ids[:corpus.max_len]
idss.append(torch.unsqueeze(torch.tensor(ids).type(torch.int64), dim=0))
ids = torch.cat(idss, dim=0)
model.eval()
with torch.no_grad():
    output = model(ids)
idx = torch.argmax(output, dim=1)
lst = []
for i in idx:
    wrd = corpus.vocab.idx2word[i]
    if wrd == "<pad>":
        break
    lst.append(wrd)

print(" ".join(lst))