import numpy as np
import pandas as pd
import torch
from vocab import Vocab
from torch.utils import data
import nltk

def read_data(data_name="php"):
    # Data paths if training models on kaggle kernels for better gpus
    data_path_en_kaggle = "/kaggle/input/dedupdata/train.en"
    data_path_de_kaggle = "/kaggle/input/dedupdata/train.de"

    data_path_en = "data/de-en_deduplicated/train.en"
    data_path_de = "data/de-en_deduplicated/train.de"

    if data_name=="php":
        with open(data_path_en_kaggle, "r") as file:
            data_en = file.read().split("\n")[:-1]

        with open(data_path_de_kaggle, "r") as file:
            data_de = file.read().split("\n")[:-1]
    else:
        data_path_kaggle = "/kaggle/input/enfrdata/fra_small.txt"
        data_path = "data/fra-eng/fra_small.txt"

        with open(data_path, "r") as file:
            return file.read()

    print(len(data_en), len(data_de))
    return data_en, data_de

# raw_text = read_data(data_name="php")
# print("abc")

## Exactly taken from the d2l book.
# def preprocess_nmt(text):
#     """Preprocess the English-French dataset."""
#     def no_space(char, prev_char):
#         return char in set(',.!?') and prev_char != ' '

#     # Replace non-breaking space with space, and convert uppercase letters to
#     # lowercase ones
#     text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
#     # Insert space between words and punctuation marks
#     out = [
#         ' ' + char if i > 0 and no_space(char, text[i - 1]) else char
#         for i, char in enumerate(text)]
#     return ''.join(out)

# preprocessed_text = preprocess_nmt(raw_text)

def read_val_data(data_name):
    source, target = [], []

    data_path_en_kaggle = "/kaggle/input/dedupdata/val.en"
    data_path_de_kaggle = "/kaggle/input/dedupdata/val.de"

    data_path_en = "data/de-en_deduplicated/val.en" 
    data_path_de = "data/de-en_deduplicated/val.de" 

    if data_name=="php":
        with open(data_path_en_kaggle, "r") as file:
            data_en = file.read().split("\n")[:-1]
        with open(data_path_de_kaggle, "r") as file:
            data_de = file.read().split("\n")[:-1]        
        source, target = data_en, data_de
    else:
        data_path_kaggle = "/kaggle/input/enfrdata/fra_small.txt"
        data_path = "data/fra-eng/fra_small.txt"
        with open(data_path_kaggle, "r") as file:
            data = file.read()
        preprocessed_text = data

        for i, line in enumerate(preprocessed_text.split("\n")):
            parts = line.split("\t")
            if len(parts) == 3:
                source.append(parts[0])
                target.append(parts[1])

    return source, target

# read_test_data(data_name="php")

def tokenize_data(text):
    source, target = [], []
    if isinstance(text, tuple):
        for sent in text[0]:
            source.append(nltk.tokenize.word_tokenize(sent.lower()))
        for sent in text[1]:
            target.append(nltk.tokenize.word_tokenize(sent.lower()))
    else:
        for i, line in enumerate(text.split("\n")):
            parts = line.split("\t")
            if len(parts) == 3:
                # source.append(parts[0].split(" "))
                # target.append(parts[1].split(" "))
                source.append(nltk.tokenize.word_tokenize(parts[0].lower()))
                target.append(nltk.tokenize.word_tokenize(parts[1].lower()))
            
    return source, target

# source, target = tokenize_data(preprocessed_text)
# print(source[:5], target[:5])
# src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
# print(len(src_vocab))

def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))

def build_array_nmt(lines, vocab, num_steps):

    lines = [vocab[l] for l in lines]
    lines = [l + [vocab["<eos>"]] for l in lines]

    arr = torch.tensor(
        [truncate_pad(l, num_steps, vocab["<pad>"]) for l in lines]
    )
    valid_len = (arr != vocab["<pad>"]).type(torch.int32).sum(1)
    return arr, valid_len


def load_data(batch_size, num_steps):
    # preprocessed_text = read_data(read_data(data_name="abc"))
    preprocessed_text = read_data(data_name="php")
    source, target = tokenize_data(preprocessed_text)
    src_vocab = Vocab(source, min_freq=1, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=1, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    dataset = data.TensorDataset(*data_arrays)
    itrtr = data.DataLoader(dataset, batch_size, shuffle=True)
    return itrtr, src_vocab, tgt_vocab

# train_iter, src_vocab, tgt_vocab = load_data(2, num_steps=10)
# for x, x_len, y, y_len in train_iter:
#     print(x)
#     print(x_len)
#     a1 = []
#     for i in x[1]:
#         a1.append(src_vocab.idx2word[i])
#     print(y)
#     print(y_len)
#     a2 = []
#     for i in y[1]:
#         a2.append(tgt_vocab.idx2word[i])
#     print(" ".join(a1), " ".join(a2))
#     break