from vocab import Vocab
import torch
import os

class Corpus:

    def __init__(self, path):
        self.vocab = Vocab()
        self.vocab.add_word('<pad>')
        self.vocab.add_all_words(os.path.join(path, "train_en.txt"))
        self.vocab.add_all_words(os.path.join(path, "train_de.txt"))

        # len1 = self.get_max_length(os.path.join(path, "news-commentary-v8.de-en.en"))
        # len2 = self.get_max_length(os.path.join(path, "news-commentary-v8.de-en.de"))
        # max_len = max(len1, len2)
        self.max_len = 30
        self.train_in = self.tokenize(os.path.join(path, "train_en.txt"))
        self.train_out = self.tokenize(os.path.join(path, "train_de.txt"))

    def get_rows(self, path):
        cnt = 0
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                cnt += 1

        print(cnt)

    def get_max_length(self, path):
        max_len = 0
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                words = line.split()
                if len(words) > max_len:
                    max_len = len(words)        

        return max_len


    def tokenize(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            idss = []
            for line in file:
                words = line.split()
                ids = []
                for word in words:
                    ids.append(self.vocab.word2idx[word])
                if len(ids) < self.max_len:
                    while len(ids) < self.max_len:
                        ids.append(self.vocab.word2idx['<pad>'])
                else:
                    ids = ids[:self.max_len]
                idss.append(torch.unsqueeze(torch.tensor(ids).type(torch.int64), dim=0))
            ids = torch.cat(idss, dim=0)
        return ids


