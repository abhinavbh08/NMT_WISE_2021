import collections

class Vocab:

    def __init__(self, tokens = None, min_freq=0, reserved_tokens = None):
        self.word2idx = dict()
        self.idx2word = []
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        if len(tokens) == 0 or isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        
        count_tokens = collections.Counter(tokens)
        self.token_freqs = sorted(count_tokens.items(), key=lambda x:x[1], reverse=True)
        self.idx2word.append("UNK")
        self.idx2word.extend(reserved_tokens)
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}
        for word, cnt in self.token_freqs:
            if cnt < min_freq:
                break
            if word not in self.word2idx:
                self.idx2word.append(word)
                self.word2idx[word] = len(self.idx2word) - 1

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        # return self.word2idx[word]

    def add_all_words(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.split()
                for word in words:
                    self.add_word(word)


    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.word2idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    @property
    def unk(self):
        return 0

    def __len__(self):
        return len(self.idx2word)

    