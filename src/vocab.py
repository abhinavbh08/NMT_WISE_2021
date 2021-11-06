class Vocab:

    def __init__(self):
        self.word2idx = dict()
        self.idx2word = []

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

    def get_sentence(self, sent):
        ids = []
        words = sent.split()
        for word in words:
            ids.append(self.word2idx[word])
        return ids

    def __len__(self):
        return len(self.idx2word)