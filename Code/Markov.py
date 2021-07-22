from nltk.util import ngrams as nltkgrams
from collections import defaultdict
import random


class markov:
    def __init__(self, n):
        '''
        self.model: contain all the data and counts
        n: is the amount of object need to make a prediction
            For example if n = 3, 3 letters must be given to decide the next item (F4| F1 F2 F3)
        '''
        self.model = defaultdict(list)
        self.n = n


    def __add__(self, ngram, nextText):
        self.model[ngram].append(nextText)


    def ngrams(self, text):
        ngram = nltkgrams(text, self.n + 1)
        self.train(ngrams)


    def train(self, ngram):
        for i in ngram:
            self.__add__(tuple(i[:self.n]), i[self.n])


    def byfreq(self, current):
        return random.choice(self.model[current])


    def getnext(self, current):
        if current in self.model:
            return self.byfreq(current)
        else:
            print("ERROR!!!!! NO combination is found")
            return ""


    def generate(self, start_with = None, length = 100, stop_when = None):
        if start_with == None:
            start_with = random.choice(list(self.model.keys()))
        result = list(start_with)
        for i in range(length):
            next_word = self.getnext(tuple(result[-self.n:]))
            result.append(next_word)
            if stop_when != None and stop_when == next_word:
                break
        return ' '.join(result)
