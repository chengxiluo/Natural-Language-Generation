import nltk
from nltk.corpus import shakespeare
from nltk import FreqDist
from nltk.util import ngrams
from collections import Counter
import Shakespeare


def remove_stopwords_inner(tokens, stopwords):
    stopwords = set(stopwords)
    return [word for word in tokens if word not in stopwords]


def remove_stopwords_nltk(tokens):
    from nltk.corpus import stopwords
    return remove_stopwords_inner(tokens, stopwords=stopwords.words('english'))


def get_bag_of_words_nltk(tokens):
    return FreqDist(tokens)


def remove_punctuation(bag_of_words):
    import string
    punctuation = set(string.punctuation)
    return Counter({k: v for k,v in bag_of_words.items() if k not in punctuation})


s = Shakespeare.shakespeare()
playList = [s.listword(p) for p in s.plays()]


n = 4
nGrams = [ngrams([j for j in shakespeare.words(i)], n) for i in shakespeare.fileids()] + [ngrams(i, n) for i in playList]
