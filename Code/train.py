import nltk
import numpy as np
import RNN
import Shakespeare
import itertools



vocabulary_size = 8000
x = Shakespeare.shakespeare()
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
min_sent_length = 5
amount_of_sent = 3

x = Shakespeare.shakespeare()
amount_of_plays = len(x.plays())/2
sentences = list()
for i in range(3):
    sentences += x.listsent(x.plays()[i])

sentences = ["%s %s %s" %(sentence_start_token, x, sentence_end_token) for x in sentences]

tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print ("Found %d unique words tokens." % len(word_freq.items()))

vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print( "Using vocabulary size %d." % vocabulary_size)
print( "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])


def generate_sentence(model):
    new_sentence = [word_to_index[sentence_start_token]]
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs, y = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str


def create_some_sentences(model):
    for sent in range(amount_of_sent):
        gen_sent = generate_sentence(model)
        while (len(gen_sent) < min_sent_length):
            gen_sent = generate_sentence(model)
        print(" ".join(word for word in gen_sent) + "\n\n")


def train_with_sgd(model, X_train, y_train, learning_rate=0.0005, nepoch=250, evaluate_loss_after=20):
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        if (epoch != 0 and epoch % evaluate_loss_after == 0):
            print ("number of examples=%d epoch=%d" % (num_examples_seen, epoch))
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1


model = RNN.RNN(vocabulary_size)
v = train_with_sgd(model, X_train, y_train)
