import preprocess
import Markov


m = Markov.markov(3)

for i in preprocess.fourGrams:
    m.train(i)

for i in range(30):
    mResult = m.generate(start_with = ['The', 'king', 'will'], length = 100)
    print(mResult, '\n')
