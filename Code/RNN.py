import numpy as np


class RNN:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        self.Uz = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.Ur = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.Uc = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V  = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.Wz = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        self.Wr = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        self.Wc = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

        self.bz = np.zeros((hidden_dim))
        self.br = np.zeros((hidden_dim))
        self.bc = np.zeros((hidden_dim))
        self.c  = np.zeros(word_dim)

        self.cUz = np.zeros(self.Uz.shape)
        self.cUr = np.zeros(self.Ur.shape)
        self.cUc = np.zeros(self.Uc.shape)
        self.cV  = np.zeros(self.V.shape)
        self.cWz = np.zeros(self.Wz.shape)
        self.cWr = np.zeros(self.Wr.shape)
        self.cWc = np.zeros(self.Wc.shape)

        self.cbz = np.zeros(self.bz.shape)
        self.cbr = np.zeros(self.br.shape)
        self.cbc = np.zeros(self.bc.shape)
        self.cc  = np.zeros(self.c.shape)


    def sigmoid(self, h):
        ''' Calculate sigmoid '''
        return 1 / (1 + np.exp(-h))


    def softmax(self, h):
        ''' Calculate softmax '''
        return np.exp(h - np.max(h))/ np.sum(np.exp(h - np.max(h)))


    def forward_prop_step(self, x_t, s_prev):
        '''GRU forward propagation'''
        r = self.sigmoid(self.Ur[:,x_t] + self.Wr.dot(s_prev) + self.br)
        z = self.sigmoid(self.Uz[:,x_t] + self.Wz.dot(s_prev) + self.bz)
        h = np.tanh(self.Uc[:,x_t] + self.Wc.dot(s_prev * r)  + self.bc)
        s = z  * s_prev + (1 - z) * h
        o = self.softmax(self.V.dot(s) + self.c)
        return o, s


    def forward_propagation(self, x):
        ''' Forward propagation function with GRU '''
        n = len(x)
        s = np.zeros((n + 1, self.hidden_dim))
        o = np.zeros((n, self.word_dim))

        for t in np.arange(n):
            r = self.sigmoid(self.Ur[:,x[t]] + self.Wr.dot(s[t-1]) + self.br)
            z = self.sigmoid(self.Uz[:,x[t]] + self.Wz.dot(s[t-1]) + self.bz)
            h = np.tanh(self.Uc[:,x[t]] + self.Wc.dot(s[t-1] * r)  + self.bc)
            s[t] = z  * s[t-1] + (1 - z) * h
            o[t] = self.softmax(self.V.dot(s) + self.c)
        return o, s


    def calculate_loss(self, x, y):
        total = 0
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            total += -1 * np.sum(np.log(correct_word_predictions))
        return total/sum((len(y_i) for y_i in y))


    def bptt(self, x, y):
        T = len(y)
        o, s = self.forward_propagation(x)
        dLdUz = np.zeros(self.Uz.shape)
        dLdUr = np.zeros(self.Ur.shape)
        dLdUc = np.zeros(self.Uc.shape)
        dLdV = np.zeros(self.V.shape)
        dLdWz = np.zeros(self.Wz.shape)
        dLdWr = np.zeros(self.Wr.shape)
        dLdWc = np.zeros(self.Wc.shape)
        dLdbz = np.zeros(self.bz.shape)
        dLdbr = np.zeros(self.br.shape)
        dLdbc = np.zeros(self.bc.shape)
        dLdc = np.zeros(self.c.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.

        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            dLdc += delta_o[t]

            delta_tz = self.V.T.dot(delta_o[t]) * (s[t]*(1 - (s[t])))
            delta_tr = self.V.T.dot(delta_o[t]) * (s[t]*(1 - (s[t])))
            delta_tc = self.V.T.dot(delta_o[t]) * (s[t]*(1 - (s[t])))

            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                dLdWz += np.outer(delta_tz, s[bptt_step-1])
                dLdWr += np.outer(delta_tr, s[bptt_step-1])
                dLdWc += np.outer(delta_tc, s[bptt_step-1])
                dLdUz[:,x[bptt_step]] += delta_tz
                dLdUr[:,x[bptt_step]] += delta_tr
                dLdUc[:,x[bptt_step]] += delta_tc
                dLdbz += delta_tz
                dLdbr += delta_tr
                dLdbc += delta_tc
                delta_tz = self.Wz.T.dot(delta_tz) * (s[bptt_step-1]*(1 - s[bptt_step-1]))
                delta_tr = self.Wr.T.dot(delta_tr) * (s[bptt_step-1]*(1 - s[bptt_step-1]))
                delta_tc = self.Wc.T.dot(delta_tc) * (s[bptt_step-1]*(1 - s[bptt_step-1]))

        return dLdUz, dLdUr, dLdUc, dLdV, dLdWz, dLdWr, dLdWc, dLdbz, dLdbr, dLdbc, dLdc


    def sgd_step(self, x, y, learning_rate):
        dLdUz, dLdUr, dLdUc, dLdV, dLdWz, dLdWr, dLdWc, dLdbz, dLdbr, dLdbc, dLdc = self.bptt(x, y)

        eps = 1e-6
        decay_rate = 0.95

        self.cUz = decay_rate * self.cUz + (1 - decay_rate) * dLdUz**2
        self.Uz += - learning_rate * dLdUz / (np.sqrt(self.cUz) + eps)
        self.cUr = decay_rate * self.cUr + (1 - decay_rate) * dLdUr**2
        self.Ur += - learning_rate * dLdUr / (np.sqrt(self.cUr) + eps)
        self.cUc = decay_rate * self.cUc + (1 - decay_rate) * dLdUc**2
        self.Uc += - learning_rate * dLdUc / (np.sqrt(self.cUc) + eps)

        self.cV = decay_rate * self.cV + (1 - decay_rate) * dLdV**2
        self.V += - learning_rate * dLdV / (np.sqrt(self.cV) + eps)

        self.cWz = decay_rate * self.cWz + (1 - decay_rate) * dLdWz**2
        self.Wz += - learning_rate * dLdWz / (np.sqrt(self.cWz) + eps)
        self.cWr = decay_rate * self.cWr + (1 - decay_rate) * dLdWr**2
        self.Wr += - learning_rate * dLdWr / (np.sqrt(self.cWr) + eps)
        self.cWc = decay_rate * self.cWc + (1 - decay_rate) * dLdWc**2
        self.Wc += - learning_rate * dLdWc / (np.sqrt(self.cWc) + eps)

       self.cbz = decay_rate * self.cbz + (1 - decay_rate) * dLdbz**2
       self.bz += - learning_rate * dLdbz / (np.sqrt(self.cbz) + eps)
       self.cbr = decay_rate * self.cbr + (1 - decay_rate) * dLdbr**2
       self.br += - learning_rate * dLdbr / (np.sqrt(self.cbr) + eps)
       self.cbc = decay_rate * self.cbc + (1 - decay_rate) * dLdbc**2
       self.bc += - learning_rate * dLdbc / (np.sqrt(self.cbc) + eps)

       self.cc  = decay_rate * self.cc + (1 - decay_rate) * dLdc**2
       self.c  += - learning_rate * dLdc / (np.sqrt(self.cc) + eps)
