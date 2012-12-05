from math import exp
import numpy as np
import numpy.random as npr

def logistic_function(x):
    return 1 / (1 + np.exp(-x))

def logistic_function_derivative(x):
    t = np.exp(-x)
    return t/(1 + t*t)

class MultilayerNetwork:

    def _init_weights(self, n, m):
        return npr.rand(n, m).transpose() * 0.4

    def __init__(self, network_size):
        """
        Matrices W are of size next_layer_cnt x cur_layer_cnt
        """
        self.weights = []
        for l in xrange(len(network_size) - 1):
            self.weights.append(self._init_weights(
                network_size[l] + 1, network_size[l + 1]
            ))

        self.network_size = network_size

    def process_learning(self, sample, result):
        """
        Back-propagation algorithm
        """
        W = self.weights
        learning_rate = 0.5

        # forward pass
        input = np.vstack((
            1,
            sample.transpose()
        ))
        output = np.vstack((
            1,
            result.transpose()
        ))

        v = {0: input}
        y = {0: input}
        for l in xrange(1, len(self.network_size)):
            v[l] = np.vstack((
                1,
                W[l - 1].dot(y[l - 1])
            ))
            y[l] = logistic_function(v[l])

        # backward pass
        error = {}
        delta = {}
        for l in reversed(xrange(len(self.network_size))):
            if l == len(self.network_size) - 1:
                error[l] = output - y[l]
            else:
                WT = W[l].transpose()
                M = np.hstack((np.zeros((WT.shape[0], 1)), WT))
                error[l] = M.dot(delta[l + 1])
            delta[l] = np.array(error[l]) * logistic_function_derivative(np.array(v[l]))

        # modify weights
        for l in reversed(xrange(1, len(self.network_size))):
            deltas_except_bias = np.vsplit(delta[l], (1,))[1]
            W[l - 1] += learning_rate * deltas_except_bias.dot(y[l - 1].transpose())

    def train_at_set(self, training_set):
        samples, values = np.hsplit(training_set, (-m,))
        for sample, value in zip(samples, values):
            self.process_learning(sample, value)

    def error_at_set(self, test_set):
        E = 0

        samples, values = np.hsplit(test_set, (-m,))
        for sample, value in zip(samples, values):
            res = self.process(sample)
            E += np.square(res - value).sum()

        return E/2

    def process(self, sample):
        """
        Straightforward computation
        """
        input = np.vstack((
            1,
            sample.transpose()
            ))
        cur = input
        for l in xrange(len(self.network_size) - 1):
            cur = np.vstack((
                1,
                logistic_function(self.weights[l].dot(cur))
            ))
        return np.vsplit(cur, (1,))[1]

def load_samples(fname):
    with open(fname, 'r') as f:
        n, m = map(int, f.readline().split())
        data = []
        for line in f.readlines():
            data.append(map(float, line.split()))
        return (n, m, np.matrix(data))


if __name__ == '__main__':
    n, m, data = load_samples('../data/circles.tsv')
    training_set, test_set = np.vsplit(data, (150, ))

    net = MultilayerNetwork((2, 30, 1))

    def calc_accuarcy():
        correct = 0
        samples, values = np.hsplit(test_set, (-m,))
        for sample, value in zip(samples, values):
            res = net.process(sample)
            if np.abs(res - value) < 0.5:
                correct += 1
        return correct * 1./ samples.shape[0]

    print calc_accuarcy()

    print net.weights
    print net.error_at_set(training_set), net.error_at_set(test_set)
    for i in xrange(50):
        net.train_at_set(training_set)
        print net.error_at_set(training_set), net.error_at_set(test_set)

    print net.weights
    print calc_accuarcy()
#    samples, values = np.hsplit(data, (-m,))
#    for sample, value in zip(samples, values):
#        print net.process(sample), value
