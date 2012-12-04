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
        return npr.rand(n, m).transpose()

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

    def process_learning(self, sample, output):
        """
        Back-propagation algorithm
        """
        W = self.weights
        learning_rate = 0.1

        # forward pass
        input = np.vstack((
            1,
            sample.transpose()
        ))
        v = {0: input}
        y = {0: input}
        for l in xrange(1, len(self.network_size)):
            v[l] = W[l - 1].dot(y[l - 1])
            y[l] = np.vstack((
                1,
                logistic_function(v[l])
            ))

        # backward pass
        error = {}
        delta = {}
        for l in reversed(xrange(len(self.network_size))):
            if l == len(self.network_size) - 1:
                error[l] = output.transpose() - y[l]
            else:
                error[l] = W[l].transpose().dot(delta[l + 1])
            delta[l] = np.array(error[l]) * logistic_function_derivative(np.array(v[l]))

        # modify weights
        for l in reversed(xrange(1, len(self.network_size))):
            W[l - 1] += learning_rate * delta[l].dot(y[l - 1].transpose())

    def train_at_set(self, training_set):
        samples, values = np.hsplit(training_set, (-m,))
        for sample, value in zip(samples, values):
            self.process_learning(sample, value)

    def process(self, sample):
        """
        Straightforward computation
        """
        cur = sample.transpose()
        for l in xrange(len(self.network_size) - 1):
            cur = logistic_function(self.weights[l].dot(cur))
        return cur

def load_samples(fname):
    with open(fname, 'r') as f:
        n, m = map(int, f.readline().split())
        data = []
        for line in f.readlines():
            data.append(map(float, line.split()))
        return (n, m, np.matrix(data))


if __name__ == '__main__':
    n, m, data = load_samples('../data/circles.tsv')
    net = MultilayerNetwork((2, 10, 1))

    net.train_at_set(data)

    samples, values = np.hsplit(data, (-m,))
    for sample, value in zip(samples, values):
        print net.process(sample), value
