import json
import numpy as np
import os
import pickle
from Networks import MultilayerNetwork

# TODO:
# * stopping criteria
# * learning rate
# * what about alpha in weights mod?

def load_samples(fname):
    with open(fname, 'r') as f:
        n, m = map(int, f.readline().split())
        data = []
        for line in f.readlines():
            data.append(map(float, line.split()))
        return (n, m, np.matrix(data))

def calc_accuarcy(net, test_set, m):
    correct = 0
    samples, values = np.hsplit(test_set, (-m,))
    for sample, value in zip(samples, values):
        res = net.process(sample)
        if np.abs(res - value) < 0.5:
            correct += 1
    return correct * 1./ samples.shape[0]

def process_data_set(net_name):
    path = os.path.join('..', 'networks', net_name)
    desc = json.load(open(os.path.join(path, 'desc.json'), 'r'))
    n, m, data = load_samples('../data/circles.tsv')
    training_set, test_set, _ = np.vsplit(data, (
        desc["training_set_size"],
        desc["training_set_size"] + desc["test_set_size"]))

    net = MultilayerNetwork(desc["layers"])

    log = []
    epoch = 0
    while True:
        epoch += 1
        net.train_at_set(training_set, m)
        tr_se, te_se = net.error_at_set(training_set, m), net.error_at_set(test_set, m)
        print "Epoch %d, training=%8.2f, test=%8.2f" % (epoch, tr_se, te_se)
        log.append({
            "t" : epoch,
            "train_E" : tr_se,
            "test_E" : te_se
        })
        if epoch > 200:
            break

    json.dump(log, open(os.path.join(path, "learn_log.json"), 'w'))
    pickle.dump(net, open(os.path.join(path, "net.pickled"), 'w'))

if __name__ == '__main__':
    process_data_set("circles_02")