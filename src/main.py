import json
from math import fabs
import numpy as np
import os
import pickle
from Networks import MultilayerNetwork

# TODO:
# * stopping criteria
# * what about alpha in weights mod?
# * move training process inside network
#
# * push network name in url
# * interface fix
#
# OPTIONAL:
# * check gradient
# * regularization
# * python load_data_file handler instead of JS processing

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

def generate_processed_data_file(fname, net, data, m):
    samples, values = np.hsplit(data, (-m,))
    res = np.vstack([net.process(sample) for sample in samples])
    data_res = np.hstack((samples, values, res))

    str_data = np.vectorize(lambda x: '%.3f' % x)(data_res)
    with open(fname, 'w') as f:
        f.write('%d %d\n' % (data.shape[1] - m, m))
        for sample in str_data:
            f.write('\t'.join(sample.flat) + '\n')

def load_net(net_name):
    path = os.path.join('..', 'networks', net_name)
    net = pickle.load(open(os.path.join(path, "net.pickled"), 'r'))
    return net

def process_data_set(net_name):
    path = os.path.join('..', 'networks', net_name)
    desc = json.load(open(os.path.join(path, 'desc.json'), 'r'))
    data_file = os.path.join(path, desc["data_file"])
    n, m, data = load_samples(data_file)
    training_set, test_set, _ = np.vsplit(data, (
        desc["training_set_size"],
        desc["training_set_size"] + desc["test_set_size"]))

    net = MultilayerNetwork(desc["layers"])

    log = []
    epoch = 0
    E_history = []
    best_E =  net.error_at_set(training_set, m)
    while True:
        epoch += 1
        net.train_at_set(training_set, m)
        tr_se, te_se = net.error_at_set(training_set, m), net.error_at_set(test_set, m)
        if tr_se < best_E:
            best_E = tr_se
            pickle.dump(net, open(os.path.join(path, "net.pickled"), 'w'))
        print "Epoch %d, training=%8.5f, test=%8.5f (LR=%8.5f)" % (epoch, tr_se, te_se, net.learning_rate)
        log.append({
            "t" : epoch,
            "train_E" : tr_se,
            "test_E" : te_se
        })

        E_history.append(tr_se)

        INC_HISTORY_SIZE = 6
        DEC_HISTORY_SIZE = 4
        SHAKE_HISTORY_SIZE = 4
        SHAKE_EPS = 1e-5

        if len(E_history) >= INC_HISTORY_SIZE:
            differences = []
            for i in xrange(len(E_history) - INC_HISTORY_SIZE, len(E_history)):
                differences.append(E_history[i] - E_history[i - 1])
            if all([e < 0 for e in differences]):
                print 'Increasing LR'
                net.learning_rate *= 1.25

        if len(E_history) >= DEC_HISTORY_SIZE:
            differences = []
            for i in xrange(len(E_history) - DEC_HISTORY_SIZE, len(E_history)):
                differences.append(E_history[i] - E_history[i - 1])
            if len(filter(lambda x: x > 0, differences)) >= 2:
                print 'Decreasing LR'
                net.learning_rate /= 1.25

        if len(E_history) >= SHAKE_HISTORY_SIZE:
            differences = []
            for i in xrange(len(E_history) - DEC_HISTORY_SIZE, len(E_history)):
                differences.append(E_history[i] - E_history[i - 1])
            if all([fabs(e) < SHAKE_EPS for e in differences]):
                print 'Shaking weights'
                E_history = []
                net.learning_rate = 0.002
                net.shake_weights()

        if epoch >= 70:
            break

    result = {
        "log" : log,
        "training_set_acc" : calc_accuarcy(net, training_set, m),
        "test_set_acc" : calc_accuarcy(net, test_set, m),
    }
    json.dump(result, open(os.path.join(path, "learn_log.json"), 'w'))
    if desc["data_visualizer"]:
        pdf_path = os.path.join(path, desc["data_visualizer"]["processed_data_file"])
        generate_processed_data_file(pdf_path, net, test_set, m)

def tmp_just_get_pdf(net_name):
    path = os.path.join('..', 'networks', net_name)
    desc = json.load(open(os.path.join(path, 'desc.json'), 'r'))
    data_file = os.path.join(path, desc["data_file"])
    pdf_path = os.path.join(path, desc["data_visualizer"]["processed_data_file"])
    net = load_net(net_name)
    n, m, data = load_samples(data_file)
    training_set, test_set, _ = np.vsplit(data, (
        desc["training_set_size"],
        desc["training_set_size"] + desc["test_set_size"]))
    generate_processed_data_file(pdf_path, net, test_set, m)

if __name__ == '__main__':
    process_data_set("rings_01")
    #tmp_just_get_pdf("circles_03")