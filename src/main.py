import json
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
    while True:
        epoch += 1
        net.train_at_set(training_set, m)
        tr_se, te_se = net.error_at_set(training_set, m), net.error_at_set(test_set, m)
        print "Epoch %d, training=%8.5f, test=%8.5f (LR=%8.5f)" % (epoch, tr_se, te_se, net.learning_rate)
        log.append({
            "t" : epoch,
            "train_E" : tr_se,
            "test_E" : te_se
        })
        E_history.append(tr_se)
        if len(E_history) >= 4:
            e1, e2, e3, e4 = E_history[-4:]
            if np.sign(e2 - e1) == np.sign(e3 - e2) == np.sign(e4 - e3):
                net.learning_rate *= 1.5
            if np.sign(e2 - e1) != np.sign(e3 - e2) and np.sign(e3 - e2) != np.sign(e4 - e3):
                net.learning_rate /= 1.5

        if epoch >= 150:
            break

    result = {
        "log" : log,
        "training_set_acc" : calc_accuarcy(net, training_set, m),
        "test_set_acc" : calc_accuarcy(net, test_set, m),
    }
    json.dump(result, open(os.path.join(path, "learn_log.json"), 'w'))
    pickle.dump(net, open(os.path.join(path, "net.pickled"), 'w'))
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
    #process_data_set("circles_03")
    tmp_just_get_pdf("circles_03")