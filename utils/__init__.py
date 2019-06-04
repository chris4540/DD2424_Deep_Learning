import time
import numpy as np
from itertools import islice
from scipy.special import softmax

def window(seq, n=2):
    """
    Returns a sliding window (of width n) over data from the iterable
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i+n]


def one_hot(labels):
    """
    Create a one hot encoding matrix from labels

    Args:
        labels (list[int])

    Return:
        one hot encoding matrix of the labels
    """
    one_idx = np.array(labels)
    nkind = len(np.unique(one_idx))
    nlabels = len(one_idx)
    ret = np.zeros((nkind, nlabels))
    ret[one_idx, np.arange(nlabels)] = 1
    return ret

def train(train_loader, net, weight_decay, scheduler):
    loss = 0
    total = 0
    correct = 0
    st = time.time()
    for inputs, labels in train_loader:
        net.train()
        out = net(inputs)
        loss += net.cross_entropy(out, labels)
        if weight_decay > 0:
            loss += weight_decay * net.L2_penalty()
        # grads = net.backward(out, labels, weight_decay)
        # net.backward(grads, lrate=scheduler.get_lr())
        net.backward(out, labels, weight_decay, lrate=scheduler.get_lr())
        scheduler.step()
        # ============================================
        # make prediction
        # apply softmax
        s_mat = softmax(out, axis=0)
        # obtain the top one
        pred = np.argmax(s_mat, axis=0)
        correct += np.sum(pred == labels)
        total += labels.shape[0]

    # print stats
    used_time = time.time() - st
    loss /= total
    acc = correct / total
    print('Train Time used: %d \t Loss: %.3f | Train Acc: %.3f%% (%d/%d)' %
        (used_time, loss, acc*100, correct, total))


def evaluate(loader, net):
    net.eval()
    correct = 0
    total = 0
    for inputs, labels in loader:
        pred = net.predict(inputs)
        correct += np.sum(pred == labels)
        total += labels.shape[0]

    ret = correct/total
    return ret