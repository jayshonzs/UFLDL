__author__ = 'xiajie'

import numpy as np
from stackedAECost import unstack_params, sigmoid


def predict(test_data, theta, levels):
    WB = unstack_params(theta, levels)
    softmax_mat = WB[-1][0]
    WB = WB[:-1]
    activations = [test_data]
    for i,wb in enumerate(WB):
        z = wb[0].dot(activations[-1]) + wb[1].reshape((levels[i+1], 1))
        a = sigmoid(z)
        activations.append(a)
    M = softmax_mat.dot(activations[-1])
    eM = np.e**(M)
    P = eM/np.sum(eM, axis=0)
    res = np.argmax(P, axis=0)
    return res

if __name__ == '__main__':
    pass
