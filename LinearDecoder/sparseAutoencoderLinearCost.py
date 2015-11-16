__author__ = 'xiajie'

import numpy as np


def mat2vec(WB):
    vec = []
    for wb in WB:
        vec.extend(wb[0].reshape(wb[0].size))
        vec.extend(wb[1])
    return np.array(vec)


def vec2mat(theta, levels):
    WB = []
    off = 0
    for i, si in enumerate(levels):
        if i == len(levels)-1:
            break
        si_plus = levels[i+1]
        w = theta[off:off+si*si_plus].reshape((si_plus, si))
        off += si*si_plus
        b = theta[off:off+si_plus]
        off += si_plus
        WB.append((w, b))
    return WB


def sigmoid(z):
    return 1./(1.+np.e**(-z))


def sigmoid_prime(a):
    return a*(1.-a)


def KL(ro, ro2):
    return ro*np.log(ro/ro2)+(1-ro)*np.log((1-ro)/(1-ro2))


def sparsity(ro, ro2):
    return (1-ro)/(1-ro2) - ro/ro2


def J(theta, X, Y, levels, lbda=3e-3, ro=0.035, beta=5):
    m = X.shape[1]
    WB = vec2mat(theta, levels)
    z2 = WB[0][0].dot(X) + WB[0][1].reshape((levels[1], 1))
    a2 = sigmoid(z2)
    z3 = WB[1][0].dot(a2) + WB[1][1].reshape((levels[2], 1))
    a3 = z3
    ro2 = np.sum(a2, axis=1)/m
    sp = beta*sparsity(ro, ro2)
    ep3 = a3 - Y
    ep2 = (np.transpose(WB[1][0]).dot(ep3) + sp.reshape(sp.size, 1))*sigmoid_prime(a2)
    dW1 = ep2.dot(np.transpose(X))/m + lbda*WB[0][0]
    dB1 = np.sum(ep2, axis=1)/m
    dW2 = ep3.dot(np.transpose(a2))/m + lbda*WB[1][0]
    dB2 = np.sum(ep3, axis=1)/m
    grad = mat2vec([(dW1, dB1), (dW2, dB2)])
    cost = 0.5*np.sum((a3-Y)**2)/m + 0.5*lbda*np.sum(WB[0][0]**2) + 0.5*lbda*np.sum(WB[1][0]**2) + beta*np.sum(KL(ro, ro2))
    print 'cost:', cost
    return cost, grad


if __name__ == '__main__':
    pass
