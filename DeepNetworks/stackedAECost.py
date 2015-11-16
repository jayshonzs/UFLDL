__author__ = 'xiajie'

import numpy as np
from scipy.sparse import coo_matrix


def stack_params(WBs, sr_mat):
    w_l1 = WBs[0][0]
    b_l1 = WBs[0][1]
    w_l2 = WBs[1][0]
    b_l2 = WBs[1][1]
    theta = w_l1.reshape(np.size(w_l1)).tolist() + b_l1.tolist()
    theta += w_l2.reshape(np.size(w_l2)).tolist() + b_l2.tolist()
    theta += sr_mat.reshape(np.size(sr_mat)).tolist()
    return np.array(theta)


def unstack_params(theta, levels):
    WB = []
    off = 0
    pre_levels = levels[:-1]
    for i, si in enumerate(pre_levels):
        if i == len(pre_levels)-1:
            break
        si_plus = pre_levels[i+1]
        w = theta[off:off+si*si_plus].reshape((si_plus, si))
        off += si*si_plus
        b = theta[off:off+si_plus]
        off += si_plus
        WB.append((w, b))
    sr_mat = theta[off:].reshape((levels[-1], levels[-2]))
    WB.append((sr_mat, np.zeros(levels[-1])))
    return WB


def sigmoid(z):
    return 1./(1.+np.e**(-z))


def sigmoid_prime(a):
    return a*(1.-a)


def stacked_cost(theta, X, Y, levels, lbda):
    m = X.shape[1]
    WB = unstack_params(theta, levels)
    softmax_mat = WB[-1][0]
    WB = WB[:-1]
    hidden_levels = len(WB)
    activations = [X]
    for i, wb in enumerate(WB):
        z = wb[0].dot(activations[-1]) + wb[1].reshape((levels[i+1], 1))
        a = sigmoid(z)
        activations.append(a)
    M = softmax_mat.dot(activations[-1])
    M -= np.max(M, axis=0)
    eM = np.e**(M)
    P = eM/np.sum(eM, axis=0)
    ground_truth = coo_matrix((np.ones(len(Y)), (Y, np.linspace(0, m-1, m)))).toarray()
    softmax_grad = -1./m*(ground_truth-P).dot(np.transpose(activations[-1])) + lbda*softmax_mat
    d = [0]*hidden_levels
    stack_grad = [None]*hidden_levels
    d[-1] = -np.transpose(softmax_mat).dot(ground_truth-P)*sigmoid_prime(activations[-1])
    for l in range(hidden_levels, 0, -1):
        d[hidden_levels-2] = np.transpose(WB[hidden_levels-1][0]).dot(d[hidden_levels-1])*sigmoid_prime(activations[hidden_levels-1])
    for l in range(0, hidden_levels):
        w = 1./m*d[l].dot(np.transpose(activations[l]))
        b = 1./m*np.sum(d[l], axis=1)
        stack_grad[l] = (w, b)
    cost = -1./m*np.sum(ground_truth*np.log(P)) + 0.5*lbda*np.sum(softmax_mat**2)
    print 'cost:', cost
    theta = stack_params(stack_grad, softmax_grad)
    return cost, theta

if __name__ == '__main__':
    pass
