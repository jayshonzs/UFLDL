__author__ = 'xiajie'

import numpy as np
from scipy.sparse import coo_matrix


def vec2mat(theta, num_classes):
    input_size = len(theta)/num_classes
    mat = theta.reshape((num_classes, input_size))
    return mat


def mat2vec(mat):
    row = mat.shape[0]
    col = mat.shape[1]
    theta = mat.reshape(row*col)
    return theta


def J(theta, args):
    X = args[0]
    Y = args[1]
    lbda = args[2]
    num_class = args[3]
    mat = vec2mat(theta, num_class)
    m = X.shape[1]
    M = mat.dot(X)
    M = M - np.max(M, axis=0)
    eM = np.e**M
    seN = np.sum(eM, axis=0)
    P = eM/seN
    H = np.log(P)
    ground_truth = coo_matrix((np.ones(len(Y)), (Y, np.linspace(0, m-1, m)))).toarray()
    decay_weight = 0.5*lbda*np.sum(mat**2)
    cost = -1./m*np.sum(ground_truth*H)+decay_weight
    print cost
    GP = ground_truth - P
    grads = -1./m*GP.dot(np.transpose(X)) + lbda*mat
    grads = grads.reshape(np.size(grads))
    return cost, grads


if __name__ == '__main__':
    pass
