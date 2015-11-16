__author__ = 'xiajie'

import numpy as np

EPSILON = 0.0001


def theta_plus(theta, i):
    global EPSILON
    ret = np.zeros(len(theta))
    ret[i] += EPSILON
    ret += theta
    return ret


def theta_minus(theta, i):
    global EPSILON
    ret = np.zeros(len(theta))
    ret[i] -= EPSILON
    ret += theta
    return ret


def compute_numerical_gradient(J, theta, X, Y, levels):
    global EPSILON
    numeric_grad = np.zeros(len(theta))
    print 'len theta:', len(theta)
    for i in range(len(theta)):
        a = J(theta_plus(theta, i), X, Y, levels)[0]
        b = J(theta_minus(theta, i), X, Y, levels)[0]
        numeric_grad[i] = (a-b)/(2*EPSILON)
    return numeric_grad
