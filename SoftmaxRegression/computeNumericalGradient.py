__author__ = 'xiajie'

import numpy as np


EPSILON = 1e-4


def theta_plus(theta, i):
    global EPSILON
    ret = np.zeros(len(theta))
    ret[i] += EPSILON
    return ret+theta


def theta_minus(theta, i):
    global EPSILON
    ret = np.zeros(len(theta))
    ret[i] -= EPSILON
    return ret+theta


def compute_numerical_gradient(J, theta, args):
    global EPSILON
    numeric_grad = np.zeros(len(theta))
    print 'len theta:', len(theta)
    for i in range(len(theta)):
        a = J(theta_plus(theta, i), args)[0]
        b = J(theta_minus(theta, i), args)[0]
        numeric_grad[i] = (a-b)/(2*EPSILON)
    return numeric_grad
