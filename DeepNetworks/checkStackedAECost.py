__author__ = 'xiajie'

import numpy as np
from stackedAECost import stacked_cost
from MNISThelper import loader


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


def compute_numerical_gradient(J, theta, X, Y, levels, lbda):
    global EPSILON
    numeric_grad = np.zeros(len(theta))
    print 'len theta:', len(theta)
    for i in range(len(theta)):
        a = J(theta_plus(theta, i), X, Y, levels, lbda)[0]
        b = J(theta_minus(theta, i), X, Y, levels, lbda)[0]
        numeric_grad[i] = (a-b)/(2*EPSILON)
    return numeric_grad


def initialization(levels):
    nt = 0
    for i, l in enumerate(levels):
        if i == len(levels)-1:
            break
        nt += levels[i+1]*l
        if i != len(levels)-2:
            nt += levels[i+1]
    return np.random.normal(0., 1., nt)


def check_numerical_gradient():
    X = loader.load_train_imgs()[:64, :10]
    Y = loader.load_train_labels()[:10]
    levels = [64, 20, 20, 10]
    theta = initialization(levels)
    print theta.shape
    lbda = 3e-3
    grad = stacked_cost(theta, X, Y, levels, lbda)[1]
    numgrad = compute_numerical_gradient(stacked_cost, theta, X, Y, levels, lbda)
    for i, g in enumerate(grad):
        print g, numgrad[i]


if __name__ == '__main__':
    check_numerical_gradient()
