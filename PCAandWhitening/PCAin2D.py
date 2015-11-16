__author__ = 'xiajie'

import matplotlib.pyplot as plt
import numpy as np


epsilon = 0.00001


def load_data(filename=u'pcaData.txt'):
    res = np.genfromtxt(filename)
    plt.scatter(res[0, :], res[1, :])
    plt.show()
    return res


def pca(X):
    m = len(X[0])
    avg = np.mean(X, axis=1).reshape((2, 1))*np.ones((2, m))
    mat = X-avg
    sigma = mat.dot(np.transpose(mat))/m
    u, s, v = np.linalg.svd(sigma)
    print u
    print s
    print v
    xRot = np.transpose(u).dot(mat)
    plt.scatter(xRot[1, :], xRot[0, :])
    plt.show()
    xTilde = np.transpose(u[:, 0]).dot(mat).reshape((1, 45))
    xHat = u[:, 0].reshape((2, 1)).dot(xTilde)
    plt.scatter(xHat[1, :], xHat[0, :])
    plt.show()
    return xRot, u, s


def pca_whitening(xRot, u, s):
    global epsilon
    diag = np.diag(1./np.sqrt(s+epsilon))
    xPCAWhite = diag.dot(xRot)
    plt.scatter(xPCAWhite[1, :], xPCAWhite[0, :])
    plt.show()


def zca_whitening(xRot, u, s):
    global epsilon
    diag = np.diag(1./np.sqrt(s+epsilon))
    xZCAWhite = u.dot(diag.dot(xRot))
    plt.scatter(xZCAWhite[1, :], xZCAWhite[0, :])
    plt.show()


if __name__ == '__main__':
    data = load_data()
    xRot, u, s = pca(data)
    pca_whitening(xRot, u, s)
    zca_whitening(xRot, u, s)
