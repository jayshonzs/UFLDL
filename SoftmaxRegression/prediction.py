__author__ = 'xiajie'

import numpy as np


def predict(mat, X):
    M = mat.dot(X)
    eM = np.e**M
    seN = np.sum(eM, axis=0)
    P = eM/seN
    res = np.argmax(P, axis=0)
    return res
