__author__ = 'xiajie'

import numpy as np
import scipy.io as sio
import random
from scipy import misc


def load_data(N=10000, size=12, filename=u'IMAGES_RAW.mat'):
    res_mat = np.zeros((size**2, N))
    data = sio.loadmat(filename)
    IMAGESr = data['IMAGESr']
    # print IMAGESr.shape
    for n in range(N):
        image_idx = random.randint(0, 9)
        start_row = random.randint(0, 500)
        start_col = random.randint(0, 500)
        patch = IMAGESr[start_row:start_row+size, start_col:start_col+size, image_idx].reshape((1, size**2))
        res_mat[:, n] = patch
    return res_mat


def sub_mean(X):
    avg = np.mean(X, axis=0).reshape((1, X.shape[1]))
    mat = X - avg
    return mat


def sub_mean2(X):
    avg = np.mean(X, axis=1).reshape((X.shape[0], 1))
    mat = X - avg
    return mat


def pca(X):
    _, c = X.shape
    mat = X.dot(np.transpose(X))/c
    u, s, _ = np.linalg.svd(mat)
    xRot = np.transpose(u).dot(X)
    return xRot, u, s


def pcaWhitening(X, R=True):
    epsilon = 0
    if R:
        epsilon = 0.001
    xRot, U, S = pca(X)
    diag = np.diag(1./np.sqrt(S+epsilon))
    xPCAWhite = diag.dot(xRot)
    return xPCAWhite, U, S


def zcaWhitening(X):
    xPCAWhitening, U, S = pcaWhitening(X)
    xZCAWhitening = U.dot(xPCAWhitening)
    return xZCAWhitening, U, S


def check_covariance(xRot):
    cov = xRot.dot(np.transpose(xRot))
    print cov[10:20, 10:20]
    misc.imsave('s.png', cov)


def find_k(S, threshold=0.99):
    total = np.sum(S)
    k_lbdas = 0.
    for i in range(len(S)):
        lbda = S[i]
        k_lbdas += lbda
        if k_lbdas/total >= threshold:
            return i+1


def dim_reduction(X, U, k):
    xTilde = np.transpose(U[:, :k]).dot(X)
    print xTilde.shape
    xHat = U[:, :k].dot(xTilde)
    return xHat


if __name__ == '__main__':
    rX = load_data()
    X = sub_mean(rX)
    # xRot, U, S = pca(X)
    # xRot, U, S = pcaWhitening(X, True)
    # check_covariance(xRot)
    '''
    k = find_k(S)
    xHat = dim_reduction(X, U, k)
    for i in range(50):
        misc.imsave('x'+str(i)+'_.png', xHat[:, i].reshape(12,12))
        misc.imsave('x'+str(i)+'__.png', X[:, i].reshape(12, 12))
    '''
    xZCAWhitening = zcaWhitening(X)[0]
    for i in range(50):
        misc.imsave('x'+str(i)+'_.png', xZCAWhitening[:, i].reshape(12,12))
        misc.imsave('x'+str(i)+'__.png', X[:, i].reshape(12, 12))
