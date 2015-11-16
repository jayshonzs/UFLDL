__author__ = 'xiajie'

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sparseAutoencoderLinearCost import J, mat2vec, vec2mat
import math
import scipy.io as sio


imageChannels = 3
patchDim = 8
numPatches = 100000
visibleSize = patchDim*patchDim*imageChannels
outputSize = visibleSize
hiddenSize = 400

ro = 0.035
lbda = 3e-3
beta = 5

epsilon = 0.1


def load_data(filename=u'../LinearDecoder/stlSampledPatches.mat'):
    data = sio.loadmat(filename)
    patches = data['patches']
    return patches


def zca_whitening(X):
    global numPatches, epsilon
    avg = np.mean(X, axis=1).reshape((X.shape[0], 1))
    mat = X - avg
    print np.sum(X, axis=1)[2]
    sigma = mat.dot(np.transpose(mat))/numPatches
    u, s, _ = np.linalg.svd(sigma)
    zca_white = u.dot(np.diag(1./np.sqrt(s+epsilon))).dot(np.transpose(u))
    print zca_white.shape, X.shape
    return zca_white.dot(X), zca_white, avg


def random_bound(levels):
    return math.sqrt(6./(levels[0]+levels[1]+1))


def initialization(levels):
    res = []
    r = random_bound(levels)
    for i, si in enumerate(levels):
        si_plus_1 = levels[i+1]
        W = np.random.uniform(-r, r, si*si_plus_1).reshape((si_plus_1, si))
        b = np.zeros(si_plus_1)
        res.append((W, b))
        if i == len(levels)-2:
            break
    return mat2vec(res)


def train(init_theta, X, Y, levels, thetapath=None):
    global lbda, ro, beta
    if thetapath is None:
        thetapath = "theta.txt"
    try:
        theta = np.genfromtxt(thetapath)
        print theta.shape
    except:
        try:
            theta, cost, info = fmin_l_bfgs_b(J, init_theta, args=(X, Y, levels, lbda, ro, beta), approx_grad=False, maxiter=400)
            np.savetxt(thetapath, theta)
            print cost
            print info
        except Exception as e:
            print e
            return None
    return theta


def load_features(thetapath=None):
    global visibleSize, hiddenSize, outputSize
    levels = [visibleSize, hiddenSize, outputSize]
    X = load_data()
    X, zca_white, avg = zca_whitening(X)
    print X.shape, zca_white.shape, avg.shape
    init_theta = initialization(levels)
    theta = train(init_theta, X, X, levels, thetapath)
    WB = vec2mat(theta, levels)
    print 'W shape:', WB[0][0].shape
    return WB, zca_white, avg


if __name__ == '__main__':
    levels = [visibleSize, hiddenSize, outputSize]
    X = load_data()
    print X.shape
    X, zca_white, _ = zca_whitening(X)
    print X.shape
    init_theta = initialization(levels)
    theta = train(init_theta, X, X, levels)
    if theta is not None:
        WB = vec2mat(theta, levels)
        print 'W shape:', WB[0][0].shape
        v = WB[0][0].dot(zca_white)
        print v.shape
        sio.savemat('W', {'W': v})
