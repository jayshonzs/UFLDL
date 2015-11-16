__author__ = 'xiajie'

from scipy.optimize import fmin_l_bfgs_b
from sparseAutoencoderCost import J, mat2vec, vec2mat
from sampleIMAGES import sample_images
import numpy as np
import math
import scipy.io as sio


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


def do_train(init_theta, X, Y, levels):
    try:
        theta, cost, info = fmin_l_bfgs_b(J, init_theta, args=(X, Y, levels), approx_grad=False, maxiter=400)
    except Exception as e:
        print e
    print theta
    print cost
    print info
    return theta

# using NG's matlab code to display the result
'''
from scipy import misc

def visualize(W1, levels):
    print W1.shape
    W1_sq = W1**2
    hiddens = levels[1]
    row = int(np.sqrt(levels[0]))
    col = row
    for i in range(hiddens):
        x = np.zeros(row*col)
        normalizer = np.sqrt(np.sum(W1_sq[i,:]))
        for j in range(levels[0]):
            x[j] = W1[i, j]/normalizer
        x = x.reshape((row, col))
        misc.imsave('s'+str(i)+'.png', x)
'''

if __name__ == '__main__':
    np.seterr(all='print')
    levels = [64, 25, 64]
    samples = sample_images(10000)
    print samples.shape
    init_theta = initialization(levels)
    print init_theta.shape
    theta = do_train(init_theta, samples, samples, levels)
    WB = vec2mat(theta, levels)
    print 'W shape:', WB[0][0].shape
    sio.savemat('W', {'W': WB[0][0]})
