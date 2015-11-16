__author__ = 'xiajie'

import numpy as np
from softMaxCost import J, vec2mat
from scipy.optimize import fmin_l_bfgs_b
from MNISThelper.loader import load_train_imgs, load_train_labels
from SoftmaxRegression.prediction import predict
from computeNumericalGradient import compute_numerical_gradient


def initialize_theta(input_size, num_classes):
    return np.random.normal(0, 1., input_size*num_classes)*0.005


def check_numerical_gradient(theta, X, Y, num_classes, lbda=1e-4):
    args = (X, Y, lbda, num_classes)
    grad = J(theta, args)[1]
    numeric_grad = compute_numerical_gradient(J, theta, args)
    for i, g in enumerate(grad):
        print g, numeric_grad[i]


def train(init_theta, X, Y, num_classes, lbda=1e-4):
    try:
        theta, cost, info = fmin_l_bfgs_b(J, init_theta, args=((X, Y, lbda, num_classes),), approx_grad=False, maxiter=400)
    except Exception as e:
        print e
    print theta
    print cost
    print info
    return theta


if __name__ == '__main__':
    inputSize = 28*28
    numClasses = 10
    lbda = 1e-4
    init_theta = initialize_theta(inputSize, numClasses)
    X = load_train_imgs(u'../MNISThelper/train-images.idx3-ubyte')
    Y = load_train_labels(u'../MNISThelper/train-labels.idx1-ubyte')
    print X.shape, Y.shape, init_theta.shape
    # numerical gradient check
    '''
    cX = X[:20, :10]
    cY = Y[:10]
    check_numerical_gradient(init_theta[:200], cX, cY, numClasses, lbda=lbda)
    '''
    theta = train(init_theta, X, Y, numClasses, lbda)
    mat = vec2mat(theta, numClasses)
    tX = load_train_imgs('../MNISThelper/t10k-images.idx3-ubyte')
    tY = load_train_labels('../MNISThelper/t10k-labels.idx1-ubyte')
    pY = predict(mat, tX)
    print tY[:20]
    print pY[:20]
    e = 0.
    for i in range(len(tY)):
        if pY[i] != tY[i]:
            e += 1
    print 'error rate', e/len(tY)
