__author__ = 'xiajie'

import numpy as np
from MNISThelper import loader
from SparseAutoencoderV.sparseAutoencoderCost import J, vec2mat
from SparseAutoencoderV.train import initialization
from scipy.optimize import fmin_l_bfgs_b
from feedForwardAutoencoder import feed_forward
from SoftmaxRegression import train as sr_train
from SoftmaxRegression.prediction import predict as sr_predict
from SoftmaxRegression.softMaxCost import vec2mat as sr_vec2mat
import scipy.io as sio

input_size = 28*28
num_labels = 5
hidden_size = 200
ro = 0.1
lbda = 3e-3
beta = 3.


def main():
    global input_size, num_labels, hidden_size, ro, lbda, beta
    levels = [input_size, hidden_size, input_size]
    mnist_data = loader.load_train_imgs()
    mnist_labels = loader.load_train_labels()
    labeled_set = np.where(mnist_labels < num_labels)[0]
    unlabeled_set = np.where(mnist_labels > num_labels-1)[0]
    train_num = np.round(len(labeled_set)/2)
    train_data = mnist_data[:, labeled_set[:train_num]]
    train_labels = mnist_labels[labeled_set[:train_num]]
    test_data = mnist_data[:, labeled_set[train_num:]]
    test_labels = mnist_labels[labeled_set[train_num:]]
    unlabeled_data = mnist_data[:, unlabeled_set]
    print "train data:", train_data.shape[1]
    print "test data:", test_data.shape[1]
    print "unlabeled data:", unlabeled_data.shape[1]
    init_theta = initialization(levels)

    theta, cost, info = fmin_l_bfgs_b(J, init_theta, args=(unlabeled_data, unlabeled_data, levels, lbda, ro, beta), approx_grad=False, maxiter=400)
    print "cost:", cost
    print "info:", info

    WB = vec2mat(theta, levels)
    sio.savemat('W', {'W': WB[0][0]})
    train_a2 = feed_forward(WB[0], train_data, hidden_size)
    test_a2 = feed_forward(WB[0], test_data, hidden_size)
    sr_init_theta = sr_train.initialize_theta(hidden_size, num_labels)
    sr_theta = sr_train.train(sr_init_theta, train_a2, train_labels, num_labels)
    sr_mat = sr_vec2mat(sr_theta, num_labels)
    pY = sr_predict(sr_mat, test_a2)
    print test_labels[:20]
    print pY[:20]
    miss = 0.
    for i, l in enumerate(test_labels):
        if l != pY[i]:
            miss += 1.
    print miss/len(test_labels)

if __name__ == '__main__':
    main()
