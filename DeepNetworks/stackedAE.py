__author__ = 'xiajie'

import numpy as np
from MNISThelper import loader
from SparseAutoencoderV.sparseAutoencoderCost import J, vec2mat
from SparseAutoencoderV.train import initialization
from scipy.optimize import fmin_l_bfgs_b
import scipy.io as sio
from SelfTaughtLearning.feedForwardAutoencoder import feed_forward
from SoftmaxRegression import train as sr_train
from SoftmaxRegression.softMaxCost import vec2mat as sr_vec2mat
import time
from stackedAECost import stack_params
from stackedAECost import stacked_cost
from stackedAEPredict import predict

input_size = 28*28
num_labels = 10
hidden_size_l1 = 200
hidden_size_l2 = 200
ro = 0.1
lbda = 3e-3
beta = 3.


def pre_train(train_data, train_labels):
    global input_size, num_labels, hidden_size_l1, hidden_size_l2, ro, lbda, beta
    levels_l1=[input_size, hidden_size_l1, input_size]
    
    print 'Step 1:', time.time()
    
    # stack 1
    init_theta = initialization(levels_l1)
    try:
        theta = np.genfromtxt("theta_l1.txt")
        print theta.shape
    except:
        theta, cost, info = fmin_l_bfgs_b(J, init_theta, args=(train_data, train_data, levels_l1, lbda, ro, beta), approx_grad=False, maxiter=400)
        np.savetxt("theta_l1.txt", theta)
        print "cost:", cost
        print "info:", info
    WB_l1 = vec2mat(theta, levels_l1)
    sio.savemat('W_l1', {'W_l1':WB_l1[0][0]})
    
    print 'Step 2:', time.time()
    
    # stack 2
    levels_l2 = [hidden_size_l1, hidden_size_l2, hidden_size_l1]
    train_l1_a2 = feed_forward(WB_l1[0], train_data, hidden_size_l1)
    init_theta = initialization(levels_l2)
    try:
        theta = np.genfromtxt("theta_l2.txt")
        print theta.shape
    except:
        theta, cost, info = fmin_l_bfgs_b(J, init_theta, args=(train_l1_a2, train_l1_a2, levels_l2, lbda, ro, beta), approx_grad=False, maxiter=400)
        np.savetxt("theta_l2.txt", theta)
        print "cost:", cost
        print "info:", info
    WB_l2 = vec2mat(theta, levels_l2)
    sio.savemat('W_l2', {'W_l2':WB_l2[0][0]})
    
    print 'Step 3:', time.time()
    
    try:
        sr_theta = np.genfromtxt("sr_theta.txt")
        print sr_theta.shape
    except:
        train_l2_a2 = feed_forward(WB_l2[0], train_l1_a2, hidden_size_l2)
        sr_init_theta = sr_train.initialize_theta(hidden_size_l2, num_labels)
        sr_theta = sr_train.train(sr_init_theta, train_l2_a2, train_labels, num_labels)
        np.savetxt("sr_theta.txt", sr_theta)
    sr_mat = sr_vec2mat(sr_theta, num_labels)
    
    return WB_l1, WB_l2, sr_mat


def fine_tuning(init_theta, train_data, train_labels, levels, lbda):
    try:
        theta = np.genfromtxt("fine_tuned.txt")
        print theta.shape
    except:
        theta, cost, info = fmin_l_bfgs_b(stacked_cost, init_theta, args=(train_data, train_labels, levels, lbda), approx_grad=False, maxiter=400)
        np.savetxt("fine_tuned.txt", theta)
        print "cost:", cost
        print "info:", info
    return theta


def train():
    global input_size, num_labels, hidden_size_l1, hidden_size_l2, lbda
    levels = [input_size, hidden_size_l1, hidden_size_l2, num_labels]
    train_data = loader.load_train_imgs()
    train_labels = loader.load_train_labels()
    WB_l1, WB_l2, sr_mat = pre_train(train_data, train_labels)
    init_theta = stack_params((WB_l1[0], WB_l2[0]), sr_mat)
    
    theta = fine_tuning(init_theta, train_data, train_labels, levels, lbda)
    
    return theta


def test(theta):
    global input_size, num_labels, hidden_size_l1, hidden_size_l2
    levels = [input_size, hidden_size_l1, hidden_size_l2, num_labels]
    train_data = loader.load_train_imgs(u'../MNISTHelper/t10k-images.idx3-ubyte')
    train_labels = loader.load_train_labels(u'../MNISTHelper/t10k-labels.idx1-ubyte')
    pl = predict(train_data, theta, levels)
    print train_labels[:20]
    print pl[:20]
    e = 0.
    for i in range(len(pl)):
        if train_labels[i] != pl[i]:
            e += 1
    print 'error rate:', e/len(pl)

if __name__ == '__main__':
    theta = train()
    test(theta)
