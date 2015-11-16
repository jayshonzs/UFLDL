__author__ = 'xiajie'

from SparseAutoencoderV.sparseAutoencoderCost import J, vec2mat
from SparseAutoencoderV.train import initialization
from MNISThelper.loader import load_train_imgs
from scipy.optimize import fmin_l_bfgs_b
import scipy.io as sio


def main():
    levels = [28*28, 196, 28*28]
    ro = 0.1
    beta = 3.
    lbda = 3e-3
    init_theta = initialization(levels)
    X = load_train_imgs()[:, :10000]
    theta, cost, info = fmin_l_bfgs_b(J, init_theta, args=(X, X, levels, lbda, ro, beta), approx_grad=False, maxiter=400)
    print theta
    print cost
    print info
    WB = vec2mat(theta, levels)
    print 'W shape:', WB[0][0].shape
    sio.savemat('W', {'W': WB[0][0]})

if __name__ == '__main__':
    import time
    print time.time()
    main()
    print time.time()
