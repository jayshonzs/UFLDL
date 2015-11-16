__author__ = 'xiajie'

from SparseAutoencoderV.sparseAutoencoderCost import sigmoid


def feed_forward(WB1, X, hidden_size):
    z2 = WB1[0].dot(X) + WB1[1].reshape((hidden_size, 1))
    a2 = sigmoid(z2)
    return a2
