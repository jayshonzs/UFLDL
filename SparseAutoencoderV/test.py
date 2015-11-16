__author__ = 'xiajie'

from computeNumericalGradient import compute_numerical_gradient
from sampleIMAGES import sample_images
from sparseAutoencoderCost import J
from train import initialization


def check_numerical_gradient(theta, X, levels=[64, 25, 64]):
    grad = J(theta, X, X, levels)[1]
    numeric_grad = compute_numerical_gradient(J, theta, X, X, levels)
    for i, g in enumerate(grad):
        print g, numeric_grad[i]


def check():
    levels = [64, 25, 64]
    theta = initialization(levels)
    X = sample_images(100)
    check_numerical_gradient(theta, X)


if __name__ == '__main__':
    check()
