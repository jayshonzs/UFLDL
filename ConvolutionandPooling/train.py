__author__ = 'xiajie'

import numpy as np
import scipy.io as sio
from LinearDecoder import train as lt
from cnnConvolve import convolve
from cnnPool import mean_pool
import math
import SoftmaxRegression.train as smrt
import random
import time

patch_dim = 8
pool_dim = 19
image_dim = 64


def load_train_images(filename=u'stlSubset/stlTrainSubset.mat'):
    data = sio.loadmat(filename)
    images = data['trainImages']
    labels = data['trainLabels']
    num_images = data['numTrainImages']
    print 'load_train_images:', images.shape, labels.shape, num_images
    return images, labels, num_images


def load_test_images(filename=u'stlSubset/stlTestSubset.mat'):
    data = sio.loadmat(filename)
    images = data['testImages']
    labels = data['testLabels']
    num_images = data['numTestImages']
    print 'load_test_images:', images.shape, labels.shape, num_images
    return images, labels, num_images


def cnn(images, num_images, w, b, zca_white, avg):
    global patch_dim, pool_dim
    num_features = w.shape[0]
    image_dim = images.shape[0]
    step_size = 10
    steps = int(math.ceil(num_features/step_size))
    pool_size = math.floor((image_dim-patch_dim+1)/pool_dim)
    
    pooled_features = np.zeros((num_features, num_images, pool_size, pool_size))
    for i in range(steps):
        feature_start = i*step_size;
        feature_end = (i+1)*step_size;
        wt = w[feature_start:feature_end, :]
        bt = b[feature_start:feature_end]
        convolved_features_this = convolve(patch_dim, step_size, images, wt, bt, zca_white, avg)
        pooled_features_this = mean_pool(pool_dim, convolved_features_this)
        pooled_features[feature_start:feature_end, :, :, :] = pooled_features_this
    return pooled_features


def train_softmax(pooled_train_features, train_num_images, train_labels):
    # train softmax regression
    softmax_lambda = 1e-4
    num_classes = 4
    softmax_x = np.transpose(pooled_train_features, (0, 2, 3, 1))
    softmax_x = softmax_x.reshape((np.size(pooled_train_features)/train_num_images, train_num_images))
    softmax_y = train_labels.reshape(np.size(train_labels)) - 1
    init_theta = smrt.initialize_theta(softmax_x.shape[0], num_classes)
    theta = smrt.train(init_theta, softmax_x, softmax_y, num_classes, softmax_lambda)
    
    mat = smrt.vec2mat(theta, num_classes)
    
    return mat


def classify(mat, pooled_test_features, test_labels, test_num_images):
    softmax_x_test = np.transpose(pooled_test_features, (0, 2, 3, 1))
    softmax_x_test = softmax_x_test.reshape((np.size(pooled_test_features)/test_num_images, test_num_images))
    softmax_y_test = test_labels.reshape(np.size(test_labels)) - 1
    predicted_labels = smrt.predict(mat, softmax_x_test)
    err = 0.
    for i in range(len(softmax_y_test)):
        if i < 100:
            print predicted_labels[i], softmax_y_test[i]
        if softmax_y_test[i] != predicted_labels[i]:
            err += 1.
    print 'error rate:', err/len(softmax_y_test)


def sigmoid(z):
    return 1./(1.+np.e**(-z))


def check(train_images):
    global patch_dim, image_dim
    WB, zca_white, avg = lt.load_features('../LinearDecoder/theta.txt')
    w = WB[0][0]
    b = WB[0][1]
    print 'zca_white:', zca_white.shape
    print 'w:', w.shape
    print 'b:', b.shape
    print 'avg:', avg.shape
    conv_images = train_images[:, :, :, :8]
    convolved_features = convolve(patch_dim, 400, conv_images, w, b, zca_white, avg)
    for _ in range(1000):
        feature_num = random.randint(0, 399)
        image_num = random.randint(0, 7)
        image_row = random.randint(0, image_dim - patch_dim)
        image_col = random.randint(0, image_dim - patch_dim)
        patch = conv_images[image_row:image_row + patch_dim, image_col:image_col + patch_dim, :, image_num]
        patch = np.concatenate((patch[:, :, 0].flatten(), patch[:, :, 1].flatten(), patch[:, :, 2].flatten()))
        patch = patch.reshape((patch.size, 1))
        patch = patch - avg
        patch = zca_white.dot(patch)
        print patch.shape
        features = w.dot(patch) + b.reshape((b.shape[0], 1))
        print features.shape
        features = sigmoid(features)
        print features.shape
        print '*******************'
        print features.shape
        print convolved_features.shape
        print feature_num, image_num, image_row, image_col
        if np.abs(features[feature_num, 0] - convolved_features[feature_num, image_num, image_row, image_col]) > 1e-9:
            print('Convolved feature does not match activation from autoencoder\n')
            print('Feature Number    : %d\n', feature_num)
            print('Image Number      : %d\n', image_num)
            print('Image Row         : %d\n', image_row)
            print('Image Column      : %d\n', image_col)
            print('Convolved feature : %0.5f\n', convolved_features[feature_num, image_num, image_row, image_col])
            print('Sparse AE feature : %0.5f\n', features[feature_num, 0])
            print('Convolved feature does not match activation from autoencoder')
    print 'Congratulations! Your convolution code passed the test.'
    
    test_matrix = np.arange(64).reshape(8, 8)
    expected_matrix = np.array([[np.mean(test_matrix[0:4, 0:4]), np.mean(test_matrix[0:4, 4:8])],
    [np.mean(test_matrix[4:8, 0:4]), np.mean(test_matrix[4:8, 4:8])]])
    test_matrix = np.reshape(test_matrix, (1, 1, 8, 8))
    pooled_features = mean_pool(4, test_matrix)
    if not (pooled_features == expected_matrix).all():
        print "Pooling incorrect"
        print "Expected matrix"
        print expected_matrix
        print "Got"
        print pooled_features
    print 'Congratulations! Your pooling code passed the test.'


if __name__ == '__main__':
    train_images, train_labels, train_numImages = load_train_images()
    # train_images = train_images[:,:,:,:100]
    # train_labels = train_labels[:100]
    # train_numImages = 100
    check(train_images)
    test_images, test_labels, test_numImages = load_test_images()
    WB, zcaWhite, avg = lt.load_features('../LinearDecoder/theta.txt')
    w = WB[0][0]
    b = WB[0][1]
    try:
        pooledTrainFeatures = np.load('pooledTrainFeatures.npy')
    except:
        print 'start train cnn...', time.time()
        pooledTrainFeatures = cnn(train_images, train_numImages, w, b, zcaWhite, avg)
        print pooledTrainFeatures
        print pooledTrainFeatures.shape
        np.save('pooledTrainFeatures', pooledTrainFeatures)
        print 'train cnn over...', time.time()
    print pooledTrainFeatures.shape
    try:
        mat = np.genfromtxt('softmax.mat')
    except:
        print 'start train_softmax...', time.time()
        mat = train_softmax(pooledTrainFeatures, train_numImages, train_labels)
        print 'train_softmax over...', time.time()
        np.savetxt('softmax.mat', mat)
    print mat.shape

    # for memory efficiency
    del pooledTrainFeatures

    # test
    try:
        pooledTestFeatures = np.load('pooledTestFeatures.npy')
    except:
        print 'start test cnn...', time.time()
        pooledTestFeatures = cnn(test_images, test_numImages, w, b, zcaWhite, avg)
        np.save('pooledTestFeatures', pooledTestFeatures)
        print 'test cnn over...', time.time()
    classify(mat, pooledTestFeatures, test_labels, test_numImages)
