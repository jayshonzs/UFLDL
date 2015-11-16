__author__ = 'xiajie'

import numpy as np
from scipy.signal import convolve2d


def sigmoid(z):
    return 1./(1.+np.e**(-z))


def convolve(patch_dim, num_features, images, W, b, zca_white, avg):
    WT = W.dot(zca_white)
    num_images = images.shape[3]
    image_dim = images.shape[0]
    image_channels = images.shape[2]
    print num_features, num_images, image_dim, patch_dim
    convolved_features = np.zeros((num_features, num_images, image_dim-patch_dim+1, image_dim-patch_dim+1))
    patch_size = patch_dim**2
    features = {}
    constant = b - WT.dot(avg).reshape(len(b))
    print b.shape, WT.shape, avg.shape, constant.shape
    for c in range(image_channels):
        features.setdefault(c, [])
        for f in range(num_features):
            print WT.shape
            feature = WT[f, c*patch_size:(c+1)*patch_size].reshape((patch_dim, patch_dim))
            feature = np.fliplr(feature)
            feature = np.flipud(feature)
            features[c].append(feature)
    for feat_idx in range(num_features):
        print 'convolve feature:', feat_idx
        for img_idx in range(num_images):
            convolved_images = []
            for channel in range(image_channels):
                feature = features[channel][feat_idx]
                im = images[:, :, channel, img_idx]
                convolved_image = convolve2d(im, feature, mode='valid')
                convolved_images.append(convolved_image)
            convolved_image = sum(convolved_images)
            convolved_image += constant[feat_idx]
            convolved_features[feat_idx, img_idx, :, :] = sigmoid(convolved_image)
    return convolved_features

if __name__ == '__main__':
    pass
