__author__ = 'xiajie'

import numpy as np
import math


def mean_pool(pool_dim, convolved_features):
    num_features = convolved_features.shape[0]
    num_images = convolved_features.shape[1]
    convolved_dim = convolved_features.shape[2]
    pool_size = int(math.floor(convolved_dim/pool_dim))
    pooled_features = np.zeros((num_features, num_images, pool_size, pool_size))
    for img_idx in range(num_images):
        for feat_idx in range(num_features):
            pooled_feature = np.zeros((pool_size, pool_size))
            for i in range(pool_size):
                for j in range(pool_size):
                    m = convolved_features[feat_idx, img_idx, i*pool_dim:(i+1)*pool_dim, j*pool_dim:(j+1)*pool_dim]
                    pooled_feature[i, j] = np.mean(m)
            pooled_features[feat_idx, img_idx, :, :] = pooled_feature
    return pooled_features

if __name__ == '__main__':
    pass
