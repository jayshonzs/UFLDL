__author__ = 'xiajie'

import scipy.io as sio
import numpy as np
import random
from scipy import misc


def normalize(images):
    mat = np.mean(images, axis=0)
    mat = images - mat
    mstd = np.std(mat.reshape(mat.size))*3.
    mat = np.where(mat > mstd, mstd, mat)
    mat = np.where(mat < -mstd, -mstd, mat)
    mat = mat/mstd
    mat = (mat + 1) * 0.4 + 0.1
    return mat


def sample_images(N=10000, size=8):
    res_mat = np.zeros((size**2, N))
    data=sio.loadmat(u'../SparseAutoencoderV/IMAGES.mat')
    images = data['IMAGES']
    # for test
    # misc.imsave('1st.png', images[:,:,0])
    for n in range(N):
        image_idx = random.randint(0, 9)
        start_row = random.randint(0, 504)
        start_col = random.randint(0, 504)
        res_mat[:, n] = images[start_row:start_row+size, start_col:start_col+size, image_idx].reshape((1, size**2))
    return normalize(res_mat)


if __name__ == '__main__':
    patch_mat = sample_images()
    s = patch_mat[:, 8998].reshape((8, 8))
    misc.imsave('lena.png', s)
