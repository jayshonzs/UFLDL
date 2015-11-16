__author__ = 'xiajie'

import numpy as np
import struct


def load_train_imgs(filename=u'../MNISThelper/train-images.idx3-ubyte'):
    train_file = open(filename, 'rb')
    buf = train_file.read()
    
    index = 0
    magic, num_images, num_rows, num_columns = struct.unpack_from('>IIII', buf, index)
    # print magic, numImages, numRows, numColumns
    ret = np.zeros((num_rows*num_columns, num_images))
    index += struct.calcsize('>IIII')
    i = 0
    while True:
        im = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')
        
        im = np.array(im)
        ret[:, i] = im
        i += 1
        if i >= num_images:
            break
    ret /= 255.
    return ret


def load_train_labels(filename=u'../MNISThelper/train-labels.idx1-ubyte'):
    label_file = open(filename, 'rb')
    buf = label_file.read()
    
    index = 0
    magic, num_labels = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')
    # print magic, numLabels
    
    ret = np.zeros(num_labels)
    for i in range(num_labels):
        ret[i] = struct.unpack_from('>B', buf, index)[0]
        index += struct.calcsize('>B')
    
    return ret

if __name__ == '__main__':
    mat = load_train_imgs()
    print mat.shape
    labels = load_train_labels()
    print labels
