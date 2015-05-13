import numpy as np
import struct

import config
from core import DatasetInfo

_mnist_data = {}


def read_mnist(fname):
    data = open(fname).read()
    magic = struct.unpack('>i', data[:4])[0]
    assert magic == 2051
    num_images = struct.unpack('>i', data[4:8])[0]
    num_rows = struct.unpack('>i', data[8:12])[0]
    num_cols = struct.unpack('>i', data[12:16])[0]
    
    pixels = struct.unpack('B' * num_images * num_rows * num_cols, data[16:])
    pixels = np.array(pixels, dtype=float).reshape((num_images, num_rows, num_cols)) / 255.

    return pixels


class MNISTInfo(DatasetInfo):
    num_images = 60000
    num_rows = 28
    num_cols = 28
    m = num_images
    n = num_rows * num_cols

    @staticmethod
    def load(fname=config.MNIST_FILE):
        if fname not in _mnist_data:
            _mnist_data[fname] = read_mnist(fname)
        return MNISTData(_mnist_data[fname])

    @staticmethod
    def load_test():
        return MNISTInfo.load(config.MNIST_TEST_FILE)

class MNISTData(MNISTInfo):
    def __init__(self, pixels):
        self.pixels = pixels
        self.pixels.flags.writeable = False
    
    def as_matrix(self):
        return self.pixels.reshape((-1, self.n))


class SubsampledMNISTInfo(MNISTInfo):
    num_rows = 14
    num_cols = 14
    n = num_rows * num_cols

    @staticmethod
    def load(fname=config.MNIST_FILE):
        if fname not in _mnist_data:
            _mnist_data[fname] = read_mnist(fname)
        pixels = _mnist_data[fname]
        pixels = 0.25 * (pixels[:, ::2, ::2] + pixels[:, 1::2, ::2] + pixels[:, ::2, 1::2] + pixels[:, 1::2, 1::2])
        return SubsampledMNISTData(pixels)

    @staticmethod
    def load_test():
        return SubsampledMNISTInfo.load(config.MNIST_TEST_FILE)

class SubsampledMNISTData(SubsampledMNISTInfo):
    def __init__(self, pixels):
        self.pixels = pixels
        self.pixels.flags.writeable = False
    
    def as_matrix(self):
        return self.pixels.reshape((-1, self.n))





    
