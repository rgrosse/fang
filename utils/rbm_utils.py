import gnumpy as gnp
import numpy as np
nax = np.newaxis


def sigmoid(t):
    return 1. / (1. + gnp.exp(-t))

def sigmoid_slow(t):
    return 1. / (1. + np.exp(-t))

def sample_units(inputs):
    return gnp.rand(inputs.shape) < sigmoid(inputs)



