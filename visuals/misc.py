import math
import numpy as np
from numpy import *
import pylab
from utils import *




def override(dicts, add=True):
    result = {}
    for dict in dicts:
        for key, val in dict.items():
            if add or dicts[0].has_key(key):
                result[key] = val
    return result


_DEFAULT_NORM_PARS = {'centered': None,
                      'mn': None,
                      'mx': None,
                      'tmn': None,
                      'tmx': None,
                      'norm_mode': 'group'}

def norm01(arr, **kwargs):
    """
    Rescale arr so that the smallest value is 0 and
    the largest value is 1.
    """
    kwargs = override((_DEFAULT_NORM_PARS, kwargs))
    centered = kwargs['centered']
    mn = kwargs['mn']
    mx = kwargs['mx']
    tmn = kwargs['tmn']
    tmx = kwargs['tmx']
    arr = asanyarray(arr, float)

    if mn is None or mx is None:
        if centered is not None:
            if mn is None:
                mn = -max(abs(arr.ravel()-centered))+centered
            if mx is None:
                mx = max(abs(arr.ravel()-centered))+centered
        else:
            if mn is None:
                mn = min(arr.ravel())
            if mx is None:
                mx = max(arr.ravel())

    if tmn is None: tmn = 0.
    if tmx is None: tmx = 1.
    assert tmn < tmx
        
    if mn == mx:
        return zeros(arr.shape)
    else:
        return tmn + (tmx-tmn)*(arr-mn)/(mx-mn)

def norm_group(arrs, **kwargs):
    kwargs = override((_DEFAULT_NORM_PARS, kwargs))
    centered = kwargs['centered']
    mn = kwargs['mn']
    mx = kwargs['mx']
    tmn = kwargs['tmn']
    tmx = kwargs['tmx']
    mode = kwargs['norm_mode']
    
    if mode == 'group':
        arr = concatenate([ar.ravel() for ar in arrs])

        if mn is None or mx is None:
            if centered is not None:
                mn = -max(abs(arr.ravel()-centered))+centered
                mx = max(abs(arr.ravel()-centered))+centered
            else:
                mn = min(arr.ravel())
                mx = max(arr.ravel())

        if tmn is None: tmn = 0.
        if tmx is None: tmx = 1.
        assert tmn < tmx
        
        if mn == mx:
            return [tmn*ones(arr.shape) for arr in arrs]
        else:
            return [tmn + (tmx-tmn)*(arr-mn)/(mx-mn) for arr in arrs]

    elif mode == 'individual':
        return [apply(norm01, (arr,), kwargs) for arr in arrs]
    else:
        assert False, 'Unrecognized mode: %s' % mode

            
            
            
_DEFAULT_JOIN_PARS = {'normalize': True,
                      'spc': 3,
                      'boundary': True,
                      'backcolor': np.zeros(3)}

def hjoin(alist, **kwargs):
    """
    Concatenate the arrays in alist horizontally, with spacing spc between them.
    If normalize is True, the arrays are all normalized to be between 0 and 1.
    Arrays may be any number of dimensions, but all the dimensions beyond the
    second must match exactly.
    """
    if len(alist) == 0:
        return zeros((0,0))
    
    kwargs = override((_DEFAULT_JOIN_PARS, kwargs))
    normalize = kwargs['normalize']
    spc = kwargs['spc']
    
    num_elts = len(alist)
    h = max(map(lambda arr: arr.shape[0], alist))
    w = sum(map(lambda arr: arr.shape[1], alist)) + spc*(num_elts-1)
    result = zeros((h, w) + alist[0].shape[2:])

    if result.ndim == 3:
        for i in range(3):
            result[:,:,i] = kwargs['backcolor'][i]

    if normalize:
        alist = apply(norm_group, (alist,), kwargs)
    
    start = 0
    for i, arr in enumerate(alist):
        m, n = arr.shape[0:2]
        result[:m, start:start+n] = arr
        start += n + spc

    return result

def vjoin(alist, **kwargs):
    """
    Concatenate the arrays in alist horizontally, with spacing spc between them.
    If normalize is True, the arrays are all normalized to be between 0 and 1.
    Arrays may be any number of dimensions, but all the dimensions beyond the
    second must match exactly.
    """
    if len(alist) == 0:
        return zeros((0,0))
    
    kwargs = override((_DEFAULT_JOIN_PARS, kwargs))
    normalize = kwargs['normalize']
    spc = kwargs['spc']
    
    num_elts = len(alist)
    h = sum(map(lambda arr: arr.shape[0], alist)) + spc*(num_elts-1)
    w = max(map(lambda arr: arr.shape[1], alist))
    result = zeros((h, w) + alist[0].shape[2:])

    if result.ndim == 3:
        for i in range(3):
            result[:,:,i] = kwargs['backcolor'][i]

    if normalize:
        alist = apply(norm_group, (alist,), kwargs)

    start = 0
    for i, arr in enumerate(alist):
        m, n = arr.shape[0:2]
        result[start:start+m, :n] = arr
        start += m + spc

    return result

def group(list, size, mode='all'):
    """
    Partitions list into groups of size, where the final
    group possibly contains fewer elements.

    Example:
        group(range(8), 3) ==> [[0, 1, 2], [3, 4, 5], [6, 7]]
    """

    # Force a copy of numpy array/matrix
    if type(list) in [array]:
        list = list.copy()
        
    result = []
    start = 0
    while True:
        end = min(start + size, len(list))
        result.append(list[start:end])
        if end == len(list):
            break
        start += size

    if mode=='truncate' and len(result[-1]) < size:
        result = result[:-1]
        
    return result


def pack(alist, ratio=1., **kwargs):
    """
    Pack the given arrays to get as close as possible to
    the desired ratio.
    """
    if len(alist) == 0:
        return zeros((0,0))
    
    kwargs = override((_DEFAULT_JOIN_PARS, kwargs))
    normalize = kwargs['normalize']
    spc = kwargs['spc']
    boundary = kwargs['boundary']
    
    rdim = max(map(lambda arr: arr.shape[0], alist))
    cdim = max(map(lambda arr: arr.shape[1], alist))
    narrs = len(alist)
    
    def score(guess):
        nrows = math.ceil(narrs/float(guess))
        ncols = guess
        return max(nrows*(rdim+spc)*ratio, ncols*(cdim+spc))

    if normalize:
        alist = apply(norm_group, (alist,), kwargs)

    ncols = argmin(map(score, range(1,narrs+1)))+1
    tiled = group(alist, ncols)

    temp = vjoin([hjoin(grp, spc=spc, normalize=False) for grp in tiled],
                 spc=spc, normalize=False)
    if boundary:
        temp = add_boundary(temp, spc=spc)
    return temp


def add_boundary(arr, spc=3, brightness=0):
    """
    Adds a boundary of width spc around arr containing all zeros.
    """
    m, n = arr.shape[0:2]
    result = brightness*ones((m+2*spc, n+2*spc) + arr.shape[2:])
    result[spc:spc+m, spc:spc+n] = arr
    return result

