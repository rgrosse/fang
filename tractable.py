import itertools
import gnumpy as gnp
import numpy as np
nax = np.newaxis

import binary_rbms
from utils import misc


def combinations_array(prefix_len):
    return gnp.garray(list(itertools.product(*[[0, 1]] * prefix_len)))


def get_scores(rbm, batch_units=10, show_progress=False):
    nhid = rbm.nhid
    assert nhid <= 30
    prefix_len = nhid - batch_units
    batch_size = 2 ** batch_units
    prefixes = combinations_array(prefix_len)
    num_batches = prefixes.shape[0]

    hid = gnp.zeros((batch_size, nhid))
    hid[:, prefix_len:] = combinations_array(batch_units)
    scores = gnp.zeros((num_batches, batch_size))

    if show_progress:
        pbar = misc.pbar(num_batches)

    for i, prefix in enumerate(prefixes):
        hid[:, :prefix_len] = prefix
        scores[i, :] = rbm.free_energy_hid(hid)

        if show_progress:
            pbar.update(i)

    if show_progress:
        pbar.finish()

    return scores
    
def exact_partition_function(rbm, batch_units=10, show_progress=False):
    return np.logaddexp.reduce(np.sort(get_scores(rbm, batch_units=batch_units,
                                                  show_progress=show_progress).as_numpy_array().ravel()))

def iter_configurations(rbm, batch_units=10, show_progress=False):
    assert rbm.nhid <= 30
    scores = get_scores(rbm, batch_units=batch_units, show_progress=show_progress).as_numpy_array()
    prefix_len = rbm.nhid - batch_units
    batch_size = 2 ** batch_units
    prefixes = combinations_array(prefix_len)

    batch_scores = np.logaddexp.reduce(scores, axis=1)    
    idxs = np.argsort(batch_scores)
    prefixes = prefixes[idxs]
    scores = scores[idxs]

    hid = gnp.zeros((batch_size, rbm.nhid))
    hid[:, prefix_len:] = combinations_array(batch_units)

    pfn = np.logaddexp.reduce(np.sort(scores.ravel()))
    normalized_scores = scores - pfn
    p = np.exp(normalized_scores)

    if show_progress:
        pbar = misc.pbar(prefixes.shape[0])

    for i, prefix in enumerate(prefixes):
        hid[:, :prefix_len] = prefix
        yield hid, p[i, :]

        if show_progress:
            pbar.update(i)

    if show_progress:
        pbar.finish()

def exact_moments(rbm, batch_units=10, show_progress=False):
    expect_vis = gnp.zeros(rbm.nvis)
    expect_hid = gnp.zeros(rbm.nhid)
    expect_prod = gnp.zeros((rbm.nvis, rbm.nhid))

    for hid, p in iter_configurations(rbm, batch_units=batch_units, show_progress=show_progress):
        cond_vis = gnp.logistic(rbm.vis_inputs(hid))
        expect_vis += gnp.dot(p, cond_vis)
        expect_hid += gnp.dot(p, hid)
        expect_prod += gnp.dot(cond_vis.T * p, hid)

    return binary_rbms.Moments(expect_vis, expect_hid, expect_prod)
        

def exact_fisher_information(rbm, batch_units=10, show_progress=False, vis_shape=None, downsample=1, return_mean=False):
    batch_size = 2 ** batch_units

    if downsample == 1:
        vis_idxs = np.arange(rbm.nvis)
    else:
        temp = np.arange(rbm.nvis).reshape((28, 28))
        mask = np.zeros((28, 28), dtype=bool)
        mask[::downsample, ::downsample] = 1
        vis_idxs = temp[mask]
    nvis = vis_idxs.size
    nhid = rbm.nhid

    num_params = nvis + nhid + nvis * nhid

    E_vis = np.zeros(nvis)
    E_hid = np.zeros(nhid)
    E_vishid = np.zeros((nvis, nhid))

    E_vis_vis = np.zeros((nvis, nvis))
    E_vis_hid = np.zeros((nvis, nhid))
    E_vis_vishid = np.zeros((nvis, nvis, nhid))
    E_hid_hid = np.zeros((nhid, nhid))
    E_hid_vishid = np.zeros((nhid, nvis, nhid))
    E_vishid_vishid = np.zeros((nvis, nhid, nvis, nhid))
    

    for hid, p in iter_configurations(rbm, batch_units=batch_units, show_progress=show_progress):
        with misc.gnumpy_conversion_check('allow'):
            cond_vis = gnp.logistic(rbm.vis_inputs(hid))
            cond_vis = gnp.garray(cond_vis.as_numpy_array()[:, vis_idxs])
            vishid = (cond_vis[:, :, nax] * hid[:, nax, :]).reshape((batch_size, nvis * nhid))
            var_vis = cond_vis * (1. - cond_vis)

            E_vis += gnp.dot(p, cond_vis)
            E_hid += gnp.dot(p, hid)
            E_vishid += gnp.dot(cond_vis.T * p, hid)

            E_vis_vis += gnp.dot(cond_vis.T * p, cond_vis)
            diag_term = gnp.dot(p, cond_vis * (1. - cond_vis))
            E_vis_vis += gnp.garray(np.diag(diag_term.as_numpy_array()))

            E_vis_hid += gnp.dot(cond_vis.T * p, hid)

            E_hid_hid += gnp.dot(hid.T * p, hid)

            E_vis_vishid += gnp.dot(cond_vis.T * p, vishid).reshape((nvis, nvis, nhid))
            diag_term = gnp.dot(var_vis.T * p, hid)
            E_vis_vishid[np.arange(nvis), np.arange(nvis), :] += diag_term

            E_hid_vishid += gnp.dot(hid.T * p, vishid).reshape((nhid, nvis, nhid))

            E_vishid_vishid += gnp.dot(vishid.T * p, vishid).reshape((nvis, nhid, nvis, nhid))
            diag_term = ((cond_vis * (1. - cond_vis))[:, :, nax, nax] * hid[:, nax, :, nax] * hid[:, nax, nax, :] * p[:, nax, nax, nax]).sum(0)
            E_vishid_vishid[np.arange(nvis), :, np.arange(nvis), :] += diag_term

    G = np.zeros((num_params, num_params))
    vis_slc = slice(0, nvis)
    hid_slc = slice(nvis, nvis + nhid)
    vishid_slc = slice(nvis + nhid, None)
    G[vis_slc, vis_slc] = E_vis_vis
    G[vis_slc, hid_slc] = E_vis_hid
    G[vis_slc, vishid_slc] = E_vis_vishid.reshape((nvis, nvis * nhid))
    G[hid_slc, vis_slc] = E_vis_hid.T
    G[hid_slc, hid_slc] = E_hid_hid
    G[hid_slc, vishid_slc] = E_hid_vishid.reshape((nhid, nvis * nhid))
    G[vishid_slc, vis_slc] = E_vis_vishid.reshape((nvis, nvis * nhid)).T
    G[vishid_slc, hid_slc] = E_hid_vishid.reshape((nhid, nvis * nhid)).T
    G[vishid_slc, vishid_slc] = E_vishid_vishid.reshape((nvis * nhid, nvis * nhid))

    s = np.concatenate([E_vis, E_hid, E_vishid.ravel()])
    G -= np.outer(s, s)

    if return_mean:
        return G, s
    else:
        return G

def exact_fisher_information_biases(rbm, batch_units=10, show_progress=False):
    batch_size = 2 ** batch_units

    nvis, nhid = rbm.nvis, rbm.nhid
    num_params = nvis + nhid

    s = gnp.zeros(num_params)
    G = gnp.zeros((num_params, num_params))

    for hid, p in iter_configurations(rbm, batch_units=batch_units, show_progress=show_progress):
        g = gnp.zeros((batch_size, num_params))
        cond_vis = gnp.logistic(rbm.vis_inputs(hid))

        g[:, :nvis] = cond_vis
        g[:, nvis:] = hid

        s += gnp.dot(p, g)
        G += gnp.dot(g.T * p, g)

        diag_term = gnp.dot(p, g * (1. - g))
        G += np.diag(diag_term.as_numpy_array())

    G -= s[:, nax] * s[nax, :]

    return G


def exact_samples(rbm, num, batch_units=10, show_progress=False):
    scores = get_scores(rbm, batch_units=batch_units).as_numpy_array()
    scores -= np.logaddexp.reduce(scores.ravel())
    p = np.exp(scores)

    prefix_len = rbm.nhid - batch_units
    prefixes = combinations_array(prefix_len).as_numpy_array()
    postfixes = combinations_array(batch_units).as_numpy_array()

    p_row = p.sum(1)
    p_row /= p_row.sum()
    cond_p_col = p / p_row[:, nax]

    cond_p_col *= (1. - 1e-8)   # keep np.random.multinomial from choking because the sum is greater than 1


    vis = np.zeros((num, rbm.nvis))
    hid = np.zeros((num, rbm.nhid))

    with misc.gnumpy_conversion_check('allow'):
        rows = np.random.multinomial(1, p_row, size=num).argmax(1)
        #cols = np.random.multinomial(1, cond_p_col[rows, :]).argmax(1)
        cols = np.array([np.random.multinomial(1, cond_p_col[row, :]).argmax()
                         for row in rows])
        hid = np.hstack([prefixes[rows, :], postfixes[cols, :]])
        vis = np.random.binomial(1, gnp.logistic(rbm.vis_inputs(hid)))

    return binary_rbms.RBMState(gnp.garray(vis), gnp.garray(hid))





