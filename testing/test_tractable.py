import gnumpy as gnp
import itertools
import numpy as np
nax = np.newaxis

import binary_rbms
import tractable
from utils import misc

NVIS = 4
NHID = 6
BATCH_UNITS = 2

def random_rbm(nvis=NVIS, nhid=NHID):
    return binary_rbms.RBM(gnp.garray(np.random.normal(size=nvis)),
                           gnp.garray(np.random.normal(size=nhid)),
                           gnp.garray(np.random.normal(size=(nvis, nhid))))

def random_base_rate_rbm(nvis=NVIS, nhid=NHID):
    return binary_rbms.RBM(gnp.garray(np.random.normal(size=nvis)),
                           gnp.garray(np.random.normal(size=nhid)),
                           gnp.zeros((nvis, nhid)))


def check_get_scores():
    with misc.gnumpy_conversion_check('allow'):
        rbm = random_rbm()
        scores = tractable.get_scores(rbm, batch_units=BATCH_UNITS)

        prefixes = tractable.combinations_array(NHID - BATCH_UNITS)
        suffixes = tractable.combinations_array(BATCH_UNITS)

        for i, pre in enumerate(prefixes):
            for j, suff in enumerate(suffixes):
                hid = np.concatenate([pre, suff])
                assert np.allclose(scores[i, j], rbm.free_energy_hid(hid[nax, :])[0])


def check_partition_function():
    with misc.gnumpy_conversion_check('allow'):
        rbm = random_rbm()
        total = -np.infty
        
        for vis_ in itertools.product(*[[0, 1]] * NVIS):
            vis = gnp.garray(vis_)
            for hid_ in itertools.product(*[[0, 1]] * NHID):
                hid = gnp.garray(hid_)
                total = np.logaddexp(total, rbm.energy(vis[nax, :], hid[nax, :])[0])

        assert np.allclose(tractable.exact_partition_function(rbm, batch_units=BATCH_UNITS), total)
        
def check_moments():
    with misc.gnumpy_conversion_check('allow'):
        rbm = random_rbm()
        pfn = tractable.exact_partition_function(rbm, batch_units=BATCH_UNITS)
        
        expect_vis = gnp.zeros(rbm.nvis)
        expect_hid = gnp.zeros(rbm.nhid)
        expect_prod = gnp.zeros((rbm.nvis, rbm.nhid))

        for hid_ in itertools.product(*[[0, 1]] * NHID):
            hid = gnp.garray(hid_)
            cond_vis = rbm.vis_expectations(hid)
            p = np.exp(rbm.free_energy_hid(hid[nax, :])[0] - pfn)

            expect_vis += p * cond_vis
            expect_hid += p * hid
            expect_prod += p * gnp.outer(cond_vis, hid)

        moments = tractable.exact_moments(rbm, batch_units=BATCH_UNITS)
        assert np.allclose(expect_vis, moments.expect_vis)
        assert np.allclose(expect_hid, moments.expect_hid)
        assert np.allclose(expect_prod, moments.expect_prod)



def assert_close(A, B, msg=''):
    if np.isscalar(A) and np.isscalar(B) and msg == '':
        assert np.max(np.abs(A - B)) < 1e-6, 'A = {}, B = {}'.format(A, B)
    else:
        assert np.max(np.abs(A - B)) < 1e-6, msg

def check_fisher_information_biases_indep():
    """Fisher information should agree with analytic solution for base rate RBM."""
    with misc.gnumpy_conversion_check('allow'):
        rbm = random_base_rate_rbm()
        E_v = gnp.logistic(rbm.vbias)
        E_h = gnp.logistic(rbm.hbias)

        G = tractable.exact_fisher_information_biases(rbm, batch_units=BATCH_UNITS)
        assert_close(G, G.T, 'G not symmetric')

        G_vis_vis = G[:NVIS, :NVIS]
        G_vis_hid = G[:NVIS, NVIS:]
        G_hid_hid = G[NVIS:, NVIS:]

        assert_close(G_vis_vis[0, 0], E_v[0] * (1. - E_v[0]))
        assert_close(G_vis_vis[0, 1], 0.)
        assert_close(G_vis_hid[0, 0], 0.)
        assert_close(G_hid_hid[0, 0], E_h[0] * (1. - E_h[0]))
        assert_close(G_hid_hid[0, 1], 0.)

def check_fisher_information_consistent():
    """The top left block of exact_fisher_information should agree with exact_fisher_information_biases."""
    with misc.gnumpy_conversion_check('allow'):
        rbm = random_rbm()
        G_bias = tractable.exact_fisher_information_biases(rbm, batch_units=BATCH_UNITS)
        G = tractable.exact_fisher_information(rbm, batch_units=BATCH_UNITS)
        assert_close(G_bias, G[:NVIS+NHID, :NVIS+NHID])


def check_fisher_information_indep():
    """Fisher information should agree with analytic solution for base rate RBM."""
    with misc.gnumpy_conversion_check('allow'):
        rbm = random_base_rate_rbm()
        E_v = gnp.logistic(rbm.vbias)
        E_h = gnp.logistic(rbm.hbias)

        G = tractable.exact_fisher_information(rbm, batch_units=BATCH_UNITS)
        assert_close(G, G.T, 'G not symmetric')

        G_vis_vishid = G[:NVIS, NVIS+NHID:].reshape((NVIS, NVIS, NHID))
        G_hid_vishid = G[NVIS:NVIS+NHID, NVIS+NHID:].reshape((NHID, NVIS, NHID))
        G_vishid_vishid = G[NVIS+NHID:, NVIS+NHID:].reshape((NVIS, NHID, NVIS, NHID))

        assert_close(G_vis_vishid[0, 0, 1], E_v[0] * (1. - E_v[0]) * E_h[1])
        assert_close(G_vis_vishid[0, 1, 2], 0.)
        assert_close(G_hid_vishid[0, 1, 0], E_h[0] * (1. - E_h[0]) * E_v[1])
        assert_close(G_hid_vishid[0, 1, 2], 0.)
        assert_close(G_vishid_vishid[0, 1, 0, 1], E_v[0] * E_h[1] * (1. - E_v[0] * E_h[1]))
        assert_close(G_vishid_vishid[0, 1, 0, 2], E_v[0] * (1. - E_v[0]) * E_h[1] * E_h[2])
        assert_close(G_vishid_vishid[0, 2, 1, 2], E_h[2] * (1. - E_h[2]) * E_v[0] * E_v[1])
        assert_close(G_vishid_vishid[0, 1, 2, 3], 0.)
        



        
