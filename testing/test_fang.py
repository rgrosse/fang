import gnumpy as gnp
import numpy as np
nax = np.newaxis

import binary_rbms
import fang
import fisher
import generic_rbms
import rbm_training
import test_tractable
import tractable
from utils import misc

N = 10
NVIS = 5
NHID = 4

def check_statistics(num_samples=1000):
    v = gnp.garray(np.random.uniform(size=(N, NVIS)))
    h = gnp.garray(np.random.uniform(size=(N, NHID)))
    stats = fang.Statistics.from_activations(v, h)

    with misc.gnumpy_conversion_check('allow'):
        g = np.zeros((num_samples, 5))
        for i in range(num_samples):
            idx = np.random.randint(N)
            curr_v = np.random.binomial(1, v[idx, :])
            curr_h = np.random.binomial(1, h[idx, :])
            g[i, :] = np.array([curr_v[0], curr_v[1], curr_h[0], curr_h[1], curr_v[0] * curr_h[1]])

        print 'm_unary v[0]',
        misc.check_expectation(stats.m_unary[0], g[:, 0])
        print 'm_unary h[0]',
        misc.check_expectation(stats.m_unary[NVIS], g[:, 2])
        print 'S_unary v[0] v[0]',
        misc.check_expectation(stats.S_unary[0, 0], g[:, 0])
        print 'S_unary v[0] v[1]',
        misc.check_expectation(stats.S_unary[0, 1], g[:, 0] * g[:, 1])
        print 'S_unary v[0] h[0]',
        misc.check_expectation(stats.S_unary[0, NVIS], g[:, 0] * g[:, 2])
        print 'S_unary h[0] h[0]',
        misc.check_expectation(stats.S_unary[NVIS, NVIS], g[:, 2])
        print 'S_unary h[0] h[1]',
        misc.check_expectation(stats.S_unary[NVIS, NVIS+1], g[:, 2] * g[:, 3])
        print

        print 'm_pair v[0]',
        misc.check_expectation(stats.m_pair[0, 1, 0], g[:, 0])
        print 'm_pair h[1]',
        misc.check_expectation(stats.m_pair[0, 1, 1], g[:, 3])
        print 'm_pair v[0] h[1]',
        misc.check_expectation(stats.m_pair[0, 1, 2], g[:, 4])
        print 'S_pair v[0] v[0]',
        misc.check_expectation(stats.S_pair[0, 1, 0, 0], g[:, 0])
        print 'S_pair v[0] h[1]',
        misc.check_expectation(stats.S_pair[0, 1, 0, 1], g[:, 0] * g[:, 3])
        print 'S_pair v[0] vh[0, 1]',
        misc.check_expectation(stats.S_pair[0, 1, 0, 2], g[:, 0] * g[:, 4])
        print 'S_pair h[1] h[1]',
        misc.check_expectation(stats.S_pair[0, 1, 1, 1], g[:, 3])
        print 'S_pair h[1] vh[0, 1]',
        misc.check_expectation(stats.S_pair[0, 1, 1, 2], g[:, 3] * g[:, 4])
        print 'S_pair vh[0, 1] vh[0, 1]',
        misc.check_expectation(stats.S_pair[0, 1, 2, 2], g[:, 4])

def test_symmetric():
    v = gnp.garray(np.random.uniform(size=(N, NVIS)))
    h = gnp.garray(np.random.uniform(size=(N, NHID)))
    stats = fang.Statistics.from_activations(v, h)

    with misc.gnumpy_conversion_check('allow'):
        assert np.allclose(stats.S_unary, stats.S_unary.T)
        assert np.allclose(stats.S_pair, stats.S_pair.as_numpy_array().swapaxes(2, 3))


def check_against_exact():
    with misc.gnumpy_conversion_check('allow'):
        rbm = test_tractable.random_rbm(NVIS, NHID)
        G, s = tractable.exact_fisher_information(rbm, return_mean=True, batch_units=2)
        rw = fisher.RegressionWeights.from_maximum_likelihood(G, NVIS, NHID)

        G, s = gnp.garray(G), gnp.garray(s)
        S = G + np.outer(s, s)
        m_unary = s[:NVIS+NHID]
        S_unary = S[:NVIS+NHID, :NVIS+NHID]

        m_pair = gnp.zeros((NVIS, NHID, 3))
        S_pair = gnp.zeros((NVIS, NHID, 3, 3))
        for i in range(NVIS):
            for j in range(NHID):
                vis_idx = i
                hid_idx = NVIS + j
                vishid_idx = NVIS + NHID + NHID * i + j
                idxs = np.array([vis_idx, hid_idx, vishid_idx])

                m_pair[i, j, :] = s[idxs]
                S_pair[i, j, :] = S[idxs[:, nax], idxs[nax, :]]

        stats = fang.Statistics(m_unary, S_unary, m_pair, S_pair)
        beta, sigma_sq = stats.compute_regression_weights()
        assert np.allclose(beta, rw.beta)
        assert np.allclose(sigma_sq, rw.sigma_sq)

        Sigma = stats.unary_covariance()
        assert np.max(np.abs(Sigma - G[:NVIS+NHID, :NVIS+NHID])) < 1e-6
                
    
def check_generic_same():
    params = rbm_training.FANGParams.defaults('pcd')
    params.start_after = 0    # otherwise the update test is silly
    WEIGHT_DECAY = 0.001
    
    with misc.gnumpy_conversion_check('allow'):
        bin_rbm = test_tractable.random_rbm(NVIS, NHID)
        v = gnp.garray(np.random.binomial(1, 0.5, size=(N, NVIS)))
        h = gnp.garray(np.random.binomial(1, 0.5, size=(N, NHID)))
        bin_stats = fang.Statistics.from_activations(v, h)

        gen_rbm = generic_rbms.BinaryRBM(generic_rbms.BinaryLayer.Params(bin_rbm.vbias.copy()),
                                         generic_rbms.BinaryLayer.Params(bin_rbm.hbias.copy()),
                                         bin_rbm.weights.copy())
        gen_stats = fang.GenericStatistics.from_activations(generic_rbms.BinaryLayer, v,
                                                            generic_rbms.BinaryLayer, h)

        assert np.allclose(bin_stats.m_unary, gen_stats.m_unary)
        assert np.allclose(bin_stats.S_unary, gen_stats.S_unary)
        assert np.allclose(bin_stats.m_pair, gen_stats.m_pair)
        assert np.allclose(bin_stats.S_pair, gen_stats.S_pair)

        bin_beta, bin_sigma_sq = bin_stats.compute_regression_weights(params.smoothing)
        gen_beta, gen_sigma_sq = gen_stats.compute_regression_weights(params.smoothing)
        assert np.allclose(bin_beta, gen_beta)
        assert np.allclose(bin_sigma_sq, gen_sigma_sq)

        assert np.allclose(bin_stats.unary_covariance(params.smoothing), gen_stats.unary_covariance(params.smoothing))

        v2 = gnp.garray(np.random.binomial(1, 0.5, size=(N, NVIS)))
        h2 = gnp.garray(np.random.binomial(1, 0.5, size=(N, NHID)))
        
        bin_pos_moments = binary_rbms.Moments.from_activations(v, h)
        bin_neg_moments = binary_rbms.Moments.from_activations(v2, h2)
        gen_pos_moments = generic_rbms.Moments(
            generic_rbms.BinaryLayer.Moments(bin_pos_moments.expect_vis),
            generic_rbms.BinaryLayer.Moments(bin_pos_moments.expect_hid),
            bin_pos_moments.expect_prod)
        gen_neg_moments = generic_rbms.Moments(
            generic_rbms.BinaryLayer.Moments(bin_neg_moments.expect_vis),
            generic_rbms.BinaryLayer.Moments(bin_neg_moments.expect_hid),
            bin_neg_moments.expect_prod)

        lrates = binary_rbms.LearningRates(2., 2., 1.5)

        bin_updater = fang.Updater(params)
        bin_updater.Lambda = np.linalg.inv(bin_stats.unary_covariance(params.smoothing))
        bin_updater.beta, bin_updater.sigma_sq = bin_beta, bin_sigma_sq
        gen_updater = fang.GenericUpdater(params)
        gen_updater.Lambda = np.linalg.inv(gen_stats.unary_covariance(params.smoothing))
        gen_updater.beta, gen_updater.sigma_sq = gen_beta, gen_sigma_sq

        bin_updater.apply_update(bin_pos_moments, bin_neg_moments, bin_rbm, WEIGHT_DECAY, lrates)
        gen_updater.apply_update(gen_pos_moments, gen_neg_moments, gen_rbm, WEIGHT_DECAY, lrates)
        assert np.allclose(bin_rbm.vbias, gen_rbm.vis.biases)
        assert np.allclose(bin_rbm.hbias, gen_rbm.hid.biases)
        assert np.allclose(bin_rbm.weights, gen_rbm.weights)
        

        
