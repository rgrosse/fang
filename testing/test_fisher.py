import numpy as np
nax = np.newaxis

import fisher
from utils import misc

D = 5


def check_gaussian_kldiv(num_samples=1000):
    """Check the Gaussian KL divergence computation by sampling."""
    Sigma_p = misc.random_psd(D)
    Sigma_q = misc.random_psd(D)
    ans = fisher.gaussian_kldiv(Sigma_p, Sigma_q)

    samples = []
    for i in range(num_samples):
        x = np.random.multivariate_normal(np.zeros(D), Sigma_p)
        samples.append(misc.gauss_loglik(x, np.zeros(D), Sigma_p) - misc.gauss_loglik(x, np.zeros(D), Sigma_q))
    misc.check_expectation(ans, samples)

    
def test_kldiv_same():
    """Check that the KL divergence computations in covariance and precision form give
    identical results."""
    Sigma_p = misc.random_psd(D)
    Sigma_q = misc.random_psd(D)
    ans1 = fisher.gaussian_kldiv(Sigma_p, Sigma_q)
    ans2 = fisher.gaussian_kldiv_info(np.linalg.inv(Sigma_p), np.linalg.inv(Sigma_q))
    assert np.allclose(ans1, ans2)


def test_regression_weights():
    """Check that two different ways of computing the precision give the same results."""
    NVIS, NHID = 3, 4
    G = misc.random_psd(NVIS + NHID + NVIS * NHID)

    rw = fisher.RegressionWeights.from_maximum_likelihood(G, NVIS, NHID)
    Lambda1 = rw.compute_Lambda()
    Lambda1[:NVIS+NHID, :NVIS+NHID] += np.linalg.inv(G[:NVIS+NHID, :NVIS+NHID])

    partial = fisher.PartialFisherInformation.from_full(G, NVIS, NHID)
    inverse = partial.compute_compact_precision()
    Lambda2 = inverse.to_full()

    assert np.allclose(Lambda1, Lambda2)


def test_precision_covariance_conversion():
    """Check that converting PartialFisherInverse to covariance form and back again yields
    the original."""
    NVIS, NHID = 3, 4
    inverse = fisher.PartialFisherInverse.random(NVIS, NHID)
    G = np.linalg.inv(inverse.to_full())
    inverse2 = fisher.PartialFisherInformation.from_full(G, NVIS, NHID).compute_compact_precision()
    assert np.allclose(inverse.Lambda_v_h, inverse2.Lambda_v_h)
    assert np.allclose(inverse.Lambda_vh_cond, inverse2.Lambda_vh_cond)

def test_precision_covariance_conversion_rc():
    """Check that converting RandomConnectivityInverse to covariance form and back again yields
    the original."""
    NVIS, NHID = 3, 4
    inverse = fisher.RandomConnectivityInverse.random(NVIS, NHID)
    G = np.linalg.inv(inverse.to_full())
    inverse2 = fisher.RandomConnectivityInverse.compute_from_G(G, NVIS, NHID, inverse.vis_idxs, inverse.hid_idxs)
    assert np.allclose(inverse.Lambda_v_h, inverse2.Lambda_v_h)
    assert np.allclose(inverse.Lambda_vh_cond, inverse2.Lambda_vh_cond)
    assert np.all(inverse.vis_idxs == inverse2.vis_idxs)
    assert np.all(inverse.hid_idxs == inverse2.hid_idxs)
    
def test_exact_rc_consistent():
    """Check that the RandomConnectivityInverse, when assigned the matching indices, gives
    the same results as the original graphical model computations."""
    NVIS, NHID = 3, 4
    G = misc.random_psd(NVIS + NHID + NVIS * NHID)
    inverse = fisher.PartialFisherInformation.from_full(G, NVIS, NHID).compute_compact_precision()
    
    vis_idxs = np.zeros((NVIS, NHID), dtype=int)
    vis_idxs[:] = np.arange(NVIS)[:, nax]
    hid_idxs = np.zeros((NVIS, NHID), dtype=int)
    hid_idxs[:] = np.arange(NHID)[nax, :]
    rc_inverse = fisher.RandomConnectivityInverse.compute_from_G(G, NVIS, NHID, vis_idxs, hid_idxs)

    assert np.allclose(inverse.Lambda_v_h, rc_inverse.Lambda_v_h)
    assert np.allclose(inverse.Lambda_vh_cond, rc_inverse.Lambda_vh_cond)

def check_uncorrelated_residuals(X, y, beta):
    """Check that each of the regressors is uncorrelated with the residuals."""
    resid = np.dot(X, beta) - y
    X = X - X.mean(0)
    resid = resid - resid.mean()
    print 'Checking beta...'
    misc.check_expectation(0., X[:, 0] * resid)
    misc.check_expectation(0., X[:, 1] * resid)

def check_calibration(X, y, beta, sigma_sq):
    """Check that the error variance is estimated correctly."""
    err = (np.dot(X, beta) - y) ** 2
    print 'Checking sigma_sq...'
    misc.check_expectation(sigma_sq, err)


def check_ml_regression_weights():
    """Check that the maximum likelihood regression weights and noise variance are
    estimated correctly."""
    NVIS, NHID = 3, 4
    VIS_IDX, HID_IDX = 1, 2
    NUM_SAMPLES = 1000000
    
    G = misc.random_psd(NVIS + NHID + NVIS * NHID)
    rw = fisher.RegressionWeights.from_maximum_likelihood(G, NVIS, NHID)

    idxs = np.array([VIS_IDX, NVIS + HID_IDX, NVIS + NHID + NHID * VIS_IDX + HID_IDX])
    G_block = G[idxs[:, nax], idxs[nax, :]]
    samples = np.random.multivariate_normal(np.zeros(3), G_block, size=NUM_SAMPLES)
    X, y = samples[:, :2], samples[:, 2]

    beta, sigma_sq = rw.beta[VIS_IDX, HID_IDX, :], rw.sigma_sq[VIS_IDX, HID_IDX]
    check_uncorrelated_residuals(X, y, beta)
    check_calibration(X, y, beta, sigma_sq)

def check_centering_trick_weights():
    """Check the correctness of the regression weights estimated using the centering
    trick. First, beta should be optimal for predicting vh from v and h, when v and h
    are sampled independently. Second, sigma_sq should reflect the true residual variance
    when v, h, and m are sampled from the true covariance G."""
    NVIS, NHID = 3, 4
    VIS_IDX, HID_IDX = 1, 2
    NUM_SAMPLES = 1000000
    
    G = misc.random_psd(NVIS + NHID + NVIS * NHID)
    expect_vis = np.random.uniform(0., 1., size=NVIS)
    expect_hid = np.random.uniform(0., 1., size=NHID)
    s = np.concatenate([expect_vis, expect_hid])
    rw = fisher.RegressionWeights.from_centering_trick(G, s, NVIS, NHID)
    beta, sigma_sq = rw.beta[VIS_IDX, HID_IDX, :], rw.sigma_sq[VIS_IDX, HID_IDX]

    # regression weights should be optimal given the independence assumption
    v = np.random.binomial(1, expect_vis[VIS_IDX], size=NUM_SAMPLES)
    h = np.random.binomial(1, expect_hid[HID_IDX], size=NUM_SAMPLES)
    vh = v * h
    X = np.array([v, h]).T
    check_uncorrelated_residuals(X, vh, beta)

    # check calibration of noise variance
    idxs = np.array([VIS_IDX, NVIS + HID_IDX, NVIS + NHID + NHID * VIS_IDX + HID_IDX])
    G_block = G[idxs[:, nax], idxs[nax, :]]
    samples = np.random.multivariate_normal(np.zeros(3), G_block, size=NUM_SAMPLES)
    X, y = samples[:, :2], samples[:, 2]
    check_calibration(X, y, beta, sigma_sq)
    
