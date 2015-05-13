import gnumpy as gnp
import numpy as np

import binary_rbms
import datasets
import generic_rbms
from utils import misc, storage
from visuals import rbm_vis


M = 4
N = 5
R = 20

def random_binary_state(m=M, n=N, r=R):
    R = 2 * (M + N)
    v = gnp.garray(np.random.binomial(1, 0.5, size=(R, M)))
    h = gnp.garray(np.random.binomial(1, 0.5, size=(R, N)))
    return v, h

def random_binary_rbms(m=M, n=N):
    vbias = gnp.garray(np.random.normal(size=m))
    hbias = gnp.garray(np.random.normal(size=n))
    weights = gnp.garray(np.random.normal(size=(m, n)))
    bin_rbm = binary_rbms.RBM(vbias.copy(), hbias.copy(), weights.copy())
    gen_rbm = generic_rbms.BinaryRBM(generic_rbms.BinaryLayer.Params(vbias),
                                     generic_rbms.BinaryLayer.Params(hbias),
                                     weights)
    return bin_rbm, gen_rbm

def check_rbms_same(bin_rbm, gen_rbm):
    with misc.gnumpy_conversion_check('allow'):
        assert np.allclose(bin_rbm.vbias, gen_rbm.vis.biases)
        assert np.allclose(bin_rbm.hbias, gen_rbm.hid.biases)
        assert np.allclose(bin_rbm.weights, gen_rbm.weights)

def random_binary_moments(m=M, n=N):
    v, h = random_binary_state()
    bin_moments = binary_rbms.Moments.from_activations(v, h)
    gen_moments = generic_rbms.BinaryMoments.from_activations(v, h)
    return bin_moments, gen_moments

def check_moments_same(bin_moments, gen_moments):
    with misc.gnumpy_conversion_check('allow'):
        assert np.allclose(bin_moments.expect_vis, gen_moments.vis.expect)
        assert np.allclose(bin_moments.expect_hid, gen_moments.hid.expect)
        assert np.allclose(bin_moments.expect_prod, gen_moments.expect_prod)

def test_from_activations_same():
    """Check that Moments.from_activations is consistent with binary_rbms module"""
    v, h = random_binary_state()
    bin_moments = binary_rbms.Moments.from_activations(v, h)
    gen_moments = generic_rbms.BinaryMoments.from_activations(v, h)
    check_moments_same(bin_moments, gen_moments)
    

def test_from_independent_same():
    """Check that Moments.from_independent is consistent with binary_rbms module"""
    expect_vis = gnp.garray(np.random.uniform(size=M))
    expect_hid = gnp.garray(np.random.uniform(size=N))
    bin_moments = binary_rbms.Moments.from_independent(expect_vis, expect_hid)
    gen_moments = generic_rbms.BinaryMoments.from_independent(
        generic_rbms.BinaryLayer.Moments(expect_vis),
        generic_rbms.BinaryLayer.Moments(expect_hid))
    check_moments_same(bin_moments, gen_moments)

def test_from_data_base_rates_same():
    """Check that Moments.from_data_base_rates is consistent with binary_rbms module"""
    v, h = random_binary_state()
    bin_moments = binary_rbms.Moments.from_data_base_rates(v, N)
    gen_moments = generic_rbms.BinaryMoments.from_data_base_rates(v, N)
    check_moments_same(bin_moments, gen_moments)

def test_full_base_rate_same():
    """Check that Moments.full_base_rate_moments is consistent with binary_rbms module"""
    bin_moments, gen_moments = random_binary_moments()
    check_moments_same(bin_moments.full_base_rate_moments(), gen_moments.full_base_rate_moments())

def test_smooth_same():
    """Check that Moments.smooth is consistent with binary_rbms module"""
    bin_moments, gen_moments = random_binary_moments()
    check_moments_same(bin_moments.smooth(0.1), gen_moments.smooth(0.1))

def test_from_moments_same():
    """Check that RBM.from_moments is consistent with binary_rbms module"""
    bin_moments, gen_moments = random_binary_moments()
    bin_rbm = binary_rbms.RBM.from_moments(bin_moments.full_base_rate_moments(), weights_std=0.)
    gen_rbm = generic_rbms.BinaryRBM.from_moments(gen_moments.full_base_rate_moments(), weights_std=0.)
    check_rbms_same(bin_rbm, gen_rbm)

def test_up_down_same():
    """Check that RBM.up and RBM.down are consistent with binary_rbms module"""
    bin_rbm, gen_rbm = random_binary_rbms()
    v, h = random_binary_state()

    with misc.gnumpy_conversion_check('allow'):
        assert np.allclose(bin_rbm.hid_inputs(v), gen_rbm.up(v) + gen_rbm.hid.biases)
        assert np.allclose(bin_rbm.vis_inputs(h), gen_rbm.down(h) + gen_rbm.vis.biases)

def test_cond_same():
    """Check that RBM.cond_vis and RBM.cond_hid are consistent with binary_rbms module"""
    bin_rbm, gen_rbm = random_binary_rbms()
    v, h = random_binary_state()
    check_moments_same(bin_rbm.cond_vis(v), gen_rbm.cond_vis(v))
    check_moments_same(bin_rbm.cond_hid(h), gen_rbm.cond_hid(h))

def test_energy_same():
    """Check that RBM.energy and RBM.free_energy_* are consistent with binary_rbms module"""
    R = 2 * (M + N)
    v = np.random.binomial(1, 0.5, size=(R, M))
    h = np.random.binomial(1, 0.5, size=(R, N))
    bin_rbm, gen_rbm = random_binary_rbms()

    with misc.gnumpy_conversion_check('allow'):
        assert np.allclose(bin_rbm.energy(v, h), gen_rbm.energy(v, h))
        assert np.allclose(bin_rbm.free_energy_vis(v), gen_rbm.free_energy_vis(v))
        assert np.allclose(bin_rbm.free_energy_hid(h), gen_rbm.free_energy_hid(h))

def test_init_partition_function_same():
    """Check that RBM.init_partition_function is consistent with binary_rbms module"""
    bin_moments, gen_moments = random_binary_moments()
    bin_rbm = binary_rbms.RBM.from_moments(bin_moments.full_base_rate_moments(), weights_std=0.)
    gen_rbm = generic_rbms.BinaryRBM.from_moments(gen_moments.full_base_rate_moments(), weights_std=0.)
    with misc.gnumpy_conversion_check('allow'):
        assert np.allclose(bin_rbm.init_partition_function(), gen_rbm.init_partition_function())

def test_dot_product_same():
    """Check that RBM.dot_product is consistent with binary_rbms module"""
    bin_moments, gen_moments = random_binary_moments()
    bin_rbm, gen_rbm = random_binary_rbms()
    with misc.gnumpy_conversion_check('allow'):
        assert np.allclose(bin_rbm.dot_product(bin_moments), gen_rbm.dot_product(gen_moments))


def test_sgd_update_same():
    """Check that RBM.sgd_update is consistent with binary_rbms module"""
    bin_rbm, gen_rbm = random_binary_rbms()
    bin_pos_moments, gen_pos_moments = random_binary_moments()
    bin_neg_moments, gen_neg_moments = random_binary_moments()
    lrates = binary_rbms.LearningRates(0.5, 0.7, 0.9)
    bin_rbm += lrates * (binary_rbms.Update.from_moments(bin_pos_moments) - binary_rbms.Update.from_moments(bin_neg_moments))
    gen_rbm.sgd_update(gen_pos_moments, gen_neg_moments, lrates)
    check_rbms_same(bin_rbm, gen_rbm)

    

def check_gibbs_sampler(num_steps=1000):
    """Check that the Gibbs sampler produces good samples"""
    bin_rbm = storage.load('data/rbms/mnist_pcd_500_rbm.pk').convert_to_garrays()
    gen_rbm = generic_rbms.BinaryRBM(
        generic_rbms.BinaryLayer.Params(bin_rbm.vbias),
        generic_rbms.BinaryLayer.Params(bin_rbm.hbias),
        bin_rbm.weights)
    data_info = datasets.mnist.MNISTInfo()

    state = bin_rbm.random_state(100)
    for i in range(num_steps):
        state = bin_rbm.step(state)
    rbm_vis.show_particles(bin_rbm, state, data_info, figname='bin', figtitle='bin')

    state = gen_rbm.random_state(100)
    for i in range(num_steps):
        state = gen_rbm.step(state)
    rbm_vis.show_particles(gen_rbm, state, data_info, figname='gen', figtitle='gen')


    



