import numpy as np

import ais
import binary_rbms
import experiments.moments
import tempered_transitions
import test_tractable
import tractable
from utils import misc


def test_schedule():
    MIN_BETA = 0.65
    NUM_STEPS = 1000
    
    with misc.gnumpy_conversion_check('allow'):
        schedule = tempered_transitions.Schedule(MIN_BETA)

        # check extreme values
        assert np.allclose(schedule.compute_beta(0.), 1.)
        assert np.allclose(schedule.compute_beta(1.), 1.)
        assert np.allclose(schedule.compute_beta(0.5), MIN_BETA)

        # should be symmetric
        assert np.allclose(schedule.compute_beta(0.3), schedule.compute_beta(0.7))

        # no large jumps
        betas = np.array([schedule.compute_beta(t) for t in np.linspace(0., 1., NUM_STEPS + 1)])
        diffs = betas[1:] - betas[:-1]
        max_diff = 2. * (1. - MIN_BETA) * 1.01 / NUM_STEPS
        assert np.all(np.abs(diffs) < max_diff)

def log_mean(values):
    return np.logaddexp.reduce(values) - np.log(len(values))

def check_rbm_path():
    """Use AIS to estimate the ratio of partition functions between an RBM and itself.
    Assuming a correct implementation (and reasonable convergence), the weights should
    have expectation 1. Prints a bootstrap confidence interval for the log ratio; this
    interval should contain 0."""
    NUM_SAMPLES = 10000
    NUM_STEPS = 100
    MIN_BETA = 0.65

    with misc.gnumpy_conversion_check('allow'):
        rbm = test_tractable.random_rbm()
        particles = tractable.exact_samples(rbm, NUM_SAMPLES, batch_units=2)
        schedule = tempered_transitions.Schedule(MIN_BETA)
        moments = tractable.exact_moments(rbm, batch_units=2)
        base_rbm = binary_rbms.RBM.from_moments(moments.full_base_rate_moments())
        path = tempered_transitions.RBMPath(base_rbm, rbm, particles, schedule)

        _, log_w, _ = ais.ais(path, np.linspace(0., 1., NUM_STEPS))
        lower, upper = misc.bootstrap(log_w, log_mean)

        if lower < 0. < upper:
            print 'Interval OK: [{:1.6f}, {:1.6f}]'.format(lower, upper)
        else:
            raise RuntimeError('Interval does not contain 0: [{:1.6f}, {:1.6f}]'.format(lower, upper))


def compare(val1, val2, threshold):
    if np.abs(val1 - val2) < threshold:
        return '{:1.5f}, {:1.5f} (OK)'.format(val1, val2)
    else:
        return '{:1.5f}, {:1.5f}  <----------------- uh-oh'.format(val1, val2)

def check_expectations():
    """Run a Markov chain using tempered transitions as the transition operator, and check that the
    units have approximately the right expectations. Uses the tractable PCD RBM. Accuracy threshold was
    set arbitrarily at 0.05."""

    # parameters chosen to have acceptance rate around 0.2
    NUM_PARTICLES = 100
    NUM_STEPS = 40
    NUM_TRANSITIONS = 500
    MIN_BETA = 0.3

    with misc.gnumpy_conversion_check('allow'):
        rbm = experiments.moments.load_trained_rbm('data/rbms/mnist_pcd_20_rbm.pk').convert_to_garrays()
        particles = tractable.exact_samples(rbm, NUM_PARTICLES, batch_units=10, show_progress=True)
        schedule = tempered_transitions.Schedule(MIN_BETA)
        exact_moments = tractable.exact_moments(rbm, batch_units=10, show_progress=True)
        base_rbm = binary_rbms.RBM.from_moments(exact_moments.full_base_rate_moments())
        tt_op = tempered_transitions.TTOperator(base_rbm, schedule, NUM_STEPS)

        total_vis = np.zeros(rbm.nvis)
        total_hid = np.zeros(rbm.nhid)
        total_prod = np.zeros((rbm.nvis, rbm.nhid))

        pbar = misc.pbar(NUM_TRANSITIONS)
        for i in range(NUM_TRANSITIONS):
            tt_op(rbm, particles)

            moments = rbm.cond_hid(particles.h)
            total_vis += moments.expect_vis
            total_hid += moments.expect_hid
            total_prod += moments.expect_prod

            pbar.update(i+1)
        pbar.finish()

        avg_vis = total_vis / NUM_TRANSITIONS
        avg_hid = total_hid / NUM_TRANSITIONS
        avg_prod = total_prod / NUM_TRANSITIONS

        # units which are not generally saturated:
        #   visibles: 400, 523
        #   hiddens: 4, 14

        print
        print 'Visible unit 400:', compare(avg_vis[400], exact_moments.expect_vis[400], 0.05)
        print 'Visible unit 523:', compare(avg_vis[523], exact_moments.expect_vis[523], 0.05)
        print 'Hidden unit 4:', compare(avg_hid[4], exact_moments.expect_hid[4], 0.05)
        print 'Hidden unit 14:', compare(avg_hid[14], exact_moments.expect_hid[14], 0.05)
        print 'Product 400, 4:', compare(avg_prod[400, 4], exact_moments.expect_prod[400, 4], 0.05)
        print 'Product 523, 14:', compare(avg_prod[523, 14], exact_moments.expect_prod[523, 14], 0.05)

