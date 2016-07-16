import math
import numpy as np

import ais
import binary_rbms
import centering
import fang
import tractable
from utils.schemas import Params, Choice
from utils import misc

HID_AVG_TIME = 100


class LRateParams(Params):
    class Fields:
        base_lrates = object                    # learning rates at time 0 (should be a binary_rbms.LRates class)
        decay_type = Choice(['fixed', 'linear', 'sqrt'])    # method of learning rate decay

    class Defaults:
        pcd = {'base_lrates': binary_rbms.LearningRates(0.1, 0.1, 0.1), 'decay_type': 'sqrt'}


class PosDataParams(Params):
    """Parameters for computing data-conditional statistics."""
    class Fields:
        mbsize = int                              # minibatch size

class PosMomentsParams(Params):
    """Obtaining the positive statistics from the target moments."""
    pass

class VarianceReductionParams(Params):
    class Fields:
        mbsize = int
        recompute_every = int

    class Defaults:
        default = {'recompute_every': 100, 'mbsize': 400}

class CDParams(Params):
    """Parameters for computing negative statistics with contrastive divergence."""
    class Fields:
        num_steps = int                           # number of CD steps
        condition_on = Choice(['vis', 'hid'])     # whether to condition on visibles or hiddens (should use 'hid')

class PCDParams(Params):
    class Fields:
        num_steps = int                           # number of PCD steps per update
        num_particles = int                       # number of PCD particles
        condition_on = Choice(['vis', 'hid'])     # whether to condition on visibles or hiddens (should use 'hid')

class AISParams(Params):
    """Compute negative statistics using AIS (experimental)"""
    class Fields:
        num_steps = int                           # number of AIS steps
        num_particles = int                       # number of AIS particles

class ExactNegParams(Params):
    """Compute exact model statistics for negative statistics."""
    class Fields:
        num_samples = int                         # number of samples to use for visualization


class SGDParams(Params):
    """Stochastic gradient descent updates."""
    class Fields:
        pass

    class Defaults:
        pass

class CenteringTrickAvgParams(Params):
    """Centering trick updates."""
    class Fields:
        timescale = float              # timescale for averaging the expectations

    class Defaults:
        default = {'timescale': 100.}

class FANGParams(Params):
    """Parameters for FANG."""
    class Fields:
        smoothing = float              # smoothing parameter for variances
        timescale = float              # timescale for averaging the expectations and covariances
        recompute_every = int          # recompute the update after this many iterations
        update_stats_every = int       # update covariances after this many iterations
        start_after = int              # use SGD until this many iterations
        init_from_data = bool          # initialize covariances from data-conditional distribution

    class Defaults:
        pcd = {'smoothing': 0.01, 'timescale': 100., 'recompute_every': 10, 'start_after': 0,
               'init_from_data': False, 'update_stats_every': 1}

class TrainingParams(Params):
    """Parameters for training an RBM from scratch."""
    class Fields:
        pos = Choice([PosDataParams, PosMomentsParams,
                      VarianceReductionParams])                 # how to compute positive statistics
        neg = Choice([CDParams, PCDParams, ExactNegParams])     # how to compute negative statistics
        num_steps = int                                         # total number of updates
        lrates = LRateParams                                    # learning rates
        updater = Choice([SGDParams, FANGParams, CenteringTrickAvgParams])    # update rule (e.g. SGD)
        momentum = float                                        # momentum parameter
        weight_decay = float                                    # weight decay
        timescale = float                                       # timescale for computing averaged RBMs

    class Defaults:
        # reasonable defaults for PCD training
        pcd = {'updater': SGDParams(), 'lrates': LRateParams.defaults('pcd'), 'momentum': 0., 'pos': PosDataParams(mbsize=400),
               'neg': PCDParams(num_steps=1, num_particles=400, condition_on='hid'), 'weight_decay': 0.002, 'num_steps': 100000,
               'timescale': 100.}



def get_lrates(lrate_params, step, time_const=1000):
    """Compute the learning rate at a given time step."""
    if lrate_params.decay_type == 'fixed':
        return lrate_params.base_lrates
    elif lrate_params.decay_type == 'linear':
        return lrate_params.base_lrates * float(time_const) / (step + time_const)
    elif lrate_params.decay_type == 'sqrt':
        return lrate_params.base_lrates * math.sqrt(float(time_const) / (step + time_const))
    else:
        raise RuntimeError('Unknown decay_type: {}'.format(lrate_params.decay_type))



class SGDUpdater:
    """Stochastic gradient descent update rule."""
    def apply_update(self, pos_moments, neg_moments, rbm, weight_decay, lrates):
        rbm.sgd_update(pos_moments, neg_moments, lrates, weight_decay)



class Trainer:
    """Training procedure for RBMs."""
    def __init__(self, rbm, params, data_matrix=None, moments=None, updater=None, fantasy_particles=None):
        self.rbm = rbm
        self.avg_rbm = rbm.copy()
        self.params = params
        self.data_matrix = data_matrix
        self.moments = moments
        if updater is None:
            updater = SGDUpdater()
        self.updater = updater

        self.nvis = rbm.nvis
        self.nhid = rbm.nhid
        if isinstance(params.neg, PCDParams):
            if fantasy_particles is None:
                self.fantasy_particles = rbm.random_state(params.neg.num_particles)
            else:
                self.fantasy_particles = fantasy_particles
        self.count = 0

    def step(self):
        """Update the PCD particles, or visualizations of samples."""
        if isinstance(self.params.neg, PCDParams):
            for i in range(self.params.neg.num_steps):
                self.fantasy_particles = self.rbm.step(self.fantasy_particles)
        if isinstance(self.params.neg, ExactNegParams) and self.params.neg.num_samples > 0:
            self.fantasy_particles = tractable.exact_samples(self.rbm, self.params.neg.num_samples)
        if isinstance(self.params.neg, AISParams) and hasattr(self, 'ais_state'):
            self.fantasy_particles = self.ais_state

    def pos_moments(self):
        """Compute the positive statistics."""
        if isinstance(self.params.pos, PosDataParams):
            num_blocks = self.data_matrix.shape[0] // self.params.pos.mbsize
            start = (self.count % num_blocks) * self.params.pos.mbsize
            end = start + self.params.pos.mbsize
            self.curr_inputs = self.data_matrix[start:end, :]   # save for negative update
            return self.rbm.cond_vis(self.curr_inputs)
        elif isinstance(self.params.pos, PosMomentsParams):
            return self.moments
        elif isinstance(self.params.pos, VarianceReductionParams):
            if self.count % self.params.pos.recompute_every == 0:
                self.batch_rbm = self.rbm.copy()
                self.batch_moments = self.rbm.cond_vis(self.data_matrix)

            num_blocks = self.data_matrix.shape[0] // self.params.pos.mbsize
            start = (self.count % num_blocks) * self.params.pos.mbsize
            end = start + self.params.pos.mbsize
            self.curr_inputs = self.data_matrix[start:end, :]   # save for negative update
            correction = self.rbm.cond_vis(self.curr_inputs) - self.batch_rbm.cond_vis(self.curr_inputs)

            return self.batch_moments + correction
        else:
            raise RuntimeError('Unknown positive update: %s' % self.params.pos)

    def neg_moments(self, particles=None):
        """Compute the negative statistics."""
        if isinstance(self.params.neg, CDParams):
            assert isinstance(self.params.pos, PosDataParams)
            state = self.rbm.sample_state(self.curr_inputs)
            for i in range(self.params.neg.num_steps):
                state = self.rbm.step(state)
            if self.params.neg.condition_on == 'hid':
                return self.rbm.cond_hid(state.hid)
            else:
                return self.rbm.cond_vis(state.vis)
        elif isinstance(self.params.neg, PCDParams):
            if particles is None:
                particles = self.fantasy_particles
            if self.params.neg.condition_on == 'hid':
                return self.rbm.cond_hid(particles.h)
            else:
                return self.rbm.cond_vis(particles.v)
        elif isinstance(self.params.neg, ExactNegParams):
            return tractable.exact_moments(self.rbm)
        elif isinstance(self.params.neg, AISParams):
            brm = self.moments.full_base_rate_moments()
            init_rbm = binary_rbms.RBM.from_moments(brm)
            seq = ais.GeometricRBMPath(init_rbm, self.rbm)
            path = ais.RBMDistributionSequence(seq, self.params.neg.num_particles, 'h')
            schedule = np.linspace(0., 1., self.params.neg.num_steps)
            state, _, _ = ais.ais(path, schedule, show_progress=True)
            self.ais_state = state
            return self.rbm.cond_hid(state.h)
        else:
            raise RuntimeError('Unknown negative update: %s' % self.neg_type)

    def update(self, particles=None):
        """Update the RBM parameters."""
        pos_moments = self.pos_moments()
        neg_moments = self.neg_moments(particles=particles)

        lrates = get_lrates(self.params.lrates, self.count)

        if hasattr(self.updater, 'recompute'):
            self.updater.recompute(self.rbm, self.fantasy_particles)

        self.updater.apply_update(pos_moments, neg_moments, self.rbm, self.params.weight_decay, lrates)

        lam = 1. / self.params.timescale
        self.avg_rbm = (1. - lam) * self.avg_rbm + lam * self.rbm

        self.count += 1

def get_updater(params, rbm, vis):
    """Wrapper for the update rules."""
    if isinstance(params, SGDParams):
        return None
    elif isinstance(params, CenteringTrickAvgParams):
        return centering.CenteringTrickAvgUpdater.from_data(rbm, vis, params)
    elif isinstance(params, FANGParams):
        if isinstance(rbm, binary_rbms.RBM):
            if params.init_from_data:
                return fang.Updater.from_data(rbm, vis, params)
            else:
                return fang.Updater(params)
        else:
            return fang.GenericUpdater(params)
    else:
        raise RuntimeError('Unknown updater params')


def train_rbm(vis, nhid, params, init_rbm=None, after_step=None, updater=None, rbm_class=None, moments_class=None,
              weights_std=0.05, show_progress=False, trainer_class=None):
    """Train an RBM from scratch."""
    assert isinstance(params, TrainingParams)

    if init_rbm is not None:
        rbm = init_rbm.copy()
    else:
        if rbm_class is None:
            rbm_class = binary_rbms.RBM
        if moments_class is None:
            moments_class = binary_rbms.Moments
            
        base_rate_moments = moments_class.from_data_base_rates(vis, nhid)
        rbm = rbm_class.from_moments(base_rate_moments, weights_std=weights_std)

    if updater is None:
        updater = get_updater(params.updater, rbm, vis)

    if trainer_class is None:
        trainer = Trainer(rbm, params, data_matrix=vis, updater=updater)
    else:
        trainer = trainer_class(rbm, params, data_matrix=vis, updater=updater)

    if show_progress:
        pbar = misc.pbar(params.num_steps)

    for i in range(params.num_steps):
        trainer.step()
        trainer.update()

        if after_step is not None:
            after_step(rbm, trainer, i)

        if show_progress:
            pbar.update(i+1)
    if show_progress:
        pbar.finish()

    return rbm, trainer.fantasy_particles





