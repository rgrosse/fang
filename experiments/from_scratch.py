if __name__ == '__main__':
    import matplotlib
    matplotlib.use('agg')

import copy
import mkl
import gnumpy as gnp
import numpy as np
import os
import pylab
import re
import sys
import time

import binary_rbms
import config
import datasets
import diagnostics
import moments as expt_moments
import rbm_training
from utils.schemas import Params, List, Choice
from utils import misc, storage
from visuals import rbm_vis



class Expt(Params):
    class Fields:
        name = str
        dataset = datasets.DatasetInfo          # dataset
        training = rbm_training.TrainingParams           # training parameters
        nhid = int                              # number of hidden units
        permute = bool                          # whether to permute training examples
        
        # initialization scheme
        #   - 'base_rates': initialize biases to log-odds of base rate moments, weights
        #        to zero (no need to break symmetries since training is convex)
        #   - TrainedRBM: previously trained RBM
        init_rbm = Choice(['base_rates', expt_moments.TrainedRBM])
        
        save_after = List(int)                 # save visualizations after each of these iterations
        show_after = List(int)                 # display visualizations after each of these iterations

        # which diagnostics to display
        #    - 'particles': PCD particles
        #    - 'gibbs_chains': Gibbs chains starting from the PCD particles (to diagnose mixing)
        #    - 'objective': estimate (training) log probs and moment matching objective using AIS
        diagnostics = List(Choice(['particles', 'gibbs_chains', 'objective']))

        # whether to save particles (this can use up lots of space)
        save_particles = bool

    def outputs_dir(self):
        """Directory containing machine-readable outputs"""
        return os.path.join(config.OUTPUTS_DIR, 'from_scratch', self.name)

    def figures_dir(self):
        """Directory containing human-readable outputs"""
        return os.path.join(config.FIGURES_DIR, 'from_scratch', self.name)

    def rbm_file(self, it):
        """The main RBM after a certain number of iterations"""
        return os.path.join(self.outputs_dir(), 'rbm_{}.pk'.format(it))

    def avg_rbm_file(self, it):
        """An RBM computed as a geometrically decaying average of the main RBMs"""
        return os.path.join(self.outputs_dir(), 'avg_rbm_{}.pk'.format(it))

    def pcd_particles_file(self, it):
        """Pickle of the PCD particles"""
        return os.path.join(self.outputs_dir(), 'particles_{}.pk'.format(it))

    def pcd_particles_figure_file(self, it):
        """Image of the PCD particles"""
        return os.path.join(self.figures_dir(), 'particles_{}.png'.format(it))

    def gibbs_chains_figure_file(self, it):
        """Image of the Gibbs chains starting from the PCD particles"""
        return os.path.join(self.figures_dir(), 'gibbs_chains_{}.png'.format(it))

    def time_file(self, it):
        """Running time up to iteration it"""
        return os.path.join(self.outputs_dir(), 'time_{}.pk'.format(it))
    
DEFAULT_PARAMS = {'dataset': datasets.MNISTInfo(),
                  'permute': False,
                  'nhid': 500,
                  'training': rbm_training.TrainingParams.defaults('pcd'),
                  'init_rbm': 'base_rates',
                  'save_after': [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000],
                  'show_after': [],
                  'diagnostics': ['particles', 'moments_err', 'gibbs_chains'],
                  'save_particles': True,
                  }

def default_parameters(name):
    params = copy.deepcopy(DEFAULT_PARAMS)
    params['name'] = name
    return params

def set_sgd(params, lrate, weights_lrate):
    """Set the learning rates for SGD."""
    base_lrates = binary_rbms.LearningRates(lrate, lrate, weights_lrate)
    params['training'].lrates.base_lrates = base_lrates

def set_fang(params, lrate, weights_lrate):
    """Switch to FANG and set the learning rates."""
    base_lrates = binary_rbms.LearningRates(lrate, lrate, weights_lrate)
    params['training'].lrates.base_lrates = base_lrates
    params['training'].updater = rbm_training.FANGParams.defaults('pcd')

def set_centering(params, bias_lrate, average=False, weights_lrate=None, rescale=False):
    """Switch to centering trick and set the learning rates.
        - average: use running averages of expectations (should be True, but keep this for backwards compatibility)
        - rescale: rescale the updates by the inverse variance
    """
    if weights_lrate is None:
        weights_lrate = bias_lrate
    base_lrates = binary_rbms.LearningRates(bias_lrate, bias_lrate, weights_lrate)
    params['training'].lrates.base_lrates = base_lrates
    if rescale:
        params['training'].updater = rbm_training.RescaledCenteringTrickParams.defaults('default')
    elif average:
        params['training'].updater = rbm_training.CenteringTrickAvgParams.defaults('default')
    else:
        params['training'].updater = rbm_training.CenteringTrickParams()

def set_updater(params, updater, bias_lrate, weights_lrate):
    if updater == 'sgd':
        set_sgd(params, bias_lrate, weights_lrate)
    elif updater == 'centering':
        set_centering(params, bias_lrate, average=True, weights_lrate=weights_lrate)
    elif updater == 'centering_rescaled':
        set_centering(params, bias_lrate, weights_lrate=weights_lrate, rescale=True)
    elif updater == 'fang':
        set_fang(params, bias_lrate, weights_lrate)
        params['training'].updater.recompute_every = 100
        params['training'].updater.update_stats_every = 10
    else:
        raise RuntimeError('Unknown updater: {}'.format(updater))


def mnist_learning_rates(algorithm):
    return {'sgd': (0.3, 0.1),
            'centering': (0.3, 0.3),
            'fang': (0.3, 0.03),
            }[algorithm]

def omniglot_learning_rates(algorithm):
    return {'sgd': (0.1, 0.1),
            'centering': (0.3, 0.3),
            'fang': (0.1, 0.01),
            }[algorithm]


def get_experiment(name):
    m = re.match(r'mnist_long2/(\w*)$', name)
    if m:
        algorithm = m.group(1)
        bias_lrate, weights_lrate = mnist_learning_rates(algorithm)
        params = default_parameters(name)
        set_updater(params, algorithm, bias_lrate, weights_lrate)
        params['permute'] = True          # should have been doing this before!
        params['training'].neg.num_particles = 2000
        params['training'].pos.mbsize = 2000
        params['training'].weight_decay = 0.
        params['training'].num_steps = 2000000
        params['save_after'] = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000,
                                200000, 500000, 1000000, 2000000]
        params['save_particles'] = False
        return Expt(**params)

    m = re.match(r'omniglot_long_500/(\w*)$', name)
    if m:
        algorithm = m.group(1)
        bias_lrate, weights_lrate = omniglot_learning_rates(algorithm)
        params = default_parameters(name)
        set_updater(params, algorithm, bias_lrate, weights_lrate)
        params['dataset'] = datasets.CharactersInfo()
        params['permute'] = True          # should have been doing this before!
        params['nhid'] = 500
        params['training'].neg.num_particles = 2000
        params['training'].pos.mbsize = 2000
        params['training'].weight_decay = 0.
        params['training'].num_steps = 2000000
        params['save_after'] = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000,
                                200000, 500000, 1000000, 2000000]
        params['save_particles'] = False
        return Expt(**params)

    raise RuntimeError('Unknown experiment: {}'.format(name))

class Visuals:
    def __init__(self, expt, vis, num_particles=100, num_steps=1000, binarize=True):
        self.expt = expt
        self.subset = expt.diagnostics
        self.t0 = time.time()

        if 'objective' in self.subset:
            self.log_prob_tracker = diagnostics.LogProbTracker(vis, None, binarize=binarize,
                                                               num_particles=num_particles, num_steps=num_steps,
                                                               compute_after=np.array(expt.show_after))

    def after_step(self, rbm, trainer, i):
        it = i + 1

        save = it in self.expt.save_after
        display = it in self.expt.show_after

        if save:
            if self.expt.save_particles:
                storage.dump(trainer.fantasy_particles, self.expt.pcd_particles_file(it))
            storage.dump(rbm, self.expt.rbm_file(it))
            if hasattr(trainer, 'avg_rbm'):
                storage.dump(trainer.avg_rbm, self.expt.avg_rbm_file(it))
            storage.dump(time.time() - self.t0, self.expt.time_file(it))

        if 'particles' in self.subset and (save or display):
            fig = rbm_vis.show_particles(rbm, trainer.fantasy_particles, self.expt.dataset, display=display,
                                         figtitle='PCD particles ({} updates)'.format(it))
            if display:
                pylab.draw()
            if save:
                misc.save_image(fig, self.expt.pcd_particles_figure_file(it))

        if 'gibbs_chains' in self.subset and (save or display):
            fig = diagnostics.show_chains(rbm, trainer.fantasy_particles, self.expt.dataset, display=display,
                                          figtitle='Gibbs chains (iteration {})'.format(it))
            if save:
                misc.save_image(fig, self.expt.gibbs_chains_figure_file(it))

        if 'objective' in self.subset:
            self.log_prob_tracker.update(rbm, trainer.fantasy_particles)

        if display:
            pylab.draw()


def run(expt, visuals=None):
    if isinstance(expt, str):
        expt = get_experiment(expt)

    storage.ensure_directory(expt.figures_dir())
    storage.ensure_directory(expt.outputs_dir())

    mkl.set_num_threads(1)
        
    v = gnp.garray(expt.dataset.load().as_matrix())
    v = 0.999 * v + 0.001 * 0.5

    if expt.permute:
        idxs = np.random.permutation(v.shape[0])
        v = v[idxs]

    if visuals is None:
        visuals = Visuals(expt, v)

    if expt.init_rbm == 'base_rates':
        init_rbm = None
    elif isinstance(expt.init_rbm, expt_moments.TrainedRBM):
        init_rbm = expt_moments.load_trained_rbm(expt.init_rbm.location).convert_to_garrays()
    else:
        raise RuntimeError('Unknown init_rbm')

    assert isinstance(expt.training, rbm_training.TrainingParams)
    rbm_training.train_rbm(v, expt.nhid, expt.training, after_step=visuals.after_step, init_rbm=init_rbm)


if __name__ == '__main__':
    expt_names = sys.argv[1:]
    for expt_name in expt_names:
        run(expt_name)




