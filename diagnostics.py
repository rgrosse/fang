import gnumpy as gnp
import numpy as np
nax = np.newaxis
import pylab

import ais
import tractable
from utils import misc
from utils.schemas import Struct
from visuals import misc as vm


FIELDS = ['vis', 'hid', 'prod_by_vis', 'prod_by_hid']

def default_save_after(num_iter, num_save):
    return np.unique(np.logspace(0., np.log10(num_iter), num_save).astype(int))



def show_chains(rbm, state, dataset, num_particles=20, num_samples=20, show_every=10, display=True,
                figname='Gibbs chains', figtitle='Gibbs chains'):
    samples = gnp.zeros((num_particles, num_samples, state.v.shape[1]))
    state = state[:num_particles, :, :]

    for i in range(num_samples):
        samples[:, i, :] = rbm.vis_expectations(state.h)
        
        for j in range(show_every):
            state = rbm.step(state)

    npix = dataset.num_rows * dataset.num_cols
    rows = [vm.hjoin([samples[i, j, :npix].reshape((dataset.num_rows, dataset.num_cols)).as_numpy_array()
                      for j in range(num_samples)],
                     normalize=False)
            for i in range(num_particles)]
    grid = vm.vjoin(rows, normalize=False)

    if display:
        pylab.figure(figname)
        pylab.matshow(grid, cmap='gray', fignum=False)
        pylab.title(figtitle)
        pylab.gcf().canvas.draw()

    return grid


DEFAULT_LOG_PROB_AFTER = default_save_after(1000000, 50)

def log_mean(values):
    return np.logaddexp.reduce(values) - np.log(len(values))


class LogZEstimateInfo(Struct):
    fields = ['counts', 'mean', 'lower', 'upper']

    @staticmethod
    def new():
        return LogZEstimateInfo(np.array([], dtype=int), np.array([]), np.array([]), np.array([]))

    def update(self, count, log_Z_vals):
        self.counts = np.concatenate([self.counts, [count]])

        log_Z_vals = log_Z_vals.as_numpy_array()
        log_Z_lower, log_Z_upper = misc.bootstrap(log_Z_vals, log_mean)
        log_Z_mean = log_mean(log_Z_vals)
        self.mean = np.concatenate([self.mean, [log_Z_mean]])
        self.lower = np.concatenate([self.lower, [log_Z_lower]])
        self.upper = np.concatenate([self.upper, [log_Z_upper]])

class ExactLogZInfo(Struct):
    fields = ['counts', 'mean']

    @staticmethod
    def new():
        return ExactLogZInfo(np.array([], dtype=int), np.array([]))

    def update(self, count, log_Z):
        self.counts = np.concatenate([self.counts, [count]])
        self.mean = np.concatenate([self.mean, [log_Z]])


class PosTermInfo(Struct):
    fields = ['counts', 'values']

    @staticmethod
    def new():
        return PosTermInfo(np.array([], dtype=int), np.array([]))

    def update(self, count, term):
        self.counts = np.concatenate([self.counts, [count]])
        self.values = np.concatenate([self.values, [term]])
        


def plot_objfn(pos_term_info, log_Z_info, color, zoom=False, label=None):
    assert np.all(pos_term_info.counts == log_Z_info.counts)
    exact = not hasattr(log_Z_info, 'lower')
    
    mean = pos_term_info.values - log_Z_info.mean
    if not exact:
        lower = pos_term_info.values - log_Z_info.upper
        upper = pos_term_info.values - log_Z_info.lower
    
    pylab.semilogx(pos_term_info.counts, mean, color=color, label=label)
    if not exact:
        pylab.errorbar(pos_term_info.counts, (lower+upper)/2., yerr=(upper-lower)/2., fmt='', ls='None', ecolor=color)
    if zoom:
        pylab.ylim(mean.max() - 50., mean.max() + 5.)





class LogProbTracker:
    def __init__(self, vis, target_moments, compute_after=DEFAULT_LOG_PROB_AFTER, num_particles=100, num_steps=1000,
                 binarize=True):
        if binarize and vis is not None:
            try:
                vis = gnp.garray(np.random.binomial(1, vis))
            except:
                vis = gnp.garray(np.random.binomial(1, vis.as_numpy_array()))
        self.vis = vis
        self.target_moments = target_moments
        self.compute_after = compute_after
        if target_moments is not None:
            self.base_rate_moments = target_moments.full_base_rate_moments()
        self.num_particles = num_particles
        self.num_steps = num_steps
        self.avg_rbm = None

        if target_moments is not None:
            nhid = self.base_rate_moments.expect_hid.size
            self.exact = (nhid <= 20)
        else:
            self.exact = False

        
        
        self.count = 0

    def init_log_Z_info(self, exact):
        if exact:
            self.log_Z_info = {name: ExactLogZInfo.new()
                               for name in ['main', 'avg']}
        else:
            self.log_Z_info = {name: LogZEstimateInfo.new()
                               for name in ['main', 'avg']}

        self.fe_info = {name: PosTermInfo.new()
                        for name in ['main', 'avg']}
        self.dp_info = {name: PosTermInfo.new()
                        for name in ['main', 'avg']}

    def update(self, rbm, particles=None):
        self.count += 1

        rbm_class = rbm.__class__

        exact = (hasattr(rbm, 'exact_partition_function') or rbm.nhid <= 20)
        if not hasattr(self, 'log_Z_info'):
            self.init_log_Z_info(exact)

        if self.avg_rbm is None:
            self.avg_rbm = rbm.copy()
        else:
            self.avg_rbm = 0.99 * self.avg_rbm + 0.01 * rbm

        if self.target_moments is not None:
            base_rate_moments = self.target_moments.full_base_rate_moments()
        else:
            curr_moments = rbm.cond_hid(particles.h)
            if hasattr(self, 'empirical_moments'):
                self.empirical_moments = 0.99 * self.empirical_moments + 0.01 * curr_moments
            else:
                self.empirical_moments = curr_moments
            base_rate_moments = self.empirical_moments.smooth().full_base_rate_moments()

        init_rbm = rbm_class.from_moments(base_rate_moments)

        if self.count in self.compute_after:
            print 'Estimating log-likelihoods...'
            for name, curr_rbm in [('main', rbm), ('avg', self.avg_rbm)]:
                if self.vis is not None:
                    self.fe_info[name].update(self.count, curr_rbm.free_energy_vis(self.vis).mean())
                if self.target_moments is not None:
                    self.dp_info[name].update(self.count, curr_rbm.dot_product(self.target_moments))
                
                if rbm.nhid <= 20:
                    log_Z = tractable.exact_partition_function(curr_rbm)
                    self.log_Z_info[name].update(self.count, log_Z)

                elif exact:
                    log_Z = curr_rbm.exact_partition_function()
                    self.log_Z_info[name].update(self.count, log_Z)

                else:
                    path = ais.GeometricRBMPath(init_rbm, curr_rbm)
                    schedule = np.linspace(0., 1., self.num_steps)
                    _, log_Z, _ = ais.ais(path, schedule, self.num_particles, show_progress=True)
                    self.log_Z_info[name].update(self.count, log_Z)

            self.plot()

            

    def plot(self):
        if self.vis is not None:
            pylab.figure('log probs')
            pylab.clf()
            plot_objfn(self.fe_info['main'], self.log_Z_info['main'], 'b', label='Raw')
            plot_objfn(self.fe_info['avg'], self.log_Z_info['avg'], 'r', label='Averaged')
            pylab.title('log probs')
            pylab.legend(loc='lower right')
            pylab.gcf().canvas.draw()

            pylab.figure('log probs (zoomed)')
            pylab.clf()
            plot_objfn(self.fe_info['main'], self.log_Z_info['main'], 'b', zoom=True, label='Raw')
            plot_objfn(self.fe_info['avg'], self.log_Z_info['avg'], 'r', label='Averaged')
            pylab.title('log probs (zoomed)')
            pylab.legend(loc='lower right')
            pylab.gcf().canvas.draw()

        if self.target_moments is not None:
            pylab.figure('moment matching objective')
            pylab.clf()
            plot_objfn(self.dp_info['main'], self.log_Z_info['main'], 'b', label='Raw')
            plot_objfn(self.dp_info['avg'], self.log_Z_info['avg'], 'r', label='Averaged')
            pylab.title('moment matching objective')
            pylab.legend(loc='lower right')
            pylab.gcf().canvas.draw()

            pylab.figure('moment matching objective (zoomed)')
            pylab.clf()
            plot_objfn(self.dp_info['main'], self.log_Z_info['main'], 'b', zoom=True, label='Raw')
            plot_objfn(self.dp_info['avg'], self.log_Z_info['avg'], 'r', label='Averaged')
            pylab.title('moment matching objective (zoomed)')
            pylab.legend(loc='lower right')
            pylab.gcf().canvas.draw()
        
