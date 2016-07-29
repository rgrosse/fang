import gnumpy as gnp
import mkl
import numpy as np
import os
import re
import sys
import time

import ais
import binary_rbms
import config
import from_scratch as expt_fs
import tractable
from utils.schemas import Params, Struct, Choice
from utils import misc, storage
from visuals import rbm_vis


AVG_VALS = [True]              # True = Polyak averaged RBM, False = raw RBM


class FromScratchRBM(Params):
    class Fields:
        expt_name = str          # name of experiment in from_scratch.py with RBMs to evaluate




class ExactParams(Params):
    """Parameters for computing log probs exactly"""
    class Fields:
        num_samples = int        # number of exact samples to display


class Expt(Params):
    class Fields:
        name = str
        dataset = str                            # dataset on which to measure log probs
        rbm_source = Choice([FromScratchRBM])            # which RBMs to evaluate
        annealing = Choice([ais.AnnealingParams, ExactParams])     # method of computing/estimating partition function
        include_test = bool                      # whether to report results on test data
        gibbs_steps = int                        # number of Gibbs steps to run starting from AIS particles

    def outputs_dir(self):
        """Machine-readable results"""
        return os.path.join(config.OUTPUTS_DIR, 'evaluation', self.name)

    def figures_dir(self):
        """Human-readable results"""
        return os.path.join(config.FIGURES_DIR, 'evaluation', self.name)

    def output_file(self, dir, name, it, avg, tp):
        if avg:
            return os.path.join(dir, '{}_{}_avg.{}'.format(name, it, tp))
        else:
            return os.path.join(dir, '{}_{}.{}'.format(name, it, tp))

    def log_Z_file(self, it, avg):
        """A vector of individual AIS estimates, or the exact partition function."""
        return self.output_file(self.outputs_dir(), 'log_Z', it, avg, 'pk')

    def final_states_file(self, it, avg):
        """States at the end of the AIS runs."""
        return self.output_file(self.outputs_dir(), 'final_states', it, avg, 'pk')

    def final_states_figure_file(self, it, avg):
        """States at the end of the AIS runs."""
        return self.output_file(self.figures_dir(), 'final_states', it, avg, 'png')

    def gibbs_states_file(self, it, avg):
        """End of Gibbs chain starting from AIS particles."""
        return self.output_file(self.outputs_dir(), 'gibbs_states', it, avg, 'pk')

    def gibbs_states_figure_file(self, it, avg):
        """End of Gibbs chain starting from AIS particles."""
        return self.output_file(self.figures_dir(), 'gibbs_states', it, avg, 'png')

    def time_file(self, it, avg):
        """Running time for AIS."""
        return self.output_file(self.outputs_dir(), 'time', it, avg, 'pk')

    def log_probs_text_file(self):
        """Text file summarizing log probs of all conditions."""
        return os.path.join(self.figures_dir(), 'log_probs.txt')


def get_experiment(name):
    m = re.match(r'from_scratch/(.*)$', name)      # approximately evaluate moment matched RBM with AIS
    if m:
        fs_expt = m.group(1)
        return Expt(
            name = name,
            dataset = expt_fs.get_experiment(fs_expt).dataset,
            rbm_source = FromScratchRBM(expt_name=fs_expt),
            annealing = ais.AnnealingParams.defaults('full'),
            include_test = False,
            gibbs_steps = 50000)

    raise RuntimeError('Unknown experiment: {}'.format(name))

def get_training_expt(expt):
    if isinstance(expt.rbm_source, FromScratchRBM):
        return expt_fs.get_experiment(expt.rbm_source.expt_name)
    else:
        raise RuntimeError('Unknown RBM source: {}'.format(expt.rbm_source))


def load_rbm(expt, it, avg):
    if isinstance(expt, str):
        expt = get_experiment(expt)
    tr_expt = get_training_expt(expt)
    if avg:
        rbm = storage.load(tr_expt.avg_rbm_file(it))
    else:
        rbm = storage.load(tr_expt.rbm_file(it))
    return rbm.convert_to_garrays()            # for RBMs saved by the CPU version of the code

def compute_moments(tr_expt, rbm):
    if isinstance(tr_expt, expt_fs.Expt):
        v = gnp.garray(tr_expt.dataset.load().as_matrix())
        return rbm.cond_vis(v).smooth()
    else:
        raise RuntimeError('Invalid tr_expt')
        

def run_ais(expt, save=True, show_progress=False):
    """Run AIS for all the RBMs, and save the estimated log partition functions and the final particles."""
    if isinstance(expt, str):
        expt = get_experiment(expt)

    mkl.set_num_threads(1)

    tr_expt = get_training_expt(expt)

    for it in tr_expt.save_after:
        for avg in AVG_VALS:
            print 'iteration', it, avg
            t0 = time.time()
            try:
                rbm = load_rbm(expt, it, avg)
            except:
                continue

            moments = compute_moments(tr_expt, rbm)
            brm = moments.full_base_rate_moments()
            init_rbm = binary_rbms.RBM.from_moments(brm)

            seq = ais.GeometricRBMPath(init_rbm, rbm)
            path = ais.RBMDistributionSequence(seq, expt.annealing.num_particles, 'h')
            schedule = np.linspace(0., 1., expt.annealing.num_steps)
            state, log_Z, _ = ais.ais(path, schedule, show_progress=show_progress)

            if save:
                storage.dump(log_Z, expt.log_Z_file(it, avg))
                storage.dump(state, expt.final_states_file(it, avg))
                storage.dump(time.time() - t0, expt.time_file(it, avg))

def save_exact_log_Z(expt):
    """Compute the exact partition functions for small RBMs."""
    if isinstance(expt, str):
        expt = get_experiment(expt)
    tr_expt = get_training_expt(expt)
        
    for it in tr_expt.save_after:
        for avg in AVG_VALS:
            print 'iteration', it, avg
            try:
                rbm = load_rbm(expt, it, avg)
            except:
                continue

            log_Z = tractable.exact_partition_function(rbm)
            storage.dump(log_Z, expt.log_Z_file(it, avg))
    

def save_log_Z(expt, save=True, show_progress=False):
    """Wrapper for partition function computation."""
    if isinstance(expt, str):
        expt = get_experiment(expt)
        
    if isinstance(expt.annealing, ExactParams):
        save_exact_log_Z(expt)
    elif isinstance(expt.annealing, ais.AnnealingParams):
        run_ais(expt, save, show_progress)
    else:
        raise RuntimeError('Unknown annealing parameter')


def run_gibbs(expt, save=True, show_progress=False):
    """Run Gibbs chains starting from the AIS particles (sampled proportionally to their
    weights), and save the final particles."""
    if isinstance(expt, str):
        expt = get_experiment(expt)
    tr_expt = get_training_expt(expt)

    for it in tr_expt.save_after:
        for avg in AVG_VALS:
            print 'Iteration', it, avg
            try:
                rbm = load_rbm(expt, it, avg)
            except:
                continue
            log_Z = storage.load(expt.log_Z_file(it, avg)).as_numpy_array()
            final_states = storage.load(expt.final_states_file(it, avg))

            # sample the states proportionally to the Z estimates
            p = log_Z - np.logaddexp.reduce(log_Z)
            p /= p.sum()    # not needed in theory, but numpy complains if it doesn't sum exactly to 1
            idxs = np.random.multinomial(1, p, size=expt.annealing.num_particles).argmax(1)
            states = binary_rbms.RBMState(final_states.v[idxs, :], final_states.h[idxs, :])

            if show_progress:
                pbar = misc.pbar(expt.gibbs_steps)

            for st in range(expt.gibbs_steps):
                states = rbm.step(states)

                if show_progress:
                    pbar.update(st)

            if show_progress:
                pbar.finish()

            if save:
                storage.dump(states, expt.gibbs_states_file(it, avg))

def save_exact_samples(expt):
    """Save exact samples from the RBM distribution."""
    if isinstance(expt, str):
        expt = get_experiment(expt)
    tr_expt = get_training_expt(expt)

    for it in tr_expt.save_after:
        for avg in AVG_VALS:
            print 'Iteration', it, avg
            try:
                rbm = load_rbm(expt, it, avg)
            except:
                continue

            states = tractable.exact_samples(rbm, expt.annealing.num_samples)
            storage.dump(states, expt.gibbs_states_file(it, avg))
                
    
def save_samples(expt, save=True, show_progress=False):
    """Wrapper for saving samples."""
    if isinstance(expt, str):
        expt = get_experiment(expt)
        
    if isinstance(expt.annealing, ExactParams):
        save_exact_samples(expt)
    elif isinstance(expt.annealing, ais.AnnealingParams):
        run_gibbs(expt, save, show_progress)
    else:
        raise RuntimeError('Unknown annealing parameter')




def log_mean(values):
    return np.logaddexp.reduce(values) - np.log(len(values))

class Results(Struct):
    fields = ['log_Z', 'log_Z_lower', 'log_Z_upper', 'train_free_energy', 'moments_dot']

    def train_logprob(self):
        return self.train_free_energy.mean() - log_mean(self.log_Z)

    def train_logprob_interval(self):
        fev = self.train_free_energy.mean()
        return fev - self.log_Z_upper, fev - self.log_Z_lower

    def moment_matching_objective(self):
        return self.moments_dot - log_mean(self.log_Z)

    def moment_matching_objective_interval(self):
        return self.moments_dot - self.log_Z_upper, self.moments_dot - self.log_Z_lower

class ExactResults(Struct):
    fields = ['log_Z', 'train_free_energy', 'moments_dot']

    def train_logprob(self):
        return self.train_free_energy.mean() - self.log_Z

    def moment_matching_objective(self):
        return self.moments_dot - self.log_Z
    

def collect_log_probs(expt, subset='test', ignore_failed=False):
    """Load the results of individual partition function estimation trials, and return
    the averaged estimates along with bootstrap confidence intervals."""
    if isinstance(expt, str):
        expt = get_experiment(expt)
    assert subset in ['train', 'test']

    if subset == 'test':
        vis = expt.dataset.load_test().as_matrix()
    else:
        vis = expt.dataset.load().as_matrix()
    vis = np.random.binomial(1, vis)

    tr_expt = get_training_expt(expt)
    results = {}
    pbar = misc.pbar(len(tr_expt.save_after) * 2)
    count = 0
    for it in tr_expt.save_after:
        for avg in AVG_VALS:
            count += 1
            #print 'iteration', it
            try:
                rbm = load_rbm(expt, it, avg)
            except:
                continue

            if ignore_failed and not os.path.exists(expt.log_Z_file(it, avg)):
                continue

            if isinstance(expt.annealing, ais.AnnealingParams):
                log_Z = storage.load(expt.log_Z_file(it, avg)).as_numpy_array()
                log_Z_lower, log_Z_upper = misc.bootstrap(log_Z, log_mean)

                train_fev = rbm.free_energy_vis(vis)

                results[it, avg] = Results(log_Z, log_Z_lower, log_Z_upper, train_fev, None)

            elif isinstance(expt.annealing, ExactParams):
                log_Z = storage.load(expt.log_Z_file(it, avg))
                train_fev = rbm.free_energy_vis(vis)
                results[it, avg] = ExactResults(log_Z, train_fev, None)

            else:
                raise RuntimeError('Unknown annealing params')

            pbar.update(count)

    pbar.finish()

    return results



def print_log_probs(expt, outstr=sys.stdout):
    """Print the estimated log probs for all conditions."""
    if isinstance(expt, str):
        expt = get_experiment(expt)

    results = collect_log_probs(expt)

    for it, avg in sorted(results.keys()):
        r = results[it, avg]

        if avg:
            print >> outstr, 'Iteration {} (averaged)'.format(it)
        else:
            print >> outstr, 'Iteration {}'.format(it)
        print >> outstr, 'Train log-likelihood: {:1.3f}'.format(r.train_logprob())
        if hasattr(r, 'train_logprob_interval'):
            print >> outstr, '    CI: [{:1.3f}, {:1.3f}]'.format(*r.train_logprob_interval())

        try:
            print >> outstr, 'Moment matching objective: {:1.3f}'.format(r.moment_matching_objective())
            if hasattr(r, 'moment_matching_objective_interval'):
                print >> outstr, '    CI: [{:1.3f}, {:1.3f}]'.format(*r.moment_matching_objective_interval())
        except:
            pass

        print >> outstr

def save_figures(expt):
    """Save visualizations of the particles."""
    if isinstance(expt, str):
        expt = get_experiment(expt)
    
    tr_expt = get_training_expt(expt)

    storage.ensure_directory(expt.figures_dir())

    for it in tr_expt.save_after:
        for avg in AVG_VALS:
            print 'Iteration', it
            try:
                rbm = load_rbm(expt, it, avg)
            except:
                continue
            final_states = storage.load(expt.final_states_file(it, avg))
            gibbs_states = storage.load(expt.gibbs_states_file(it, avg))

            fig = rbm_vis.show_particles(rbm, final_states, expt.dataset)
            misc.save_image(fig, expt.final_states_figure_file(it, avg))

            fig = rbm_vis.show_particles(rbm, gibbs_states, expt.dataset)
            misc.save_image(fig, expt.gibbs_states_figure_file(it, avg))

    print_log_probs(expt, open(expt.log_probs_text_file(), 'w'))
        

        
if __name__ == '__main__':
    expt_names = sys.argv[1:-1]
    command = sys.argv[-1]
    assert expt_names
    for expt_name in expt_names:
        if command == 'ais':
            save_log_Z(expt_name)
        elif command == 'gibbs':
            save_samples(expt_name)
        elif command == 'all':
            save_log_Z(expt_name)
            save_samples(expt_name)
        else:
            raise RuntimeError('Unknown command: {}'.format(command))

