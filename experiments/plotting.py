import itertools
import os
import numpy as np
import pylab

import config
import evaluation
import from_scratch
from utils import storage

COLORS = ['b', 'g', 'r', 'c', 'k', 'm', 'y']
LINE_STYLES = ['-', '--', ':']
ALGORITHM_LABELS = {'sgd': 'SGD', 'centering': 'centering', 'fang': 'FANG'}

PREFIX = 'from_scratch'

def load_wall_clock_time(expt_name):
    eval_expt = evaluation.get_experiment(expt_name)
    tr_expt = from_scratch.get_experiment(eval_expt.rbm_source.expt_name)
    return [storage.load(tr_expt.time_file(it)) for it in tr_expt.save_after
            if os.path.exists(tr_expt.time_file(it))]

class Plotter:
    def __init__(self, logscale, average):
        self.logscale = logscale
        self.average = average
        self.labels = set()

    def plot_results(self, results, xloc, color, ls, label):
        iter_counts = sorted(set([it for it, av in results.keys() if av == self.average]))
        sorted_results = [results[it, self.average] for it in iter_counts]

        avg = np.array([r.train_logprob() for r in sorted_results])
        if hasattr(r, 'train_logprob_interval'):
            lower = np.array([r.train_logprob_interval()[0] for r in sorted_results])
            upper = np.array([r.train_logprob_interval()[1] for r in sorted_results])

        if self.logscale:
            plot_cmd = pylab.semilogx
        else:
            plot_cmd = pylab.plot

        xloc = xloc[:len(avg)]

        lw = 2.

        if label not in self.labels:
            plot_cmd(xloc, avg, color=color, ls=ls, lw=lw, label=label)
        else:
            plot_cmd(xloc, avg, color=color, ls=ls, lw=lw)

        self.labels.add(label)

        pylab.xticks(fontsize='xx-large')
        pylab.yticks(fontsize='xx-large')

        try:
            pylab.errorbar(xloc, (lower+upper)/2., yerr=(upper-lower)/2., fmt='', ls='None', ecolor=color)
        except:
            pass


def collect_names(name):
    d = os.path.join(config.OUTPUTS_DIR, 'evaluation', PREFIX, name)
    conditions = os.listdir(d)
    return [os.path.join(name, c) for c in sorted(conditions)]


def plot_log_probs(expt_names, average=True, logscale=True, subset='test',
                   use_wall_clock=True, labels='auto', colors=COLORS, time_unit='seconds'):
    pylab.figure()
    pylab.clf()
    assert time_unit in ['seconds', 'minutes', 'hours']
    assert subset in ['train', 'test']

    plotter = Plotter(logscale, average)

    if labels == 'auto' or labels is None:
        labels = expt_names
    
    for name, (style, color), label in zip(expt_names, itertools.product(LINE_STYLES, colors), labels):
        full_expt_name = '{}/{}'.format(PREFIX, name)
        print full_expt_name
        expt = evaluation.get_experiment(full_expt_name)
        results = evaluation.collect_log_probs(expt, subset=subset, ignore_failed=True)
        iter_counts = sorted(set([k[0] for k in results.keys()]))

        if use_wall_clock:
            xloc = np.array(load_wall_clock_time(full_expt_name))
            if time_unit == 'hours':
                xloc /= 3600.
            elif time_unit == 'minutes':
                xloc /= 60.
        else:
            xloc = np.array(iter_counts)

        plotter.plot_results(results, xloc, color, style, label)

    pylab.legend(loc='lower right', fontsize='xx-large')

def get_ylim(expt_base):
    if 'mnist' in expt_base:
        return -100., -80.
    elif 'omniglot' in expt_base:
        return -120., -95.
    else:
        raise RuntimeError('Unable to determine y-axis scaling')

def show_comparison(expt_base, algorithms, subset, ymin=None, ymax=None):
    expt_names = ['/'.join([expt_base, alg]) for alg in algorithms]
    labels = [ALGORITHM_LABELS[alg] for alg in algorithms]
    plot_log_probs(expt_names, subset=subset, labels=labels)
    pylab.xlim(1e1, 1e5)
    if ymin is None:
        ymin, ymax = get_ylim(expt_base)
    pylab.ylim(ymin, ymax)
    pylab.xlabel('Wall clock time (seconds)', fontsize='x-large')
    pylab.ylabel('{} log-probabilities'.format({'train': 'Training', 'test': 'Test'}[subset]),
                 fontsize='x-large')
    pylab.tight_layout()
    

                
