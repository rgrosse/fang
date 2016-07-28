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

def load_wall_clock_time(expt_name, prefix='mm_rbms'):
    eval_expt = evaluation.get_experiment(expt_name)
    tr_expt = from_scratch.get_experiment(eval_expt.rbm_source.expt_name)
    return [storage.load(tr_expt.time_file(it)) for it in tr_expt.save_after
            if os.path.exists(tr_expt.time_file(it))]

class Plotter:
    def __init__(self, logscale, average):
        self.logscale = logscale
        self.average = average
        self.labels = set()

    def plot_results(self, results, xloc, color, ls, label, presentation=False, subsample=False):
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

        if presentation:
            lw = 2.
        else:
            lw = 1.

        if subsample:
            xloc = xloc[1::4]
            avg = avg[1::4]
            lower = lower[1::4]
            upper = upper[1::4]

        if label not in self.labels:
            plot_cmd(xloc, avg, color=color, ls=ls, lw=lw, label=label)
        else:
            plot_cmd(xloc, avg, color=color, ls=ls, lw=lw)

        self.labels.add(label)

        if presentation:
            pylab.xticks(fontsize='xx-large')
            pylab.yticks(fontsize='xx-large')

        try:
            pylab.errorbar(xloc, (lower+upper)/2., yerr=(upper-lower)/2., fmt='', ls='None', ecolor=color)
        except:
            pass


def plot_log_probs(expt_names, prefix='mm_rbms', average=True, figtitle='Log prob comparison', logscale=True, use_test=False,
                   use_wall_clock=False, presentation=False):
    #pylab.figure(figtitle)
    pylab.figure()
    pylab.clf()

    plotter = Plotter(logscale, average)

    for name, color in zip(expt_names, COLORS):
        results_dir = os.path.join(config.OUTPUTS_DIR, 'evaluation', prefix, name)
        subdirs = os.listdir(results_dir)
        for subdir, ls in zip(subdirs, LINE_STYLES):
            full_expt_name = '{}/{}/{}'.format(prefix, name, subdir)
            print full_expt_name, ls
            expt = evaluation.get_experiment(full_expt_name)
            results = evaluation.collect_log_probs(expt, use_test=use_test, ignore_failed=True)
            iter_counts = sorted(set([k[0] for k in results.keys()]))

            if use_wall_clock:
                xloc = load_wall_clock_time(full_expt_name)
            else:
                xloc = iter_counts

            plotter.plot_results(results, xloc, color, ls, name, presentation=presentation)

            
    if presentation:
        pylab.legend(loc='lower right', fontsize='x-large')
    else:
        pylab.legend(loc='lower right')
        pylab.title(figtitle)


def collect_names(name, prefix='mm_rbms'):
    d = os.path.join(config.OUTPUTS_DIR, 'evaluation', prefix, name)
    conditions = os.listdir(d)
    return [os.path.join(name, c) for c in sorted(conditions)]


def plot_log_probs2(expt_names, prefix='mm_rbms', average=True, figtitle='Log prob comparison', logscale=True, use_test=False,
                    use_wall_clock=False, presentation=False, labels='auto', colors=COLORS, minutes=False,
                    hours=False, subsample=False, show_legend=True):
    #pylab.figure(figtitle)
    pylab.figure()
    pylab.clf()

    plotter = Plotter(logscale, average)

    if labels == 'auto' or labels is None:
        labels = expt_names
    
    for name, (style, color), label in zip(expt_names, itertools.product(LINE_STYLES, colors), labels):
        results_dir = os.path.join(config.OUTPUTS_DIR, 'evaluation', prefix, name)
        full_expt_name = '{}/{}'.format(prefix, name)
        print full_expt_name
        expt = evaluation.get_experiment(full_expt_name)
        results = evaluation.collect_log_probs(expt, use_test=use_test, ignore_failed=True)
        iter_counts = sorted(set([k[0] for k in results.keys()]))

        if use_wall_clock:
            xloc = np.array(load_wall_clock_time(full_expt_name, prefix))
            if hours:
                xloc /= 3600.
            elif minutes:
                xloc /= 60.
        else:
            xloc = np.array(iter_counts)

        plotter.plot_results(results, xloc, color, style, label, presentation=presentation,
                             subsample=subsample)

    #assert False
    if show_legend:
        if presentation:
            pylab.legend(loc='lower right', fontsize='xx-large')
        else:
            pylab.legend(loc='lower right')
            pylab.title(figtitle)    
