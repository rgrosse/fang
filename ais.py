import numpy as np

import binary_rbms
from utils import misc, rbm_utils
from utils.schemas import Params

DEBUGGER = None

class AnnealingParams(Params):
    """Parameters for estimating log probs with AIS"""
    class Fields:
        num_particles = int      # number of AIS runs
        num_steps = int          # number of AIS steps

    class Defaults:
        quick = {'num_particles': 100, 'num_steps': 10000}
        full = {'num_particles': 400, 'num_steps': 50000}



def sigmoid_schedule(num, rad=4):
    t = np.linspace(-rad, rad, num)
    sigm = 1. / (1. + np.exp(-t))
    return (sigm - sigm.min()) / (sigm.max() - sigm.min())


def ais(path, schedule, show_progress=False):
    state = path.init_sample()
    pf = path.init_partition_function()
    terms = []

    if show_progress:
        pbar = misc.pbar(len(schedule))
    
    for it, (t0, t1) in enumerate(zip(schedule[:-1], schedule[1:])):
        delta = path.joint_prob(state, t1) - path.joint_prob(state, t0)
        pf += delta
        terms.append(delta)

        state = path.step(state, t1)

        if hasattr(DEBUGGER, 'after_ais_step'):
            DEBUGGER.after_ais_step(vars())

        if show_progress:
            pbar.update(it + 1)

    if show_progress:
        pbar.finish()

    return state, pf, terms




class RBMPath:
    def __init__(self, sequence, num_samples, energy_of):
        self.sequence = sequence
        self.num_samples = num_samples
        assert energy_of in ['v', 'h', 'vh']
        self.energy_of = energy_of

    def init_partition_function(self):
        return self.sequence.get_rbm(0.).init_partition_function()

    def init_sample(self):
        return self.sequence.get_rbm(0).init_samples(self.num_samples)

    def joint_prob(self, state, t):
        rbm = self.sequence.get_rbm(t)
        if self.energy_of == 'v':
            return rbm.free_energy_vis(state.v)
        elif self.energy_of == 'h':
            return rbm.free_energy_hid(state.h)
        else:
            assert self.energy_of == 'vh'
            return rbm.energy(state.v, state.h)

    def step(self, state, t):
        rbm = self.sequence.get_rbm(t)

        # if we use free energies, the order in which we sample is significant
        if self.energy_of == 'v':
            h = rbm_utils.sample_units(rbm.hid_inputs(state.v))
            v = rbm_utils.sample_units(rbm.vis_inputs(h))
            return binary_rbms.RBMState(v, h)
        else:
            v = rbm_utils.sample_units(rbm.vis_inputs(state.h))
            h = rbm_utils.sample_units(rbm.hid_inputs(v))
            return binary_rbms.RBMState(v, h)

    def half_step_up(self, state, t):
        rbm = self.sequence.get_rbm(t)
        h = rbm_utils.sample_units(rbm.hid_inputs(state.v))
        return binary_rbms.RBMState(state.v, h)

    def half_step_down(self, state, t):
        rbm = self.sequence.get_rbm(t)
        v = rbm_utils.sample_units(rbm.vis_inputs(state.h))
        return binary_rbms.RBMState(v, state.h)




