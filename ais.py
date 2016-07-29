import binary_rbms
from utils import misc, rbm_utils


def ais(sequence, schedule, num_particles, show_progress=False):
    state = sequence.init_sample(num_particles)
    pf = sequence.init_partition_function()
    terms = []

    if show_progress:
        pbar = misc.pbar(len(schedule))
    
    for it, (t0, t1) in enumerate(zip(schedule[:-1], schedule[1:])):
        delta = sequence.joint_prob(state, t1) - sequence.joint_prob(state, t0)
        pf += delta
        terms.append(delta)

        state = sequence.step(state, t1)

        if show_progress:
            pbar.update(it + 1)

    if show_progress:
        pbar.finish()

    return state, pf, terms



class GeometricRBMPath:
    """Geometric averages path with a linear schedule."""
    
    def __init__(self, init_rbm, target_rbm):
        self.init_rbm = init_rbm
        self.target_rbm = target_rbm

    def get_rbm(self, t):
        return self.init_rbm * (1. - t) + self.target_rbm * t

    def init_partition_function(self):
        return self.get_rbm(0.).init_partition_function()

    def init_sample(self, num_particles):
        return self.get_rbm(0).init_samples(num_particles)

    def joint_prob(self, state, t):
        rbm = self.get_rbm(t)
        return rbm.free_energy_hid(state.h)

    def step(self, state, t):
        rbm = self.get_rbm(t)
        v = rbm_utils.sample_units(rbm.vis_inputs(state.h))
        h = rbm_utils.sample_units(rbm.hid_inputs(v))
        return binary_rbms.RBMState(v, h)




