import gnumpy as gnp
import numpy as np
nax = np.newaxis

from utils import rbm_utils, schemas




class RBMState(schemas.Struct):
    fields = ['v', 'h']
    
    def copy(self):
        return RBMState(self.v.copy(), self.h.copy())

    @staticmethod
    def random_init(rbm, num_fantasy):
        v = gnp.garray(np.random.binomial(1, 0.5, size=(num_fantasy, rbm.nvis)))
        h = gnp.garray(np.random.binomial(1, 0.5, size=(num_fantasy, rbm.nhid)))
        return RBMState(v, h)

    def __getitem__(self, slc):
        if isinstance(slc, tuple) and len(slc) == 3:
            pslc, vslc, hslc = slc
            return RBMState(self.v[pslc, vslc].copy(), self.h[pslc, hslc].copy())
        else:
            return RBMState(self.v[slc].copy(), self.h[slc].copy())

    def __setitem__(self, slc, other):
        if isinstance(slc, tuple) and len(slc) == 3:
            pslc, vslc, hslc = slc
            self.v[pslc, vslc] = other.v
            self.h[pslc, hslc] = other.h
        else:
            self.v[slc] = other.v
            self.h[slc] = other.h


class RBM(schemas.ArrayStruct):
    fields = ['vbias', 'hbias', 'weights']
    check_op_class = False
    require_garrays = True
    
    def __init__(self, *args, **kwargs):
        schemas.ArrayStruct.__init__(self, *args, **kwargs)
        self.nvis = self.vbias.size
        self.nhid = self.hbias.size

    def allclose(self, other):
        return np.allclose(self.vbias, other.vbias) and np.allclose(self.hbias, other.hbias) \
               and np.allclose(self.weights, other.weights)

    def vis_inputs(self, hid):
        return gnp.dot(hid, self.weights.T) + self.vbias

    def hid_inputs(self, vis):
        return gnp.dot(vis, self.weights) + self.hbias

    def hid_expectations(self, v):
        return gnp.logistic(self.hid_inputs(v))

    def vis_expectations(self, h):
        return gnp.logistic(self.vis_inputs(h))

    def sample_state(self, vis):
        hid = rbm_utils.sample_units(self.hid_inputs(vis))
        return RBMState(vis, hid)

    def step(self, state):
        v = rbm_utils.sample_units(self.vis_inputs(state.h))
        h = rbm_utils.sample_units(self.hid_inputs(v))
        return RBMState(v, h)



    def energy(self, vis, hid):
        assert hid.ndim == 2
        #return (vis * self.vbias[nax, :]).sum(1) + \
        #       (hid * self.hbias[nax, :]).sum(1) + \
        #           (vis[:, :, nax] * self.weights[nax, :, :] * hid[:, nax, :]).sum(2).sum(1)
        return gnp.dot(vis, self.vbias) + \
               gnp.dot(hid, self.hbias) + \
               gnp.sum(vis * gnp.dot(hid, self.weights.T), 1)

    def free_energy_vis(self, vis):
        #vis_term = (self.vbias[nax, :] * vis).sum(1)
        vis_term = gnp.dot(vis, self.vbias)
        #hid_term = np.logaddexp(0., self.hid_inputs(vis)).sum(1)
        hid_term = gnp.sum(gnp.log_1_plus_exp(self.hid_inputs(vis)), 1)
        return vis_term + hid_term

    def free_energy_hid(self, hid):
        #hid_term = (self.hbias[nax, :] * hid).sum(1)
        hid_term = gnp.dot(hid, self.hbias)
        #vis_term = np.logaddexp(0., self.vis_inputs(hid)).sum(1)
        vis_term = gnp.sum(gnp.log_1_plus_exp(self.vis_inputs(hid)), 1)
        return hid_term + vis_term


    def cond_vis(self, vis, redundant=False):
        expect_hid = rbm_utils.sigmoid(self.hid_inputs(vis))
        if redundant:
            return RedundantMoments.from_activations(vis, expect_hid)
        else:
            return Moments.from_activations(vis, expect_hid)

    def cond_hid(self, hid, redundant=False):
        expect_vis = rbm_utils.sigmoid(self.vis_inputs(hid))
        if redundant:
            return RedundantMoments.from_activations(expect_vis, hid)
        else:
            return Moments.from_activations(expect_vis, hid)

    def copy(self):
        return RBM(self.vbias.copy(), self.hbias.copy(), self.weights.copy())

    @staticmethod
    def from_moments(moments, weights_std=0.):
        """Initialize an RBM so the visible and hidden biases match the given moments and the weights are
        set to small random values."""
        assert isinstance(moments, Moments)
        assert np.allclose(moments.expect_prod.as_numpy_array(),
                           gnp.outer(moments.expect_vis, moments.expect_hid).as_numpy_array())
        vbias = gnp.log(moments.expect_vis) - gnp.log(1. - moments.expect_vis)
        hbias = gnp.log(moments.expect_hid) - gnp.log(1. - moments.expect_hid)
        assert np.all(np.isfinite(vbias.as_numpy_array())) and np.all(np.isfinite(hbias.as_numpy_array()))
        
        if weights_std > 0.:
            weights = gnp.garray(np.random.normal(0., weights_std, size=(vbias.size, hbias.size)))
        else:
            weights = gnp.zeros((vbias.size, hbias.size))

        return RBM(vbias, hbias, weights)

    def random_state(self, num):
        return RBMState.random_init(self, num)

    def zero_update(self):
        return Update.zeros(self.nvis, self.nhid)

    def zero_moments(self):
        return Moments.zeros(self.nvis, self.nhid)

    def init_partition_function(self):
        """Compute the partition function assuming the weights are zero, i.e. all units
        are independent."""
        assert np.allclose(self.weights.as_numpy_array(), 0.)
        return gnp.sum(gnp.log_1_plus_exp(self.vbias)) + gnp.sum(gnp.log_1_plus_exp(self.hbias))

    def init_samples(self, num):
        """Generate exact samples from the model assuming the weights are all zero, i.e.
        all units are independent."""
        assert np.allclose(self.weights.as_numpy_array(), 0.)
        vis = rbm_utils.sample_units(gnp.outer(gnp.ones(num), self.vbias))
        hid = rbm_utils.sample_units(gnp.outer(gnp.ones(num), self.hbias))
        return RBMState(vis, hid)

    def dot_product(self, moments):
        """Compute the dot product of the natural parameter representation with the moments"""
        return gnp.dot(self.vbias, moments.expect_vis) + \
               gnp.dot(self.hbias, moments.expect_hid) + \
               gnp.sum(self.weights * moments.expect_prod)

    def sgd_update(self, pos_moments, neg_moments, lrates, weight_decay=0.):
        update = Update.from_moments(pos_moments) - Update.from_moments(neg_moments)
        update.weights -= weight_decay * self.weights
        self += lrates * update
        

    def update_from_moments(self, moments):
        #return Update(moments.expect_vis, moments.expect_hid, moments.expect_prod)
        return Update.from_moments(moments)



class Update(schemas.ArrayStruct):
    fields = ['vbias', 'hbias', 'weights']
    check_op_class = False
    require_garrays = True
    
    def copy(self):
        return Update(self.vbias.copy(), self.hbias.copy(), self.weights.copy())

    @staticmethod
    def zeros(nvis, nhid):
        return Update(gnp.zeros(nvis), gnp.zeros(nhid), gnp.zeros((nvis, nhid)))

    @staticmethod
    def from_moments(moments):
        return Update(moments.expect_vis, moments.expect_hid, moments.expect_prod)

class Moments(schemas.ArrayStruct):
    fields = ['expect_vis', 'expect_hid', 'expect_prod']
    check_op_class = False
    require_garrays = True
    
    def __init__(self, *args, **kwargs):
        schemas.ArrayStruct.__init__(self, *args, **kwargs)
        self.nvis = self.expect_vis.size
        self.nhid = self.expect_hid.size

    @classmethod
    def from_activations(cls, vis, hid):
        expect_vis = vis.mean(0)
        expect_hid = hid.mean(0)
        expect_prod = gnp.dot(vis.T, hid) / vis.shape[0]
        return cls(expect_vis, expect_hid, expect_prod)

    @classmethod
    def from_independent(cls, expect_vis, expect_hid):
        return cls(expect_vis, expect_hid, gnp.outer(expect_vis, expect_hid))

    @classmethod
    def from_data_base_rates(cls, vis, nhid):
        expect_vis = vis.mean(0)
        expect_hid = 0.5 * gnp.ones(nhid)
        return cls.from_independent(expect_vis, expect_hid)

    @classmethod
    def uniform(cls, nvis, nhid):
        return cls.from_independent(0.5 * gnp.ones(nvis), 0.5 * gnp.ones(nhid))

    @classmethod
    def zeros(cls, nvis, nhid):
        return cls.from_independent(gnp.zeros(nvis), gnp.zeros(nhid))

    def transpose(self):
        return self.__class__(self.expect_hid, self.expect_vis, self.expect_prod.T)

    def smooth_old(self, eps=0.001):
        moments = (1. - eps) ** 2 * self
        moments += eps * (1. - eps) * self.__class__.from_independent(0.5 * gnp.ones(moments.expect_vis.size), moments.expect_hid)
        moments += eps * (1. - eps) * self.__class__.from_independent(moments.expect_vis, 0.5 * gnp.ones(moments.expect_hid.size))
        moments += eps ** 2 * self.__class__.uniform(moments.expect_vis.size, moments.expect_hid.size)
        return moments

    def smooth(self, eps=0.001):
        moments = (1. - eps) ** 2 * self
        moments += eps * (1. - eps) * self.__class__.from_independent(0.5 * gnp.ones(moments.expect_vis.size), self.expect_hid)
        moments += eps * (1. - eps) * self.__class__.from_independent(self.expect_vis, 0.5 * gnp.ones(moments.expect_hid.size))
        moments += eps ** 2 * self.__class__.uniform(moments.expect_vis.size, moments.expect_hid.size)
        return moments

    def base_rate_moments(self):
        return self.__class__.from_independent(self.expect_vis.copy(), 0.5 * np.ones(self.expect_hid.size))

    def full_base_rate_moments(self):
        return self.__class__.from_independent(self.expect_vis.copy(), self.expect_hid.copy())

    def dot_product(self, rbm):
        """Return the dot product of the moments with the natural parameter representation
        of the RBM"""
        return gnp.dot(self.expect_vis * rbm.vbias) + \
               gnp.dot(self.expect_hid * rbm.hbias) + \
               gnp.sum(self.expect_prod * rbm.weights)

    def to_update(self):
        return Update(self.expect_vis, self.expect_hid, self.expect_prod)



class LearningRates(schemas.ArrayStruct):
    fields = ['vbias', 'hbias', 'weights']
    check_op_class = False
    require_garrays = True


