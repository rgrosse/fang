import gnumpy as gnp

from utils import schemas





class CenteringTrickAvgUpdater:
    def __init__(self, params, expect_vis=None, expect_hid=None):
        self.params = params
        self.expect_vis = expect_vis
        self.expect_hid = expect_hid

    @classmethod
    def from_data(cls, rbm, v, params):
        v = gnp.garray(v)
        h = rbm.hid_expectations(v)
        expect_vis = v.mean(0)
        expect_hid = h.mean(0)
        return cls(params, expect_vis, expect_hid)

    def recompute(self, rbm, particles):
        lam = 1. / self.params.timescale
        self.expect_vis = (1. - lam) * self.expect_vis + \
                          lam * particles.v.mean(0)
        self.expect_hid = (1. - lam) * self.expect_hid + \
                          lam * particles.h.mean(0)
        

    def apply_update(self, pos_moments, neg_moments, rbm, weight_decay, lrates):
        pos_prods = pos_moments.expect_prod + \
                    -gnp.outer(pos_moments.expect_vis, self.expect_hid) + \
                    -gnp.outer(self.expect_vis, pos_moments.expect_hid) + \
                    gnp.outer(self.expect_vis, self.expect_hid)
        neg_prods = neg_moments.expect_prod + \
                    -gnp.outer(neg_moments.expect_vis, self.expect_hid) + \
                    -gnp.outer(self.expect_vis, neg_moments.expect_hid) + \
                    gnp.outer(self.expect_vis, self.expect_hid)

        weights_update = lrates.weights * (pos_prods - neg_prods) + \
                         -lrates.weights * weight_decay * rbm.weights
        vbias_update = lrates.vbias * (pos_moments.expect_vis - neg_moments.expect_vis) + \
                       -gnp.dot(weights_update, neg_moments.expect_hid)
        hbias_update = lrates.hbias * (pos_moments.expect_hid - neg_moments.expect_hid) + \
                       -gnp.dot(neg_moments.expect_vis, weights_update)

        rbm.vbias += vbias_update
        rbm.hbias += hbias_update
        rbm.weights += weights_update


        




