import gnumpy as gnp
import numpy as np
nax = np.newaxis
import scipy.linalg

import binary_rbms
from utils import misc, schemas



class Statistics(schemas.ArrayStruct):
    fields = ['m_unary', 'S_unary', 'm_pair', 'S_pair']
    require_garrays = True

    def __init__(self, *args, **kwargs):
        schemas.ArrayStruct.__init__(self, *args, **kwargs)
        self.nvis, self.nhid = self.S_pair.shape[:2]
        

    def unary_covariance(self, smoothing=0.):
        Sigma_unary = self.S_unary - self.m_unary[:, nax] * self.m_unary[nax, :]
        return Sigma_unary + smoothing * gnp.eye(self.nvis + self.nhid)

    def compute_regression_weights(self, smoothing=0.):
        Sigma_pair = self.S_pair - self.m_pair[:, :, :, nax] * self.m_pair[:, :, nax, :]
        Sigma_pair += smoothing * gnp.eye(3)
        beta = np.linalg.solve(Sigma_pair[:, :, :2, :2].as_numpy_array(), Sigma_pair[:, :, 2, :2].as_numpy_array())
        beta = gnp.garray(beta)
        sigma_sq = Sigma_pair[:, :, 2, 2] - (Sigma_pair[:, :, :2, :2] * beta[:, :, :, nax] * beta[:, :, nax, :]).sum(-1).sum(-1)
        return beta, sigma_sq

    @classmethod
    def from_activations(cls, v, h):
        nvis, nhid = v.shape[1], h.shape[1]
        v_mean = v.mean(0)
        h_mean = h.mean(0)
        vh = gnp.concatenate([v, h], axis=1)
        m_unary = vh.mean(0)

        S_unary = gnp.dot(vh.T, vh) / vh.shape[0]
        S_unary[:nvis, :nvis] += gnp.diagflat((v * (1. - v)).mean(0))
        S_unary[nvis:, nvis:] += gnp.diagflat((h * (1. - h)).mean(0))

        m_pair = gnp.zeros((nvis, nhid, 3))
        m_pair[:, :, 0] = v_mean[:, nax]
        m_pair[:, :, 1] = h_mean[nax, :]
        m_pair[:, :, 2] = gnp.dot(v.T, h) / h.shape[0]

        S_pair = gnp.zeros((nvis, nhid, 3, 3))
        S_pair[:] = S_unary[:nvis, nvis:, nax, nax]
        S_pair[:, :, 0, 0] = v_mean[:, nax]
        S_pair[:, :, 1, 1] = h_mean[nax, :]

        return cls(m_unary, S_unary, m_pair, S_pair)
        

    @classmethod
    def from_particles(cls, rbm, particles):
        h = particles.h
        v = rbm.vis_expectations(particles.h)
        return cls.from_activations(v, h)

        
        
class Updater:
    def __init__(self, params, stats=None):
        self.stats = stats
        self.params = params
        self.count = 0

    @staticmethod
    def from_data(rbm, v, params):
        h = rbm.hid_expectations(v)
        stats = Statistics.from_activations(v, h)
        return Updater(params, stats)

    def recompute(self, rbm, particles):
        if self.count % self.params.update_stats_every == 0:
            new_stats = Statistics.from_particles(rbm, particles)
            if self.stats is not None:
                lam = self.params.update_stats_every / self.params.timescale
                self.stats = (1. - lam) * self.stats + lam * new_stats
            else:
                self.stats = new_stats

        if self.count % self.params.recompute_every == 0:
            Sigma = self.stats.unary_covariance(self.params.smoothing)
            self.Lambda = np.linalg.inv(Sigma.as_numpy_array())
            self.beta, self.sigma_sq = self.stats.compute_regression_weights(self.params.smoothing)

        self.count += 1

    def apply_update(self, pos_moments, neg_moments, rbm, weight_decay, lrate):
        assert np.allclose(lrate.vbias, lrate.hbias)

        if self.count < self.params.start_after:
            rbm.sgd_update(pos_moments, neg_moments, lrate)
            return

        # base rates
        ds = gnp.concatenate([pos_moments.expect_vis - neg_moments.expect_vis,
                              pos_moments.expect_hid - neg_moments.expect_hid])
        dbias = lrate.vbias * gnp.dot(self.Lambda, ds.as_numpy_array())
        da, db = dbias[:rbm.nvis], dbias[rbm.nvis:]

        residuals = pos_moments.expect_prod - neg_moments.expect_prod + \
                    -weight_decay * rbm.weights + \
                    -self.beta[:, :, 0] * (pos_moments.expect_vis - neg_moments.expect_vis)[:, nax] + \
                    -self.beta[:, :, 1] * (pos_moments.expect_hid - neg_moments.expect_hid)[nax, :]
        lam = 1. / self.sigma_sq

        dw = lrate.weights * lam * residuals
        da -= lrate.weights * (lam * residuals * self.beta[:, :, 0]).sum(1)
        db -= lrate.weights * (lam * residuals * self.beta[:, :, 1]).sum(0)

        update = binary_rbms.Update(da, db, dw)
        rbm += update


        
        
