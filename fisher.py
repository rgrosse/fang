import gnumpy as gnp
import numpy as np
nax = np.newaxis

from utils import misc, schemas

def logdet(Sigma):
    return np.linalg.slogdet(Sigma)[1]


def gaussian_kldiv(Sigma_p, Sigma_q):
    """Compute KL(p || q) for zero-centered Gaussians."""
    D = Sigma_p.shape[0]
    Lambda_q = np.linalg.inv(Sigma_q)
    return -0.5 * logdet(Sigma_p) + \
           -0.5 * D + \
           0.5 * logdet(Sigma_q) + \
           0.5 * np.sum(Sigma_p * Lambda_q)

def gaussian_kldiv_info(Lambda_p, Lambda_q):
    """Compute KL(p || q) for zero-centered Gaussians in information form."""
    D = Lambda_p.shape[0]
    Sigma_p = np.linalg.inv(Lambda_p)
    return 0.5 * logdet(Lambda_p) + \
           -0.5 * D + \
           -0.5 * logdet(Lambda_q) + \
           0.5 * np.sum(Sigma_p * Lambda_q)


class PartialFisherInverse(schemas.Struct):
    """Compact representation of the graphical model approximation to an inverse Fisher
    information matrix. Lambda_v_h is a dense PSD matrix of dimension nvis + nhid representing
    the inverse marginal covariance of v and h. Lambda_vh_cond represents the conditional
    distributions of pairwise (m_ij) given unary. It is of size nvis x nhid x 3 x 3, where the
    [i, j, :, :] block represents the rank-one quadratic form (m_ij - a v_i - b h_j)^2 / sigma_ij^2
    and the 3 indices correspond to v_i, h_j, and m_ij, respectively."""
    fields = ['Lambda_v_h', 'Lambda_vh_cond']

    def __init__(self, Lambda_v_h, Lambda_vh_cond):
        schemas.Struct.__init__(self, Lambda_v_h, Lambda_vh_cond)
        self.nvis, self.nhid = Lambda_vh_cond.shape[:2]

    def to_full(self):
        """Compute the full precision matrix."""
        D = self.nvis + self.nhid + self.nvis * self.nhid
        G_inv = np.zeros((D, D))

        G_inv[:self.nvis+self.nhid, :self.nvis+self.nhid] = self.Lambda_v_h

        for i in range(self.nvis):
            for j in range(self.nhid):
                vis_idx = i
                hid_idx = self.nvis + j
                vishid_idx = self.nvis + self.nhid + self.nhid * i + j
                idxs = np.array([vis_idx, hid_idx, vishid_idx])
                G_inv[idxs[:, nax], idxs[nax, :]] += self.Lambda_vh_cond[i, j, :, :]

        return G_inv

    @staticmethod
    def random(nvis, nhid):
        """Return a random instance for testing purposes."""
        Lambda_v_h = misc.random_psd(nvis + nhid)
        temp = np.random.normal(size=(nvis, nhid, 3))
        Lambda_vh_cond = temp[:, :, :, nax] * temp[:, :, nax, :]
        return PartialFisherInverse(Lambda_v_h, Lambda_vh_cond)
        


class RandomConnectivityInverse(schemas.Struct):
    """Compact representation of the graphical model approximation to an inverse Fisher
    information matrix where the pairwise statistics are assigned random parents. vis_idxs
    and hid_idxs are nvis x nhid matrices which give the indices of the visible and hidden
    parents of a given pairwise statistic. Lambda_v_h and Lambda_vh_cond have the same
    meaning as in PartialFisherInverse."""
    fields = ['Lambda_v_h', 'Lambda_vh_cond', 'vis_idxs', 'hid_idxs']

    def __init__(self, *args, **kwargs):
        schemas.Struct.__init__(self, *args, **kwargs)
        self.nvis, self.nhid = self.Lambda_vh_cond.shape[:2]

    @staticmethod
    def compute_from_G(G, nvis, nhid, vis_idxs=None, hid_idxs=None):
        """Fit the model to a full Fisher information matrix G."""
        if vis_idxs is None:
            vis_idxs = np.random.randint(0, nvis, size=(nvis, nhid))
        if hid_idxs is None:
            hid_idxs = np.random.randint(0, nhid, size=(nvis, nhid))

        Lambda_v_h = np.linalg.inv(G[:nvis+nhid, :nvis+nhid])
        
        Lambda_vh_cond = np.zeros((nvis, nhid, 3, 3))
        for i in range(nvis):
            for j in range(nhid):
                child_vis_idx = i
                child_hid_idx = j
                child_vishid_idx = nvis + nhid + nhid * i + j
                parent_vis_idx = vis_idxs[child_vis_idx, child_hid_idx]
                parent_hid_idx = nvis + hid_idxs[child_vis_idx, child_hid_idx]
                G_idxs = np.array([parent_vis_idx, parent_hid_idx, child_vishid_idx])

                G_block = G[G_idxs[:, nax], G_idxs[nax, :]]
                Lambda_all = np.linalg.inv(G_block)
                Lambda_parents = np.linalg.inv(G_block[:2, :2])
                Lambda_vh_cond[child_vis_idx, child_hid_idx, :, :] += Lambda_all
                Lambda_vh_cond[child_vis_idx, child_hid_idx, :2, :2] -= Lambda_parents

        return RandomConnectivityInverse(Lambda_v_h, Lambda_vh_cond, vis_idxs, hid_idxs)
        

        

    def to_full(self):
        """Compute the full precision matrix."""
        D = self.nvis + self.nhid + self.nvis * self.nhid
        G_inv = np.zeros((D, D))

        G_inv[:self.nvis+self.nhid, :self.nvis+self.nhid] = self.Lambda_v_h

        for i in range(self.nvis):
            for j in range(self.nhid):
                child_vis_idx = i
                child_hid_idx = j
                child_vishid_idx = self.nvis + self.nhid + self.nhid * i + j
                parent_vis_idx = self.vis_idxs[child_vis_idx, child_hid_idx]
                parent_hid_idx = self.nvis + self.hid_idxs[child_vis_idx, child_hid_idx]
                G_inv_idxs = np.array([parent_vis_idx, parent_hid_idx, child_vishid_idx])
                
                G_inv[G_inv_idxs[:, nax], G_inv_idxs[nax, :]] += self.Lambda_vh_cond[child_vis_idx, child_hid_idx, :, :]

        return G_inv

    @staticmethod
    def random(nvis, nhid):
        """Return a random instance for testing purposes."""
        vis_idxs = np.random.randint(0, nvis, size=(nvis, nhid))
        hid_idxs = np.random.randint(0, nhid, size=(nvis, nhid))
        Lambda_v_h = misc.random_psd(nvis + nhid)
        temp = np.random.normal(size=(nvis, nhid, 3))
        Lambda_vh_cond = temp[:, :, :, nax] * temp[:, :, nax, :]
        return RandomConnectivityInverse(Lambda_v_h, Lambda_vh_cond, vis_idxs, hid_idxs)
    



class PartialFisherInformation(schemas.Struct):
    """Compact representation of the parts of the Fisher information matrix needed to estimate
    the graphical model. Sigma_v_h is a PSD matrix of dimension nvis x nhid, representing
    the covariance of the unary statistics. Sigma_v_h_vh is an nvis x nhid x 3 x 3 matrix,
    where the [i, j, :, :] block represents the marginal covariance of v[i], h[j], and v[i] * h[j].
    Note that this representation is redundant, since the covariance of v[i] and h[j] is
    stored in both Sigma_v_h and Sigma_v_h_vh."""
    fields = ['Sigma_v_h', 'Sigma_v_h_vh']

    @staticmethod
    def from_full(G, nvis, nhid):
        """Extract the relevant entries from the full Fisher information."""
        Sigma_v_h = G[:nvis+nhid, :nvis+nhid]

        Sigma_v_h_vh = np.zeros((nvis, nhid, 3, 3))
        for i in range(nvis):
            for j in range(nhid):
                vis_idx = i
                hid_idx = nvis + j
                vishid_idx = nvis + nhid + nhid * i + j
                idxs = np.array([vis_idx, hid_idx, vishid_idx])
                Sigma_v_h_vh[i, j, :, :] = G[idxs[:, nax], idxs[nax, :]]

        return PartialFisherInformation(Sigma_v_h, Sigma_v_h_vh)
                

    def compute_compact_precision(self):
        Lambda_v_h = np.linalg.inv(self.Sigma_v_h)

        Lambda_vh_cond = np.linalg.inv(self.Sigma_v_h_vh)
        Lambda_v_h_marg = np.linalg.inv(self.Sigma_v_h_vh[:, :, :2, :2])
        Lambda_vh_cond[:, :, :2, :2] -= Lambda_v_h_marg

        return PartialFisherInverse(Lambda_v_h, Lambda_vh_cond)
    
        

class RegressionWeights(schemas.Struct):
    """The regression parameters needed for the CPDs in the Gaussian graphical model."""
    fields = ['beta', 'sigma_sq']

    def __init__(self, *args, **kwargs):
        schemas.Struct.__init__(self, *args, **kwargs)
        self.nvis, self.nhid = self.beta.shape[:2]

    @staticmethod
    def from_centering_trick(G, s, nvis, nhid):
        """Assign the regression weights using the centering trick, and compute
        the conditional covariances matrices from the full Fisher information."""
        beta = np.zeros((nvis, nhid, 2))
        sigma_sq = np.zeros((nvis, nhid))

        for i in range(nvis):
            for j in range(nhid):
                vis_idx = i
                hid_idx = nvis + j
                vishid_idx = nvis + nhid + nhid * i + j
                idxs = np.array([vis_idx, hid_idx, vishid_idx])
                Sigma = G[idxs[:, nax], idxs[nax, :]]

                beta[i, j, :] = [s[hid_idx], s[vis_idx]]
                sigma_sq[i, j] = Sigma[2, 2] + \
                                 -2. * np.dot(beta[i, j, :], Sigma[2, :2]) + \
                                 np.dot(beta[i, j, :], np.dot(Sigma[:2, :2], beta[i, j, :]))

        return RegressionWeights(beta, sigma_sq)

        

    @staticmethod
    def from_maximum_likelihood(G, nvis, nhid):
        """Compute the maximum likelihood parameter estimates from the full covariance."""
        beta = np.zeros((nvis, nhid, 2))
        sigma_sq = np.zeros((nvis, nhid))

        for i in range(nvis):
            for j in range(nhid):
                vis_idx = i
                hid_idx = nvis + j
                vishid_idx = nvis + nhid + nhid * i + j
                idxs = np.array([vis_idx, hid_idx, vishid_idx])
                Sigma = G[idxs[:, nax], idxs[nax, :]]

                beta[i, j, :] = np.linalg.solve(Sigma[:2, :2], Sigma[2, :2])
                sigma_sq[i, j] = Sigma[2, 2] - np.dot(Sigma[2, :2], beta[i, j, :])
                
        return RegressionWeights(beta, sigma_sq)

        

    def compute_Lambda(self):
        """Compute the quadratic form representation of the CPDs."""
        D = self.nvis + self.nhid + self.nvis * self.nhid
        Lambda = np.zeros((D, D))

        for i in range(self.nvis):
            for j in range(self.nhid):
                vis_idx = i
                hid_idx = self.nvis + j
                vishid_idx = self.nvis + self.nhid + self.nhid * i + j
                idxs = np.array([vis_idx, hid_idx, vishid_idx])

                v = np.array([self.beta[i, j, 0], self.beta[i, j, 1], -1])
                Lambda[idxs[:, nax], idxs[nax, :]] += np.outer(v, v) / self.sigma_sq[i, j]

        return Lambda



def correlation_fraction(g, s, nvis, nhid):
    with misc.gnumpy_conversion_check('allow'):
        expect_vis = s[:nvis]
        expect_hid = s[nvis:nvis+nhid]
        da = g[:nvis]
        db = g[nvis:nvis+nhid]
        dW = g[nvis+nhid:].reshape((nvis, nhid))

        first_order_expl = gnp.outer(da, expect_hid) + gnp.outer(expect_vis, db)
        first_order_norm = gnp.sum(da**2) + gnp.sum(db**2) + gnp.sum(first_order_expl**2)

        dcorr = dW - first_order_expl
        dcorr_norm = gnp.sum(dcorr**2)
        g_norm = gnp.sum(g**2)
        #return first_order_norm, dcorr_norm, g_norm
        return dcorr_norm / (dcorr_norm + first_order_norm)
    





