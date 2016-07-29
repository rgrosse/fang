import numpy as np
nax = np.newaxis
import pylab
import scipy.linalg

import fisher
import tractable
from utils import misc, storage
from visuals import misc as vm

SMALL_RBM_FILE = 'trained_rbms/mnist_small.pk'


def load_small_rbm():
    return storage.load(SMALL_RBM_FILE)

    
def compute_G(rbm=None):
    if rbm is None:
        rbm = load_small_rbm()
    return tractable.exact_fisher_information(rbm, batch_units=10, show_progress=True, return_mean=True)



def visualize_row(g, nvis=196, nhid=20, title=''):
    ncols = np.sqrt(nvis).astype(int)

    title = 'vishid'
    vishid = g[nvis+nhid:].reshape((nvis, nhid))
    imgs = [vishid[:, j].reshape((ncols, ncols)) for j in range(nhid)]
    pylab.matshow(vm.pack(imgs), cmap='gray')
    pylab.title(title)


def show_eigenvectors(G, num=5):
    d, Q = scipy.linalg.eigh(G)
    d = d[::-1]
    Q = Q[:, ::-1]

    for eignum in range(num):
        visualize_row(Q[:, eignum], title='Eigenvector {}'.format(eignum+1))




def plot_eigenspectrum(G, s, nvis, nhid):
    with misc.gnumpy_conversion_check('allow'):
        dim = G.shape[0]
        d, Q = scipy.linalg.eigh(G)
        d = d[::-1]
        Q = Q[:, ::-1]

        pts = np.unique(np.floor(np.logspace(0., np.log10(dim-1), 500)).astype(int)) - 1

        cf = [fisher.correlation_fraction(Q[:, i], s, nvis, nhid) for i in pts]

        pylab.figure()
        pylab.subplot(2, 1, 1)
        pylab.loglog(range(1, dim+1), d, 'b-', lw=2.)
        pylab.xticks([])
        pylab.yticks(fontsize='large')

        pylab.subplot(2, 1, 2)
        pylab.semilogx(pts+1, cf, 'r-', lw=2.)
        pylab.xticks(fontsize='x-large')
        pylab.yticks(fontsize='large')


def generate_eigenvalue_figure(G=None, s=None):
    if G is None:
        G, s = compute_G()

    # Figure 2, eigenspectrum plots
    plot_eigenspectrum(G, s, 196, 20)

    # Figure 2, top eigenvector
    show_eigenvectors(G)
        


def num_params_symmetric(d):
    return d * (d+1) // 2
    
    
def compare_kldiv(G=None, s=None, nvis=196, nhid=20, eps=1e-6):
    if G is None:
        G, s = compute_G()
        
    D = G.shape[0]
    assert D == nvis + nhid + nvis * nhid

    results = []
    
    G = G + eps * np.eye(D)

    print 'isotropic...'
    G_iso = np.mean(np.diag(G)) * np.eye(D)
    results.append(('isotropic', fisher.gaussian_kldiv(G, G_iso), 1))

    print 'diagonal...'
    G_diag = np.diag(np.diag(G))
    results.append(('diagonal', fisher.gaussian_kldiv(G, G_diag), D))

    print 'block diagonal...'
    G_v_v = G[:nvis, :nvis]
    G_h_h = G[nvis:nvis+nhid, nvis:nvis+nhid]
    G_vh_h = G[nvis+nhid:, nvis:nvis+nhid].reshape((nvis, nhid, nhid))
    G_vh_vh = G[nvis+nhid:, nvis+nhid:].reshape((nvis, nhid, nvis, nhid))

    G_block_diag_rsh = np.zeros((nvis + 1, nhid, nvis + 1, nhid))
    for j in range(nhid):
        G_block_diag_rsh[1:, j, 1:, j] = G_vh_vh[:, j, :, j]
        G_block_diag_rsh[0, j, 1:, j] = G_vh_h[:, j, j]
        G_block_diag_rsh[1:, j, 0, j] = G_vh_h[:, j, j]
        G_block_diag_rsh[0, j, 0, j] = G_h_h[j, j]

    G_block_diag = np.zeros(G.shape)
    G_block_diag[:nvis, :nvis] = G_v_v
    G_block_diag[nvis:, nvis:] = G_block_diag_rsh.reshape(((nvis+1) * nhid, (nvis+1) * nhid))

    num_params = num_params_symmetric(nvis) + nhid * num_params_symmetric(nvis + 1)
    results.append(('block diag', fisher.gaussian_kldiv(G, G_block_diag), num_params))
    
    for rank in [50, 200]:
        print 'rank', rank, '...'
        d, Q = scipy.linalg.eigh(G)
        d = d[::-1]
        Q = Q[:, ::-1]

        lam = np.mean(d[rank:])
        G_low_rank = np.dot(Q[:, :rank] * (d[:rank] - lam), Q[:, :rank].T)
        G_low_rank += lam * np.eye(D)
        num_params = rank * D + 1
        results.append(('rank {}'.format(rank), fisher.gaussian_kldiv(G, G_low_rank), num_params))

    print 'rand connect...'
    inverse = fisher.RandomConnectivityInverse.compute_from_G(G, nvis, nhid)
    G_inv = inverse.to_full()
    G_random = np.linalg.inv(G_inv)
    num_params = num_params_symmetric(nvis + nhid) + 3 * nvis * nhid
    results.append(('rand conn', fisher.gaussian_kldiv(G, G_random), num_params))

    print 'centering...'
    rw = fisher.RegressionWeights.from_centering_trick(G, s, nvis, nhid)
    Lambda_ctr = rw.compute_Lambda()
    Lambda_ctr[:nvis+nhid, :nvis+nhid] += np.diag(1. / np.diag(G[:nvis+nhid, :nvis+nhid]))
    G_ctr = np.linalg.inv(Lambda_ctr)
    num_params = 2. * nvis + 2. * nhid + nvis * nhid
    results.append(('centering', fisher.gaussian_kldiv(G, G_ctr), num_params))
    
    print 'FANG...'
    partial = fisher.PartialFisherInformation.from_full(G, nvis, nhid)
    inverse = partial.compute_compact_precision()
    G_inv = inverse.to_full()
    G_factorized = np.linalg.inv(G_inv)
    num_params = num_params_symmetric(nvis + nhid) + 3 * nvis * nhid
    results.append(('FANG', fisher.gaussian_kldiv(G, G_factorized), num_params))

    return results



