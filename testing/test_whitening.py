import numpy as np
import gnumpy as gnp

from utils import misc
import whitening



def generate_data(m=20, n=5):
    return gnp.garray(np.random.normal(size=(m, n)))


def test_diagonal_whitener():
    """The whitened data should be unit variance."""
    np.random.seed(0)
    with misc.gnumpy_conversion_check('allow'):
        X = generate_data()
        p = whitening.DiagonalWhitener.fit(X, eps=0.)
        Xp = p.apply(X)
        assert np.allclose(np.var(Xp, 0), 1.)

def test_pca_whitener():
    """PCAWhitener test"""
    np.random.seed(0)
    K = 5
    with misc.gnumpy_conversion_check('allow'):
        X = generate_data(30, 10)
        d, Q = whitening.pca(X)
        p = whitening.PCAWhitener.fit(X, K, eps=0.)
        Xp = p.apply(X)
        Xpp = np.dot(Xp, Q)
        proj_var = np.var(Xpp, 0)
        
        # first K components should be white
        assert np.allclose(proj_var[:K], 1.)

        # K + 1st component should be white
        assert np.allclose(proj_var[K], 1.)

        # remaining components should have decreasing variance
        assert np.allclose(proj_var[K:], np.sort(proj_var[K:])[::-1])


def test_iso_pca_whitener():
    """IsoPCAWhitener should agree with applying DiagonalWhitener and PCAWhitener in sequence."""
    np.random.seed(0)
    K = 2
    with misc.gnumpy_conversion_check('allow'):
        X = generate_data()
        p = whitening.DiagonalWhitener.fit(X, eps=0.)
        Xp = p.apply(X)
        pp = whitening.PCAWhitener.fit(Xp, K, eps=0.)
        Xpp = pp.apply(Xp)

        ppp = whitening.IsoPCAWhitener.fit(X, K=K, eps=0.)
        Xppp = ppp.apply(X)
        assert np.allclose(Xpp, Xppp)

def test_apply_to_gradient():
    for version in ['diagonal', 'pca']:
        yield misc.Callable(check_apply_to_gradient, version)

def check_apply_to_gradient(version):
    """apply_to_gradient for DiagonalWhitener and PCAWhitener should match the result of calling apply twice."""
    np.random.seed(0)
    K = 2
    with misc.gnumpy_conversion_check('allow'):
        X = generate_data()

        if version == 'diagonal':
            p = whitening.DiagonalWhitener.fit(X)
        elif version == 'pca':
            p = whitening.PCAWhitener.fit(X, K=K)
        else:
            raise RuntimeError('Unknown version: {}'.format(version))

        Xp1 = p.apply_to_gradient(X)
        Xp2 = p.apply(p.apply(X))
        assert np.max(np.abs(Xp1 - Xp2)) < 1e-5     # np.allclose fails because of numerical instability




            
        



        
        
        
    


    





