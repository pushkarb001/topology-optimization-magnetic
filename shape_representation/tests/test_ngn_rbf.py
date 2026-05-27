import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ngn import make_grid, make_centers, ngn_density, threshold
from rbf import rbf_density, normalize

XX, YY = make_grid(40)
centers = make_centers(4)
N = len(centers)


def test_ngn_ones():
    # all weights = 1 should give rho = 1 everywhere due to normalization
    rho = ngn_density(np.ones(N), XX, YY, centers, sigma=0.2)
    assert np.allclose(rho, 1.0, atol=1e-6)


def test_ngn_zeros():
    # all weights = 0 should give rho = 0 everywhere
    rho = ngn_density(np.zeros(N), XX, YY, centers, sigma=0.2)
    assert np.allclose(rho, 0.0, atol=1e-6)


def test_ngn_output_shape():
    rho = ngn_density(np.random.rand(N), XX, YY, centers)
    assert rho.shape == XX.shape


def test_threshold_is_binary():
    rho = ngn_density(np.random.rand(N), XX, YY, centers)
    B = threshold(rho)
    assert set(np.unique(B)).issubset({0.0, 1.0})


def test_rbf_normalize():
    # after normalizing, all three RBF types should sit within [0,1]
    w = np.random.randn(N)
    for rbf_type in ['thinplate', 'multiquadric', 'gaussian']:
        rho = rbf_density(w, XX, YY, centers, rbf_type=rbf_type)
        rho_n = normalize(rho)
        assert rho_n.min() >= -1e-9 and rho_n.max() <= 1 + 1e-9


def test_rbf_output_shape():
    rho = rbf_density(np.random.randn(N), XX, YY, centers)
    assert rho.shape == XX.shape