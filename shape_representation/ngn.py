import numpy as np

def make_grid(M=60):
    # creates an MxM grid of sample points over the unit square [0,1]x[0,1]
    # M=60 gives 3600 points which is fine resolution for visualization
    x = np.linspace(0, 1, M)
    y = np.linspace(0, 1, M)
    return np.meshgrid(x, y)


def make_centers(N_side=4):
    # places kernel centers on a regular N_side x N_side sub-grid
    # N_side=4 gives 16 centers, so the GA chromosome will have 16 weights
    # increasing N_side gives finer shape control but makes the GA search harder
    c = np.linspace(0, 1, N_side)
    CX, CY = np.meshgrid(c, c)
    return np.stack([CX.ravel(), CY.ravel()], axis=1)


def ngn_density(weights, XX, YY, centers, sigma=0.2):
    # compute a Gaussian bump for each kernel center
    # sigma controls how wide each bump spreads across the domain
    N = len(centers)
    G = np.zeros((N, *XX.shape))

    for k, (cx, cy) in enumerate(centers):
        r2 = (XX - cx)**2 + (YY - cy)**2
        G[k] = np.exp(-r2 / (2 * sigma**2))

    # normalize: divide each Gaussian by the sum of all Gaussians at that point
    # after this step, summing all phi_k at any point gives exactly 1.0
    # this is what keeps rho bounded in [0,1] when weights are in [0,1]
    G_sum = G.sum(axis=0)
    G_sum = np.where(G_sum < 1e-12, 1e-12, G_sum)
    phi = G / G_sum

    # weighted sum — this is the final density field
    rho = np.einsum('k,kmn->mn', weights, phi)
    return rho


def threshold(rho, tau=0.5):
    # anything above tau becomes iron (1.0), everything else is air (0.0)
    # this binary map is what gets sent to FEMM later
    return (rho > tau).astype(float)