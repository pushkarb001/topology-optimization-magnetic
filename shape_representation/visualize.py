import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

from ngn import make_grid, ngn_density, threshold
from rbf import rbf_density, normalize


def plot_density(weights, centers, sigma=0.2, tau=0.5, title=''):
    XX, YY = make_grid(60)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # each kernel scaled by its weight — shows which centers dominate before summing
    for k, (cx, cy) in enumerate(centers):
        r2 = (XX - cx)**2 + (YY - cy)**2
        G = np.exp(-r2 / (2 * sigma**2))
        axes[0].contourf(XX, YY, G * weights[k], alpha=0.35, cmap='RdBu')
    axes[0].scatter(centers[:, 0], centers[:, 1], c='black', s=18, zorder=5)
    axes[0].set_title('Individual kernels')
    axes[0].set_aspect('equal')

    # full NGN density field after all kernels are summed and normalized
    rho_ngn = ngn_density(weights, XX, YY, centers, sigma)
    cf = axes[1].contourf(XX, YY, rho_ngn, levels=50, cmap='viridis')
    plt.colorbar(cf, ax=axes[1])
    axes[1].set_title('NGN density field')
    axes[1].set_aspect('equal')

    # same weights through RBF — compare how differently it represents the same vector
    rho_rbf = rbf_density(weights, XX, YY, centers, rbf_type='multiquadric')
    rho_rbf_n = normalize(rho_rbf)
    cf2 = axes[2].contourf(XX, YY, rho_rbf_n, levels=50, cmap='plasma')
    plt.colorbar(cf2, ax=axes[2])
    axes[2].set_title('RBF density field (multiquadric)')
    axes[2].set_aspect('equal')

    # binary iron/air map after thresholding
    B = threshold(rho_ngn, tau)
    axes[3].imshow(B, cmap='gray_r', origin='lower', extent=[0, 1, 0, 1])
    axes[3].set_title(f'Binary map (tau={tau})')
    axes[3].set_aspect('equal')

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def check_connectivity(B):
    # n_components > 1 means there are multiple disconnected iron regions, which is bad for manufacturing
    labeled, n_components = label(B)
    return n_components <= 1, n_components