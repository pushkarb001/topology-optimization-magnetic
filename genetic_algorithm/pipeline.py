import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shape_representation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'femm_automation'))

from ngn import make_grid, make_centers, ngn_density, threshold
from rbf import rbf_density, normalize
from simulator import simulate
from ga import run_ga


def make_fitness_fn(method='ngn', N_side=4, sigma=0.2, tau=0.5, grid_shape=(15, 5)):
    XX, YY  = make_grid(60)
    centers = make_centers(N_side)

    def fitness_fn(weights):
        if method == 'ngn':
            rho = ngn_density(weights, XX, YY, centers, sigma)
        else:
            rho = rbf_density(weights, XX, YY, centers, rbf_type='multiquadric')
            rho = normalize(rho)

        B_full = threshold(rho, tau)

        # resize 60x60 binary map down to the FEMM grid size
        from skimage.transform import resize
        B = resize(B_full, grid_shape, order=0, anti_aliasing=False)
        B = (B > 0.5).astype(float)

        return simulate(B, verbose=False)

    return fitness_fn


if __name__ == '__main__':
    print("starting NGN + GA + FEMM pipeline")
    print("pop_size=10, generations=5 — just checking it runs end to end\n")

    fitness_fn = make_fitness_fn(method='ngn', N_side=4, sigma=0.2, tau=0.5)

    best_w, best_f, hist_best, hist_avg = run_ga(
        fitness_fn    = fitness_fn,
        n_weights     = 16,
        pop_size      = 10,
        n_generations = 5,
        mutation_rate = 0.1,
        seed          = 42
    )

    print(f'\nBest flux linkage: {best_f:.6f} Wb')
    print(f'Best weights: {best_w.round(4)}')

    plt.figure(figsize=(8, 4))
    plt.plot(hist_best, label='best')
    plt.plot(hist_avg,  label='average')
    plt.xlabel('Generation')
    plt.ylabel('Flux linkage (Wb)')
    plt.title('GA convergence — NGN + FEMM')
    plt.legend()
    plt.tight_layout()
    plt.savefig('convergence_ngn.png')
    plt.show()