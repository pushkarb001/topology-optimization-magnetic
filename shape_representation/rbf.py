import numpy as np


def rbf_density(weights, XX, YY, centers, rbf_type='multiquadric', c=0.3):
    # unlike NGN there is no normalization here, so the field is unbounded
    # always call normalize() on the output before passing to threshold()
    N = len(centers)
    phi = np.zeros((N, *XX.shape))

    for k, (cx, cy) in enumerate(centers):
        # RBF formulas use actual distance r, not squared distance
        r = np.sqrt((XX - cx)**2 + (YY - cy)**2)

        if rbf_type == 'thinplate':
            # formula is r² log(r) — blows up at r=0 so we clamp those points to 0
            phi[k] = np.where(r < 1e-12, 0.0, r**2 * np.log(r + 1e-12))

        elif rbf_type == 'multiquadric':
            # c controls the spread — larger c means each center influences a wider area
            phi[k] = np.sqrt(r**2 + c**2)

        elif rbf_type == 'gaussian':
            # most local of the three — influence drops off quickly beyond distance c
            phi[k] = np.exp(-r**2 / c**2)

        else:
            raise ValueError(f"rbf_type '{rbf_type}' not recognised. "
                             f"Use thinplate, multiquadric, or gaussian.")

    rho = np.einsum('k,kmn->mn', weights, phi)
    return rho


def normalize(rho):
    # rescales the field to [0,1] so tau=0.5 is always a meaningful threshold
    mn, mx = rho.min(), rho.max()
    return (rho - mn) / (mx - mn + 1e-12)