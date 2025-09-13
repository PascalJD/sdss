from __future__ import annotations
from typing import List, Tuple, Sequence
import math

import jax.numpy as jnp
import jax
import chex
# import matplotlib
#
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import wandb

from targets.base_target import Target


# Taken from FAB code
class Energy:
    """
    https://zenodo.org/record/3242635#.YNna8uhKjIW
    """

    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def _energy(self, x):
        raise NotImplementedError()

    def energy(self, x, temperature=None):
        assert x.shape[-1] == self._dim, "`x` does not match `dim`"
        if temperature is None:
            temperature = 1.
        return self._energy(x) / temperature

    def force(self, x, temperature=None):
        e_func = lambda x: jnp.sum(self.energy(x, temperature=temperature))
        return -jax.grad(e_func)(x)


class DoubleWellEnergy(Energy):
    def __init__(self, a: float = -0.5, b: float = -6.0, c: float = 1.):
        dim = 2
        super().__init__(dim)
        self._a = a
        self._b = b
        self._c = c

    def _energy(self, x):
        d = x[:, [0]]
        v = x[:, 1:]
        e1 = self._a * d + self._b * d ** 2 + self._c * d ** 4
        e2 = jnp.sum(0.5 * v ** 2, axis=-1, keepdims=True)
        return e1 + e2

    def log_prob(self, x):
        if len(x.shape) == 1:
            x = jnp.expand_dims(x, axis=0)
        return jnp.squeeze(-self.energy(x))

    @property
    def log_Z(self):
        log_Z_dim0 = jnp.log(11784.50927)
        log_Z_dim1 = 0.5 * jnp.log(2 * jnp.pi)
        return log_Z_dim0 + log_Z_dim1


class ManyWellEnergy(Target):
    def __init__(self, a: float = -0.5, b: float = -6.0, c: float = 1., dim=32, can_sample=False, sample_bounds=None) -> None:
        assert dim % 2 == 0
        self.n_wells = dim // 2
        self.double_well_energy = DoubleWellEnergy(a, b, c)

        log_Z = self.double_well_energy.log_Z * self.n_wells
        super().__init__(dim=dim, log_Z=log_Z, can_sample=can_sample)

        self.centre = 1.7
        self.max_dim_for_all_modes = 40  # otherwise we get memory issues on huge test set
        if self.dim < self.max_dim_for_all_modes:
            dim_1_vals_grid = jnp.meshgrid(*[jnp.array([-self.centre, self.centre]) for _ in
                                             range(self.n_wells)])
            dim_1_vals = jnp.stack([dim.flatten() for dim in dim_1_vals_grid], axis=-1)
            n_modes = 2 ** self.n_wells
            assert n_modes == dim_1_vals.shape[0]
            test_set = jnp.zeros((n_modes, dim))
            test_set = test_set.at[:, jnp.arange(dim) % 2 == 0].set(dim_1_vals)
            self.test_set = test_set
        else:
            raise NotImplementedError("still need to implement this")

        self.shallow_well_bounds = [-1.75, -1.65]
        self.deep_well_bounds = [1.7, 1.8]
        self._plot_bound = 3.

    def log_prob(self, x):
        batched = x.ndim == 2

        if not batched:
            x = x[None,]

        log_probs = jnp.sum(jnp.stack([self.double_well_energy.log_prob(x[..., i * 2:i * 2 + 2]) for
                                       i in range(self.n_wells)], axis=-1), axis=-1, keepdims=True).reshape((-1,))

        if not batched:
            log_probs = jnp.squeeze(log_probs, axis=0)
        return log_probs

    def log_prob_2D(self, x):
        """Marginal 2D pdf - useful for plotting."""
        return self.double_well_energy.log_prob(x)

    def visualise(self, samples: chex.Array = None, axes=None, show=False, prefix='') -> dict:
        """Visualise samples from the model."""
        plotting_bounds = (-3, 3)
        grid_width_n_points = 100
        fig, axs = plt.subplots(2, 2, sharex="row", sharey="row")
        samples = jnp.clip(samples, plotting_bounds[0], plotting_bounds[1])
        for i in range(2):
            for j in range(2):
                # plot contours
                def _log_prob_marginal_pair(x_2d, i, j):
                    x = jnp.zeros((x_2d.shape[0], self.dim))
                    x = x.at[:, i].set(x_2d[:, 0])
                    x = x.at[:, j].set(x_2d[:, 1])
                    return self.log_prob(x)

                xx, yy = jnp.meshgrid(
                    jnp.linspace(plotting_bounds[0], plotting_bounds[1], grid_width_n_points),
                    jnp.linspace(plotting_bounds[0], plotting_bounds[1], grid_width_n_points)
                )
                x_points = jnp.column_stack([xx.ravel(), yy.ravel()])
                log_probs = _log_prob_marginal_pair(x_points, i, j + 2)
                log_probs = jnp.clip(log_probs, -1000, a_max=None).reshape((grid_width_n_points, grid_width_n_points))
                axs[i, j].contour(xx, yy, log_probs, levels=20)

                # plot samples
                axs[i, j].plot(samples[:, i], samples[:, j + 2], "o", alpha=0.5)

                if j == 0:
                    axs[i, j].set_ylabel(f"$x_{i + 1}$")
                if i == 1:
                    axs[i, j].set_xlabel(f"$x_{j + 1 + 2}$")

        wb = {"figures/vis": [wandb.Image(fig)]}
        if show:
            plt.show()
        else:
            plt.close()

        return wb


    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        return None
    

# -----------------------------------------------------------------------------#
#                            helpers                                           #
# -----------------------------------------------------------------------------#
def _rejection_sampling(key, shape, target_logpdf,
                        proposal_sample, proposal_logpdf, log_M):
    N = int(jnp.prod(jnp.array(shape)))

    samples = jnp.empty((N, 1))                    # pre-allocate
    i       = jnp.array(0, dtype=jnp.int32)        # **scalar** counter

    def cond(state):
        i, *_ = state
        return i < N

    def body(state):
        i, key, samples = state
        key, subk1, subk2 = jax.random.split(key, 3)

        # draw candidate
        x = proposal_sample(subk1)                 # shape (1,)
        log_u = jnp.log(jax.random.uniform(subk2))

        accept_vec = log_u < (target_logpdf(x) - proposal_logpdf(x) - log_M)  # (1,)
        accept     = jnp.squeeze(accept_vec, axis=0)                          # ()  ✅ scalar

        # write into buffer only if accepted
        samples = jax.lax.cond(
            accept,
            lambda s: s.at[i].set(x),   # when True
            lambda s: s,                # when False
            samples
        )

        # increment counter — cast to scalar to keep shape 0-D
        i = i + accept.astype(jnp.int32)           # still shape ()

        return i, key, samples

    i, key, samples = jax.lax.while_loop(cond, body, (i, key, samples))
    return samples.reshape(shape + (1,)), key


# -----------------------------------------------------------------------------#
#                            Double well: dim = 1                              #
# -----------------------------------------------------------------------------#
class DoubleWell(Target):
    r"""
    log ρ(x) = - (x² - separation)²    (up to a constant)

    Two modes at ±√separation, optional global `shift`.
    """
    def __init__(self,
                 separation: float = 2.0,
                 shift: float = 0.0,
                 rejection_scaling: float = 3.0,
                 can_sample: bool = False):
        self.separation        = float(separation)
        self.shift             = float(shift)
        self.rejection_scaling = float(rejection_scaling)

        # proposal: symmetric mixture of two 1-D Gaussians
        s = math.sqrt(self.separation)
        self._prop_loc   = jnp.array([[-s], [s]]) + self.shift
        self._prop_scale = jnp.array([[1 / s], [1 / s]])

        super().__init__(dim=1, log_Z=None, can_sample=can_sample)

    # -------- unnormalised log-density ----------------------------------- #
    def unnorm_log_prob(self, x):
        x = x - self.shift
        return - (x**2 - self.separation) ** 2

    def log_prob(self, x):
        return self.unnorm_log_prob(x)

    # -------- score ------------------------------------------------------- #
    def score(self, x):
        x = x - self.shift
        return -4.0 * (x**2 - self.separation) * x

    # -------- proposal: log-pdf & sample ---------------------------------- #
    def _proposal_logpdf(self, x):
        s = math.sqrt(self.separation)
        locs  = self.shift + jnp.array([-s, s])    # shape (2,)
        scales = 1.0 / s                           # scalar
        logpdf_each = jax.scipy.stats.norm.logpdf(
            (x[..., None] - locs) * s              # broadcast, shape (..., 2)
        ) - 0.5 * jnp.log(2.0)
        return jax.scipy.special.logsumexp(logpdf_each, axis=-1)

    def _proposal_sample(self, key):
        key, subk1, subk2 = jax.random.split(key, 3)

        comp  = jax.random.bernoulli(subk1)        # Boolean
        # turn it into ±1
        sign  = jnp.where(comp, 1.0, -1.0)

        s     = math.sqrt(self.separation)
        mean  = self.shift + sign * s              # ±√sep + shift
        scale = 1.0 / s                            # shared for both modes

        return mean + scale * jax.random.normal(subk2, (1,))

    # -------- sampling ----------------------------------------------------- #
    def sample(self, key: chex.PRNGKey, shape: Sequence[int]):
        if not self.can_sample:
            raise RuntimeError("Sampling disabled (can_sample=False).")
        log_M = jnp.log(self.rejection_scaling)
        return _rejection_sampling(
            key=key,
            shape=tuple(shape),
            target_logpdf=self.unnorm_log_prob,
            proposal_sample=lambda k: self._proposal_sample(k),
            proposal_logpdf=self._proposal_logpdf,
            log_M=log_M,
        )[0]

    def visualise(self,
                samples: chex.Array,
                axes=None,
                show: bool = False,
                prefix: str = '') -> dict:
        """
        1-D plot of the double-well density and a histogram of `samples`.
        """
        # Flatten to (N,)
        samples_1d = samples.reshape(-1)

        # ------ figure scaffold ------
        fig, ax = plt.subplots()

        # Histogram of samples
        ax.hist(samples_1d, bins=100, density=True,
                alpha=0.4, label="samples")

        # Analytical pdf curve
        xs = jnp.linspace(-3.0 + self.shift, 3.0 + self.shift, 400)
        ys = jnp.exp(self.unnorm_log_prob(xs[:, None]))
        ys = ys / jnp.trapz(ys, xs)          # normalise numerically
        ax.plot(xs, ys, label="pdf")

        ax.set_xlabel(r"$x$")
        ax.set_ylabel("density")
        ax.legend()

        wb = {f"{prefix}figures/vis": [wandb.Image(fig)]}
        if show:
            plt.show()
        else:
            plt.close(fig)
        return wb


# -----------------------------------------------------------------------------#
#                     Multi-well  (d = m + n_gauss)                            #
# -----------------------------------------------------------------------------#
class MultiWell(Target):
    """
    *dim*     total dimension **d**
    *m*       number of double-well coordinates (1 ≤ m ≤ d)
    *delta*   additive shift in exp(-(x²-δ))  **OR**
    *separation*/*shift* pair as in the PyTorch code.
    """
    def __init__(self,
                 dim: int,
                 m: int,
                 *,
                 # you may specify either (delta) or (separation, shift)
                 delta: float | None = None,
                 separation: float | None = None,
                 shift: float = 0.0,
                 can_sample: bool = False):
        if not (1 <= m <= dim):
            raise ValueError(f"`m` must be between 1 and dim={dim}")

        if delta is not None:
            # match exp(−(x²−δ)²) by setting separation=δ and shift=0
            separation = float(delta)
            shift      = 0.0
        elif separation is None:
            separation = 2.0    # default

        self.m            = m
        self.n_gauss      = dim - m
        self.double_well  = DoubleWell(separation=separation,
                                       shift=shift,
                                       can_sample=can_sample)
        self.shift        = shift
        self.separation   = separation
        super().__init__(dim=dim, log_Z=None, can_sample=can_sample)

    # -------- log-density -------------------------------------------------- #
    def log_prob(self, x: chex.Array) -> chex.Array:
        if x.ndim == 1:
            x = x[None, :]
        lp = jnp.sum(
            self.double_well.unnorm_log_prob(
                x[..., :self.m].reshape((-1, 1))
            ).reshape(x.shape[0], self.m),
            axis=-1
        )
        if self.n_gauss:
            quad = jnp.sum(x[..., self.m:] ** 2, axis=-1)
            lp  += -0.5 * quad
        return lp

    # -------- score -------------------------------------------------------- #
    def score(self, x: chex.Array):
        score_dw = jax.vmap(self.double_well.score)(
            x[..., :self.m].reshape((-1, 1))
        ).reshape(x.shape[:-1] + (self.m,))
        if self.n_gauss:
            score_gauss = -x[..., self.m:]
            return jnp.concatenate([score_dw, score_gauss], axis=-1)
        return score_dw

    # -------- sampling ----------------------------------------------------- #
    def sample(self, key: chex.PRNGKey, shape: Sequence[int]):
        if not self.can_sample:
            raise RuntimeError("Sampling disabled (can_sample=False).")
        key_dw, key_g = jax.random.split(key)
        # double-well coords
        samples_dw = self.double_well.sample(
            key_dw, shape + (self.m,)
        )
        samples_dw = samples_dw.squeeze(-1)           # remove trailing dim
        # Gaussian coords
        if self.n_gauss:
            samples_gauss = jax.random.normal(
                key_g, shape + (self.n_gauss,)
            ) + self.shift
            samples = jnp.concatenate([samples_dw, samples_gauss], axis=-1)
        else:
            samples = samples_dw
        return samples

    # -------- visualise (2×2 contour) ------------------------------------- #
    def visualise(self, samples: chex.Array, axes=None,
                  show=False, prefix='') -> dict:
        plotting_bounds = (-3, 3)
        grid_width = 100
        pairs = min(self.dim // 2, 2)
        fig, axs = plt.subplots(pairs, pairs, squeeze=False,
                                sharex="row", sharey="row")
        samples = jnp.clip(samples, *plotting_bounds)

        xx, yy = jnp.meshgrid(
            jnp.linspace(*plotting_bounds, grid_width),
            jnp.linspace(*plotting_bounds, grid_width)
        )
        pts = jnp.column_stack([xx.ravel(), yy.ravel()])

        def lp_pair(p, i, j):
            x = jnp.zeros((p.shape[0], self.dim))
            x = x.at[:, i].set(p[:, 0])
            x = x.at[:, j].set(p[:, 1])
            return self.log_prob(x)

        for r in range(pairs):
            for c in range(pairs):
                i, j = 2 * r, 2 * c + 1
                if j >= self.dim:
                    axs[r, c].axis('off')
                    continue
                lp = lp_pair(pts, i, j).reshape(grid_width, grid_width)
                lp = jnp.clip(lp, -1000)
                axs[r, c].contour(xx, yy, lp, levels=20)
                axs[r, c].plot(samples[:, i], samples[:, j],
                               "o", alpha=0.5)
                if c == 0:
                    axs[r, c].set_ylabel(rf"$x_{{{i+1}}}$")
                if r == pairs - 1:
                    axs[r, c].set_xlabel(rf"$x_{{{j+1}}}$")

        wb = {f"{prefix}figures/vis": [wandb.Image(fig)]}
        plt.close()
        return wb