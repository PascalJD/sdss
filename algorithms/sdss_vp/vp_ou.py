# algorithms/sdss_vp/vp_ou.py
# Utilities for VP (OU) exact/noising kernels.

from __future__ import annotations
import jax
import jax.numpy as jnp


def _gaussian_log_prob_diag(
    x: jnp.ndarray,mean: jnp.ndarray,var: jnp.ndarray
) -> jnp.ndarray:
    """Log N(x; mean, var I) aggregated over the last dim."""
    d = x.shape[-1]
    inv_var = 1.0 / var
    quad = jnp.sum((x - mean) ** 2, axis=-1) * inv_var
    return -0.5 * (d * jnp.log(2.0 * jnp.pi * var) + quad)


# Linear schedule: closed-form VP
def vp_linear_shrink(
    t0: jnp.ndarray,
    t1: jnp.ndarray,
    beta_min: float,
    beta_max: float
) -> jnp.ndarray:
    """s_{t0|t1} for linear beta(t) on [0,1]."""
    d_beta = beta_max - beta_min
    integ = 0.5 * beta_min * (t1 - t0) + 0.25 * d_beta * (t1 * t1 - t0 * t0)
    return jnp.exp(-integ)


def vp_linear_var(
    t0: float, t1: float, sigma0: float, beta_min: float, beta_max: float
) -> jnp.ndarray:
    """(1 - s^2) * sigma0^2."""
    s = vp_linear_shrink(t0, t1, beta_min, beta_max)
    return (1.0 - s * s) * (sigma0 ** 2)


def vp_ou_backward_logprob_linear(
    x0: jnp.ndarray, 
    x1: jnp.ndarray, 
    t0: jnp.ndarray, 
    t1: jnp.ndarray,
    sigma0: float, 
    beta_min: float, 
    beta_max: float
) -> jnp.ndarray:
    """log p_{t0|t1}(x0 | x1) for linear beta."""
    s = vp_linear_shrink(t0, t1, beta_min, beta_max)
    var = vp_linear_var(t0, t1, sigma0, beta_min, beta_max)
    mean = s * x1
    return _gaussian_log_prob_diag(x0, mean, var)


# General schedule: numerical beta integration
def _integrate_beta_trapz(
    beta_fn, 
    total_steps: int,
    t0: jnp.ndarray, 
    t1: jnp.ndarray, 
    n: int = 4097
) -> jnp.ndarray:
    """Approximate int_{t0}^{t1} beta(s) ds using trapezoidal rule."""
    ts = jnp.linspace(t0, t1, n)
    steps = ts * float(total_steps)
    betas = jax.vmap(beta_fn)(steps.astype(jnp.float32))
    return jnp.trapz(betas, ts)


def vp_shrink_from_schedule(
    beta_fn, 
    total_steps: int,
    t0: jnp.ndarray,
    t1: jnp.ndarray,
    n_trapz: int = 4097
) -> jnp.ndarray:
    integ = _integrate_beta_trapz(beta_fn, total_steps, t0, t1, n_trapz)
    return jnp.exp(-0.5 * integ)


def vp_var_from_schedule(
    beta_fn, total_steps: int, sigma0: float, t0: float, t1: float
) -> jnp.ndarray:
    s = vp_shrink_from_schedule(beta_fn, total_steps, t0, t1)
    return (1.0 - s * s) * (sigma0 ** 2)


def vp_ou_backward_logprob_from_schedule(
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    beta_fn,
    total_steps: int,
    sigma0: float,
    t0: float = 0.0,
    t1: float = 1.0,
    n_trapz: int = 4097
) -> jnp.ndarray:
    integ = _integrate_beta_trapz(beta_fn, total_steps, t0, t1, n_trapz)
    s = jnp.exp(-integ)
    var = (1.0 - s * s) * (sigma0 ** 2)
    mean = s * x1
    return _gaussian_log_prob_diag(x0, mean, var)


# Weights adjuster
def make_ou_weight_fn(
    *, 
    prior_std: float, 
    noise_schedule, 
    schedule_type: str,
    num_steps: int, 
    beta_min: float | None = None,
    beta_max: float | None = None,
    n_trapz: int = 257
):
    """
    Return a function that maps (model_state, params, paths, log_w_em) -> log_w_ou
    Replaces EM backward kernels by exact OU backward kernels per step.
    """
    T = float(num_steps)

    if schedule_type.lower() == "linear":
        # Exact form
        compute_s = lambda t0, t1: vp_linear_shrink(
            t0, t1, float(beta_min), float(beta_max)
        )
    else:
        # e.g., "cosine", numerical integration
        compute_s = lambda t0, t1: vp_shrink_from_schedule(
            noise_schedule, num_steps, t0, t1, n_trapz
        )

    def weight_fn(model_state, params, paths, log_w_em):
        B, Kp1, D = paths.shape
        k = Kp1 - 1
        if (num_steps % k) != 0:
            raise ValueError(f"num_steps={num_steps} must be divisible by k={k}.")
        d  = num_steps // k
        dt = float(d) / T

        # Training scan uses descending codes [T, T-d, ..., d]
        codes_desc = jnp.arange(d, num_steps + 1, d, dtype=jnp.int32)[::-1]  # (k,)
        codes_f32  = codes_desc.astype(jnp.float32)
        t1 = codes_f32 / T  # (k,)
        t0 = t1 - dt  # (k,)

        # Path slices per step
        Xn = paths[:, :-1, :]  # (B, k, D)
        Xnp1 = paths[:,  1:, :]  # (B, k, D)

        # EM backward kernel
        beta_t = jax.vmap(lambda c: noise_schedule(c))(codes_f32)  # (k,)
        var_em = beta_t * (prior_std ** 2) * dt  # (k,)
        mean_em = Xnp1 - 0.5 * (beta_t * dt)[None, :, None] * Xnp1  # (B,k,D)
        log_bwd_em = _gaussian_log_prob_diag(Xn, mean_em, var_em[None, :])  # (B,k)

        # OU backward params per step
        s_vec = compute_s(t0, t1)
        var_ou  = (1.0 - s_vec * s_vec) * (prior_std ** 2)  # (k,)
        mean_ou = s_vec[None, :, None] * Xnp1  # (B,k,D)
        log_bwd_ou = _gaussian_log_prob_diag(Xn, mean_ou, var_ou[None, :]) # (B,k)

        # Weight correction
        delta = jnp.sum(log_bwd_ou - log_bwd_em, axis=1) 
        log_w_ou = log_w_em + delta
        return log_w_ou 

    return weight_fn