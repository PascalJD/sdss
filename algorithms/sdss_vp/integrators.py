# algorithms/sdss_vp/integrators.py
from __future__ import annotations
from typing import NamedTuple, Literal
import jax
import jax.numpy as jnp

from algorithms.sdss_vp.vp_ou import vp_linear_shrink, vp_shrink_from_schedule

Kind = Literal["em", "midpoint_em"]

class StepParams(NamedTuple):
    t0_norm: jnp.ndarray
    t1_norm: jnp.ndarray
    dt: jnp.ndarray
    s: jnp.ndarray  # shrink over [t0,t1]
    bar_beta: jnp.ndarray  # average beta over [t0,t1]
    sigma_bar: jnp.ndarray  # sigma0 * sqrt(bar_beta)
    beta_t1: jnp.ndarray  # beta at right endpoint (for EM)
    sigma_t1: jnp.ndarray  # sigma0 * sqrt(beta_t1)

def _compute_s(
    schedule_type: str,
    noise_schedule,
    num_steps: int,
    t0_norm: jnp.ndarray,
    t1_norm: jnp.ndarray,
    beta_min: float | None,
    beta_max: float | None,
    n_trapz: int,
) -> jnp.ndarray:
    if schedule_type.lower() == "linear" and (beta_min is not None) and (beta_max is not None):
        return vp_linear_shrink(t0_norm, t1_norm, float(beta_min), float(beta_max))
    # general schedule via trapz
    s_sq = vp_shrink_from_schedule(noise_schedule, num_steps, t0_norm, t1_norm, n_trapz)
    return jnp.sqrt(s_sq)

def compute_step_params(
    *,
    kind: Kind,
    t1_code: jnp.ndarray,  # float32 "code"
    dt: jnp.ndarray,  # normalized dt in [0,1]
    prior_std: float,
    noise_schedule,
    num_steps: int,
    schedule_type: str,
    beta_min: float | None,
    beta_max: float | None,
    n_trapz: int,
) -> StepParams:
    """Return the time/variance quantities shared by the kernels on [t0,t1]."""
    t_float = float(num_steps)
    t1_norm = t1_code / t_float
    t0_norm = t1_norm - dt

    # shrink over the cell, then bar_beta = (-2 log s)/dt
    s = jnp.clip(
        _compute_s(
            schedule_type, 
            noise_schedule, 
            num_steps,
            t0_norm,
            t1_norm, 
            beta_min, 
            beta_max, 
            n_trapz
        ),
        1e-12, 
        1.0 - 1e-12
    )
    integ_beta = -2.0 * jnp.log(s)  # \int beta
    bar_beta = integ_beta / dt  # average beta
    sigma_bar = prior_std * jnp.sqrt(jnp.maximum(bar_beta, 1e-20))

    # endpoint beta/sigma for EM
    beta_t1 = noise_schedule(t1_code)
    sigma_t1 = prior_std * jnp.sqrt(jnp.maximum(beta_t1, 1e-20))

    return StepParams(t0_norm=t0_norm, t1_norm=t1_norm, dt=dt,
                      s=s, bar_beta=bar_beta, sigma_bar=sigma_bar,
                      beta_t1=beta_t1, sigma_t1=sigma_t1)

def control_time(kind: Kind, sp: StepParams) -> jnp.ndarray:
    """Normalized time used to evaluate the control u(x, t, dt)."""
    if kind == "midpoint_em":
        return 0.5 * (sp.t0_norm + sp.t1_norm)
    # default: EM uses right endpoint
    return sp.t1_norm

def sigma_for_lgv(kind: Kind, sp: StepParams) -> jnp.ndarray:
    """Sigma scalar to feed your lgv_init gradient."""
    if kind == "midpoint_em":
        return sp.sigma_bar
    return sp.sigma_t1

def scale(kind: Kind, sp: StepParams) -> jnp.ndarray:
    """Std dev of the Gaussian step."""
    if kind == "midpoint_em":
        return sp.sigma_bar * jnp.sqrt(sp.dt)
    return sp.sigma_t1 * jnp.sqrt(sp.dt)

def fwd_mean(
        kind: Kind, x: jnp.ndarray, u: jnp.ndarray, sp: StepParams
    ) -> jnp.ndarray:
    """Mean of p_{n+1|n}."""
    if kind == "midpoint_em":
        return x + (sp.sigma_bar * u + 0.5 * sp.bar_beta * x) * sp.dt
    # EM (right-point) mean
    return x + (sp.sigma_t1 * u + 0.5 * sp.beta_t1 * x) * sp.dt

def bwd_mean(kind: Kind, x_next: jnp.ndarray, sp: StepParams) -> jnp.ndarray:
    """Mean of p_{n|n+1}."""
    if kind == "midpoint_em":
        return x_next - 0.5 * sp.bar_beta * x_next * sp.dt
    # EM (right-point) backward mean
    return x_next - 0.5 * sp.beta_t1 * x_next * sp.dt

def ode_drift(kind, x, u, sp):
    # Probability-flow ODE drift over [t0,t1] with the integrator's coefficients
    if kind == "midpoint_em":
        return x + (0.5 * sp.sigma_bar * u + 0.5 * sp.bar_beta * x) * sp.dt
    # default: EM, right-end coefficients
    return x + (0.5 * sp.sigma_t1 * u + 0.5 * sp.beta_t1 * x) * sp.dt