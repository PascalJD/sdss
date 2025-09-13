# algorithms/dss/dss_reparam.py
import jax
import jax.numpy as jnp

def _get_scalings(t_sigma, *, sigma_data=0.5, sigma_min=1e-3):
    t_hat = jnp.maximum(t_sigma - sigma_min, 0.0)
    c_skip = (sigma_data**2) / (t_hat**2 + sigma_data**2)
    c_out  = (sigma_data * t_hat) / jnp.sqrt(sigma_data**2 + t_sigma**2)
    return c_skip, c_out

def cm_apply(
    base_apply_fn,
    params,
    x,
    time_code, 
    lgv_term,
    *,
    sigma_data=0.5,
    sigma_min=None,
    sigma_lookup=None,
):
    """
    If sigma_lookup is None, 'time_code' is interpreted as σ directly.
    Otherwise, 'time_code' is an integer (or float index) looked up in the table.
    """
    if sigma_lookup is None:
        sigma_t = jnp.maximum(time_code, 0.0)
    else:
        idx = jnp.clip(
            jnp.rint(time_code), 0, sigma_lookup.shape[0] - 1
        ).astype(jnp.int32)
        sigma_t = jnp.take(sigma_lookup, idx)

    if sigma_min is None:
        sigma_min \
            = float(sigma_lookup[-2]) if sigma_lookup is not None else 2e-3

    c_skip, c_out = _get_scalings(
        sigma_t, sigma_data=sigma_data, sigma_min=sigma_min
    )
    g = base_apply_fn(params, x, time_code, lgv_term)  # the raw network
    return c_skip * x + c_out * g