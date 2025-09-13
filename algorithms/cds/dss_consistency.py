# --- algorithms/dss/dss_consistency.py ---

import jax
import jax.numpy as jnp
from functools import partial

from algorithms.dss.dss_reparam import cm_apply

def consistency_loss(
    rng,
    teacher_state, 
    teacher_params,
    student_state, 
    student_params,
    *,
    batch_size: int,
    aux_tuple,
    target,
    sigmas,  # shape (N+1,) : [sigma_max, ..., sigma_min, 0]
    d_sigmas,  # shape (N,)
    per_batch_t: bool = False,
    teacher_use_sde: bool = True,
    terminal_weighting: bool = False,
    w_clip: float = 8.0,
    clip_x_std: float = 4.0,
    cm_sigma_data: float = 0.5,
    cm_sigma_min: float | None = None,
    return_pairs: bool = False,
):
    init_std, init_sampler, init_log_prob = aux_tuple
    N = sigmas.shape[0] - 1  # number of updates

    def langevin_init_fn(
        x, 
        sigma_scalar, 
        sigma_for_grad, 
        Smax, 
        initial_log_prob, 
        target_log_prob
    ):
        tr = jnp.clip(sigma_scalar / (Smax + 1e-12), 0.0, 1.0)
        return sigma_for_grad * (
            (1.0 - tr) * target_log_prob(x) + tr * initial_log_prob(x)
        )
    
    Smax = sigmas[0]
    langevin_init = partial(
        langevin_init_fn,
        Smax=Smax,
        initial_log_prob=init_log_prob,
        target_log_prob=target.log_prob,
    )

    def _teacher_sde_step(x, i, key):
        sigma = sigmas[i]
        d_sigma = jnp.maximum(d_sigmas[i], 1e-12)
        diff = jnp.sqrt(2.0 * jnp.maximum(sigma, 0.0))
        grad_x = lambda xi: jax.grad(langevin_init)(xi, sigma, diff)
        lgv = jax.lax.stop_gradient(jax.vmap(grad_x)(x))
        timecode = jnp.full((x.shape[0], 1), sigma, dtype=x.dtype)
        u = teacher_state.apply_fn(teacher_params, x, timecode, lgv)
        mean = x + diff * u * d_sigma
        scale = diff * jnp.sqrt(d_sigma)
        key, kn = jax.random.split(key)
        noise = jax.random.normal(kn, shape=x.shape)
        x_new = mean + scale * noise
        return x_new, key

    def _teacher_pf_ode_step(x, i):
        sigma = sigmas[i]
        d_sigma = jnp.maximum(d_sigmas[i], 1e-12)
        diff = jnp.sqrt(2.0 * jnp.maximum(sigma, 0.0))
        grad_x = lambda xi: jax.grad(langevin_init)(xi, sigma, diff)
        lgv = jax.lax.stop_gradient(jax.vmap(grad_x)(x))
        timecode = jnp.full((x.shape[0], 1), sigma, dtype=x.dtype)
        u = teacher_state.apply_fn(teacher_params, x, timecode, lgv)
        return x + 0.5 * diff * u * d_sigma

    # Sample prior batch
    rng, kx = jax.random.split(rng)
    x0 = init_sampler(seed=kx, sample_shape=(batch_size,))
    x0 = jnp.clip(x0, -clip_x_std * init_std, clip_x_std * init_std)

    # Draw indices i and build (X_t, X_{t+1})
    rng, kt = jax.random.split(rng)
    if per_batch_t:
        i = jax.random.randint(kt, (), 0, N)
        step_start, step_stop = 0, i 
        if teacher_use_sde:
            def body(k, carry):
                x, key = carry
                x, key = _teacher_sde_step(x, k, key)
                return (jax.lax.stop_gradient(x), key)
            (x_t, rng) = jax.lax.fori_loop(
                step_start, step_stop, body, (x0, rng)
            )
        else:
            def body(k, x):
                x = _teacher_pf_ode_step(x, k)
                return jax.lax.stop_gradient(x)
            x_t = jax.lax.fori_loop(step_start, step_stop, body, x0)
        x_tp1 = _teacher_pf_ode_step(x_t, i)  # one deterministic step
        sigma_L = jnp.full((batch_size, 1), sigmas[i], dtype=jnp.float32)
        sigma_R = jnp.full((batch_size, 1), sigmas[i+1], dtype=jnp.float32)
    else:
        idxs = jnp.arange(0, N, dtype=jnp.int32)
        def scan_step(carry, k):
            x, key = carry
            if teacher_use_sde:
                x, key = _teacher_sde_step(x, k, key)
            else:
                x = _teacher_pf_ode_step(x, k)
            return (jax.lax.stop_gradient(x), key), x
        (xN, rng), xs_after = jax.lax.scan(scan_step, (x0, rng), idxs, length=N)
        states = jnp.swapaxes(jnp.concatenate([x0[None, ...], xs_after], axis=0), 0, 1)  # (B,N+1,D)

        i_b = jax.random.randint(kt, (batch_size,), 0, N)   # per-sample interval
        x_t = jax.vmap(lambda s, k: s[k])(states, i_b)
        x_tp1 = jax.vmap(lambda xi, k: _teacher_pf_ode_step(xi[None, ...], k).squeeze(0))(x_t, i_b)
        sigma_L = jnp.take(sigmas, i_b)[:, None]
        sigma_R = jnp.take(sigmas, i_b + 1)[:, None]

    # student predictions with CM reparameterization
    def lgv_at(x, sigma):
        diff = jnp.sqrt(2.0 * jnp.maximum(sigma, 0.0))
        return jax.grad(langevin_init)(x, sigma, diff)
    lgv_L = jax.vmap(lgv_at, in_axes=(0, 0))(x_t,   sigma_L.squeeze(-1))
    lgv_R = jax.vmap(lgv_at, in_axes=(0, 0))(x_tp1, sigma_R.squeeze(-1))
    lgv_L = jax.lax.stop_gradient(lgv_L)
    lgv_R = jax.lax.stop_gradient(lgv_R)
    pred_left = cm_apply(
        student_state.apply_fn, student_params,
        x_t, sigma_L, lgv_L,
        sigma_data=cm_sigma_data, sigma_min=cm_sigma_min
    )
    pred_right = cm_apply(
        student_state.apply_fn, student_params,
        x_tp1, sigma_R, lgv_R,
        sigma_data=cm_sigma_data, sigma_min=cm_sigma_min
    )
    pred_right = jax.lax.stop_gradient(pred_right)  # target

    # loss
    per_ex_mse = jnp.sum((pred_right - pred_left) ** 2, axis=-1)
    if terminal_weighting:
        log_w = init_log_prob(x0) - target.log_prob(pred_right)
        # w = jnp.exp(jnp.clip(log_w, -w_clip, w_clip))
        # w = jax.lax.stop_gradient(w)
        per_ex_mse = per_ex_mse + log_w
    loss = jnp.mean(per_ex_mse)

    # logs
    metrics = {
        "loss": loss,
        "mse_raw": jnp.mean(jnp.sum((pred_right - pred_left) ** 2, axis=-1)),
        "cm/sigma_left_mean": jnp.mean(sigma_L),
    }
    if terminal_weighting:
        metrics.update({
            "weight/mean": jnp.mean(log_w),
            "weight/max": jnp.max(log_w),
            "weight/min": jnp.min(log_w),
        })
    if return_pairs:
        metrics["pairs/pleft"] = pred_left
        metrics["pairs/pright"] = pred_right
        metrics["pairs/xt"]  = x_t

    return loss, metrics