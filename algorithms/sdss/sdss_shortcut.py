import jax
import jax.numpy as jnp
from functools import partial

def sample_d_and_i(
    key,
    num_steps,
    mode="inference",
):
    N = jnp.asarray(num_steps)
    if mode == "inference":
        # Restrict to relevant steps used during sampling
        # 1) Pick t even in [0, N)
        key, keygen = jax.random.split(key)
        n_evens = (N + 1) // 2
        i = jax.random.randint(key, shape=(), minval=0, maxval=n_evens)
        i = i * 2
        # 2) Pick d a power of 2 s.t. (T - t) mod d = 0 
        key, keygen = jax.random.split(keygen)
        remaining = N - i
        lsb = remaining & (-remaining)
        k_max = jnp.log2(lsb).astype(jnp.int32)
        k = jax.random.randint(key, shape=(), minval=1, maxval=k_max + 1)
        d = 1 << k
    else:
        # No constraint
        key_d, key_i = jax.random.split(key)
        k_max = jnp.log2(N).astype(jnp.int32)
        k = jax.random.randint(key_d, shape=(), minval=1, maxval=k_max + 1)
        d = 1 << k
        i_max_p1 = jnp.maximum(N - d + 1, 1)
        i = jax.random.randint(key_i, shape=(), minval=0, maxval=i_max_p1)
    return d, i


def per_sample_mse(
    seed,
    model_state,
    params,    
    paths,
    aux_tuple,
    target,
    sigmas,
):
    # Setup
    init_std, init_sampler, init_log_prob = aux_tuple
    target_log_prob = target.log_prob
    num_steps = sigmas.shape[0] - 1
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
        target_log_prob=target_log_prob,
    )
    def pf_ode_step(state, sigma, d_sigma, params):
        x, key_gen = state
        diff = jnp.sqrt(2.0 * jnp.maximum(sigma, 0.0))
        lgv = jax.lax.stop_gradient(jax.grad(langevin_init)(x, sigma, diff))
        time_code = jnp.full((1, 1), sigma, dtype=x.dtype)
        d_code = jnp.full((1, 1), d_sigma, dtype=x.dtype)
        u = model_state.apply_fn(params, x, time_code, d_code, lgv)
        x_new = x + (0.5 * diff * u) * d_sigma
        return x_new, key_gen
    
    # Sample time and step indexes
    d, i = sample_d_and_i(seed, num_steps)
    half = d >> 1
    x_t = paths[i, :]  # shape (dim,)
    state_start = (x_t, seed)  # seed is not re-used 
    sigma_start = sigmas[i]
    sigma_half = sigmas[i + half]
    sigma_end = sigmas[i + d]
    d_sigma_start = sigma_start - sigma_half
    d_sigma_end = sigma_half - sigma_end
    d_sigma_shortcut = sigma_start - sigma_end

    # Two step target
    params_sg = jax.tree.map(jax.lax.stop_gradient, params)
    state_half = pf_ode_step(
        state_start, sigma_start, d_sigma_start, params_sg
    )
    state_target = pf_ode_step(
        state_half, sigma_half, d_sigma_end, params_sg
    )

    # Shortcut
    state_pred = pf_ode_step(
        state_start, sigma_start, d_sigma_shortcut, params
    )

    # MSE 
    x_target, _ = state_target
    x_pred, _ = state_pred
    return jnp.mean((x_target - x_pred)**2)


def shortcut_loss(
    key,
    model_state,
    params,
    paths,   # shape (batch_size, N+1, dim)
    batch_size,
    aux_tuple,
    target,
    sigmas,
):
    keys = jax.random.split(key, num=batch_size)
    mse = jax.vmap(
        per_sample_mse,
        in_axes=(0, None, None, 0, None, None, None)
    )(keys, model_state, params, paths, aux_tuple, target, sigmas)
    return jnp.mean(mse)