import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from functools import partial


def sample_d_and_t(rng, num_steps: int):
    n = jnp.asarray(num_steps, dtype=jnp.int32)
    rng_d, rng_t = jax.random.split(rng)

    k_max = jnp.int32(jnp.log2(n))
    k = jax.random.randint(rng_d, (), 1, k_max + 1)
    d = jnp.int32(1 << k)

    choices = n // d
    m = jax.random.randint(rng_t, (), 0, choices)
    t = m * d
    return d, t


def distillation_loss(
    rng,
    model_state,
    params,
    teacher_params,
    paths,
    batch_size,
    prior_tuple,
    target,
    betas,
    num_steps,
):
    rngs = jax.random.split(rng, num=batch_size)
    mse = jax.vmap(
        _per_sample_mse,
        in_axes=(0, None, None, None, 0, None, None, None, None),
    )(
        rngs,
        model_state,
        params,
        teacher_params,
        paths,
        prior_tuple,
        target,
        betas,
        num_steps,
    )
    return jnp.mean(mse)


def _per_sample_mse(
    rng,
    model_state,
    params,
    teacher_params,
    paths,
    prior_tuple,
    target,
    betas,
    num_steps,
):
    prior_std, prior_sample, prior_logp = prior_tuple
    target_logp = target.log_prob
    paths = jax.lax.stop_gradient(paths)

    def lgv_init(x, t_code, sigma_t, t_total):
        tr = t_code / t_total
        return sigma_t * (
            (1.0 - tr) * target_logp(x) + tr * prior_logp(x)
        )

    lgv_init = partial(lgv_init, t_total=num_steps)

    def pf_ode_step(state, t_code, d_code, params_local):
        x, rng_local = state

        beta_t = betas(t_code)
        sigma_t = jnp.sqrt(beta_t) * prior_std

        lgv = jax.lax.stop_gradient(
            jax.grad(lgv_init)(x, t_code, sigma_t)
        )

        t_norm = (t_code / num_steps) * jnp.ones(1)
        d_norm = (d_code / num_steps) * jnp.ones(1)

        u = model_state.apply_fn(
            params_local, x, t_norm, d_norm, lgv
        )
        x_new = x + (0.5 * sigma_t * u + 0.5 * beta_t * x) * (d_code / num_steps)
        return x_new, rng_local

    d_code, t_idx = sample_d_and_t(rng, num_steps)
    d_half = d_code >> 1
    t_half_idx = t_idx + d_half

    t_code = num_steps - t_idx
    t_half_code = num_steps - t_half_idx

    x_t = paths[t_idx, :]
    state0 = (x_t, rng)

    params_sg = jtu.tree_map(jax.lax.stop_gradient, teacher_params)

    state_half = pf_ode_step(state0, t_code, d_half, params_sg)
    state_tgt = pf_ode_step(state_half, t_half_code, d_half, params_sg)

    state_pred = pf_ode_step(state0, t_code, d_code, params)

    x_tgt, _ = state_tgt
    x_pred, _ = state_pred
    return jnp.mean((x_tgt - x_pred) ** 2)