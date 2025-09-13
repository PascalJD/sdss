import jax
import jax.numpy as jnp
from functools import partial
import numpyro.distributions as npdist


def sample_kernel(rng_key, mean, scale):
    eps = jax.random.normal(rng_key, shape=mean.shape)
    return mean + scale * eps


def log_prob_kernel(x, mean, scale):
    dist = npdist.Independent(npdist.Normal(loc=mean, scale=scale), 1)
    return dist.log_prob(x)


def per_sample_rnd(
    seed,
    model_state,
    params,
    aux_tuple,
    target,
    sigmas, 
    d_sigmas,
    stop_grad=False,
    prior_to_target=True,
    use_ode=True,
):
    init_std, init_sampler, init_log_prob = aux_tuple
    target_log_prob = target.log_prob

    N = sigmas.shape[0] - 1

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

    def apply_single(x, sigma, d_sigma, diff, *, params):
        # Lift to batch=1 for the model call.
        lgv = jax.lax.stop_gradient(jax.grad(langevin_init)(x, sigma, diff))  # (D,)
        xb = x[None, ...]  # (1, D)
        lvgb = lgv[None, ...]  # (1, D)
        tcode = jnp.array([[sigma]], dtype=xb.dtype)  # (1,1)
        dcode = jnp.array([[d_sigma]], dtype=xb.dtype)  # (1,1)
        u_b = model_state.apply_fn(params, xb, tcode, dcode, lvgb)  # (1, D)
        u = u_b[0]  # (D,)
        return u

    def step_prior_to_target(state, i):
        x, log_w, key_gen = state
        sigma = sigmas[i]
        d_sigma = jnp.maximum(d_sigmas[i], 1e-12)  # Δσ_i > 0

        if stop_grad:
            x = jax.lax.stop_gradient(x)

        # SDE terms
        diff = jnp.sqrt(2.0 * jnp.maximum(sigma, 0.0))
        u = apply_single(x, sigma, d_sigma, diff, params=params)        
        scale = diff * jnp.sqrt(d_sigma)
        mean_fwd = x + diff * u * d_sigma

        # Forward step
        key, key_gen = jax.random.split(key_gen)
        if use_ode:
            x_new = x + 0.5 * diff * u * d_sigma
        else: 
            x_new = sample_kernel(key, mean_fwd, scale)
        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        # Backward mean
        mean_bwd = x_new # Drift 0 for karras

        # RN terms (reverse proposal has mean=x_new, same scale)
        fwd_lp = log_prob_kernel(x_new, mean_fwd, scale)
        bwd_lp = log_prob_kernel(x, mean_bwd, scale)
        log_w = log_w + (bwd_lp - fwd_lp)
        return (x_new, log_w, key_gen), x_new

    # reverse; sigma increases
    def step_target_to_prior(state, i):
        x, log_w, key_gen = state
        sigma = sigmas[i]
        d_sigma = jnp.maximum(d_sigmas[i], 1e-12)

        # SDE terms
        diff = jnp.sqrt(2.0 * jnp.maximum(sigma, 0.0))
        scale = diff * jnp.sqrt(d_sigma)
        mean_bwd = x

        # reverse proposal (drift=0 -> mean=x)
        key, key_gen = jax.random.split(key_gen)
        x_new = sample_kernel(key, mean_bwd, scale)
        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        # forward kernel for RN denominator
        u = apply_single(x, sigma, d_sigma, diff, params=params)
        mean_fwd = x_new + diff * u * d_sigma

        fwd_lp = log_prob_kernel(x, mean_fwd, scale)
        bwd_lp = log_prob_kernel(x_new, mean_bwd, scale)
        log_w = log_w + (bwd_lp - fwd_lp)
        return (x_new, log_w, key_gen), x_new

    key, key_gen = jax.random.split(seed)
    idx_fwd = jnp.arange(N)  # 0..N-1
    idx_rev = idx_fwd[::-1]  # N-1..0 (naming convention in SDSS)

    if prior_to_target:
        init_x = jnp.clip(init_sampler(seed=key), -4 * init_std, 4 * init_std)
        aux = (init_x, 0.0, key)
        (final_x, log_ratio, _), xs_after = jax.lax.scan(step_prior_to_target, aux, idx_fwd)
        terminal_cost = init_log_prob(init_x) - target_log_prob(final_x)
    else:
        init_x = target.sample(key, ())
        aux = (init_x, 0.0, key)
        (final_x, log_ratio, _), xs_after = jax.lax.scan(step_target_to_prior, aux, idx_rev)
        terminal_cost = init_log_prob(final_x) - target_log_prob(init_x)

    running_cost = -log_ratio
    stochastic_costs = jnp.zeros_like(running_cost)
    traj = jnp.concatenate([init_x[None, ...], xs_after], axis=0)  # (N+1, D)
    return final_x, running_cost, stochastic_costs, terminal_cost, traj


def rnd(
    key,
    model_state,
    params,
    batch_size,
    aux_tuple,
    target,
    sigmas, 
    d_sigmas,
    stop_grad=False,
    prior_to_target=True,
    use_ode=False,
    return_traj=False,
):
    seeds = jax.random.split(key, num=batch_size)
    x_T, run_c, stoch_c, term_c, traj = jax.vmap(
        per_sample_rnd,
        in_axes=(
            0, 
            None, None, None, None, None, None, None, None, None)
    )(
        seeds, model_state, params, aux_tuple, target, sigmas, d_sigmas,
        stop_grad, prior_to_target, use_ode,
    )
    if return_traj:
        return x_T, run_c, stoch_c, term_c, traj
    else:
        return x_T, run_c, stoch_c, term_c


def neg_elbo(
    key,
    model_state,
    params,
    batch_size,
    aux_tuple,
    target,
    sigmas,
    d_sigmas,
    stop_grad=False,
):
    x_T, run_c, _, term_c = rnd(
        key, model_state, params, batch_size, aux_tuple, target,
        sigmas, d_sigmas, stop_grad, prior_to_target=True, use_ode=False, return_traj=False,
    )
    neg_elbo_val = run_c + term_c
    return jnp.mean(neg_elbo_val), (neg_elbo_val, x_T)