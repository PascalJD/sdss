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

    def step_prior_to_target(state, i):
        x, log_w, key_gen = state
        sigma_i = sigmas[i]
        d_sigma = jnp.maximum(d_sigmas[i], 1e-12)  # Δσ_i > 0
        g = jnp.sqrt(2.0 * jnp.maximum(sigma_i, 0.0))
        scale = g * jnp.sqrt(d_sigma)

        if stop_grad:
            x = jax.lax.stop_gradient(x)

        # side input (score-like)
        lgv = jax.lax.stop_gradient(jax.grad(langevin_init)(x, sigma_i, g))
        time_code = jnp.full((x.shape[0], 1), sigma_i, dtype=x.dtype)
        u = model_state.apply_fn(params, x, time_code, lgv)

        # forward kernel (Euler–Maruyama for SDE, PF-ODE if use_ode)
        mean_fwd = x + g * u * d_sigma
        key, key_gen = jax.random.split(key_gen)
        x_new = x + 0.5 * g * u * d_sigma if use_ode else sample_kernel(key, mean_fwd, scale)
        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        # RN terms (reverse proposal has mean=x_new, same scale)
        fwd_lp = log_prob_kernel(x_new, mean_fwd, scale)
        bwd_lp = log_prob_kernel(x, x_new, scale)
        log_w = log_w + (bwd_lp - fwd_lp)
        return (x_new, log_w, key_gen), x_new

    # reverse; sigma increases
    def step_target_to_prior(state, i):
        x, log_w, key_gen = state
        sigma_i = sigmas[i]
        d_sigma = jnp.maximum(d_sigmas[i], 1e-12)
        g = jnp.sqrt(2.0 * jnp.maximum(sigma_i, 0.0))
        scale = g * jnp.sqrt(d_sigma)

        # reverse proposal (μ=0 ⇒ mean=x)
        key, key_gen = jax.random.split(key_gen)
        x_new = sample_kernel(key, x, scale)
        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        # forward kernel for RN denominator
        lgv = jax.lax.stop_gradient(jax.grad(langevin_init)(x_new, sigma_i, g))
        time_code = jnp.full((x.shape[0], 1), sigma_i, dtype=x.dtype)
        u = model_state.apply_fn(params, x, time_code, lgv)
        mean_fwd = g * u * d_sigma

        fwd_lp = log_prob_kernel(x, mean_fwd, scale)
        bwd_lp = log_prob_kernel(x_new, x, scale)
        log_w = log_w + (bwd_lp - fwd_lp)
        return (x_new, log_w, key_gen), x_new

    key, key_gen = jax.random.split(seed)

    if prior_to_target:
        init_x = jnp.clip(init_sampler(seed=key), -4 * init_std, 4 * init_std)
        aux = (init_x, 0.0, key)
        (final_x, log_ratio, _), xs_after = jax.lax.scan(step_prior_to_target, aux, jnp.arange(sigmas.shape[0] - 1))
        terminal_cost = init_log_prob(init_x) - target_log_prob(final_x)
    else:
        init_x = target.sample(key, ())
        aux = (init_x, 0.0, key)
        (final_x, log_ratio, _), xs_after = jax.lax.scan(step_target_to_prior, aux, jnp.arange(sigmas.shape[0] - 1))
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
        in_axes=(0, None, None, None, None, None, None, None, None, None)
    )(
        seeds, model_state, params, aux_tuple, target, sigmas, d_sigmas,
        stop_grad, prior_to_target, use_ode
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
        sigmas, d_sigmas, stop_grad, prior_to_target=True, use_ode=False, return_traj=False
    )
    neg_elbo_val = run_c + term_c
    return jnp.mean(neg_elbo_val), (neg_elbo_val, x_T)