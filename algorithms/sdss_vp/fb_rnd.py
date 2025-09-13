import jax
import jax.numpy as jnp
from functools import partial
import numpyro.distributions as npdist


def sample_kernel(rng, mean, scale):
    eps = jax.random.normal(rng, shape=(mean.shape[0],))
    return mean + scale * eps


def log_prob_kernel(x, mean, scale):
    dist = npdist.Independent(npdist.Normal(loc=mean, scale=scale), 1)
    return dist.log_prob(x)


def _make_step_codes(total_steps: int,
                     eval_steps: int | None) -> jnp.ndarray:
    t = int(total_steps)
    if eval_steps is None:
        k = t
    else:
        k = int(eval_steps)
        if k > t:
            raise ValueError(f"eval_steps={k} > total_steps={t}.")
        if (t % k) != 0:
            raise ValueError(f"T={t} must be divisible by k={k}.")
    d = t // k
    return jnp.arange(d, t + 1, d, dtype=jnp.int32)


def per_sample_rnd(
    rng,
    model_state,
    params,
    prior_tuple,
    target,
    num_steps,
    noise_schedule,
    eval_steps=None,
    stop_grad=False,
    prior_to_target=True,
    use_ode=True,
):
    prior_std, prior_sample, prior_logp = prior_tuple
    target_logp = target.log_prob

    step_codes = _make_step_codes(num_steps, eval_steps)
    k = int(step_codes.shape[0])
    t_float = float(num_steps)
    d = t_float / float(k)
    dt = d / t_float

    def lgv_init(x, t_code, sigma_t, t_total):
        tr = t_code / t_total
        return sigma_t * (
            (1.0 - tr) * target_logp(x) + tr * prior_logp(x)
        )

    lgv_init = partial(lgv_init, t_total=t_float)

    def sim_prior_to_target(state, t_code):
        x, log_w, rng_inner = state
        t_code = t_code.astype(jnp.float32)

        if stop_grad:
            x = jax.lax.stop_gradient(x)

        beta_t = noise_schedule(t_code)
        sigma_t = jnp.sqrt(2.0 * beta_t) * prior_std

        lgv = jax.lax.stop_gradient(
            jax.grad(lgv_init)(x, t_code, sigma_t)
        )

        u = model_state.apply_fn(
            params,
            x,
            (t_code / t_float) * jnp.ones(1),
            (d / t_float) * jnp.ones(1),
            lgv,
        )

        fwd_mean = x + (sigma_t * u + beta_t * x) * dt
        scale = sigma_t * jnp.sqrt(dt)

        rng_inner, sub = jax.random.split(rng_inner)
        if use_ode:
            x_new = x + (0.5 * sigma_t * u + beta_t * x) * dt
        else:
            x_new = sample_kernel(sub, fwd_mean, scale)

        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        bwd_mean = x_new - beta_t * x_new * dt
        fwd_lp = log_prob_kernel(x_new, fwd_mean, scale)
        bwd_lp = log_prob_kernel(x, bwd_mean, scale)
        log_w = log_w + (bwd_lp - fwd_lp)

        return (x_new, log_w, rng_inner), x_new

    def sim_target_to_prior(state, t_code):
        x, log_w, rng_inner = state
        t_code = t_code.astype(jnp.float32)

        beta_t = noise_schedule(t_code)
        sigma_t = jnp.sqrt(2.0 * beta_t) * prior_std
        scale = sigma_t * jnp.sqrt(dt)

        bwd_mean = x - beta_t * x * dt
        rng_inner, sub = jax.random.split(rng_inner)
        x_new = sample_kernel(sub, bwd_mean, scale)

        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        lgv = jax.lax.stop_gradient(
            jax.grad(lgv_init)(x_new, t_code, sigma_t)
        )

        u = model_state.apply_fn(
            params,
            x_new,
            (t_code / t_float) * jnp.ones(1),
            (d / t_float) * jnp.ones(1),
            lgv,
        )

        fwd_mean = x_new + (sigma_t * u + beta_t * x) * dt
        fwd_lp = log_prob_kernel(x, fwd_mean, scale)
        bwd_lp = log_prob_kernel(x_new, bwd_mean, scale)
        log_w = log_w + (bwd_lp - fwd_lp)

        return (x_new, log_w, rng_inner), x_new

    rng, sub = jax.random.split(rng)

    if prior_to_target:
        codes_desc = step_codes[::-1]
        x0 = jnp.clip(prior_sample(seed=sub), -4 * prior_std, 4 * prior_std)
        state0 = (x0, 0.0, rng)
        (xT, log_ratio, _), xs = jax.lax.scan(
            sim_prior_to_target, state0, codes_desc
        )
        term_c = prior_logp(x0) - target_logp(xT)
    else:
        codes_asc = step_codes
        x0 = target.sample(sub, ())
        state0 = (x0, 0.0, rng)
        (xT, log_ratio, _), xs = jax.lax.scan(
            sim_target_to_prior, state0, codes_asc
        )
        term_c = prior_logp(xT) - target_logp(x0)

    run_c = -log_ratio
    stoch_c = jnp.zeros_like(run_c)
    traj = jnp.concatenate([x0[None, ...], xs], axis=0)

    return xT, run_c, stoch_c, term_c, traj


def rnd(
    rng,
    model_state,
    params,
    batch_size,
    prior_tuple,
    target,
    num_steps,
    noise_schedule,
    eval_steps=None,
    stop_grad=False,
    prior_to_target=True,
    use_ode=False,
    return_traj=False,
):
    rngs = jax.random.split(rng, num=batch_size)
    xT, rc, sc, tc, traj = jax.vmap(
        per_sample_rnd,
        in_axes=(0, None, None, None, None, None, None,
                 None, None, None, None),
    )(
        rngs,
        model_state,
        params,
        prior_tuple,
        target,
        num_steps,
        noise_schedule,
        eval_steps,
        stop_grad,
        prior_to_target,
        use_ode,
    )
    if return_traj:
        return xT, rc, sc, tc, traj
    return xT, rc, sc, tc


def neg_elbo(
    rng,
    model_state,
    params,
    batch_size,
    prior_tuple,
    target,
    num_steps,
    noise_schedule,
    eval_steps=None,
    stop_grad=False,
):
    xT, rc, _, tc = rnd(
        rng,
        model_state,
        params,
        batch_size,
        prior_tuple,
        target,
        num_steps,
        noise_schedule,
        eval_steps=eval_steps,
        stop_grad=stop_grad,
        prior_to_target=True,
        use_ode=False,
        return_traj=False,
    )
    val = rc + tc
    return jnp.mean(val), (val, xT)