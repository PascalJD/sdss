import jax
import jax.numpy as jnp
from functools import partial
import numpyro.distributions as npdist

from algorithms.sdss_vp.vp_ou import vp_linear_shrink, vp_shrink_from_schedule


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
    *,
    return_ou_weight: bool = False,
    schedule_type: str = "linear",
    beta_min: float | None = None,
    beta_max: float | None = None,
    n_trapz: int = 1025,
):
    prior_std, prior_sample, prior_logp = prior_tuple
    target_logp = target.log_prob

    step_codes = _make_step_codes(num_steps, eval_steps)
    k = int(step_codes.shape[0])
    t_float = float(num_steps)
    d = t_float / float(k)
    dt = d / t_float

    if schedule_type.lower() == "linear" and (beta_min is not None) and (beta_max is not None):
        def compute_s(t0_norm, t1_norm):
            return vp_linear_shrink(t0_norm, t1_norm, float(beta_min), float(beta_max))
    else:
        def compute_s(t0_norm, t1_norm):
            s_sq = vp_shrink_from_schedule(noise_schedule, num_steps, t0_norm, t1_norm, n_trapz)
            return jnp.sqrt(s_sq)

    def lgv_init(x, t_code, sigma_t, t_total):
        tr = t_code / t_total
        return sigma_t * (
            (1.0 - tr) * target_logp(x) + tr * prior_logp(x)
        )

    lgv_init = partial(lgv_init, t_total=t_float)

    def sim_prior_to_target(state, t_code):
        if return_ou_weight:
            x, log_w_em, log_w_ou, rng_inner = state
        else:
            x, log_w_em, rng_inner = state

        t_code = t_code.astype(jnp.float32)

        if stop_grad:
            x = jax.lax.stop_gradient(x)

        beta_t = noise_schedule(t_code)
        sigma_t = jnp.sqrt(beta_t) * prior_std
        t1_norm = t_code / t_float
        t0_norm = t1_norm - dt

        # control
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

        # Forward EM kernel (always EM forward)
        fwd_mean = x + (sigma_t * u + 0.5 * beta_t * x) * dt
        scale = sigma_t * jnp.sqrt(dt)

        rng_inner, sub = jax.random.split(rng_inner)
        if use_ode:
            x_new = x + (0.5 * sigma_t * u + 0.5 * beta_t * x) * dt
        else:
            x_new = sample_kernel(sub, fwd_mean, scale)

        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        # Backward EM kernel
        bwd_em_mean = x_new - 0.5 * beta_t * x_new * dt
        fwd_lp = log_prob_kernel(x_new, fwd_mean, scale)
        bwd_em_lp = log_prob_kernel(x, bwd_em_mean, scale)
        log_w_em = log_w_em + (bwd_em_lp - fwd_lp)

        if return_ou_weight:
            # Backward OU kernel (exact noising)
            s = compute_s(t0_norm, t1_norm)
            scale_ou = jnp.sqrt(jnp.maximum(1.0 - s * s, 1e-20)) * prior_std
            bwd_ou_mean = s * x_new
            bwd_ou_lp = log_prob_kernel(x, bwd_ou_mean, scale_ou)
            log_w_ou = log_w_ou + (bwd_ou_lp - fwd_lp)
            return (x_new, log_w_em, log_w_ou, rng_inner), x_new
        else:
            return (x_new, log_w_em, rng_inner), x_new

    def sim_target_to_prior(state, t_code):
        if return_ou_weight:
            x, log_w_em, log_w_ou, rng_inner = state
        else:
            x, log_w_em, rng_inner = state

        t_code = t_code.astype(jnp.float32)

        beta_t = noise_schedule(t_code)
        sigma_t = jnp.sqrt(beta_t) * prior_std
        scale = sigma_t * jnp.sqrt(dt)

        t1_norm = t_code / t_float
        t0_norm = t1_norm - dt

        # Backward EM step (sample)
        bwd_mean = x - 0.5 * beta_t * x * dt
        rng_inner, sub = jax.random.split(rng_inner)
        x_new = sample_kernel(sub, bwd_mean, scale)

        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        # control at right endpoint
        lgv = jax.lax.stop_gradient(jax.grad(lgv_init)(x_new, t_code, sigma_t))
        u = model_state.apply_fn(
            params,
            x_new,
            (t_code / t_float) * jnp.ones(1),
            (d / t_float) * jnp.ones(1),
            lgv,
        )

        # Forward EM kernel
        fwd_mean = x_new + (sigma_t * u + 0.5 * beta_t * x) * dt
        fwd_lp = log_prob_kernel(x, fwd_mean, scale)
        bwd_lp = log_prob_kernel(x_new, bwd_mean, scale)
        log_w_em = log_w_em + (bwd_lp - fwd_lp)

        if return_ou_weight:
            # Replace backward by OU kernel (exact)
            s = compute_s(t0_norm, t1_norm)
            scale_ou = jnp.sqrt(jnp.maximum(1.0 - s * s, 1e-20)) * prior_std
            bwd_ou_mean = s * x
            bwd_ou_lp = log_prob_kernel(x_new, bwd_ou_mean, scale_ou)
            log_w_ou = log_w_ou + (bwd_ou_lp - fwd_lp)
            return (x_new, log_w_em, log_w_ou, rng_inner), x_new
        else:
            return (x_new, log_w_em, rng_inner), x_new

    rng, sub = jax.random.split(rng)

    if prior_to_target:
        codes_desc = step_codes[::-1]
        x0 = jnp.clip(prior_sample(seed=sub), -4 * prior_std, 4 * prior_std)
        if return_ou_weight:
            state0 = (x0, 0.0, 0.0, rng)
            (xT, log_ratio_em, log_ratio_ou, _), xs = jax.lax.scan(
                sim_prior_to_target, state0, codes_desc
            )
        else:
            state0 = (x0, 0.0, rng)
            (xT, log_ratio_em, _), xs = jax.lax.scan(
                sim_prior_to_target, state0, codes_desc
            )
        term_c = prior_logp(x0) - target_logp(xT)
    else:
        codes_asc = step_codes
        x0 = target.sample(sub, ())
        if return_ou_weight:
            state0 = (x0, 0.0, 0.0, rng)
            (xT, log_ratio_em, log_ratio_ou, _), xs = jax.lax.scan(
                sim_target_to_prior, state0, codes_asc
            )
        else:
            state0 = (x0, 0.0, rng)
            (xT, log_ratio_em, _), xs = jax.lax.scan(
                sim_target_to_prior, state0, codes_asc
            )
        term_c = prior_logp(xT) - target_logp(x0)

    traj = jnp.concatenate([x0[None, ...], xs], axis=0)
    stoch_c = jnp.zeros_like(log_ratio_em)
    run_c_em = -log_ratio_em

    if return_ou_weight:
        run_c_ou = -log_ratio_ou
        return xT, run_c_em, run_c_ou, stoch_c, term_c, traj
    else:
        return xT, run_c_em, stoch_c, term_c, traj


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
    *,
    return_ou_weight: bool = False,
    schedule_type: str = "linear",
    beta_min: float | None = None,
    beta_max: float | None = None,
    n_trapz: int = 1025,
):
    rngs = jax.random.split(rng, num=batch_size)
    if return_ou_weight:
        xT, rc_em, rc_ou, sc, tc, traj = jax.vmap(
            per_sample_rnd,
            in_axes=(0, None, None, None, None, None, None,
                     None, None, None, None, None,
                     None, None, None, None),  # all trailing kwargs shared
        )(
            rngs, model_state, params, prior_tuple, target,
            num_steps, noise_schedule, eval_steps, stop_grad,
            prior_to_target, use_ode,
            True, schedule_type, beta_min, beta_max, n_trapz
        )
        if return_traj:
            return xT, rc_em, rc_ou, sc, tc, traj
        return xT, rc_em, rc_ou, sc, tc
    else:
        xT, rc, sc, tc, traj = jax.vmap(
            per_sample_rnd,
            in_axes=(0, None, None, None, None, None, None,
                     None, None, None, None),  # original signature
        )(
            rngs, model_state, params, prior_tuple, target,
            num_steps, noise_schedule, eval_steps, stop_grad,
            prior_to_target, use_ode,
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