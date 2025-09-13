# ==================== DSS single-step evaluation (no control) ====================

import jax
import jax.numpy as jnp

from algorithms.common.eval_methods.utils import (
    moving_averages, save_samples, compute_reverse_ess
)
from algorithms.common.ipm_eval import discrepancies
from algorithms.dss.dss_reparam import cm_apply


def _build_langevin_init(prior_log_prob, target_log_prob, Smax):
    """
    Same side-input used during training. We will call it only at sigma=Smax.
    Returns a function f(x, sigma_scalar, sigma_for_grad) -> scalar
    """
    def body(x, sigma_scalar, sigma_for_grad):
        tr = jnp.clip(sigma_scalar / (Smax + 1e-12), 0.0, 1.0)
        return sigma_for_grad * (
            (1.0 - tr) * target_log_prob(x) + tr * prior_log_prob(x)
        )
    return body


def get_dss_eval_fn(
        *,
        aux_tuple,          # (init_std, prior_sampler, prior_log_prob)
        target,
        target_samples,
        cfg,
        sigmas,             # Karras array; we only use sigmas[0] = sigma_max
        cm_sigma_data=0.5,
        cm_sigma_min=1e-8,
        clip_x_std=4.0,
):
    """
    Returns (short_eval, logger) similar to get_eval_fn(), but wired for
    DSS single-step state mapping. Computes Sinkhorn/ΔlogZ/ESS, etc.
    """
    sigma_max = float(sigmas[0])
    init_std, prior_sampler, prior_log_prob = aux_tuple
    target_log_prob = target.log_prob

    # Build the side-input function; we'll close over the log-probs (no jit args).
    langevin_init = _build_langevin_init(prior_log_prob, target_log_prob, sigma_max)

    # Tiny jitted core that *only* depends on arrays/scalars & params:
    def _cm_step_apply(apply_fn, params, x0):
        diff = jnp.sqrt(2.0 * jnp.maximum(jnp.asarray(sigma_max, x0.dtype), 0.0))
        sigma_scalar = jnp.asarray(sigma_max, x0.dtype)

        # lgv = ∇_x [ diff * ((1 - tr) log p_tgt(x) + tr log p_prior(x)) ] at sigma=Smax
        def lgv_single(xi):
            return jax.grad(langevin_init)(xi, sigma_scalar, diff)

        lgv = jax.vmap(lgv_single)(x0)
        time_code = jnp.full((x0.shape[0], 1), sigma_max, dtype=x0.dtype)

        xT = cm_apply(
            apply_fn, params,
            x0, time_code, lgv,
            sigma_data=cm_sigma_data, sigma_min=cm_sigma_min
        )
        return xT

    _cm_step_apply_jit = jax.jit(_cm_step_apply, static_argnames=("apply_fn",))

    logger = {
        'KL/elbo': [],
        'KL/eubo': [],                    # not computed here
        'logZ/delta_forward': [],
        'logZ/forward': [],
        'logZ/delta_reverse': [],
        'logZ/reverse': [],
        'ESS/forward': [],
        'ESS/reverse': [],
        'discrepancies/mmd': [],
        'discrepancies/sd': [],
        'discrepancies/sinkhorn': [],
        'other/target_log_prob': [],
        'other/EMC': [],
        "stats/step": [],
        "stats/wallclock": [],
        "stats/nfe": [],
    }

    def short_eval(model_state, key):
        # Handle possible tuple(model_state1, model_state2) like your other eval.
        if isinstance(model_state, tuple):
            model_state = model_state[0]  # use first by default
        params = model_state.params

        # 1) Prior sampling (NO jit): avoids passing Python callables into jit.
        x0 = prior_sampler(seed=key, sample_shape=(cfg.eval_samples,))
        x0 = jnp.clip(x0, -clip_x_std * init_std, clip_x_std * init_std)

        # 2) Single-step state map via CM (jit-compiled core)
        xT = _cm_step_apply_jit(model_state.apply_fn, params, x0)

        # 3) Boundary-only costs (running/stochastic terms vanish for 1 step)
        # terminal_costs = log p_prior(x0) - log p_tgt(xT)
        prior_lp  = prior_log_prob(x0)
        target_lp = target_log_prob(xT)
        terminal_costs   = prior_lp - target_lp
        running_costs    = jnp.zeros_like(terminal_costs)
        stochastic_costs = jnp.zeros_like(terminal_costs)

        # 4) Importance weights & scalars
        # log w = log p_tgt(xT) - log p_prior(x0) = -(running+stochastic+terminal)
        log_is_weights = target_lp - prior_lp
        ln_z = jax.scipy.special.logsumexp(log_is_weights) - jnp.log(cfg.eval_samples)
        elbo = -jnp.mean(running_costs + terminal_costs)  # boundary-only ELBO

        # Fill reverse-direction metrics (primary for DSS)
        logger['logZ/reverse'].append(ln_z)
        if getattr(target, "log_Z", None) is not None:
            logger['logZ/delta_reverse'].append(jnp.abs(ln_z - target.log_Z))
        logger['KL/elbo'].append(elbo)
        logger['ESS/reverse'].append(compute_reverse_ess(log_is_weights, cfg.eval_samples))
        logger['other/target_log_prob'].append(jnp.mean(target_lp))

        # Forward metrics: not defined here (we only do single-step reverse)
        # Leave 'KL/eubo', 'logZ/forward', 'ESS/forward' empty.

        # Visuals & extras from the target helper
        logger.update(target.visualise(samples=xT, show=cfg.visualize_samples))

        if cfg.compute_emc and getattr(target, "has_entropy", False):
            logger['other/EMC'].append(target.entropy(xT))

        # Discrepancies (MMD/SD/Sinkhorn)
        for d in cfg.discrepancies:
            key_name = f'discrepancies/{d}'
            try:
                val = getattr(discrepancies, f'compute_{d}')(target_samples, xT, cfg) \
                      if target_samples is not None else jnp.inf
            except AttributeError:
                val = jnp.nan
            logger[key_name].append(val)

        # Always compute Sinkhorn if not already requested
        if 'sinkhorn' not in cfg.discrepancies:
            try:
                sink = discrepancies.compute_sinkhorn(target_samples, xT, cfg) \
                       if target_samples is not None else jnp.inf
            except AttributeError:
                sink = jnp.nan
            logger['discrepancies/sinkhorn'].append(sink)

        if cfg.moving_average.use_ma:
            logger.update(moving_averages(logger, window_size=cfg.moving_average.window_size))

        if cfg.save_samples:
            save_samples(cfg, logger, xT)

        return logger

    return short_eval, logger