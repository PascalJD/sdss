# algorithms/sdss/sdss_trainer.py

from functools import partial
from time import time
import logging

import distrax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu  
import wandb

from algorithms.common.diffusion_related.init_model import init_model
from algorithms.common.diffusion_related.noise_schedule import build_karras_sigmas
from algorithms.common.eval_methods.stochastic_oc_methods import get_eval_fn
from algorithms.common.eval_methods.utils import extract_last_entry
from algorithms.sdss.sdss_is_weights import rnd
from algorithms.sdss.sdss_shortcut import shortcut_loss
from utils.print_util import print_results

log = logging.getLogger(__name__)
logging.getLogger("orbax.checkpoint").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)


def sdss_trainer(cfg, target, checkpointer):
    key_gen = jax.random.PRNGKey(cfg.seed)
    dim = target.dim
    alg_cfg = cfg.algorithm
    batch_size = alg_cfg.batch_size

    # Prior & target
    initial_density = distrax.MultivariateNormalDiag(
        jnp.zeros(dim), jnp.ones(dim) * alg_cfg.init_std
    )
    aux_tuple = (alg_cfg.init_std, initial_density.sample, initial_density.log_prob)
    target_samples = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    # Model 
    key, key_gen = jax.random.split(key_gen)
    model_state = init_model(key, dim, alg_cfg)

    # Schedule
    sigma_max = float(alg_cfg.get("sigma_max", alg_cfg.init_std))
    sigma_min = float(alg_cfg.get("sigma_min", 2e-3))
    rho = float(alg_cfg.get("rho", 7.0))
    sigmas = build_karras_sigmas(
            num_steps=alg_cfg.num_steps,  # number of EM steps 
            sigma_max=sigma_max,  # T
            sigma_min=sigma_min,  # epsilon
            rho=rho
        )  # N+1 timesteps, N steps
    N_train = sigmas.shape[0] - 1  # number of EM steps
    # pick n_eval_steps+1 boundaries (including both endpoints)
    n_eval_steps = alg_cfg.eval_num_steps  # number of EM steps (inference) 
    idx = jnp.rint(jnp.linspace(0, N_train, n_eval_steps + 1, endpoint=True)).astype(jnp.int32)  # N_train is the last timestep index
    sigmas_eval = sigmas[idx]
    d_sigmas_eval = jnp.maximum(sigmas_eval[:-1] - sigmas_eval[1:], 0.0)
    d_sigmas = jnp.abs(sigmas[:-1] - sigmas[1:]) 
    d_sigmas_eval = jnp.abs(sigmas_eval[:-1] - sigmas_eval[1:]) 
    # print(f"\n\nSIGMAS:\n {sigmas}, length={sigmas.shape[0]}")
    # print(f"\nEVAL SIGMAS:\n {sigmas_eval}, length={sigmas_eval.shape[0]}\n\n")

    # Objective 
    def sampling_loss(key, model_state, params):
        key, key_gen = jax.random.split(key)
        key, key_gen = jax.random.split(key)
        (
            final_x, run_costs, stoch_costs, term_costs, full_paths
        ) = rnd(
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
            return_traj=True,
        )
        neg_elbo_vals = run_costs + term_costs
        elbo_loss = jnp.mean(neg_elbo_vals)
        return elbo_loss, (key_gen, full_paths)
    grad_sampling_loss = jax.jit(
        jax.value_and_grad(sampling_loss, argnums=2, has_aux=True)
    )
    def self_distillation_loss(key, model_state, params, paths):
        key, key_gen = jax.random.split(key)
        mse = shortcut_loss(
            key,
            model_state,
            params,
            paths,
            batch_size,
            aux_tuple,
            target,
            sigmas,
        )
        return jnp.mean(mse), key_gen
    grad_self_distillation_loss = jax.jit(
        jax.value_and_grad(self_distillation_loss, argnums=2, has_aux=True)
    )

    rnd_short = partial(
        rnd,
        batch_size=cfg.eval_samples, 
        aux_tuple=aux_tuple,
        target=target, 
        sigmas=sigmas_eval,
        d_sigmas=d_sigmas_eval,
        stop_grad=True, 
        use_ode=True,
        return_traj=False,
    )
    eval_fn, logger = get_eval_fn(rnd_short, target, target_samples, cfg)
    eval_freq = max(alg_cfg.iters // cfg.n_evals, 1)
    logger["train/sc"] = []
    logger["train/grad_diff"] = []
    logger["train/grad_sc"] = []
    logger["train/sc"] = []
    logger["KL/neg_elbo"] = []

    # Gradient clipping helper
    def clip_tree(tree):
        g_flat, _ = jax.flatten_util.ravel_pytree(tree)
        factor = jnp.minimum(1., alg_cfg.grad_clip / (1e-9 + jnp.linalg.norm(g_flat)))
        return jtu.tree_map(lambda x: x * factor, tree)

    # Training loop
    eval_freq = max(alg_cfg.iters // cfg.n_evals, 1)
    timer = 0.0
    for step in range(alg_cfg.iters):
        key, key_gen = jax.random.split(key_gen)
        t0 = time()
        # Training step sampling loss
        (
            (elbo_loss, (key, full_paths)), grads_sampling
        ) = grad_sampling_loss(key, model_state, model_state.params)
        diff_grad_norm = jnp.sqrt(
                sum(jnp.vdot(p, p) for p in jtu.tree_leaves(grads_sampling))
        )
        grads_sampling  = clip_tree(grads_sampling)
        model_state = model_state.apply_gradients(grads=grads_sampling)

        # Training step self-distillation loss
        (
            (sc_loss, key), grads_sd
        ) = grad_self_distillation_loss(
            key, model_state, model_state.params, full_paths
        )
        sc_grad_norm = jnp.sqrt(
                sum(jnp.vdot(p, p) for p in jtu.tree_leaves(grads_sd))
        )
        grads_sd = clip_tree(grads_sd)
        model_state = model_state.apply_gradients(grads=grads_sd)
        timer += time() - t0

        # Eval / logging
        if (step % eval_freq == 0) or (step == alg_cfg.iters - 1):
            key, key_gen = jax.random.split(key_gen)
            logger["stats/step"].append(step)
            logger["stats/wallclock"].append(timer)
            logger["stats/nfe"].append((step + 1) * alg_cfg.batch_size)
            logger["KL/neg_elbo"].append(elbo_loss)
            logger["train/sc"].append(sc_loss)
            logger["train/grad_diff"].append(diff_grad_norm)
            logger["train/grad_sc"].append(sc_grad_norm)

            logger.update(eval_fn(model_state, key))
            print_results(step, logger, cfg)

            if cfg.use_wandb:
                wandb.log(extract_last_entry(logger))

            # Checkpoint  
            if step > 0 :
                metrics = {}
                if logger.get('KL/elbo_mov_avg'):
                    metrics['KL/elbo_mov_avg'] = float(logger['KL/elbo_mov_avg'][-1])
                if logger.get('KL/neg_elbo'):
                    metrics['KL/neg_elbo'] = float(logger['KL/neg_elbo'][-1])
                if logger.get('discrepancies/sd'):
                    metrics['discrepancies/sd'] = float(logger['discrepancies/sd'][-1])

                pkg = dict(
                    model_state=jax.device_get(model_state),
                    key_gen=key_gen,
                    step=step,
                    timer=timer,
                )
                future = checkpointer.save(step, pkg, metrics=metrics)
                print(f"[Orbax] step={step} saved with metrics={metrics} (queued={future})")