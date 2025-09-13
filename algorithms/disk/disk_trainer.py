# algorithms/disk/disk_trainer.py

from functools import partial
from time import time
import logging
from pathlib import Path

import distrax
import jax
import jax.numpy as jnp
import wandb
import orbax.checkpoint as ocp
from orbax.checkpoint._src.checkpoint_managers import preservation_policy as _pp

from algorithms.common.diffusion_related.init_model import init_model
from algorithms.common.diffusion_related.noise_schedule import build_karras_sigmas
from algorithms.common.eval_methods.stochastic_oc_methods import get_eval_fn
from algorithms.common.eval_methods.utils import extract_last_entry
from algorithms.disk.disk_is_weights import rnd, neg_elbo
from algorithms.disk.disk_debug import fig_traj_over_time, fig_paths_2d
from utils.print_util import print_results

log = logging.getLogger(__name__)
logging.getLogger("orbax.checkpoint").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

class _Compat:
    LatestN = _pp.LatestN
    BestN   = _pp.BestN
    AnyPreservationPolicy = _pp.AnyPreservationPolicy
p = _Compat

def best_elbo_fn(metrics):
    return float(metrics.get('elbo_mov_avg', -jnp.inf))

def disk_trainer(cfg, target, checkpointer):
    key_gen = jax.random.PRNGKey(cfg.seed)
    dim = target.dim
    alg_cfg = cfg.algorithm

    # Prior & target
    initial_density = distrax.MultivariateNormalDiag(
        jnp.zeros(dim), jnp.ones(dim) * alg_cfg.init_std
    )
    aux_tuple = (alg_cfg.init_std, initial_density.sample, initial_density.log_prob)
    target_samples = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    # Model 
    key, key_gen = jax.random.split(key_gen)
    model_state = init_model(key, dim, alg_cfg)

    # NOTE: For EDM, set init_std == sigma_max (so prior ~ N(0, T^2 I)).
    kcfg = getattr(alg_cfg, "karras", {})
    sigma_max = float(kcfg.get("sigma_max", alg_cfg.init_std))
    sigma_min = float(kcfg.get("sigma_min", 2e-3))
    rho = float(kcfg.get("rho", 7.0))
    sigmas = build_karras_sigmas(
        num_steps=alg_cfg.num_steps,
        sigma_max=sigma_max,  # T
        sigma_min=sigma_min,  # epsilon
        rho=rho
    )
    d_sigmas = jnp.abs(sigmas[:-1] - sigmas[1:]) 

    # Objective / eval
    loss = jax.jit(
        jax.grad(neg_elbo, 2, has_aux=True),
        static_argnums=(3, 4, 5),
    )
    rnd_short = partial(
        rnd,
        batch_size=cfg.eval_samples,
        aux_tuple=aux_tuple,
        target=target,
        sigmas=sigmas,
        d_sigmas=d_sigmas,
        stop_grad=True,
        use_ode=False,
        return_traj=False,
    )
    eval_fn, logger = get_eval_fn(rnd_short, target, target_samples, cfg)

    # Training loop
    eval_freq = max(alg_cfg.iters // cfg.n_evals, 1)
    timer = 0.0
    for step in range(alg_cfg.iters):
        key, key_gen = jax.random.split(key_gen)
        t0 = time()
        grads, _ = loss(
            key, 
            model_state, 
            model_state.params, 
            alg_cfg.batch_size,
            aux_tuple, 
            target,
            sigmas=sigmas, 
            d_sigmas=d_sigmas,
        )
        timer += time() - t0
        model_state = model_state.apply_gradients(grads=grads)

        # Eval / logging
        if (step % eval_freq == 0) or (step == alg_cfg.iters - 1):
            key, key_gen = jax.random.split(key_gen)
            logger["stats/step"].append(step)
            logger["stats/wallclock"].append(timer)
            logger["stats/nfe"].append((step + 1) * alg_cfg.batch_size)

            logger.update(eval_fn(model_state, key))
            print_results(step, logger, cfg)

            # quick viz of trajectories using the SAME schedule
            if getattr(alg_cfg, "debug", {}).get("traj_batch", 0) and cfg.use_wandb:
                K = min(int(alg_cfg.debug.traj_batch), cfg.eval_samples)
                key, key_gen = jax.random.split(key_gen)
                xT, rc, sc, tc, traj = rnd(
                    key=key,
                    model_state=model_state,
                    params=model_state.params,
                    batch_size=K,
                    aux_tuple=aux_tuple,
                    target=target,
                    sigmas=sigmas,
                    d_sigmas=d_sigmas,
                    stop_grad=True,
                    prior_to_target=True,
                    use_ode=False,
                    return_traj=True,
                )
                fig_t = fig_traj_over_time(traj, which_dim=0, title="DISK trajectories (dim 0)")
                wandb.log({"debug/vis/disk_traj_dim0": wandb.Image(fig_t)}, step=step)
                if target.dim >= 2:
                    fig_xy = fig_paths_2d(traj, dims=(0, 1), title="DISK 2D paths")
                    wandb.log({"debug/vis/disk_paths_2d": wandb.Image(fig_xy)}, step=step)

                # one-time log of the sigma grid
                if step == 0:
                    wandb.log({
                        "debug/karras/sigma_max": float(sigmas[0]),
                        "debug/karras/sigma_min": float(sigmas[alg_cfg.num_steps-1]),
                    }, step=step)

            if cfg.use_wandb:
                wandb.log(extract_last_entry(logger))

        # Checkpoint
        if (step % cfg.checkpoint.save_every == 0) or (step == alg_cfg.iters - 1):
            if jax.process_index() == 0:
                val = float(logger['KL/elbo_mov_avg'][-1]) if logger['KL/elbo_mov_avg'] else -float("inf")
                metrics = {'elbo_mov_avg': val}
                pkg = dict(
                    model_state=jax.device_get(model_state),
                    key_gen=key_gen,
                    step=step,
                    timer=timer,
                )
                future = checkpointer.save(step, pkg, metrics=metrics)
                print(f"[Orbax] step={step}  queued={future} metric={val}")