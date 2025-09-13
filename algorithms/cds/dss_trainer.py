# algorithms/dss/dss_trainer.py
from pathlib import Path
import logging
import time

import numpy as np
import jax
import jax.numpy as jnp
import distrax
import wandb
import orbax.checkpoint as ocp

from algorithms.common.diffusion_related.init_model import init_model
from algorithms.common.diffusion_related.noise_schedule import (
    build_karras_sigmas, make_eval_sigmas
)
from algorithms.dss.dss_eval import get_dss_eval_fn
from algorithms.common.eval_methods.utils import extract_last_entry
from algorithms.dss.dss_consistency import consistency_loss

log = logging.getLogger(__name__)
logging.getLogger("orbax.checkpoint").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def load_orbax_checkpoint(ckpt_dir: Path, step: int | None):
    path = Path(ckpt_dir).expanduser().resolve()
    # Accept both ".../checkpoints" and ".../checkpoints/<step>"
    if path.name.isdigit():
        guessed_step = int(path.name)
        if step is None:
            step = guessed_step
        path = path.parent  # manager root
    mngr = ocp.CheckpointManager(
        path,
        ocp.PyTreeCheckpointer(),
        options=ocp.CheckpointManagerOptions(create=False),
    )
    if step is None:
        step = mngr.latest_step()
    if step is None:
        existing = sorted(
            int(p.name) for p in path.iterdir()
            if p.is_dir() and p.name.isdigit()
        )
        raise FileNotFoundError(
            f"No checkpoint found in {path}. Existing steps: {existing}"
        )
    pkg = mngr.restore(step)
    return pkg, step

def build_teacher_from_ckpt(cfg, target, *, rng=None):
    if rng is None:
        rng = jax.random.PRNGKey(0)
    tcfg = cfg.algorithm.teacher
    ckpt_dir = Path(tcfg.ckpt_uri).expanduser().resolve()
    pkg, step = load_orbax_checkpoint(ckpt_dir, tcfg.step)
    raw_ms = pkg["model_state"]
    teacher_params = raw_ms.get("params", raw_ms) if isinstance(raw_ms, dict) else raw_ms.params
    teacher_state = init_model(rng, target.dim, cfg.algorithm).replace(params=teacher_params)
    print(f"[Teacher] loaded {tcfg.name} checkpoint step={step} from {ckpt_dir}")
    return teacher_state, step

def _global_norm(tree):
    sq = 0.0
    for x in jax.tree_util.tree_leaves(tree):
        sq = sq + jnp.sum(jnp.square(x))
    return jnp.sqrt(sq + 1e-8)

def _maybe_clip(grads, max_norm):
    if not max_norm or max_norm <= 0:
        return grads, _global_norm(grads), 1.0
    gnorm = _global_norm(grads)
    scale = jnp.minimum(1.0, max_norm / (gnorm + 1e-8))
    grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
    return grads, gnorm, scale

def dss_trainer(cfg, target, checkpointer):
    key_gen = jax.random.PRNGKey(int(cfg.seed))
    dim = target.dim
    alg_cfg = cfg.algorithm

    # Prior & target
    initial_density = distrax.MultivariateNormalDiag(
        jnp.zeros(dim), jnp.ones(dim) * alg_cfg.init_std
    )
    aux_tuple = (alg_cfg.init_std, initial_density.sample, initial_density.log_prob)
    target_samples = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    # Teacher (frozen)
    key, key_gen = jax.random.split(key_gen)
    teacher_state, _ = build_teacher_from_ckpt(cfg, target, rng=key)
    teacher_params = teacher_state.params

    # Student
    key, key_gen = jax.random.split(key_gen)
    model_state = init_model(key, dim, alg_cfg)
    if getattr(alg_cfg, "init_from_teacher", True):
        model_state = model_state.replace(params=teacher_params)
        print("[DSS] student params initialized from teacher.")

    # Build Karras EDM schedule identical to DISK
    kcfg = alg_cfg.teacher.karras
    sigma_max = float(alg_cfg.init_std)
    sigma_min = float(kcfg.sigma_min)
    rho = float(kcfg.get("rho", 7.0))
    sigmas = build_karras_sigmas(
        alg_cfg.num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho
    )  # (N+1,)
    d_sigmas  = jnp.maximum(sigmas[:-1] - sigmas[1:], 0.0)  # (N,)
    cm_sigma_data = getattr(alg_cfg, "cm", {}).get("sigma_data", 0.5)
    cm_sigma_min = sigma_min # last non-zero sigma

    # Loss
    loss_and_grads = jax.jit(
        jax.value_and_grad(consistency_loss, argnums=4, has_aux=True),
        static_argnames=(
            "batch_size","aux_tuple","target",
            "per_batch_t","teacher_use_sde","terminal_weighting",
            "w_clip","clip_x_std","cm_sigma_data","cm_sigma_min","return_pairs",
        ),
    )

    # Eval
    eval_fn_single, logger = get_dss_eval_fn(
        aux_tuple=aux_tuple,
        target=target,
        target_samples=target_samples,
        cfg=cfg,
        sigmas=sigmas,                      # we use sigmas[0] as σ_max
        cm_sigma_data=cm_sigma_data,
        cm_sigma_min=cm_sigma_min,                  # recommended: identity only at ~0
    )
    logger["train/grad_global_norm"] = []
    logger["train/clip_scale"] = []
    logger["loss/value"] = []

    # Training loop
    eval_freq = max(alg_cfg.iters // cfg.n_evals, 1)
    timer = 0.0
    for step in range(alg_cfg.iters):
        key, key_gen = jax.random.split(key_gen)
        t0 = time.time()

        (loss_value, metrics), grads = loss_and_grads(
            key,
            teacher_state, teacher_params,
            model_state,  model_state.params,
            batch_size=alg_cfg.batch_size,
            aux_tuple=aux_tuple,
            target=target,
            sigmas=sigmas,
            d_sigmas=d_sigmas,
            per_batch_t=getattr(alg_cfg, "per_batch_t", False),
            terminal_weighting=getattr(alg_cfg, "terminal_weighting", False),
            teacher_use_sde=getattr(alg_cfg, "teacher_use_sde", False),
            cm_sigma_data=cm_sigma_data,
            cm_sigma_min=cm_sigma_min,
            return_pairs=False,
    )

        grads, gnorm, clip_scale = _maybe_clip(grads, getattr(alg_cfg, "grad_clip", 0.0))
        model_state = model_state.apply_gradients(grads=grads)
        timer += time.time() - t0

        # Eval + logging
        if (step % eval_freq == 0) or (step == alg_cfg.iters - 1):
            key, key_gen = jax.random.split(key_gen)

            logger["stats/step"].append(step)
            logger["stats/wallclock"].append(timer)
            logger["stats/nfe"].append((step + 1) * alg_cfg.batch_size)
            logger["train/grad_global_norm"].append(float(gnorm))
            logger["train/clip_scale"].append(float(clip_scale))
            logger["loss/value"].append(float(loss_value))

            # merge metrics so W&B shows cm/* scalars
            for k, v in metrics.items():
                try:
                    logger.setdefault(k, []).append(float(jnp.asarray(v)))
                except Exception:
                    pass

            logger.update(eval_fn_single(model_state, key))

            # Prints + W&B
            from utils.print_util import print_results
            print_results(step, logger, cfg)
            if cfg.use_wandb:
                wandb.log(extract_last_entry(logger))
        
        # Checkpoint
        if (step % cfg.checkpoint.save_every == 0) or (step == alg_cfg.iters - 1):
            if jax.process_index() == 0:
                val = float(logger['KL/elbo_mov_avg'][-1]) if logger['KL/elbo_mov_avg'] else -float("inf")
                pkg = dict(model_state=jax.device_get(model_state), key_gen=key_gen, step=step, timer=timer)
                future = checkpointer.save(step, pkg, metrics={'elbo_mov_avg': val})
                print(f"[Orbax] step={step}  queued={future} metric={val}")