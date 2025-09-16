# algorithms/sdss_vp/eval.py
from __future__ import annotations
from functools import partial
from pathlib import Path
import time
import csv

import numpy as np
import jax
import jax.numpy as jnp
import distrax
import matplotlib.pyplot as plt
import wandb

from algorithms.common.eval_methods.utils import (
    moving_averages, save_samples, compute_reverse_ess
)
from algorithms.sdss_vp.fb_rnd import rnd
from algorithms.sdss_vp.vp_ou import make_ou_weight_fn
from algorithms.common.ipm_eval import discrepancies


def plot_paths(full_paths, *, max_paths=64, wandb_key="figures/paths"):
    arr = np.asarray(jax.device_get(full_paths))
    if arr.ndim == 2:
        arr = arr[None, ...]
    bsz, t1, d = arr.shape
    t = np.arange(t1)

    if bsz > max_paths:
        idx = np.linspace(0, bsz - 1, max_paths, dtype=int)
        arr = arr[idx]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for p in arr:
        ax.plot(t, p[:, 0], lw=0.8, alpha=0.8)
    ax.set_xlabel("integration step")
    ax.set_ylabel(r"$x_0$")
    ax.set_title("Trajectories (first coordinate)")

    wb_img = wandb.Image(fig)
    plt.close(fig)
    return {wandb_key: [wb_img]}


def get_multi_eval_fn(
    rnd_base_ode,
    rnd_base_sde,
    target,
    target_samples,
    cfg,
    eval_budgets,
    viz_budget,
):
    # JIT one reverse per budget; ensure we get paths back
    rnd_rev_ode = {
        k: jax.jit(partial(
            rnd_base_ode, prior_to_target=True, eval_steps=k, return_traj=True
        ))
        for k in eval_budgets
    }
    rnd_rev_sde = {
        k: jax.jit(partial(
            rnd_base_sde, prior_to_target=True, eval_steps=k, return_traj=True
        ))
        for k in eval_budgets
    }

    logger = {
        "KL/elbo": [],
        "KL/eubo": [],
        "logZ/delta_forward": [],
        "logZ/forward": [],
        "logZ/delta_reverse": [],
        "logZ/reverse": [],
        "ESS/forward": [],
        "ESS/reverse": [],
        "discrepancies/mmd": [],
        "discrepancies/sd": [],
        "other/target_log_prob": [],
        "other/EMC": [],
        "stats/step": [],
        "stats/wallclock": [],
        "stats/nfe": [],
    }

    def _suffix(key, k):
        return f"{key}@k={k}"

    def _append(dct, key, val):
        dct.setdefault(key, []).append(val)

    def eval_once(model_state, rng):
        samples_for_plot = None

        for k in eval_budgets:
            # ODE pass: samples for discrepancies/visuals
            rng, sub = jax.random.split(rng)
            (samples_ode, _, _, _, paths_ode) = rnd_rev_ode[k](
                sub, model_state, model_state.params
            )

            # Discrepancies per budget (on ODE samples)
            for d in cfg.discrepancies:
                key = f"discrepancies/{d}"
                val = getattr(discrepancies, f"compute_{d}")(
                    target_samples, samples_ode, cfg
                )
                _append(logger, _suffix(key, k), val)
                if k == viz_budget:
                    _append(logger, key, val)

            # Optional path plots for viz_budget
            if k == viz_budget:
                logger.update(plot_paths(paths_ode, wandb_key="figures/paths"))
                _append(logger, "other/target_log_prob",
                        jnp.mean(target.log_prob(samples_ode)))
                samples_for_plot = samples_ode

            # SDE pass: compute EM and OU weights directly from rnd()
            rng, sub = jax.random.split(rng)
            # rnd_base_sde was created with return_ou_weight=True
            (_, run_c_em, run_c_ou, stoch_c, term_c, _paths_sde) = rnd_rev_sde[k](
                sub, model_state, model_state.params
            )

            # EM metrics
            log_w_em = -(run_c_em + stoch_c + term_c)
            ln_z_em = jax.scipy.special.logsumexp(log_w_em) - jnp.log(cfg.eval_samples)
            elbo_em = jnp.mean(log_w_em)
            ess_em = compute_reverse_ess(log_w_em, cfg.eval_samples)

            _append(logger, _suffix("logZ/reverse", k), ln_z_em)
            _append(logger, _suffix("KL/elbo", k), elbo_em)
            _append(logger, _suffix("ESS/reverse", k), ess_em)

            # Un‑suffixed for viz_budget
            if k == viz_budget:
                _append(logger, "logZ/reverse", ln_z_em)
                _append(logger, "KL/elbo", elbo_em)
                _append(logger, "ESS/reverse", ess_em)

            # OU metrics (computed independently, no correction)
            log_w_ou = -(run_c_ou + stoch_c + term_c)
            ln_z_ou = jax.scipy.special.logsumexp(log_w_ou) - jnp.log(cfg.eval_samples)
            elbo_ou = jnp.mean(log_w_ou)
            ess_ou = compute_reverse_ess(log_w_ou, cfg.eval_samples)

            _append(logger, _suffix("logZ/reverse_ou", k), ln_z_ou)
            _append(logger, _suffix("KL/elbo_ou", k), elbo_ou)
            _append(logger, _suffix("ESS/reverse_ou", k), ess_ou)

        # Visuals and extras for viz_budget
        if samples_for_plot is not None:
            logger.update(
                target.visualise(samples=samples_for_plot, show=cfg.visualize_samples)
            )
        if cfg.compute_emc and cfg.target.has_entropy:
            _append(logger, "other/EMC", target.entropy(samples_for_plot))

        if cfg.moving_average.use_ma:
            logger.update(
                moving_averages(
                    logger, window_size=cfg.moving_average.window_size
                )
            )
        if cfg.save_samples and (samples_for_plot is not None):
            save_samples(cfg, logger, samples_for_plot)

        return logger

    return eval_once, logger


def run_eval_sdss_vp(cfg, target, model_state):
    """Evaluate at multiple budgets; compute EM and OU weights directly from rnd(); save CSVs."""
    rng = jax.random.PRNGKey(cfg.seed)

    dim = target.dim
    alg_cfg = cfg.algorithm
    eval_cfg = cfg.eval
    out_root = Path("/home/pascal/projects/single-step-diffusion-samplers/logs/eval/runs")
    out_root.mkdir(parents=True, exist_ok=True)

    # Prior
    prior = distrax.MultivariateNormalDiag(
        jnp.zeros(dim), jnp.ones(dim) * alg_cfg.init_std
    )
    prior_tuple = (alg_cfg.init_std, prior.sample, prior.log_prob)

    # Target samples for discrepancies
    target_samples = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    # Eval budgets
    eval_budgets = list(
        getattr(alg_cfg, "multi_eval_steps", [1, 2, 4, 8, 16, 32, 64, 128])
    )
    viz_budget = getattr(alg_cfg, "viz_eval_steps", max(eval_budgets))
    for k in eval_budgets:
        if alg_cfg.num_steps % k != 0:
            raise ValueError(
                f"num_steps={alg_cfg.num_steps} not divisible by k={k}"
            )

    # Two RND bases: ODE for samples; SDE for weights (EM+OU) from scratch
    rnd_base_ode = partial(
        rnd,
        batch_size=cfg.eval_samples,
        prior_tuple=prior_tuple,
        target=target,
        num_steps=alg_cfg.num_steps,
        noise_schedule=alg_cfg.noise_schedule,
        stop_grad=True,
        use_ode=True,
        return_traj=True,
        # return_ou_weight left False (default) to avoid extra work here
    )

    # schedule metadata for OU shrink:
    schedule_type = getattr(alg_cfg, "schedule_type", "linear")
    bmin = getattr(alg_cfg, "beta_min", None)
    bmax = getattr(alg_cfg, "beta_max", None)

    rnd_base_sde = partial(
        rnd,
        batch_size=cfg.eval_samples,
        prior_tuple=prior_tuple,
        target=target,
        num_steps=alg_cfg.num_steps,
        noise_schedule=alg_cfg.noise_schedule,
        stop_grad=True,
        use_ode=False,
        return_traj=True,
        # ask rnd to compute OU weights alongside EM
        return_ou_weight=True,
        schedule_type=schedule_type,
        beta_min=bmin,
        beta_max=bmax,
        n_trapz=1025,
    )

    eval_fn, logger = get_multi_eval_fn(
        rnd_base_ode=rnd_base_ode,
        rnd_base_sde=rnd_base_sde,
        target=target,
        target_samples=target_samples,
        cfg=cfg,
        eval_budgets=eval_budgets,
        viz_budget=viz_budget,
    )

    # Repeat loop (logger accumulates across calls)
    n_repeat = int(eval_cfg.n_repeat)
    t0 = time.time()
    for rep in range(n_repeat):
        rng, sub = jax.random.split(rng)
        _ = eval_fn(model_state, sub)  # appends to `logger`

        # Optional lightweight W&B snapshot each rep (viz budget only)
        if cfg.use_wandb and logger.get("logZ/reverse"):
            wb_payload = {
                "eval/rep": rep,
                "eval/logZ_em": float(logger["logZ/reverse"][-1]),
                "eval/ELBO_em": float(logger["KL/elbo"][-1]),
                "eval/ESS_em": float(logger["ESS/reverse"][-1]),
            }
            if "figures/paths" in logger and logger["figures/paths"]:
                wb_payload["figures/paths"] = logger["figures/paths"][-1]
            for d in cfg.discrepancies:
                if logger.get(f"discrepancies/{d}"):
                    wb_payload[f"discrepancies/{d}"] = float(logger[f"discrepancies/{d}"][-1])
            wandb.log(wb_payload)

    # Save CSVs
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"{alg_cfg.name}_{cfg.target.name}_seed{cfg.seed}"
    raw_path = out_root / f"{tag}_raw_{ts}.csv"
    agg_path = out_root / f"{tag}_agg_{ts}.csv"

    # RAW dump: per k, per metric, per rep.
    raw_rows = []
    def _to_np(x):  # DeviceArray -> float
        return float(np.asarray(x))

    for k in eval_budgets:
        em_lnz = logger.get(f"logZ/reverse@k={k}", [])
        em_elbo = logger.get(f"KL/elbo@k={k}", [])
        em_ess = logger.get(f"ESS/reverse@k={k}", [])

        ou_lnz = logger.get(f"logZ/reverse_ou@k={k}", [])
        ou_elbo = logger.get(f"KL/elbo_ou@k={k}", [])
        ou_ess = logger.get(f"ESS/reverse_ou@k={k}", [])

        # Rows per rep for EM
        for i in range(len(em_lnz)):
            raw_rows.append({
                "k": k, "kind": "weights_em",
                "rep": i,
                "logZ": _to_np(em_lnz[i]),
                "ELBO": _to_np(em_elbo[i]),
                "ESS": _to_np(em_ess[i]),
            })
        # Rows per rep for OU
        for i in range(len(ou_lnz)):
            raw_rows.append({
                "k": k, "kind": "weights_ou",
                "rep": i,
                "logZ": _to_np(ou_lnz[i]),
                "ELBO": _to_np(ou_elbo[i]),
                "ESS": _to_np(ou_ess[i]),
            })
        # Discrepancies on ODE samples
        for d in cfg.discrepancies:
            disc_key = f"discrepancies/{d}@k={k}"
            disc_vals = logger.get(disc_key, [])
            for i in range(len(disc_vals)):
                raw_rows.append({
                    "k": k, "kind": "samples_ode",
                    "rep": i,
                    f"disc/{d}": _to_np(disc_vals[i]),
                })

    # Save RAW
    raw_cols = sorted({c for r in raw_rows for c in r.keys()})
    with raw_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=raw_cols)
        w.writeheader()
        for r in raw_rows:
            w.writerow(r)

    # Save AGG (mean/std per (kind,k))
    agg_rows = []
    from collections import defaultdict
    bucket = defaultdict(list)
    for r in raw_rows:
        bucket[(r["kind"], r["k"])].append(r)
    for (kind, k), rows in sorted(bucket.items(), key=lambda z: (z[0][1], z[0][0])):
        out = {"kind": kind, "k": k, "count": len(rows)}
        # Numerics present
        num_keys = set()
        for r in rows:
            for kk in r:
                if kk in ("logZ", "ELBO", "ESS") or kk.startswith("disc/"):
                    num_keys.add(kk)
        for kk in sorted(num_keys):
            vals = np.array([r[kk] for r in rows if kk in r], dtype=float)
            out[f"{kk}_mean"] = float(vals.mean()) if vals.size else np.nan
            out[f"{kk}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        agg_rows.append(out)

    agg_cols = sorted({c for r in agg_rows for c in r.keys()})
    with agg_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_cols)
        w.writeheader()
        for r in agg_rows:
            w.writerow(r)

    elapsed = time.time() - t0
    if cfg.use_wandb:
        wandb.log({"eval/runtime_sec": elapsed})

    print(f"[eval] Wrote RAW rows to: {raw_path}")
    print(f"[eval] Wrote aggregated stats to: {agg_path}")
    return {"raw_csv": str(raw_path), "agg_csv": str(agg_path)}