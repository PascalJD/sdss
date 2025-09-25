# algorithms/sdss_vp/eval.py
from __future__ import annotations
from functools import partial
from pathlib import Path
from typing import Any
import time
import csv
import json

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
# from algorithms.sdss_vp.vp_ou import make_ou_weight_fn
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
    rnd_fwd_ode = {
        k: jax.jit(partial(
            rnd_base_ode, prior_to_target=False, eval_steps=k, return_traj=True
        ))
        for k in eval_budgets
    }
    rnd_fwd_sde = {
        k: jax.jit(partial(
            rnd_base_sde, prior_to_target=False, eval_steps=k, return_traj=True
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
        "logZ/detflow": [],
        "logZ/delta_detflow": [],
        "ESS/detflow": [],
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
            (samples_ode, _rc_ode, logdet_ode, term_c_ode, paths_ode) = rnd_rev_ode[k](
                sub, model_state, model_state.params
            )

            # Deterministic-flow weights and logZ
            log_w_det = -term_c_ode + logdet_ode
            ln_z_det = jax.scipy.special.logsumexp(log_w_det) - jnp.log(cfg.eval_samples)
            ess_det = compute_reverse_ess(log_w_det, cfg.eval_samples)
            elbo_det  = jnp.mean(log_w_det)
            _append(logger, _suffix("logZ/detflow", k), ln_z_det)
            _append(logger, _suffix("ESS/detflow", k), ess_det)
            if k == viz_budget:
                _append(logger, "logZ/detflow", ln_z_det)
                _append(logger, "ESS/detflow", ess_det)
                if target.log_Z is not None:
                    _append(logger, "logZ/delta_detflow", jnp.abs(ln_z_det - target.log_Z))

            # Deterministic forward (EUBO) 
            rng, sub = jax.random.split(rng)
            (_, _rc_fwd_ode, logdet_fwd_ode, term_c_fwd_ode, _paths_fwd_ode) = rnd_fwd_ode[k](
                sub, model_state, model_state.params
            )
            log_w_det_fwd = -term_c_fwd_ode + logdet_fwd_ode

            # Upper bound and a second logZ estimator
            eubo_det = jnp.mean(log_w_det_fwd) 
            ln_z_det_fwd = -(jax.scipy.special.logsumexp(-log_w_det_fwd) - jnp.log(cfg.eval_samples))

            _append(logger, _suffix("KL/elbo_detflow", k), elbo_det)  # reverse ELBO
            _append(logger, _suffix("KL/eubo_detflow", k), eubo_det)  # forward EUBO
            _append(logger, _suffix("logZ/forward_detflow", k), ln_z_det_fwd)

            if k == viz_budget:
                _append(logger, "KL/elbo_detflow", elbo_det)
                _append(logger, "KL/eubo_detflow", eubo_det)
                _append(logger, "logZ/forward_detflow", ln_z_det_fwd)
                if target.log_Z is not None:
                    _append(logger, "logZ/delta_forward_detflow", jnp.abs(ln_z_det_fwd - target.log_Z))

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

            # SDE pass: compute integrator pass 
            rng, sub = jax.random.split(rng)
            (_, run_c_em, stoch_c, term_c, _paths_sde) = rnd_rev_sde[k](
                sub, model_state, model_state.params
            )

            # Rev metrics
            log_w_em = -(run_c_em + stoch_c + term_c)
            ln_z_em = jax.scipy.special.logsumexp(log_w_em) - jnp.log(cfg.eval_samples)
            elbo_em = jnp.mean(log_w_em)
            ess_em = compute_reverse_ess(log_w_em, cfg.eval_samples)
            _append(logger, _suffix("logZ/reverse", k), ln_z_em)
            _append(logger, _suffix("KL/elbo", k), elbo_em)
            _append(logger, _suffix("ESS/reverse", k), ess_em)
            if k == viz_budget:
                _append(logger, "logZ/reverse", ln_z_em)
                _append(logger, "KL/elbo", elbo_em)
                _append(logger, "ESS/reverse", ess_em)
                if target.log_Z is not None:
                    _append(logger, "logZ/delta_reverse", jnp.abs(ln_z_em - target.log_Z))

            # Fwd metrics 
            rng, sub = jax.random.split(rng)
            (_, run_c_fwd, stoch_c_fwd, term_c_fwd, _) = rnd_fwd_sde[k](sub, model_state, model_state.params)
            fwd_log_w = -(run_c_fwd + stoch_c_fwd + term_c_fwd)
            eubo = jnp.mean(fwd_log_w)
            ln_z_fwd = -(jax.scipy.special.logsumexp(-fwd_log_w) - jnp.log(cfg.eval_samples))
            ess_fwd = jnp.exp(ln_z_fwd - (jax.scipy.special.logsumexp(fwd_log_w) - jnp.log(cfg.eval_samples)))
            _append(logger, _suffix("KL/eubo", k), eubo)
            _append(logger, _suffix("logZ/forward", k), ln_z_fwd)
            _append(logger, _suffix("ESS/forward", k), ess_fwd)
            if k == viz_budget:
                _append(logger, "KL/eubo", eubo)
                _append(logger, "logZ/forward", ln_z_fwd)
                _append(logger, "ESS/forward", ess_fwd)
                if target.log_Z is not None:
                    _append(logger, "logZ/delta_forward", jnp.abs(ln_z_fwd - target.log_Z))


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


def _is_scalar_metric(x: Any) -> bool:
    """Return True if x is a scalar we can safely put in CSV."""
    # Skip wandb images/objects
    if isinstance(x, getattr(wandb, "Image", ())):
        return False
    # Plain scalars
    if isinstance(x, (int, float, bool, str)):
        return True
    # NumPy / JAX scalars
    if isinstance(x, np.generic):
        return True
    if isinstance(x, (np.ndarray, jnp.ndarray)):
        return np.asarray(x).shape == ()
    return False


def _to_python(x: Any) -> Any:
    """Convert JAX/NumPy types to vanilla Python; arrays -> lists (for JSON)."""
    if isinstance(x, (np.ndarray, jnp.ndarray)):
        arr = np.asarray(x)
        return arr.item() if arr.shape == () else arr.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def _last_scalar_row(logger: dict[str, list]) -> dict[str, Any]:
    """Pick the last scalar value for each metric key (for CSV append)."""
    row = {}
    for k, v in logger.items():
        if not isinstance(v, list) or len(v) == 0:
            continue
        last = v[-1]
        # Skip non-serializable artifacts (e.g., wandb.Image) or non-scalars
        if _is_scalar_metric(last):
            row[k] = _to_python(last)
    return row


def _full_history_rows(logger: dict[str, list], *, scalars_only: bool = True) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Build full history as rows. If scalars_only=True, keep only scalar-valued metrics
    (good for CSV). If False, convert arrays to lists (good for JSON).
    """
    # Determine number of evaluation rows from max list length
    lengths = [len(v) for v in logger.values() if isinstance(v, list)]
    n = max(lengths) if lengths else 0

    # Establish the set of keys to include
    keys: list[str] = []
    for k, v in logger.items():
        if not isinstance(v, list):
            continue
        if scalars_only:
            # Keep if the last recorded value is scalar; this is a heuristic
            if len(v) > 0 and _is_scalar_metric(v[-1]):
                keys.append(k)
        else:
            keys.append(k)

    # Build rows
    rows: list[dict[str, Any]] = []
    for i in range(n):
        r: dict[str, Any] = {}
        for k in keys:
            seq = logger.get(k, [])
            val = seq[i] if i < len(seq) else None
            if scalars_only:
                r[k] = _to_python(val) if _is_scalar_metric(val) else None
            else:
                # JSON path: allow lists; convert arrays to lists
                r[k] = _to_python(val)
        rows.append(r)

    return keys, rows


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def save_metrics(
    logger: dict[str, list],
    cfg,
    *,
    fmt: str = "csv",  # "csv" or "json"
    mode: str = "append",  # "append" or "overwrite" (CSV); JSON always overwrites
    filename: str | None = None,  # default: metrics.csv / metrics.json
) -> str:
    # Resolve directory from Hydra config and ensure it exists
    metrics_dir = Path(getattr(cfg.paths, "metrics_dir"))
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Pick filename
    if filename is None:
        filename = "metrics.csv" if fmt.lower() == "csv" else "metrics.json"
    path = metrics_dir / filename

    fmt = fmt.lower()
    mode = mode.lower()

    if fmt == "json":
        # Save the entire (sanitized) history as a single JSON file
        sanitized: dict[str, list] = {}
        for k, v in logger.items():
            if isinstance(v, list):
                sv = []
                for item in v:
                    if isinstance(item, getattr(wandb, "Image", ())):
                        sv.append(None)  # drop images
                    else:
                        sv.append(_to_python(item))
                sanitized[k] = sv
        _atomic_write_text(path, json.dumps(sanitized, indent=2))
        return str(path)

    if fmt != "csv":
        raise ValueError("fmt must be 'csv' or 'json'")

    if mode == "overwrite":
        # Rebuild whole scalar history and rewrite CSV
        header, rows = _full_history_rows(logger, scalars_only=True)
        header = sorted(header)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, None) for k in header})
        tmp.replace(path)
        return str(path)

    if mode != "append":
        raise ValueError("mode must be 'append' or 'overwrite' for CSV")

    # Append one last-row snapshot of scalar metrics
    row = _last_scalar_row(logger)

    # If file doesn't exist, write header + first row
    if not path.exists():
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return str(path)

    # If header changed (new keys), rebuild entire file to keep one consistent CSV
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        existing_header = next(reader, None)
    if existing_header is None or set(existing_header) != set(row.keys()):
        # Rewrite whole file from history to keep a single consistent file
        return save_metrics(logger, cfg, fmt="csv", mode="overwrite", filename=filename)

    # Normal append
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)
    return str(path)
