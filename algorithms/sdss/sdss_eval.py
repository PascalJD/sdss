# algorithms/sdss/sdss_eval.py
import os, json, logging
from pathlib import Path
from collections import defaultdict
from functools import partial
from numbers import Number

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import distrax
from omegaconf import DictConfig

from algorithms.common.eval_methods.stochastic_oc_methods import get_eval_fn
from algorithms.common.diffusion_related.noise_schedule import build_karras_sigmas
from algorithms.sdss.sdss_is_weights import rnd

log = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def _is_scalar(x):
    if isinstance(x, Number):
        return True
    if isinstance(x, (np.ndarray, jax.Array)) and x.ndim == 0:
        return True
    return False

def _powers_of_two_up_to(n):
    ks = []
    k = 1
    while k <= n:
        ks.append(k)
        k *= 2
    return ks

def sdss_eval(cfg: DictConfig, model_state, step: int) -> None:
    """Evaluate SDSS at NFE in {1,2,4,8,16,32,64,128} (capped by training steps)."""
    eval_cfg = cfg.get("eval", {})
    n_repeat  = int(eval_cfg.get("n_repeat", 1))
    batch_size = int(eval_cfg.get("batch_size", cfg.eval_samples))

    # ---- Target & initial density (same as trainer) ----
    target = cfg.target.fn
    dim = target.dim
    init_std = float(cfg.algorithm.init_std)

    initial_density = distrax.MultivariateNormalDiag(
        jnp.zeros(dim), jnp.ones(dim) * init_std
    )
    aux_tuple = (init_std, initial_density.sample, initial_density.log_prob)

    target_samples = target.sample(jax.random.PRNGKey(0), (batch_size,))

    # ---- Rebuild training schedule (same as trainer) ----
    alg_cfg = cfg.algorithm
    sigma_max = float(alg_cfg.get("sigma_max", init_std))
    sigma_min = float(alg_cfg.get("sigma_min", 2e-3))
    rho       = float(alg_cfg.get("rho", 7.0))

    sigmas = build_karras_sigmas(
        num_steps=alg_cfg.num_steps,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        rho=rho,
    )
    N_train = int(sigmas.shape[0]) - 1
    if N_train <= 0:
        raise ValueError(f"Invalid training schedule (N_train={N_train}).")

    # Evaluate at powers of two up to min(128, N_train)
    max_k_cap = min(128, N_train)
    k_values = _powers_of_two_up_to(max_k_cap)

    key_gen = jax.random.PRNGKey(cfg.seed)
    all_rows = []

    print(f"[SDSS EVAL] step={step}  repeats={n_repeat}  NFE in {k_values}")
    for k in k_values:
        # Subsample to k steps => k+1 boundaries (inclusive)
        idx = jnp.rint(jnp.linspace(0, N_train, k + 1, endpoint=True)).astype(jnp.int32)
        sigmas_eval = sigmas[idx]
        d_sigmas_eval = jnp.abs(sigmas_eval[:-1] - sigmas_eval[1:])

        rnd_short = partial(
            rnd,
            batch_size=batch_size,
            aux_tuple=aux_tuple,
            target=target,
            sigmas=sigmas_eval,
            d_sigmas=d_sigmas_eval,
            stop_grad=True,
            use_ode=True,
            return_traj=False,
        )
        eval_fn, _ = get_eval_fn(rnd_short, target, target_samples, cfg)

        metrics = defaultdict(list)
        for _ in range(n_repeat):
            key, key_gen = jax.random.split(key_gen)
            logdict = eval_fn(model_state, key)
            for name, series in logdict.items():
                if series:
                    v = series[-1]
                    if _is_scalar(v):
                        metrics[name].append(float(v))

        for m, vals in metrics.items():
            arr = np.asarray(vals, float)
            all_rows.append({
                "metric": m,
                "nfe": int(k),
                "mean": arr.mean() if arr.size else np.nan,
                "std":  arr.std(ddof=1) if arr.size > 1 else np.nan,
                "n":    int(arr.size),
                "step": int(step),
            })

        print(f"[SDSS EVAL] ... finished NFE={k}")

    df = pd.DataFrame(all_rows).sort_values(["metric", "nfe"])
    out_dir = Path("results") / cfg.algorithm.name
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{cfg.target.name}_seed{cfg.seed}_step{step}"
    df.to_csv(out_dir / f"{base}.csv", index=False)

    with open(out_dir / f"{base}.raw.json", "w") as f:
        json.dump({"rows": all_rows}, f, indent=2)

    print(f"[SDSS EVAL] wrote {out_dir / f'{base}.csv'}")