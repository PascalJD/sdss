# Entirely vibe coded

# utils/ckpt_utils.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Sequence, Any

import orbax.checkpoint as ocp

def _all_checkpoint_steps(root: Path) -> List[int]:
    if not root.exists():
        return []
    steps = [int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    steps.sort()
    return steps

def _load_metric(root: Path, step: int, metric_key: str) -> Optional[float]:
    """Try a few common locations/orbax layouts for metrics."""
    # 1) metrics/<something>.json (Orbax JSON handler dir)
    mdir = root / str(step) / "metrics"
    if mdir.is_dir():
        for jf in mdir.glob("*.json"):
            try:
                with jf.open("r") as f:
                    data = json.load(f)
                if metric_key in data:
                    return float(data[metric_key])
            except Exception:
                pass
    # 2) metrics.json directly under the step dir
    mfile = root / str(step) / "metrics.json"
    if mfile.exists():
        try:
            with mfile.open("r") as f:
                data = json.load(f)
            if metric_key in data:
                return float(data[metric_key])
        except Exception:
            pass
    return None

def _load_first_available_metric(
    root: Path, step: int, keys: Sequence[str]
) -> Tuple[Optional[str], Optional[float]]:
    for k in keys:
        v = _load_metric(root, step, k)
        if v is not None:
            return k, v
    return None, None

def select_eval_steps(
    manager: ocp.CheckpointManager,
    use_best: bool,
    max_ckpts: int,
    metric_key: str = "elbo_mov_avg",
    mode: str = "max",
) -> List[int]:
    """
    When training used preservation_policy (LatestN/BestN), Orbax doesn't expose
    a 'best step' API. We select it by scanning step dirs and reading metrics.
    """
    root = Path(manager.directory)
    steps = _all_checkpoint_steps(root)
    if not steps:
        raise RuntimeError(f"No checkpoints found in {root}")

    if use_best:
        candidates = [metric_key, "KL/elbo_mov_avg", "KL/elbo"]
        scored: List[Tuple[int, float]] = []
        used_key: Optional[str] = None

        # First pass: try metric_key exactly
        vals = []
        for s in steps:
            v = _load_metric(root, s, candidates[0])
            if v is not None:
                vals.append((s, v))
        if not vals:
            # Fallbacks
            vals = []
            for s in steps:
                k, v = _load_first_available_metric(root, s, candidates[1:])
                if v is not None:
                    used_key = k
                    vals.append((s, v))
        else:
            used_key = candidates[0]

        if vals:
            reverse = (mode.lower() == "max")
            vals.sort(key=lambda kv: kv[1], reverse=reverse)
            chosen = [s for s, _ in vals[:max_ckpts]]
            print(f"[EVAL] Selecting best by '{used_key}' (mode={mode}); steps={chosen}")
            return chosen

        print("[EVAL] Warning: no metrics found for best-selection; falling back to latest.")

    # Fallback: latest N
    return steps[-max_ckpts:]


def _extract_params_from_restored(obj: Any):
    """Return params pytree if we can find it inside what Orbax restored."""
    # If training saved dict(pkg) with {"model_state": <something>}
    if isinstance(obj, dict):
        ms = obj.get("model_state", None)
        if ms is not None:
            # ms may be TrainState-like or a plain dict (apply_fn not serializable)
            if hasattr(ms, "params"):
                return ms.params
            if isinstance(ms, dict) and "params" in ms:
                return ms["params"]
        # Also handle the case where the top-level dict *is* the TrainState fields
        if "params" in obj:
            return obj["params"]
    # If training saved the TrainState directly
    if hasattr(obj, "params"):
        return obj.params
    return None


def restore_model_state(
    manager: ocp.CheckpointManager,
    step: int,
    prototype_state=None,
):
    """
    Restore a usable TrainState for eval:
      - Load checkpoint (often a dict with 'model_state' but without apply_fn).
      - Extract 'params' pytree.
      - Return prototype_state.replace(params=restored_params) if prototype supplied.
      - If a full TrainState somehow survived (rare), return it as-is.
    """
    obj = manager.restore(step)

    # If a valid TrainState slipped through serialization (unlikely):
    if hasattr(obj, "params") and hasattr(obj, "apply_fn"):
        return obj

    # Normal path: pull params and inject them into prototype
    params = _extract_params_from_restored(obj)
    if params is not None:
        if prototype_state is None:
            raise TypeError(
                "Restored params, but no prototype_state was provided. "
                "Pass a TrainState from init_model(...), then I'll replace its params."
            )
        try:
            return prototype_state.replace(params=params)
        except Exception as e:
            raise TypeError(
                "Failed to inject restored params into prototype_state. "
                "Ensure prototype_state is a flax TrainState and compatible with the saved params."
            ) from e

    # Nothing matched
    raise TypeError(
        f"Checkpoint step {step} does not contain recognizable params. "
        "Ensure training saved either a TrainState or at least its .params."
    )