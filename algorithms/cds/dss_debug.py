import io
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import wandb
from PIL import Image

def _to_cpu(x):
    return np.asarray(jax.device_get(x))

def _extent_from_samples(xs2d, pad=0.15):
    # xs2d: (B,2) numpy
    lo = xs2d.min(axis=0)
    hi = xs2d.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    lo = lo - pad * span
    hi = hi + pad * span
    return [lo[0], hi[0], lo[1], hi[1]]

def _make_density_bg(target, extent, n=200):
    xmin, xmax, ymin, ymax = extent
    gx = np.linspace(xmin, xmax, n)
    gy = np.linspace(ymin, ymax, n)
    X, Y = np.meshgrid(gx, gy)
    P = np.stack([X, Y], axis=-1).reshape(-1, 2)
    # log_prob expects (D,) or (N,D)
    lp = _to_cpu(target.log_prob(jnp.asarray(P))).reshape(n, n)
    # Use exp of centered log_prob for contrast without overflow
    lp = lp - np.max(lp)
    dens = np.exp(lp)
    return X, Y, dens

def fig_student_vs_target(samples_2d, target, title="student vs target", extent=None):
    xs = _to_cpu(samples_2d)
    if extent is None:
        extent = _extent_from_samples(xs)
    X, Y, dens = _make_density_bg(target, extent, n=200)

    fig = plt.figure(figsize=(5, 3.5), dpi=120)
    ax = plt.gca()
    ax.imshow(dens, origin="lower",
              extent=[extent[0], extent[1], extent[2], extent[3]],
              aspect="equal")
    ax.scatter(xs[:, 0], xs[:, 1], s=8, c="r", alpha=0.55, marker="x")
    ax.set_title(title)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    return fig

def fig_consistency_segments(p_left, p_right, Xt, title="consistency pairs", max_lines=300):
    # p_left = fθ(Xt,t), p_right = stopgrad fθ(Xt+1,t+1)
    A = _to_cpu(p_left)
    B = _to_cpu(p_right)
    X = _to_cpu(Xt)
    # choose a subset for visibility
    idx = np.random.default_rng(0).choice(A.shape[0], size=min(max_lines, A.shape[0]), replace=False)
    A = A[idx]; B = B[idx]; X = X[idx]
    fig = plt.figure(figsize=(5, 3.5), dpi=120)
    ax = plt.gca()
    ax.scatter(X[:,0], X[:,1], s=6, c="k", alpha=0.3, label="Xt")
    for a, b in zip(A, B):
        ax.plot([a[0], b[0]], [a[1], b[1]], lw=0.8, alpha=0.7)
    ax.set_aspect("equal")
    ax.set_title(title)
    return fig

def fig_trajectories_over_time(xs_seq, which_dim=0, title="teacher trajectories (dim0)"):
    # xs_seq: (T+1, K, D) on cpu
    X = _to_cpu(xs_seq)[..., which_dim]  # (T+1, K)
    T1, K = X.shape
    fig = plt.figure(figsize=(6, 3.2), dpi=120)
    ax = plt.gca()
    t = np.arange(T1)
    for k in range(K):
        ax.plot(t, X[:, k], lw=1.0, alpha=0.6)
    ax.set_xlabel("step n"); ax.set_ylabel(f"x[{which_dim}]")
    ax.set_title(title)
    return fig

def wandb_image(fig_or_array, *, close=True, dpi=150):
    # case 1: already an array
    if isinstance(fig_or_array, np.ndarray):
        return wandb.Image(fig_or_array)

    # case 2: a matplotlib Figure
    if hasattr(fig_or_array, "savefig"):
        buf = io.BytesIO()
        fig_or_array.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        if close:
            plt.close(fig_or_array)
        buf.seek(0)
        img = Image.open(buf).convert("RGBA")  # ensure 3/4 channels
        return wandb.Image(img)

    # fallback: let wandb try (it can handle raw FigureArtist etc.)
    return wandb.Image(fig_or_array)

def make_teacher_steps(teacher_state, teacher_params, noise_schedule, init_std, target, num_steps):
    dt = 1.0 / num_steps
    betas = noise_schedule

    def langevin_init_fn(x, t, sigma_t, T, initial_log_prob, target_log_prob):
        # Only target is used here for grad; initial_log_prob not needed for plotting
        tr = t / T
        return sigma_t * ((1. - tr) * target_log_prob(x))  # + tr * initial_log_prob(x))

    def sde_step(x, step_f, key):
        beta_t = betas(step_f); sigma_t = jnp.sqrt(2. * beta_t) * init_std
        lgv = jax.vmap(lambda xi: jax.grad(langevin_init_fn)(
            xi, step_f, sigma_t, num_steps, None, target.log_prob))(x)
        u = teacher_state.apply_fn(teacher_params, x, jnp.full((x.shape[0],1), step_f, x.dtype), lgv)
        key, kn = jax.random.split(key)
        noise = jnp.clip(jax.random.normal(kn, shape=x.shape), -4., 4.)
        x_new = x + (sigma_t * u + beta_t * x) * dt + sigma_t * jnp.sqrt(dt) * noise
        return x_new, key

    def pf_ode_step(x, step_f):
        beta_t = betas(step_f); sigma_t = jnp.sqrt(2. * beta_t) * init_std
        lgv = jax.vmap(lambda xi: jax.grad(langevin_init_fn)(
            xi, step_f, sigma_t, num_steps, None, target.log_prob))(x)
        u = teacher_state.apply_fn(teacher_params, x, jnp.full((x.shape[0],1), step_f, x.dtype), lgv)
        x_new = x + (0.5 * sigma_t * u + beta_t * x) * dt
        return x_new

    return sde_step, pf_ode_step

def fig_teacher_vs_target(x_2d, target, title="Teacher vs. target", bins=200, pad=0.2):
    """x_2d: (B,2) array of teacher terminal samples (projected to first two dims)."""
    x = np.asarray(x_2d)
    # Heatmap bounds from samples (plus a little padding)
    mn = x.min(axis=0); mx = x.max(axis=0)
    span = np.maximum(mx - mn, 1.0)
    lo = mn - pad * span; hi = mx + pad * span

    # Target log-density on a grid
    xs = np.linspace(lo[0], hi[0], bins)
    ys = np.linspace(lo[1], hi[1], bins)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    grid = np.stack([X, Y], axis=-1).reshape(-1, 2)
    logp = target.log_prob(jnp.asarray(grid)).reshape(bins, bins)
    Z = np.exp(np.asarray(logp - logp.max()))  # stable-ish unnormalized heatmap

    # Figure
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=140)
    ax.imshow(Z, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower", aspect="equal")
    ax.scatter(x[:, 0], x[:, 1], s=10, c="r", marker="x", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("x[0]"); ax.set_ylabel("x[1]")
    return fig