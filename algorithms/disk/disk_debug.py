# algorithms/disk/disk_debug.py (optional)
import numpy as np
import matplotlib.pyplot as plt

def fig_traj_over_time(traj_BTD, which_dim=0, title="Trajectories vs. step"):
    # traj_BTD: (B, T+1, D)
    B, T1, _ = traj_BTD.shape
    xs = np.asarray(traj_BTD)[:, :, which_dim]
    fig, ax = plt.subplots(figsize=(6,4))
    for b in range(B):
        ax.plot(range(T1), xs[b], lw=1, alpha=0.6)
    ax.set_xlabel("step index n")
    ax.set_ylabel(f"x[{which_dim}]")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    return fig

def fig_paths_2d(traj_BTD, dims=(0,1), title="2D paths"):
    # draw a few paths in 2D
    x = np.asarray(traj_BTD)[:, :, list(dims)]  # (B, T+1, 2)
    fig, ax = plt.subplots(figsize=(5,5))
    for b in range(min(64, x.shape[0])):
        ax.plot(x[b,:,0], x[b,:,1], lw=0.8, alpha=0.5)
        ax.scatter(x[b,0,0], x[b,0,1], s=8)     # start
        ax.scatter(x[b,-1,0],x[b,-1,1], s=8)    # end
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    return fig