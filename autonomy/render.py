"""Render a run to an animated GIF: ground truth (left) vs. the robot's belief map (right)."""
from __future__ import annotations

import io

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .lidar import Lidar
from .mapping import OccupancyMap
from .sim import Result
from .world import World

BG, ACC, BLUE, FG = "#030a12", "#6ec6ff", "#3dffc4", "#e8f4ff"
# free = near-black, unknown = slate, occupied = bright
BELIEF = matplotlib.colors.LinearSegmentedColormap.from_list("belief", ["#040d17", "#0f2233", "#e8f4ff"])


def _replay_maps(world: World, result: Result, lidar: Lidar) -> list[np.ndarray]:
    m = OccupancyMap(world.h, world.w, world.res)
    out = []
    for f in result.frames:
        m.update(*f.pose, lidar.angles, f.scan, lidar.max_range)
        out.append(m.probability().copy())
    return out


def to_gif(world: World, result: Result, goal, path: str, fps: int = 20, stride: int = 2) -> None:
    lidar = Lidar()
    maps = _replay_maps(world, result, lidar)
    ext = [0, world.w * world.res, 0, world.h * world.res]
    trail = np.array(result.trail)
    frames = []
    for i in range(0, len(result.frames), stride):
        f = result.frames[i]
        x, y, th = f.pose
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=72, facecolor=BG)
        for ax, title in zip(axes, ["ground truth + lidar", "robot's belief (log-odds map)"]):
            ax.set_facecolor(BG); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#0f2233")
            ax.set_title(title, color="#7f9bb3", fontsize=10, family="monospace", loc="left")
        axes[0].imshow(world.grid, origin="lower", extent=ext, cmap=matplotlib.colors.ListedColormap([BG, "#1a3a55"]))
        axes[1].imshow(maps[i], origin="lower", extent=ext, cmap=BELIEF, vmin=0, vmax=1)
        # lidar rays
        ang = th + lidar.angles
        hx, hy = x + np.cos(ang) * f.scan, y + np.sin(ang) * f.scan
        for a, b in zip(hx[::3], hy[::3]):
            axes[0].plot([x, a], [y, b], color=ACC, alpha=.12, lw=.6)
        axes[0].scatter(hx, hy, s=3, c=ACC)
        n = int(f.t / 0.1) + 1
        for ax in axes:
            if f.path is not None:
                ax.plot(f.path[:, 0], f.path[:, 1], color=BLUE, lw=1.4, alpha=.9)
            ax.plot(trail[:n, 0], trail[:n, 1], color=FG, lw=1, alpha=.5)
            ax.add_patch(plt.Circle((x, y), 0.4, color=ACC))
            ax.plot([x, x + np.cos(th) * .8], [y, y + np.sin(th) * .8], color=FG, lw=1.5)
            ax.scatter([goal[0]], [goal[1]], marker="*", s=180, c=FG, zorder=5)
        fig.suptitle(f"t = {f.t:5.1f}s   replans = {sum(fr.replanned for fr in result.frames[:i+1])}",
                     color=FG, family="monospace", fontsize=10, x=0.02, ha="left")
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=BG); plt.close(fig)
        frames.append(imageio.imread(buf.getvalue()))
    frames += [frames[-1]] * fps  # hold the final frame
    imageio.mimsave(path, frames, duration=1 / fps, loop=0)
