"""2D LiDAR model: ray-marched range readings with Gaussian noise."""
from __future__ import annotations

import numpy as np

from .world import World


class Lidar:
    def __init__(self, n_beams: int = 120, max_range: float = 8.0, fov: float = 2 * np.pi,
                 noise_std: float = 0.03, seed: int = 0):
        self.n = n_beams
        self.max_range = max_range
        self.angles = np.linspace(-fov / 2, fov / 2, n_beams, endpoint=fov < 2 * np.pi)
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def scan(self, world: World, x: float, y: float, theta: float) -> np.ndarray:
        """Return range per beam. A beam that hits nothing returns max_range."""
        step = world.res * 0.5
        ts = np.arange(step, self.max_range + step, step)                  # (S,)
        ang = theta + self.angles                                          # (N,)
        px = x + np.cos(ang)[:, None] * ts[None, :]                        # (N, S)
        py = y + np.sin(ang)[:, None] * ts[None, :]
        c = np.clip((px / world.res).astype(int), 0, world.w - 1)
        r = np.clip((py / world.res).astype(int), 0, world.h - 1)
        hit = world.grid[r, c].astype(bool)
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), len(ts) - 1)
        any_hit = hit.any(axis=1)
        noisy = ts[first] + self.rng.normal(0, self.noise_std, self.n)
        # no-return beams report exactly max_range so the mapper never mistakes noise for a hit
        return np.where(any_hit, np.clip(noisy, 0.0, self.max_range - 0.01), self.max_range)
