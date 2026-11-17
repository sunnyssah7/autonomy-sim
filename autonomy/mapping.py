"""Log-odds occupancy grid mapping from LiDAR scans (known pose)."""
from __future__ import annotations

import numpy as np

L_OCC = 0.85      # log-odds added to a cell where a beam ended on an obstacle
L_FREE = -0.4     # log-odds added to cells a beam passed through
L_MIN, L_MAX = -4.0, 4.0


class OccupancyMap:
    def __init__(self, h: int, w: int, res: float):
        self.h, self.w, self.res = h, w, res
        self.logodds = np.zeros((h, w), dtype=np.float32)

    def update(self, x: float, y: float, theta: float, angles: np.ndarray,
               ranges: np.ndarray, max_range: float) -> None:
        step = self.res * 0.5
        for a, rng in zip(theta + angles, ranges):
            ts = np.arange(0, rng, step)
            cs = ((x + np.cos(a) * ts) / self.res).astype(int)
            rs = ((y + np.sin(a) * ts) / self.res).astype(int)
            ok = (cs >= 0) & (cs < self.w) & (rs >= 0) & (rs < self.h)
            free = np.unique(np.stack([rs[ok], cs[ok]], 1), axis=0)
            if len(free):
                self.logodds[free[:, 0], free[:, 1]] += L_FREE
            if rng < max_range - 1e-3:  # beam actually hit something
                ce, re = int((x + np.cos(a) * rng) / self.res), int((y + np.sin(a) * rng) / self.res)
                if 0 <= ce < self.w and 0 <= re < self.h:
                    self.logodds[re, ce] += L_OCC
        np.clip(self.logodds, L_MIN, L_MAX, out=self.logodds)

    def probability(self) -> np.ndarray:
        return 1.0 - 1.0 / (1.0 + np.exp(self.logodds))

    def obstacles(self, thresh: float = 0.65) -> np.ndarray:
        """Cells believed occupied. Unknown space is treated as free (optimistic planning)."""
        return self.probability() > thresh
