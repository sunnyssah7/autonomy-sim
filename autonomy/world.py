"""Ground-truth world: a 2D occupancy grid the robot cannot see directly."""
from __future__ import annotations

import numpy as np


class World:
    """Binary occupancy grid. 1 = obstacle, 0 = free. Cells are `res` meters wide."""

    def __init__(self, width_m: float = 40.0, height_m: float = 30.0, res: float = 0.25, seed: int = 7):
        self.res = res
        self.w = int(width_m / res)
        self.h = int(height_m / res)
        self.grid = np.zeros((self.h, self.w), dtype=np.uint8)
        self.rng = np.random.default_rng(seed)
        self._border()

    def _border(self) -> None:
        self.grid[0, :] = self.grid[-1, :] = 1
        self.grid[:, 0] = self.grid[:, -1] = 1

    def add_rect(self, x0: float, y0: float, x1: float, y1: float) -> None:
        r0, r1 = sorted((int(y0 / self.res), int(y1 / self.res)))
        c0, c1 = sorted((int(x0 / self.res), int(x1 / self.res)))
        self.grid[r0:r1, c0:c1] = 1

    def scatter(self, n: int, keep_clear: list[tuple[float, float]], clear_r: float = 3.0) -> None:
        """Drop `n` random rectangular obstacles, keeping a radius around key points free."""
        placed = 0
        while placed < n:
            x, y = self.rng.uniform(2, self.w * self.res - 2), self.rng.uniform(2, self.h * self.res - 2)
            sw, sh = self.rng.uniform(0.8, 4.0), self.rng.uniform(0.8, 4.0)
            if any(np.hypot(x - px, y - py) < clear_r + max(sw, sh) for px, py in keep_clear):
                continue
            self.add_rect(x - sw / 2, y - sh / 2, x + sw / 2, y + sh / 2)
            placed += 1

    def occupied(self, x: float, y: float) -> bool:
        c, r = int(x / self.res), int(y / self.res)
        if r < 0 or c < 0 or r >= self.h or c >= self.w:
            return True
        return bool(self.grid[r, c])

    @classmethod
    def demo(cls, seed: int = 7) -> "World":
        """Warehouse-style map: two shelving walls with gaps plus random clutter."""
        w = cls(seed=seed)
        w.add_rect(12, 0, 13, 20)       # wall with a gap at the top
        w.add_rect(26, 10, 27, 30)      # wall with a gap at the bottom
        w.scatter(22, keep_clear=[(3, 3), (37, 27), (12.5, 25), (26.5, 5)])
        return w
