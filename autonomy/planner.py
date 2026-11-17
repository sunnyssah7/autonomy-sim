"""A* path planning on an inflated 8-connected occupancy grid."""
from __future__ import annotations

import heapq
import math

import numpy as np
from scipy.ndimage import binary_dilation

SQRT2 = math.sqrt(2)
MOVES = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
         (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


def inflate(obstacles: np.ndarray, radius_cells: int) -> np.ndarray:
    """Grow obstacles by the robot radius so the planner can treat the robot as a point."""
    if radius_cells <= 0:
        return obstacles.copy()
    yy, xx = np.mgrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
    disk = (xx ** 2 + yy ** 2) <= radius_cells ** 2
    return binary_dilation(obstacles, structure=disk)


def astar(blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """Return a list of (row, col) cells from start to goal, or None if unreachable.

    Octile-distance heuristic: admissible and consistent on an 8-connected grid,
    so the first time the goal is popped the path is optimal.
    """
    h, w = blocked.shape
    if blocked[goal] or blocked[start]:
        return None

    def heur(r: int, c: int) -> float:
        dr, dc = abs(r - goal[0]), abs(c - goal[1])
        return (dr + dc) + (SQRT2 - 2) * min(dr, dc)

    g = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    openq = [(heur(*start), 0.0, start)]
    closed = set()
    while openq:
        _, gc, cur = heapq.heappop(openq)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]
        closed.add(cur)
        r, c = cur
        for dr, dc, cost in MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or blocked[nr, nc]:
                continue
            if dr and dc and (blocked[r, nc] or blocked[nr, c]):  # no corner cutting
                continue
            ng = gc + cost
            if ng < g.get((nr, nc), math.inf):
                g[(nr, nc)] = ng
                parent[(nr, nc)] = cur
                heapq.heappush(openq, (ng + heur(nr, nc), ng, (nr, nc)))
    return None


def path_blocked(path: list[tuple[int, int]], blocked: np.ndarray) -> bool:
    return any(blocked[r, c] for r, c in path)
