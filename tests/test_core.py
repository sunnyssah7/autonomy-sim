import math

import numpy as np

from autonomy.lidar import Lidar
from autonomy.planner import astar, inflate
from autonomy.sim import run
from autonomy.world import World


def test_astar_straight_line_is_optimal():
    grid = np.zeros((10, 10), bool)
    path = astar(grid, (0, 0), (0, 9))
    assert len(path) == 10


def test_astar_diagonal_cost():
    grid = np.zeros((10, 10), bool)
    path = astar(grid, (0, 0), (9, 9))
    assert len(path) == 10  # pure diagonal


def test_astar_routes_around_wall():
    grid = np.zeros((10, 10), bool)
    grid[0:9, 5] = True
    path = astar(grid, (0, 0), (0, 9))
    assert path is not None and all(not grid[r, c] for r, c in path)
    assert any(r == 9 for r, _ in path)  # had to go through the gap


def test_astar_unreachable():
    grid = np.zeros((10, 10), bool)
    grid[:, 5] = True
    assert astar(grid, (0, 0), (0, 9)) is None


def test_inflate_grows_obstacle():
    g = np.zeros((9, 9), bool); g[4, 4] = True
    assert inflate(g, 2).sum() > 1 and inflate(g, 2)[4, 6]


def test_lidar_measures_wall_distance():
    w = World(10, 10, 0.1)
    w.add_rect(6, 0, 7, 10)
    lidar = Lidar(n_beams=4, fov=2 * math.pi, noise_std=0.0)
    r = lidar.scan(w, 3.0, 5.0, 0.0)
    forward = r[np.argmin(np.abs(lidar.angles))]
    assert abs(forward - 3.0) < 0.15


def test_end_to_end_reaches_goal():
    res = run(World.demo(seed=7))
    assert res.reached
