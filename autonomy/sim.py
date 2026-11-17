"""Sense -> map -> plan -> act loop."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .control import PurePursuit, Robot
from .lidar import Lidar
from .mapping import OccupancyMap
from .planner import astar, inflate, path_blocked
from .world import World


@dataclass
class Frame:
    t: float
    pose: tuple[float, float, float]
    scan: np.ndarray
    path: np.ndarray | None
    target: np.ndarray | None
    replanned: bool


@dataclass
class Result:
    reached: bool
    sim_time: float
    distance: float
    replans: int
    plan_ms: list[float] = field(default_factory=list)
    frames: list[Frame] = field(default_factory=list)
    trail: list[tuple[float, float]] = field(default_factory=list)
    map: OccupancyMap | None = None


def run(world: World, start=(3.0, 3.0, 0.0), goal=(37.0, 27.0), dt: float = 0.1,
        t_max: float = 180.0, record_every: int = 2) -> Result:
    robot = Robot(*start)
    lidar = Lidar()
    omap = OccupancyMap(world.h, world.w, world.res)
    ctrl = PurePursuit()
    infl = int(np.ceil((robot.radius + 0.15) / world.res))
    goal_rc = (int(goal[1] / world.res), int(goal[0] / world.res))

    path_rc, path_xy = None, None
    res = Result(False, 0.0, 0.0, 0)
    t, k = 0.0, 0
    while t < t_max:
        scan = lidar.scan(world, robot.x, robot.y, robot.theta)                 # sense
        omap.update(robot.x, robot.y, robot.theta, lidar.angles, scan, lidar.max_range)  # map

        blocked = inflate(omap.obstacles(), infl)
        here = (int(robot.y / world.res), int(robot.x / world.res))
        blocked[max(0, here[0] - 1):here[0] + 2, max(0, here[1] - 1):here[1] + 2] = False
        replanned = False
        if path_rc is None or path_blocked(path_rc, blocked):                    # plan
            t0 = time.perf_counter()
            path_rc = astar(blocked, here, goal_rc)
            res.plan_ms.append((time.perf_counter() - t0) * 1e3)
            if path_rc is None:
                break
            path_xy = (np.array(path_rc)[:, ::-1] + 0.5) * world.res
            res.replans += 1
            replanned = True

        v, w, target = ctrl.command(robot, path_xy)                              # act
        px, py = robot.x, robot.y
        robot.step(v, w, dt)
        if world.occupied(robot.x, robot.y):  # safety: never drive into ground truth
            robot.x, robot.y = px, py
            path_rc = None
        res.distance += float(np.hypot(robot.x - px, robot.y - py))
        res.trail.append((robot.x, robot.y))

        if k % record_every == 0:
            res.frames.append(Frame(t, (robot.x, robot.y, robot.theta), scan, path_xy, target, replanned))
        if np.hypot(robot.x - goal[0], robot.y - goal[1]) < 0.5:
            res.reached = True
            res.frames.append(Frame(t, (robot.x, robot.y, robot.theta), scan, path_xy, target, False))
            break
        t += dt
        k += 1

    res.sim_time = t
    res.map = omap
    return res
