"""Unicycle robot model + pure pursuit path tracking."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Robot:
    x: float
    y: float
    theta: float
    radius: float = 0.4
    v_max: float = 1.6       # m/s
    w_max: float = 2.5       # rad/s

    def step(self, v: float, w: float, dt: float) -> None:
        v = float(np.clip(v, 0.0, self.v_max))
        w = float(np.clip(w, -self.w_max, self.w_max))
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta = (self.theta + w * dt + np.pi) % (2 * np.pi) - np.pi


class PurePursuit:
    """Steer toward a lookahead point on the path.

    Curvature to hit a point at lateral offset y_L and distance L in the robot frame
    is k = 2*y_L / L^2, so w = v * k. Speed drops on sharp turns.
    """

    def __init__(self, lookahead: float = 1.2):
        self.L = lookahead

    def command(self, robot: Robot, path_xy: np.ndarray) -> tuple[float, float, np.ndarray]:
        d = np.hypot(path_xy[:, 0] - robot.x, path_xy[:, 1] - robot.y)
        nearest = int(d.argmin())
        ahead = np.where(d[nearest:] >= self.L)[0]
        target = path_xy[nearest + ahead[0]] if len(ahead) else path_xy[-1]

        dx, dy = target[0] - robot.x, target[1] - robot.y
        c, s = np.cos(-robot.theta), np.sin(-robot.theta)
        x_r, y_r = c * dx - s * dy, s * dx + c * dy           # target in robot frame
        L2 = max(x_r ** 2 + y_r ** 2, 1e-6)
        k = 2.0 * y_r / L2

        heading_err = abs(np.arctan2(y_r, x_r))
        v = robot.v_max * max(0.15, np.cos(min(heading_err, np.pi / 2)))
        if heading_err > 1.2:                                  # turn in place if facing away
            v = 0.0
            return v, np.sign(y_r) * robot.w_max, target
        return v, v * k, target
