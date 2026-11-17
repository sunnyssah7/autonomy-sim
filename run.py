"""Run the demo: python run.py [--seed N] [--gif media/demo.gif]"""
import argparse

import numpy as np

from autonomy.render import to_gif
from autonomy.sim import run
from autonomy.world import World

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gif", default=None)
    args = ap.parse_args()

    world = World.demo(seed=args.seed)
    goal = (37.0, 27.0)
    res = run(world, goal=goal)

    straight = np.hypot(goal[0] - 3, goal[1] - 3)
    print(f"reached goal   : {res.reached}")
    print(f"sim time       : {res.sim_time:.1f} s")
    print(f"distance       : {res.distance:.1f} m  ({res.distance / straight:.2f}x straight-line)")
    print(f"replans        : {res.replans}")
    print(f"A* median time : {np.median(res.plan_ms):.1f} ms")
    if args.gif:
        to_gif(world, res, goal, args.gif)
        print(f"wrote {args.gif}")
