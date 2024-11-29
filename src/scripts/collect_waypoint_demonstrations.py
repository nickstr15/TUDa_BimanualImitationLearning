"""
Script to collect waypoints for the Panda handover task.

Arguments:
    --environment -e: environment to use
    --robots -r: robots to use, e.g "Panda Panda" or "Baxter"
    --num_success -ns: number of successful demonstrations to collect, default is 100
    --visualize -v: flag to use human render mode
"""

import argparse
import datetime
import os
import numpy as np

import robosuite as suite

from src.demonstration.waypoints import ENV_TO_WAYPOINT_EXPERT
from src.demonstration.waypoints.core.waypoint_expert import TwoArmWaypointExpertBase
from src.utils.paths import DEMOS_DIR


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--environment", "-e", type=str, default="TwoArmPickPlace", help="Environment to use")
    parser.add_argument("--waypoints", "-wp", type=str, default="two_arm_pick_place_wp.yaml", help="Waypoints file to use")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--num_success", "-ns", type=int, default=100, help="Number of successful demonstrations to collect")
    parser.add_argument("--visualize", "-v", action="store_true", help="Flag to use human render mode")

    args = parser.parse_args()

    robots = args.robots.split()

    folder = "_".join(
            args.robots.split() + [args.environment, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")]
    )

    out_dir = str(
        os.path.join(
            DEMOS_DIR, "wp", folder
        )
    )

    env_config = dict(
        env_name=args.environment,
        robots=robots,
    )

    env = suite.make(
        **env_config,
        use_object_obs=True,
        use_camera_obs=False,
        has_renderer=args.visualize,
        has_offscreen_renderer=False,
    )

    expert: TwoArmWaypointExpertBase = ENV_TO_WAYPOINT_EXPERT[args.environment](
        env,
        waypoints_file=args.waypoints,
    )

    expert.collect_data(
        out_dir=out_dir,
        num_successes=args.num_success,
        render=args.visualize,
        env_config=env_config
    )

if __name__ == "__main__":
    main()