"""
Script to collect waypoints for the Panda handover task.

Arguments:
    --expert_id -e: id of the expert agent
    --visualize -v: flag to use human render mode
    --visualize_targets -vt: flag to visualize the targets, only applies if visualize flag is set
    --action_mode -am: action mode for the environment (0 for absolute, 1 for relative)
    --real_time -rt: flag to display the expert in real time, only applies if visualize flag is set
"""

import argparse
import datetime
import os

from src.data.waypoints.panda_handover_wp_expert import PandaHandoverWpExpert
from src.environments.core.enums import ActionMode
from src.utils.paths import DEMOS_DIR

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--expert_id", "-e", type=int, default=1, help="id of the expert agent")
    parser.add_argument("--visualize", "-v", action="store_true", help="flag to use human render mode")
    parser.add_argument("--visualize_targets", "-vt", action="store_true", help="flag to visualize the targets")
    parser.add_argument("--action_mode", "-am", type=int, default=1, help="action mode for the environment (0 for absolute, 1 for relative)")
    parser.add_argument("--real_time", "-rt", action="store_true", help="flag to display the expert in real time")

    args = parser.parse_args()

    env_args = dict(
        visualize_targets=args.visualize_targets,
        action_mode = ActionMode(args.action_mode),
        render_mode = "human" if args.visualize else "rgb_array"
    )

    out_dir = os.path.join(DEMOS_DIR, "panda_handover", "wp", str(args.expert_id).zfill(3))

    expert = PandaHandoverWpExpert(args.expert_id, env_args)
    expert.collect_data(out_dir, args.visualize, args.real_time)
    expert.dispose()

if __name__ == "__main__":
    main()