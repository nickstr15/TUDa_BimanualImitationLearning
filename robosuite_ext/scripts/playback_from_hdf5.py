"""
Script to play back demonstrations for the Panda handover task.

Arguments:
    --hdf5_folder -f: path to the hdf5 file containing the demonstrations in root/data/panda_handover,
        e.g. -f "sample" => root/data/panda_handover/sample/demo.hdf5
    --use_actions -ua: flag to use the actions for playback instead of loading the simulation states one by one
    --target_real_time -rt: flag to play back the demonstrations in real time
    --num_episodes -ne: number of episodes to play back. If None, all episodes are played back.
"""

import argparse
import os

import robosuite_ext.environments
from robosuite_ext.demonstration.utils.play_back import play_back_from_hdf5
from robosuite_ext.utils.paths import DEMOS_DIR


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hdf5_folder", "-f", type=str, required=True,
                        help="path to the hdf5 file in repo-root/data containing the demonstrations." + \
                             "Example: -f 'sample' => root/data/sample/demo.hdf5")
    parser.add_argument("--use_actions", "-ua", action="store_true",
                        help="flag to use the actions for playback instead of loading the simulation states one by one")
    parser.add_argument("--target_real_time", "-rt", action="store_true",
                        help="flag to play back the demonstrations in real time")
    parser.add_argument("--num_episodes", "-ne", type=int, default=None,
                        help="number of episodes to play back. If None, all episodes are played back.")

    args = parser.parse_args()

    file = str(
        os.path.join(
            DEMOS_DIR, args.hdf5_folder, "demo.hdf5"
        )
    )

    play_back_from_hdf5(
        hdf5_path = file,
        use_actions = args.use_actions,
        num_episodes = args.num_episodes,
        target_rt = args.target_real_time
    )

if __name__ == "__main__":
    main()