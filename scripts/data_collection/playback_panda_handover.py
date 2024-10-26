"""
Script to play back demonstrations for the Panda handover task.

Arguments:
    --hdf5_file -f: path to the hdf5 file containing the demonstrations
    --use_actions -ua: flag to use the actions for playback instead of loading the simulation states one by one
    --target_real_time -rt: flag to play back the demonstrations in real time
"""

import argparse
import os

from src.demonstration.data_collection.hdf5 import playback_demonstrations_from_hdf5
from src.utils.paths import DEMOS_DIR


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hdf5_file", "-f", type=str, required=True, help="path to the hdf5 file in root/data/panda_handover containing the demonstrations")
    parser.add_argument("--use_actions", "-ua", action="store_true", help="flag to use the actions for playback instead of loading the simulation states one by one")
    parser.add_argument("--target_real_time", "-rt", action="store_true", help="flag to play back the demonstrations in real time")

    args = parser.parse_args()

    file = os.path.join(DEMOS_DIR, "panda_handover", args.hdf5_file)
    playback_demonstrations_from_hdf5(file, args.use_actions, args.target_real_time)

if __name__ == "__main__":
    main()