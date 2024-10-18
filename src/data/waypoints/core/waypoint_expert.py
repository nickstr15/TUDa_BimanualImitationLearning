import os.path
import time

import yaml
from transforms3d.quaternions import qmult, qinverse

from src.control.utils.arm_state import ArmState
from src.data.data_collection.data_collection_wrapper import DataCollectionWrapper
from src.data.data_collection.hdf5 import gather_demonstrations_as_hdf5
from src.data.waypoints.core.waypoint import Waypoint
from src.environments.core.action import OSAction
from src.environments.core.enums import ActionMode
from src.environments.core.environment_interface import IEnvironment
from src.utils.clipping import clip_translation, clip_quat
from src.utils.constants import MAX_DELTA_TRANSLATION, MAX_DELTA_ROTATION
from src.utils.paths import WAYPOINTS_DIR
from src.utils.real_time import RealTimeHandler


class WaypointExpertBase:
    """
    Class for an expert agent that acts in the environment
    by following a predefined trajectory of waypoints.
    """
    def __init__(
        self,
        environment : IEnvironment,
        waypoints_file : str,
        max_delta_translation : float = MAX_DELTA_TRANSLATION,
        max_delta_rotation : float = MAX_DELTA_ROTATION,
    ) -> None:
        """
        Constructor for the WaypointExpert class.
        :param environment: Environment in which the expert agent acts
        :param waypoints_file: File containing the waypoints n $WAYPOINTS_DIR
        :param max_delta_translation: Maximum translation distance between current position and action output
        :param max_delta_rotation: Maximum rotation angle between current orientation and action output
        """
        full_waypoints_path = os.path.join(WAYPOINTS_DIR, waypoints_file)
        assert os.path.isfile(full_waypoints_path), f"Waypoints file {full_waypoints_path} not found"
        self._env = environment

        self._max_delta_translation = max_delta_translation
        self._max_delta_rotation = max_delta_rotation

        # set control
        with open(full_waypoints_path, 'r') as f:
            self._control_config = yaml.safe_load(f)

        self._initial_config = self._load_initial_config()
        self._waypoints = self._load_waypoints()

        self._action_mode = self._env.action_mode

    def _load_initial_config(self) -> dict:
        """
        Load the initial configuration from the control config.
        :return: Initial configuration
        """
        return self._control_config["initial_configuration"]

    def _load_waypoints(self) -> list[Waypoint]:
        """
        Load the waypoints from the control config.
        :return: List of waypoints
        """
        waypoints_list = []
        for waypoint_data in self._control_config["waypoints"]:
            waypoints_list.append(Waypoint(waypoint_data))

        return waypoints_list

    def _get_action(self, current_state: dict, waypoint: Waypoint) -> OSAction:
        """
        Get the action to reach the waypoint.
        The output is a clipped version of the Waypoint state to
        respect self._max_delta_translation and self._max_delta_rotation.
        :param current_state: Current state of the device
        :param waypoint: Waypoint to reach
        :return: Action to reach the waypoint
        """
        assert current_state.keys() == waypoint.targets.keys(), "Current state and waypoint targets do not match"

        targets = {
            name : ArmState()
            for name in waypoint.targets.keys()
        }

        for name, target in waypoint.targets.items():
            current = current_state[name]

            # Clip the translation
            pos_target = target.get_xyz()
            pos_current = current.get_xyz()
            pos_delta = clip_translation(pos_target - pos_current, self._max_delta_translation)
            if self._action_mode == ActionMode.ABSOLUTE:
                targets[name].set_xyz(pos_delta + pos_current)
            elif self._action_mode == ActionMode.RELATIVE:
                targets[name].set_xyz(pos_delta)

            # Clip the rotation
            quat_target = target.get_quat()
            quat_current = current.get_quat()
            quat_delta = clip_quat(
                qmult(quat_target, qinverse(quat_current)),
                self._max_delta_rotation
            )
            if self._action_mode == ActionMode.ABSOLUTE:
                targets[name].set_quat(qmult(quat_delta, quat_current))
            else:
                targets[name].set_quat(quat_delta)

            # Set GripperState
            targets[name].set_gripper_state(target.get_gripper_state())

        return OSAction(targets)

    def _run(
        self,
        render : bool = False,
        target_real_time : bool = False
    ) -> None:
        """
        Run the expert agent in the environment.
        :param render: Whether to render the environment
        :param target_real_time: Whether to render in real time
        """
        if target_real_time and not (render and self._env.render_mode == "human"):
            print("[Warning] Real time rendering requires rendering in human mode, ignoring real time rendering.")
            target_real_time = False

        # minimum number of steps in done state
        min_steps_terminated = int(1.0 * self._env.render_fps)
        steps_terminated = 0

        # set the initial configuration
        self._env.reset()
        self._env.set_seed(
            seed=self._initial_config.get("seed", 0)
        )
        positions = self._initial_config.get("positions", [])
        self._env.set_initial_config(positions)

        if render:
            self._env.render()


        dt = 1.0 / self._env.render_fps

        rt = RealTimeHandler(self._env.render_fps)

        current_state = self._env.get_device_states()
        for waypoint in self._waypoints:
            step = 0
            reached = False
            rt.reset()
            while not reached:
                action = self._get_action(current_state, waypoint)
                observation, _, terminated, _, _ = self._env.step(action)
                if render:
                    self._env.render()
                step += 1
                current_state = self._env.get_device_states()
                reached = waypoint.is_reached_by(
                    current_state, step * dt
                )

                if terminated:
                    steps_terminated += 1
                    if steps_terminated >= min_steps_terminated:
                        print("Done.")
                        return
                else:
                    steps_terminated = 0

                if target_real_time:
                    rt.sleep()

    def dispose(self) -> None:
        """
        Dispose the expert agent.
        """
        self._env.close()

    def visualize(self) -> None:
        """
        Visualize the expert agent in the environment.
        """
        self._run(render=True)

    def collect_data(self,
        out_dir : str,
        render: bool = False,
        target_real_time: bool = False
    ) -> None:
        """
        Collect data from the expert agent in the environment.

        :param out_dir: Output directory for the data
        :param render: Whether to render the environment
        :param target_real_time: Whether to render in real time
        """
        tmp_directory = "/tmp/bil/{}".format(str(time.time()).replace(".", "_"))
        self._env = DataCollectionWrapper(self._env, tmp_directory)
        self._run(render=render, target_real_time=target_real_time)
        self._env.close()

        gather_demonstrations_as_hdf5(tmp_directory, out_dir, self._env.args)

        self._env.clean_up()




