import json
import os.path
import shutil
import time
from abc import ABC
from collections import OrderedDict

import numpy as np
import yaml

import robosuite as suite
from robosuite.controllers.parts.arm import OperationalSpaceController
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.utils.transform_utils import quat2axisangle, quat_multiply, quat_inverse
from robosuite.wrappers import DataCollectionWrapper

from robosuite_ext.demonstration.waypoints.core.waypoint import Waypoint
from robosuite_ext.demonstration.waypoints.utils.null_orientation import get_two_arm_null_orientation
from robosuite_ext.utils.clipping import clip_translation, clip_quat_by_axisangle
from robosuite_ext.utils.paths import WAYPOINTS_DIR, RECORDING_DIR
from robosuite_ext.utils.real_time import RealTimeHandler
from robosuite_ext.utils.robot_states import TwoArmEEState, EEState
from robosuite_ext.utils.robot_targets import GripperTarget
from robosuite_ext.demonstration.utils.gather_demonstrations import gather_demonstrations_as_hdf5
from robosuite_ext.wrappers.recording_wrapper import RecordingWrapper
from robosuite_ext.wrappers.target_visualization_wrapper import TargetVisualizationWrapper

def get_limits(
        abs_limit: float | np.ndarray | list | None,
        lower_min: float | np.ndarray,
        upper_max: float | np.ndarray
) -> tuple:
    """
    Get the limits for a value.
    :param abs_limit: Absolute limit
    :param lower_min: Lower limit
    :param upper_max: Upper limit
    :return: Tuple with the limits
    """
    if abs_limit is None:
        return lower_min, upper_max

    # assert type of lower_min and lower_max is the same
    assert type(upper_max) == type(lower_min), \
        f"Expected upper_max and lower_min to have the same type, got {type(upper_max)} and {type(lower_min)}"

    # if abs_limit is a scalar and lower_min and upper_max are arrays, then make abs_limit an array
    if type(abs_limit) == float and type(lower_min) == np.ndarray:
        abs_limit = np.array([abs_limit] * len(lower_min))

    if type(abs_limit) == list:
        abs_limit = np.array(abs_limit)

    limit_low = -1*abs_limit
    limit_high = abs_limit

    # clip the abs_limit to the range of lower_min and upper_max
    limit_low = np.clip(limit_low, lower_min, upper_max)
    limit_high = np.clip(limit_high, lower_min, upper_max)

    return limit_low, limit_high


class TwoArmWaypointExpertBase(ABC):
    """
    Class for an expert agent that acts in the environment
    by following a predefined trajectory of waypoints.

    Only applicable for two-arm environments with OSC for each arm.
    """
    target_env_name = None

    def __init__(
        self,
        environment : TwoArmEnv,
        waypoints_file : str,
    ) -> None:
        """
        Constructor for the WaypointExpert class.
        :param environment: Environment in which the expert agent acts. MUST use use_object_obs=True.
        :param waypoints_file: File containing the waypoints n $WAYPOINTS_DIR
        """
        full_waypoints_path = os.path.join(WAYPOINTS_DIR, waypoints_file)
        assert os.path.isfile(full_waypoints_path), f"Waypoints file {full_waypoints_path} not found"

        assert environment.env_configuration in ["parallel", "single-robot"], \
            f"Invalid env_configuration {environment.env_configuration}. Only 'parallel' or 'single-robot' supported."

        self._env = environment

        # get action mode + input/output ranges
        self._evaluate_controller_setup()

        self._null_quat_right, self._null_quat_left = get_two_arm_null_orientation(
            [r.name for r in self._env.robots]
        )

        with open(full_waypoints_path, 'r') as f:
            self._waypoint_cfg = yaml.safe_load(f)

        # minimum number of steps in done state (-> 1 second)
        self._min_steps_terminated = int(1.0 * self._env.control_freq)
        # minimum number of steps in @ last waypoint before aborting the episode (-> 2 seconds)
        self._min_steps_last_wp = int(2.0 * self._env.control_freq)
        # timing and real time handler
        self._dt = 1.0 / self._env.control_freq
        self._rt_handler = RealTimeHandler(self._env.control_freq)

        self.ee_target_methods = self._create_ee_target_methods_dict()

    def _evaluate_controller_setup(self) -> None:
        """
        Evaluates the controller setup to get
        - controller input + output ranges
        - controller input type
        :return:
        """
        controller_dict = self.__get_controllers()
        self._get_action_mode(controller_dict)
        self._get_controller_ranges(controller_dict)

    def _get_action_mode(self, controllers_dict: dict[str, OperationalSpaceController]) -> None:
        """
        Check the action mode and store it in self._action_mode.
        It must be the same for all devices and can be either 'delta' or 'absolute'.

        :param controllers_dict: Dictionary with the controllers for the left and right arm
        """
        mode = None
        for controller in controllers_dict.values():
            if mode is None:
                mode = controller.input_type
            assert mode == controller.input_type, \
                f"[WP] Inconsistent action modes for the controllers.  {mode}, got {controller.input_type}"

        assert mode in ["delta", "absolute"], \
            f"[WP] Invalid action mode {mode}. Expected 'delta' or 'absolute' for all controllers."

        self._action_mode = mode

    def _get_controller_ranges(self, controllers_dict: dict[str, OperationalSpaceController]) -> None:
        """
        Get the input and output ranges of the controllers.

        :param controllers_dict: Dictionary with the controllers for the left and right arm
        """
        self._ctrl_input_min = {
            "left": controllers_dict["left"].input_min,
            "right": controllers_dict["right"].input_min
        }
        self._ctrl_input_max = {
            "left": controllers_dict["left"].input_max,
            "right": controllers_dict["right"].input_max
        }

        self._ctrl_output_min = {
            "left": controllers_dict["left"].output_min,
            "right": controllers_dict["right"].output_min
        }

        self._ctrl_output_max = {
            "left": controllers_dict["left"].output_max,
            "right": controllers_dict["right"].output_max
        }

    def __get_controllers(self) -> dict[str, OperationalSpaceController]:
        """
        Get the input and output ranges of the controllers.

        :return: Dictionary with the controllers for the left and right arm
        """
        controllers = {
            "left": None,
            "right": None
        }
        if len(self._env.robots) == 1:
            # single robot with two arms
            for name, part_controller in self._env.robots[0].composite_controller.part_controllers.items():
                if name not in ["left", "right"]:
                    continue
                assert type(part_controller) == OperationalSpaceController, \
                    f"[WP] Only OperationalSpaceController is supported, got {type(part_controller)}"
                controllers[name] = part_controller

        elif len(self._env.robots) == 2:
            mapped_names = ["right", "left"]
            # two separate robot arms
            for robot, mapped_name in zip(self._env.robots, mapped_names):
                for name, part_controller in robot.composite_controller.part_controllers.items():
                    if name not in ["right"]:
                        continue
                    assert type(part_controller) == OperationalSpaceController, \
                        f"[WP] Only OperationalSpaceController is supported, got {type(part_controller)}"

                    controllers[mapped_name] = part_controller

        else:
            raise ValueError(f"[WP] Invalid number of robots {len(self._env.robots)}. Expected 1 or 2.")

        return controllers

    def _create_waypoints(self, obs: OrderedDict = None) -> list[Waypoint]:
        """
        Create the list of waypoints.
        This method can be dependent on the environments initial state.
        :param obs: Observation after environment reset
        :return: List of waypoints
        """
        waypoints : list[Waypoint] = []

        for waypoint_dict in self._waypoint_cfg:
            waypoints.append(
                Waypoint(
                    waypoint_dict=waypoint_dict,
                    known_waypoints=waypoints,
                    obs=obs,
                    waypoint_expert=self)
            )

        return waypoints

    def _get_action(self, current_obs: OrderedDict, waypoint: Waypoint) -> np.ndarray:
        """
        Get the action to reach the waypoint.
        The output is a clipped version of the Waypoint state and is scaled to the controller input range, e.g. [-1, 1].

        robosuite expects the following action components:
        - position action [x, y, z] or [dx, dy, dz]
        - orientation action as a 3D axis angle [vx, vy, vz] or [dvx, dvy, dvz],
            with the magnitude of the axis angle being the rotation in radians
        - gripper action [grip] (-1 => open, 1 => closed)

        - position action and orientation action are concatenated to the part action for the OSC
        - the gripper action is a separate action for the gripper controller

        :param current_obs: Current observation
        :param waypoint: Waypoint to reach
        :return: Action to reach the waypoint
        """
        current = TwoArmEEState.from_dict(current_obs, self._env.env_configuration)

        ctrl_input_min_lr = [self._ctrl_input_min["left"], self._ctrl_input_min["right"]]
        ctrl_input_max_lr = [self._ctrl_input_max["left"], self._ctrl_input_max["right"]]
        ctrl_output_min_lr = [self._ctrl_output_min["left"], self._ctrl_output_min["right"]]
        ctrl_output_max_lr = [self._ctrl_output_max["left"], self._ctrl_output_max["right"]]
        target_lr = [waypoint.target.left, waypoint.target.right]
        current_lr = [current.left, current.right]
        part_action_lr = []
        grip_action_lr = []

        for target, current, ctrl_input_min, ctrl_input_max, ctrl_output_min, ctrl_output_max, in zip(
                target_lr, current_lr, ctrl_input_min_lr, ctrl_input_max_lr, ctrl_output_min_lr, ctrl_output_max_lr
        ):
            # Clip the translation
            pos_target = target.xyz
            pos_current = current.xyz

            v_min, v_max = get_limits(target.max_vel_pos, ctrl_output_min[:3], ctrl_output_max[:3])

            pos_delta = clip_translation(pos_target - pos_current, v_min, v_max)
            if self._action_mode == "absolute":
                pos_action = pos_current + pos_delta
            elif self._action_mode == "delta":
                pos_action = pos_delta
            else:
                raise ValueError(f"[WP] Invalid action mode {self._action_mode}. Expected 'delta' or 'absolute'")

            quat_target = target.quat
            quat_current = current.quat

            v_min, v_max = get_limits(target.max_vel_ori, ctrl_output_min[3:], ctrl_output_max[3:])

            quat_delta = clip_quat_by_axisangle( # output quat of type wxyz
                quat_multiply(quat_target, quat_inverse(quat_current)),
                v_min,
                v_max
            )
            if self._action_mode == "absolute":
                rot_target = quat_multiply(quat_delta, quat_current)
                rot_action = quat2axisangle(rot_target)
            else:
                rot_action = quat2axisangle(quat_delta)

            part_action = np.concatenate([pos_action, rot_action])

            # Rescale the input range
            if self._action_mode == "delta":
                # this is the reverse of
                # https://github.com/ARISE-Initiative/robosuite/blob/0926cbec81bf19ff7667d387b55da8b8714647ea/robosuite/controllers/parts/controller.py#L149
                action_scale = abs(ctrl_output_max - ctrl_output_min) / abs(ctrl_input_max - ctrl_input_min)
                action_output_transform = (ctrl_output_max + ctrl_output_min) / 2.0
                action_input_transform = (ctrl_input_max + ctrl_input_min) / 2.0

                part_action = (part_action - action_output_transform) / action_scale + action_input_transform

            part_action_lr.append(part_action)

            # Set GripperState
            grip_action = np.array([target.grip])
            grip_action_lr.append(grip_action)

        action = self.__compose_action(part_action_lr, grip_action_lr)

        return action

    def __compose_action(self, part_action_lr: list[np.ndarray], grip_action_lr: list[np.ndarray]) -> np.ndarray:
        """
        Compose the action for the two arms.

        robosuite order: right before left

        :param part_action_lr: the action controlling the position and orientation of the end-effector
        :param grip_action_lr: the action controlling the gripper state
        :return: total action
        """
        dof = self._env.action_spec[0].shape[0]
        if dof == 12: # no gripper
            return np.concatenate(part_action_lr[::-1])

        if dof == 14: # gripper + single arms
            if self._env.env_configuration == "single-robot":
                # first the part actions, then the gripper actions
                return np.concatenate(part_action_lr[::-1] + grip_action_lr[::-1])
            else:
                # part + gripper + part + gripper
                return np.concatenate([
                    part_action_lr[1], grip_action_lr[1], part_action_lr[0], grip_action_lr[0]
                ])

        raise NotImplementedError(f"[WP] Unsupported action space with {dof} DoF.")

    def _run_episode(
        self,
        render : bool = False,
        target_real_time : bool = False,
        randomize : bool = False
    ) -> bool:
        """
        Run the expert agent in the environment.
        :param render: Whether to render the environment
        :param target_real_time: Whether to render in real time
        :param randomize: Whether to randomize the environment

        :return: True if the episode was successful
        """
        if target_real_time and not (render and self._env.has_renderer):
            print("[TwoArmWaypointExpertBase - INFO] Ignoring real time rendering as the environment " + \
                  "does not have a renderer and/or render==False.")
            target_real_time = False

        # reset the environment
        self._env.deterministic_reset = not randomize
        obs = self._env.reset()

        waypoints = self._create_waypoints(obs)

        steps_terminated = 0
        steps_in_last_wp = 0
        success = False
        unreachable = False
        for n, waypoint in enumerate(waypoints):
            is_last_wp = n == len(waypoints) - 1
            wp_step = 0
            self._rt_handler.reset()
            while True:
                if waypoint.uses_feedback:
                    waypoint.update(obs)

                action = self._get_action(obs, waypoint)
                obs, _, _, _ = self._env.step(action)
                done = self._env._check_success()
                if done:
                    steps_terminated += 1
                    if steps_terminated >= self._min_steps_terminated:
                        success = True
                        break
                else:
                    steps_terminated = 0

                if render:
                    self._env.render()
                wp_step += 1
                reached, unreachable = waypoint.is_reached_by(
                    obs, self._env.env_configuration, wp_step * self._dt
                )

                if reached and not is_last_wp:
                    break
                elif reached and is_last_wp:
                    steps_in_last_wp += 1
                    if steps_in_last_wp >= self._min_steps_last_wp:
                        success = done
                        break
                elif unreachable:
                    break

                if target_real_time:
                    self._rt_handler.sleep()

            if success:
                break
            if unreachable:
                print(f"[INFO] Waypoint {waypoint.id} ({waypoint.description}) could not be reached. Aborting this episode.")
                break

        if success:
            print(f"[INFO] Episode finished successful.")
        else:
            print(f"[INFO] Episode finished unsuccessful.")

        if self._env.viewer is not None:
            self._env.viewer.close()

        return success

    def visualize(
        self,
        num_episodes : int = 1,
        visualize_targets : bool = False,
        num_recording_episodes : int = 0,
    ) -> None:
        """
        Visualize the expert agent in the environment.
        :param num_episodes: Number of episodes to visualize
        :param visualize_targets: Whether to visualize the target positions
        :param num_recording_episodes: Number of episodes to record
        """
        if visualize_targets:
            self._env = TargetVisualizationWrapper(self._env)

        num_recording_episodes = min(num_recording_episodes, num_episodes)
        if num_recording_episodes > 0:
            self._env = RecordingWrapper(self._env)
            self._env.start_recording(directory=RECORDING_DIR)

        for i in range(num_episodes):

            if (num_recording_episodes > 0) and (i == num_recording_episodes):
                self._env.stop_recording()

            _ = self._run_episode(
                render=True,
                target_real_time=True,
                randomize=True
            )

        if visualize_targets and num_recording_episodes > 0:
            self._env = self._env.unwrapped().unwrapped()
        elif num_recording_episodes > 0:
            self._env = self._env.unwrapped()
        elif visualize_targets:
            self._env = self._env.unwrapped()

    def collect_data(self,
        out_dir : str,
        num_successes : int,
        env_config: dict,
        render: bool = False,
        target_real_time: bool = False
    ):
        """
        Collect demonstration from the expert agent in the environment.

        :param out_dir: Output directory for the data
        :param num_successes: Number of successful episodes to collect
        :param env_config: Configuration of the environment, needed to ensure reproducibility
        :param render: Whether to render the environment
        :param target_real_time: Whether to render in real time
        """
        tmp_dir = os.path.join("/tmp", "bil_wp", str(time.time()).replace(".", "_"))

        self._env = DataCollectionWrapper(
            self._env,
            directory=tmp_dir,
        )

        success_count = 0
        count = 0
        while success_count < num_successes:
            success_count += self._run_episode(
                render=render,
                target_real_time=target_real_time,
                randomize=True
            )
            count += 1
            print(f"[INFO] Collected {success_count}/{num_successes} successful demonstrations after {count} episodes.")

        self._env.close()

        env_info = json.dumps(env_config)
        gather_demonstrations_as_hdf5(tmp_dir, out_dir, env_info)

        shutil.rmtree(tmp_dir)


    def _create_ee_target_methods_dict(self) -> dict:
        """
        Create a dictionary of methods that return the position, orientation, and gripper state of a device.

        All methods take the observation after reset as input.
        :return: Dictionary of methods
        """
        return {
            "initial_state_left": self._initial_state_left,
            "initial_state_right": self._initial_state_right,
        }

    def _initial_state_left(self, obs: OrderedDict = None) -> dict:
        """
        Initial state for the left arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        state: EEState = TwoArmEEState.from_dict(obs, env_config=self._env.env_configuration).left
        dct = {
            "pos": state.xyz,
            "quat": state.quat,
            "grip": GripperTarget.OPEN_VALUE
        }
        return dct

    def _initial_state_right(self, obs: OrderedDict = None) -> dict:
        """
        Initial state for the right arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        state: EEState = TwoArmEEState.from_dict(obs, env_config=self._env.env_configuration).right
        dtc = {
            "pos": state.xyz,
            "quat": state.quat,
            "grip": GripperTarget.OPEN_VALUE
        }
        return dtc

    @classmethod
    def example(
        cls,
        waypoints_file: str,
        num_episodes: int = 10,
        robots: str | list[str] = ["Panda"] * 2,
        gripper_types: str | list[str] | None = ["default", "default"],
        num_recording_episodes: int = 0,
    ):
        """
        Run an example for the expert agent in the environment.
        :param waypoints_file: file containing the waypoints
        :param num_episodes: Number of episodes to visualize
        :param robots: robot configuration
        :param gripper_types: gripper configuration
        :param num_recording_episodes: number of episodes to record
        """
        if cls.__class__.__name__ == "TwoArmLiftWaypointExpertBase":
            # exit because the TwoArmLiftWaypointExpertBase does not implement a task
            print("[INFO] Skipping example for TwoArmLiftWaypointExpertBase.")
            return

        env = suite.make(
            env_name=cls.target_env_name,
            gripper_types=gripper_types,
            robots=robots,
            env_configuration="parallel",
            has_renderer=True,
            camera_names="frontview" if num_recording_episodes > 0 else "agentview",
            has_offscreen_renderer=num_recording_episodes > 0,
            use_camera_obs=num_recording_episodes > 0,
            camera_widths=1280 if num_recording_episodes > 0 else 256,
            camera_heights=720 if num_recording_episodes > 0 else 256,
        )

        expert = cls(
            environment=env,
            waypoints_file=waypoints_file,
        )
        expert.visualize(
            num_episodes=num_episodes,
            num_recording_episodes=num_recording_episodes,
        )




