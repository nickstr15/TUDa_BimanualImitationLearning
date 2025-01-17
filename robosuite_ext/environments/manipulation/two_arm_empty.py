from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as tu
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask


class TwoArmEmpty(TwoArmEnv):
    """
    This class corresponds to a bimanual pick and place task, requiring the robot to pick up a
    hammer, hand it over to the other robot, and place it in a bin. The task is considered successful if the hammer is
    placed in the bin.

    Args:
        robots (str or list of str): Specification for specific robot(s)
            Note: Must be either 2 robots or 1 bimanual robot!

        env_configuration (str): Specifies how to position the robots within the environment if two robots inputted. Can be either:

            :`'parallel'`: Sets up the two robots next to each other on the -x side of the table
            :`'opposed'`: Sets up the two robots opposed from each others on the opposite +/-y sides of the table.

        Note that "default" "parallel" if two robots are used.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        arm_distances (float): the distance between the two arms. Default is 0.7. Only used if two robots are used and
            the env_configuration is "parallel".

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using offscreen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set too False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        horizon (int): Every episode lasts for exactly @horizon time-steps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list.

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        ValueError: [Invalid number of robots specified]
        ValueError: [Invalid env configuration]
        ValueError: [Invalid robots for specified env configuration]
    """

    def __init__(
            self,
            robots,
            env_configuration="default",
            controller_configs=None,
            gripper_types="default",
            initialization_noise="default",
            arm_distances=0.7,
            bin_size=(0.3, 0.3, 0.15),
            position_tol=(0.05, 0.05),
            orientation_tol=np.pi/4,
            use_camera_obs=True,
            use_object_obs=True,
            reward_scale=1.0,
            reward_shaping=False,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="frontview",
            render_collision_mesh=False,
            render_visual_mesh=True,
            render_gpu_device_id=-1,
            control_freq=20,
            lite_physics=True,
            horizon=1000,
            ignore_done=False,
            hard_reset=True,
            camera_names="agentview",
            camera_heights=256,
            camera_widths=256,
            camera_depths=False,
            camera_segmentations=None,  # {None, instance, class, element}
            renderer="mjviewer",
            renderer_config=None,
    ):

        # setting for arm position
        self.arm_distances = arm_distances

        # settings for bin
        self.bin_size = np.array(bin_size)

        # initial sampling range for object positions
        self.position_tol = position_tol
        self.orientation_tol = orientation_tol

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # make default env_configuration "parallel" if two robots are used
        if len(robots) == 2 and env_configuration == "default":
            env_configuration = "parallel"

        self.placement_initializer = None

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )


    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:
            0 if hammer is not in the bin
            1 if hammer is in the bin

        Un-normalized summed components if using reward shaping:
            Distance: in [-1, 0], the distance between the hammer and the bin, normalized
                by initial distance (-1 at start), 0 when hammer is in the bin
            Success: 0 if hammer is not in the bin, 1 if hammer is in the bin

        Note that the final reward is normalized and scaled by reward_scale / 1.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        return 0


    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # load model for table-top workspace
        mujoco_arena = TableArena(
            table_offset=(0, 0, 0.05),
        )

        # Adjust base pose(s) accordingly
        if self.env_configuration == "single-robot":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](mujoco_arena.table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "opposed":
                # Set up robots facing towards each other by rotating them from their default position
                for robot, rotation in zip(self.robots, (np.pi / 2, -np.pi / 2)):
                    xpos = robot.robot_model.base_xpos_offset["table"](mujoco_arena.table_full_size[0])
                    rot = np.array((0, 0, rotation))
                    xpos = tu.euler2mat(rot) @ np.array(xpos)
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            else:  # "parallel" configuration setting
                # Set up robots parallel to each other but offset from the center
                half_arm_distance = self.arm_distances / 2.0
                offsets = (-half_arm_distance, half_arm_distance)
                for robot, offset in zip(self.robots, offsets):
                    xpos = robot.robot_model.base_xpos_offset["table"](mujoco_arena.table_full_size[0])
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.8894354364730311, -3.481824231498976e-08, 1.7383813133506494],
            quat=[0.6530981063842773, 0.2710406184196472, 0.27104079723358154, 0.6530979871749878],
        )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena = mujoco_arena,
            mujoco_robots = [robot.robot_model for robot in self.robots],
            mujoco_objects = [],
        )



    def _check_success(self):
        return False







