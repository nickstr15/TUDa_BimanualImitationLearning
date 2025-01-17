from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.transform_utils import quat2mat, euler2mat, mat2euler, mat2quat

from robosuite_ext.models.objects.quad_insert_objects import QuadBracket, QuadPeg


class TwoArmQuadInsert(TwoArmEnv):
    """
    This class corresponds to a bimanual pick and place task, requiring the robot to pick up a bracket with four holes
    that must be placed on a peg. The task is considered successful if the bracket is successfully inserted.

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

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            each table.

        arm_distances (float): the distance between the two arms. Default is 0.6. Only used if two robots are used and
            the env_configuration is "parallel".

        position_tol_bracket (2-tuple): the tolerance for the position of the bracket at the start of the episodes.
            The position will be sampled uniformly in the range for x and y dimension.
            Default is (0.02, 0.02).

        orientation_tol_bracket (2-tuple): the tolerance for the rotation of the bracket at the start of the episodes.
            The rotation will be sampled uniformly in the range around the z axis.
            Default is [-pi/8, pi/8].

        position_tol_peg (2-tuple): the tolerance for the position of the peg at the start of the episodes.
            The position will be sampled uniformly in the range for x and y dimension.
            Default is (0.02, 0.02).

        orientation_tol_peg (2-tuple): the tolerance for the rotation of the peg at the start of the episodes.
            The rotation will be sampled uniformly in the range around the z axis.
            Default is [-pi/8, 0].

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

        flipped (bool): Whether to flip the environment in such a way that the responsibilities of the two arms are swapped.

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
            table_full_size=(0.8, 1.5, 0.05),
            table_friction=(1.0, 5e-3, 1e-4),
            arm_distances=0.6,
            position_tol_bracket=(0.02, 0.02),
            orientation_tol_bracket=(-np.pi/12, np.pi/12),
            position_tol_peg=(0.02, 0.02),
            orientation_tol_peg=(-np.pi/12, np.pi/12),
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
        # fixed settings
        self.handle_distance = 0.5

        # settings for table-top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # setting for arm position
        self.arm_distances = arm_distances

        # initial sampling range for object positions
        self.position_tol_bracket = position_tol_bracket
        self.orientation_tol_bracket = orientation_tol_bracket
        self.position_tol_peg = position_tol_peg
        self.orientation_tol_peg = orientation_tol_peg

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # make default env_configuration "parallel" if two robots are used
        if len(robots) == 2 and env_configuration == "default":
            env_configuration = "parallel"

        self.placement_initializer = None

        self.flipped = False

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
            0 if bracket is not inserted
            1 if bracket is inserted

        Un-normalized summed components if using reward shaping:
            Not implemented for this task

        Note that the final reward is normalized and scaled by reward_scale / 1.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward_ = 0

        # check for goal completion: hammer in the bin
        if self._check_success():
            reward_ += 1

        # if using reward shaping, add distance component
        if self.reward_shaping:
            raise NotImplementedError("Reward shaping is not implemented for this task.")

        if self.reward_scale is not None:
            reward_ *= self.reward_scale / 1.0

        return reward_


    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose(s) accordingly
        if self.env_configuration == "single-robot":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "opposed":
                # Set up robots facing towards each other by rotating them from their default position
                for robot, rotation in zip(self.robots, (np.pi / 2, -np.pi / 2)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    rot = np.array((0, 0, rotation))
                    xpos = euler2mat(rot) @ np.array(xpos)
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            else:  # "parallel" configuration setting
                # Set up robots parallel to each other but offset from the center
                half_arm_distance = self.arm_distances / 2.0
                offsets = (-half_arm_distance, half_arm_distance)
                for robot, offset in zip(self.robots, offsets):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)

        # load model for table-top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.8894354364730311, -3.481824231498976e-08, 1.7383813133506494],
            quat=[0.6530981063842773, 0.2710406184196472, 0.27104079723358154, 0.6530979871749878],
        )

        # initialize objects of interest
        self.bracket = QuadBracket(name="bracket")
        self.peg = QuadPeg(name="peg")

        self.placement_initializer = self._get_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena = mujoco_arena,
            mujoco_robots = [robot.robot_model for robot in self.robots],
            mujoco_objects = [self.bracket, self.peg],
        )

    def _get_placement_initializer(self):
        """
        Helper function to return a placement initializer for the task.
        """

        # Create a placement initializer
        placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # Pre-define settings for each object's placement
        objects = [self.bracket, self.peg]
        x_centers = [-0.05, -0.15]
        y_centers = [-0.4, 0.35] if not self.flipped else [0.4, -0.35]
        x_tols = [self.position_tol_bracket[0], self.position_tol_peg[0]]
        y_tols = [self.position_tol_bracket[1], self.position_tol_peg[1]]
        rot_tols = [self.orientation_tol_bracket, self.orientation_tol_peg]
        rot_axes = ["z", "z"]
        for obj, x, y, x_tol, y_tol, rot_tol, r_axis in zip(
            objects,
            x_centers, y_centers,
            x_tols, y_tols,
            rot_tols, rot_axes
        ):
            # Create a sampler for the object
            sampler = UniformRandomSampler(
                name=f"{obj.name}ObjectSampler",
                mujoco_objects=obj,
                x_range=[x-x_tol, x+x_tol],
                y_range=[y-y_tol, y+y_tol],
                rotation=[rot_tol[0], rot_tol[1]],
                rotation_axis=r_axis,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
            )
            placement_initializer.append_sampler(sampler)

        return placement_initializer

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Hammer object references from this env
        self.bracket_body_id = self.sim.model.body_name2id(self.bracket.root_body)

        # General env references
        self.table_top_id = self.sim.model.site_name2id("table_top")

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment.

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            modality = "object"

            #position and rotation of hammer
            @sensor(modality=modality)
            def bracket_xpos(_):
                return np.array(self._bracket_xpos)

            @sensor(modality=modality)
            def bracket_quat(_):
                return np.array(self._bracket_quat)

            #position and rotation of bracket handles
            @sensor(modality=modality)
            def handle0_xpos(_):
                return np.array(self._bracket_handle0_xpos)

            @sensor(modality=modality)
            def handle1_xpos(_):
                return np.array(self._bracket_handle1_xpos)

            @sensor(modality=modality)
            def target_xpos(_):
                return np.array(self._peg_target_xpos)

            @sensor(modality=modality)
            def target_quat(_):
                return np.array(self._peg_target_quat)

            @sensor(modality=modality)
            def flap_a_hole_xpos(_):
                return np.array(self._flap_a_hole_xpos)

            @sensor(modality=modality)
            def flap_b_hole_xpos(_):
                return np.array(self._flap_b_hole_xpos)

            @sensor(modality=modality)
            def flap_c_hole_xpos(_):
                return np.array(self._flap_c_hole_xpos)

            @sensor(modality=modality)
            def flap_d_hole_xpos(_):
                return np.array(self._flap_d_hole_xpos)

            @sensor(modality=modality)
            def flap_a_peg_xpos(_):
                return np.array(self._flap_a_peg_xpos)

            @sensor(modality=modality)
            def flap_b_peg_xpos(_):
                return np.array(self._flap_b_peg_xpos)

            @sensor(modality=modality)
            def flap_c_peg_xpos(_):
                return np.array(self._flap_c_peg_xpos)

            @sensor(modality=modality)
            def flap_d_peg_xpos(_):
                return np.array(self._flap_d_peg_xpos)

            sensors = [
                bracket_xpos, bracket_quat,
                handle0_xpos, handle1_xpos,
                target_xpos, target_quat,
                flap_a_hole_xpos, flap_b_hole_xpos, flap_c_hole_xpos, flap_d_hole_xpos,
                flap_a_peg_xpos, flap_b_peg_xpos, flap_c_peg_xpos, flap_d_peg_xpos
            ]
            names = [s.__name__ for s in sensors]

            arm_sensor_fns = []
            if self.env_configuration == "single-robot":
                # If single-robot, we only have one robot. gripper 0 is always right and gripper 1 is always left
                pf0 = self.robots[0].robot_model.naming_prefix + "right_"
                pf1 = self.robots[0].robot_model.naming_prefix + "left_"
                prefixes = [pf0, pf1]
                arm_sensor_fns = [
                    self._get_obj_eef_sensor(full_pf, f"handle_xpos", f"gripper{idx}_to_handle", modality)
                    for idx, full_pf in enumerate(prefixes)
                ]
            else:
                # If not single-robot, we have two robots. gripper 0 is always the first robot's gripper and
                # gripper 1 is always the second robot's gripper. However, must account for the fact that
                # each robot may have multiple arms/grippers
                robot_arm_prefixes = [self._get_arm_prefixes(robot, include_robot_name=False) for robot in self.robots]
                robot_full_prefixes = [self._get_arm_prefixes(robot, include_robot_name=True) for robot in self.robots]
                for idx, (arm_prefixes, full_prefixes) in enumerate(zip(robot_arm_prefixes, robot_full_prefixes)):
                    arm_sensor_fns += [
                        self._get_obj_eef_sensor(full_pf, f"handle_xpos", f"{arm_pf}gripper{idx}_to_handle", modality)
                        for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
                    ]

            sensors += arm_sensor_fns
            names += [s.__name__ for s in arm_sensor_fns]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from a xml
        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()
            # Loop through all objects and reset their positions
            for pos, quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(pos), np.array(quat)])
                )
            self.sim.step()

    def _check_success(self):
        """
        Check if the task has been completed. For this task, this means the hammer is in the bin.

        Returns:
            bool: True if task is successful (hammer is in the bin), False otherwise.
        """

        return self._bracket_inserted and not self._grasping_bracket

    @property
    def _bracket_xpos(self):
        """
        Returns the position of the hammer in the world frame.
        """
        return self.sim.data.site_xpos[
            self.sim.model.site_name2id(self.bracket.important_sites["center"])
        ]

    @property
    def _bracket_quat(self):
        """
        Returns the orientation of the hammer in the world frame.
        """
        xmat = self.sim.data.site_xmat[
            self.sim.model.site_name2id(self.bracket.important_sites["center"])
        ]

        return mat2quat(xmat.reshape((3, 3)))

    @property
    def _bracket_handle0_xpos(self):
        """
        Returns the position of the first handle of the hammer in the world frame.
        """
        return self.sim.data.site_xpos[
            self.sim.model.site_name2id(self.bracket.important_sites["handle0"])
        ]

    @property
    def _bracket_handle1_xpos(self):
        """
        Returns the position of the second handle of the hammer in the world frame.
        """
        return self.sim.data.site_xpos[
            self.sim.model.site_name2id(self.bracket.important_sites["handle1"])
        ]

    @property
    def _peg_target_xpos(self):
        """
        Returns the position of the target peg in the world frame.
        """
        return self.sim.data.site_xpos[
            self.sim.model.site_name2id(self.peg.important_sites["target"])
        ]

    @property
    def _peg_target_quat(self):
        """
        Returns the orientation of the target peg in the world frame.
        """
        xmat = self.sim.data.site_xmat[
            self.sim.model.site_name2id(self.peg.important_sites["target"])
        ]
        return mat2quat(xmat.reshape((3, 3)))

    @property
    def _flap_a_hole_xpos(self):
        """
        Returns the position of the first hole of the bracket in the world frame.
        """
        return self.sim.data.site_xpos[
            self.sim.model.site_name2id(self.bracket.important_sites["flap_a_hole"])
        ]

    @property
    def _flap_b_hole_xpos(self):
        """
        Returns the position of the second hole of the bracket in the world frame.
        """
        return self.sim.data.site_xpos[
            self.sim.model.site_name2id(self.bracket.important_sites["flap_b_hole"])
        ]

    @property
    def _flap_c_hole_xpos(self):
        """
        Returns the position of the third hole of the bracket in the world frame.
        """
        return self.sim.data.site_xpos[
            self.sim.model.site_name2id(self.bracket.important_sites["flap_c_hole"])
        ]

    @property
    def _flap_d_hole_xpos(self):
        """
        Returns the position of the fourth hole of the bracket in the world frame.
        """
        return self.sim.data.site_xpos[
            self.sim.model.site_name2id(self.bracket.important_sites["flap_d_hole"])
        ]

    @property
    def _flap_a_peg_xpos(self):
        """
        Returns the position of the first hole of the peg in the world frame.
        """
        return self.sim.data.site_xpos[
            self.sim.model.site_name2id(self.peg.important_sites["flap_a_peg"])
        ]

    @property
    def _flap_b_peg_xpos(self):
        """
        Returns the position of the second hole of the peg in the world frame.
        """
        return self.sim.data.site_xpos[
            self.sim.model.site_name2id(self.peg.important_sites["flap_b_peg"])
        ]

    @property
    def _flap_c_peg_xpos(self):
        """
        Returns the position of the third hole of the peg in the world frame.
        """
        return self.sim.data.site_xpos[
            self.sim.model.site_name2id(self.peg.important_sites["flap_c_peg"])
        ]

    @property
    def _flap_d_peg_xpos(self):
        """
        Returns the position of the fourth hole of the peg in the world frame.
        """
        return self.sim.data.site_xpos[
            self.sim.model.site_name2id(self.peg.important_sites["flap_d_peg"])
        ]


    @property
    def _bracket_inserted(self):
        """
        Returns True if the bracket is inserted on the peg.
        """
        hole_pos = np.array([
            self._flap_a_hole_xpos, self._flap_b_hole_xpos, self._flap_c_hole_xpos, self._flap_d_hole_xpos
        ])

        peg_pos = np.array([
            self._flap_a_peg_xpos, self._flap_b_peg_xpos, self._flap_c_peg_xpos, self._flap_d_peg_xpos
        ])

        pos_ok = np.all(np.linalg.norm(hole_pos - peg_pos, axis=1) < 0.002) # 2mm tolerance

        return pos_ok

    @property
    def _grasping_bracket(self):
        """
        Returns True if any arm is grasping the hammer, False otherwise.
        """
        # Check if any Arm's gripper is grasping the bracket
        (g0, g1) = (
            (self.robots[0].gripper["right"], self.robots[0].gripper["left"])
            if self.env_configuration == "single-robot"
            else (self.robots[0].gripper, self.robots[1].gripper)
        )
        return (
                self._check_grasp(gripper=g0, object_geoms=self.bracket) or
                self._check_grasp(gripper=g1, object_geoms=self.bracket)
        )

if __name__ == "__main__":
    from robosuite_ext.environments.utils.visualize import visualize_static

    visualize_static("TwoArmQuadInsert", robots=["Panda", "Panda"])







