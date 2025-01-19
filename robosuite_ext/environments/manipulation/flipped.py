import numpy as np
from typing_extensions import override

from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import EmptyArena, TableArena
from robosuite.models.objects import PlateWithHoleObject, CylinderObject, HammerObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import find_elements, array_to_string, CustomMaterial
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.transform_utils import euler2mat
from robosuite.environments.manipulation.two_arm_peg_in_hole import TwoArmPegInHole, \
    _OBJECT_POS_OFFSET_FN as PEG_HOLE_OFFSET_FN
from robosuite.environments.manipulation.two_arm_lift import TwoArmLift
from robosuite.environments.manipulation.two_arm_handover import TwoArmHandover
from robosuite.environments.manipulation.two_arm_transport import TwoArmTransport

from robosuite_ext.environments.manipulation.two_arm_quad_insert import TwoArmQuadInsert
from robosuite_ext.environments.manipulation.two_arm_ball_insert import TwoArmBallInsert
from robosuite_ext.environments.manipulation.two_arm_hinged_bin import TwoArmHingedBin
from robosuite_ext.environments.manipulation.two_arm_pick_place import TwoArmPickPlace

##############################################################################################################
# PEG IN HOLE FLIPPED ########################################################################################
##############################################################################################################
class TwoArmPegInHoleFlipped(TwoArmPegInHole):
    @override
    def _load_model(self):
        """
        Loads a xml model, puts it in self.model
        """
        TwoArmEnv._load_model(self)

        # Adjust base pose(s) accordingly
        if self.env_configuration == "single-robot":
            xpos = self.robots[0].robot_model.base_xpos_offset["empty"]
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "opposed":
                # Set up robots facing towards each other by rotating them from their default position
                for robot, rotation in zip(self.robots, (np.pi / 2, -np.pi / 2)):
                    xpos = robot.robot_model.base_xpos_offset["empty"]
                    rot = np.array((0, 0, rotation))
                    xpos = euler2mat(rot) @ np.array(xpos)
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            else:  # "parallel" configuration setting
                # Set up robots parallel to each other but offset from the center
                for robot, offset in zip(self.robots, (-0.25, 0.25)):
                    xpos = robot.robot_model.base_xpos_offset["empty"]
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)

        # Add arena and robot
        mujoco_arena = EmptyArena()

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[1.0666432116509934, 1.4903257668114777e-08, 2.0563394967349096],
            quat=[0.6530979871749878, 0.27104058861732483, 0.27104055881500244, 0.6530978679656982],
        )

        # initialize objects of interest
        self.hole = PlateWithHoleObject(name="hole")
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.peg = CylinderObject(
            name="peg",
            size_min=(self.peg_radius[0], self.peg_length),
            size_max=(self.peg_radius[1], self.peg_length),
            material=greenwood,
            rgba=[0, 1, 0, 1],
            joints=None,
        )

        # Load hole object
        hole_obj = self.hole.get_obj()
        hole_obj.set("quat", "0 0 0.707 0.707")
        hole_robot = self.robots[0] if self.env_configuration == "single-robot" else self.robots[1]
        # Get offset function for hole object based on the robot type
        hole_offset_fn = PEG_HOLE_OFFSET_FN.get(self.robots[0].robot_model.__class__.__name__, {}).get(
            "hole", lambda x: x
        )
        hole_obj.set("pos", array_to_string(hole_offset_fn([0.11, 0, 0.17])))

        # Load peg object
        peg_obj = self.peg.get_obj()
        # Get offset function for peg object based on the robot type
        peg_pos_offset_fn = PEG_HOLE_OFFSET_FN.get(hole_robot.robot_model.__class__.__name__, {}).get(
            "peg", lambda x: x
        )
        peg_obj.set("pos", array_to_string(peg_pos_offset_fn([0, 0, self.peg_length])))

        # Append appropriate objects to arms
        if self.env_configuration == "single-robot":
            eef0, eef1 = [self.robots[0].robot_model.eef_name[arm] for arm in self.robots[0].arms]
            model0, model1 = [self.robots[0].robot_model, self.robots[0].robot_model]
        else:
            # Always place object on the right arm when 2 robots are used
            eef0, eef1 = [robot.robot_model.eef_name["right"] for robot in self.robots]
            model0, model1 = [self.robots[0].robot_model, self.robots[1].robot_model]
        body1 = find_elements(root=model0.worldbody, tags="body", attribs={"name": eef0}, return_first=True)
        body2 = find_elements(root=model1.worldbody, tags="body", attribs={"name": eef1}, return_first=True)
        body1.append(hole_obj)
        body2.append(peg_obj)

        # task includes arena, robot, and objects of interest
        # We don't add peg and hole directly since they were already appended to the robots
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )

        # Make sure to add relevant assets from peg and hole objects
        self.model.merge_assets(self.hole)
        self.model.merge_assets(self.peg)


##############################################################################################################
# LIFT FLIPPED ###############################################################################################
##############################################################################################################
TwoArmLiftFlipped = TwoArmLift


##############################################################################################################
# HANDOVER FLIPPED ###########################################################################################
##############################################################################################################
class TwoArmHandoverFlipped(TwoArmHandover):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.table_offset[1] *= -1

##############################################################################################################
# TRANSPORT FLIPPED ##########################################################################################
##############################################################################################################
class TwoArmTransportFlipped(TwoArmTransport):
    @override
    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds
        """
        # Create placement initializer
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # Pre-define settings for each object's placement
        object_names = ["start_bin", "lid", "payload", "target_bin", "trash", "trash_bin"]
        table_nums = [1, 1, 1, 0, 0, 0]
        x_centers = [
            self.table_full_size[0] * 0.25,
            0,  # gets overridden anyways
            0,  # gets overridden anyways
            -self.table_full_size[0] * 0.25,
            0,  # gets overridden anyways
            self.table_full_size[0] * 0.25,
        ]
        pos_tol = 0.005
        rot_centers = [0, 0, np.pi / 2, 0, 0, 0]
        rot_tols = [0, 0, np.pi / 6, 0, 0.3 * np.pi, 0]
        rot_axes = ["z", "z", "y", "z", "z", "z"]
        for obj_name, x, r, r_tol, r_axis, table_num in zip(
            object_names, x_centers, rot_centers, rot_tols, rot_axes, table_nums
        ):
            # Get name and table
            obj = self.transport.objects[obj_name]
            table_pos = self.table_offsets[table_num]
            # Create sampler for this object and add it to the sequential sampler
            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"{obj_name}ObjectSampler",
                    mujoco_objects=obj,
                    x_range=[x - pos_tol, x + pos_tol],
                    y_range=[-pos_tol, pos_tol],
                    rotation=[r - r_tol, r + r_tol],
                    rotation_axis=r_axis,
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                    reference_pos=table_pos,
                    z_offset=0.001,
                )
            )

##############################################################################################################
# BALL INSERT FLIPPED ########################################################################################
##############################################################################################################
TwoArmBallInsertFlipped = TwoArmBallInsert

##############################################################################################################
# QUAD INSERT FLIPPED ########################################################################################
##############################################################################################################
class TwoArmQuadInsertFlipped(TwoArmQuadInsert):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flipped = True

##############################################################################################################
# HINGED BIN FLIPPED #########################################################################################
##############################################################################################################
class TwoArmHingedBinFlipped(TwoArmHingedBin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flipped = True

##############################################################################################################
# PICK PLACE FLIPPED #########################################################################################
##############################################################################################################
class TwoArmPickPlaceFlipped(TwoArmPickPlace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flipped = True