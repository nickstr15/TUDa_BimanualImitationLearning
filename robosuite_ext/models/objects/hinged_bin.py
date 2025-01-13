import numpy as np

from robosuite_ext.models.objects import CompositeBodyObject, BoxObject, CylinderObject
from robosuite_ext.utils.mjcf_utils import CustomMaterial, add_to_dict, array_to_string
from robosuite_ext.utils.transform_utils import convert_quat, axisangle2quat


class HingedBin(CompositeBodyObject):
    """
    Generates a four-walled bin container with a hinged lid on top.
    The lid has an additional handle attached to it.
    rgs:
        name (str): Name of this Bin object
        full_bin_size (3-array): (x,y,z) full size of bin without the handle
        wall_thickness (float): How thick to make walls of bin
        handle_height (float): Height of the handle
        handle_width (float): Width of the handle
        handle_thickness (float): Thickness of the handle
        transparent_walls (bool): If True, walls will be semi-translucent
        friction (3-array or None): If specified, sets friction values for this bin. None results in default values
        bin_density (float): Density value to use for all geoms of the bin. Defaults to 100,000
        lid_density (float): Density value to use for all geoms of the lid. Defaults to 250
        use_texture (bool): If true, geoms will be defined by realistic textures and rgba values will be ignored
        rgba (4-array or None): If specified, sets rgba values for all geoms. None results in default values
    """

    def __init__(
        self,
        name,
        full_bin_size=(0.3, 0.3, 0.15),
        wall_thickness=0.01,
        handle_height=0.04,
        handle_width=0.1,
        handle_thickness=0.02,
        transparent_walls=True,
        friction=None,
        bin_density=100000.0,
        lid_density=250.0,
        use_texture=True,
        rgba=(0.2, 0.1, 0.0, 1.0),
    ):
        # Set name
        self._name = name

        # Set object attributes
        self.full_bin_size = np.array(full_bin_size)

        self.wall_thickness = wall_thickness
        self.handle_height = handle_height
        self.handle_width = handle_width
        self.handle_thickness = handle_thickness

        self.transparent_walls = transparent_walls
        self.friction = friction if friction is None else np.array(friction)
        self.bin_density = bin_density
        self.lid_density = lid_density
        self.use_texture = use_texture
        self.rgba = rgba

        # Element references
        self._base_geom = "base"

        # Other private attributes
        self._important_sites = {}

        # Material
        # Define materials we want to use for this object
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }
        self.wood_mat = CustomMaterial(
            texture="WoodDark",
            tex_name="wood_tex",
            mat_name="wood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        # Create dictionary of values to create geoms for composite object and run super init
        super().__init__(**self._get_object_attrs())


        # Manually fix top, bottom, and radius
        self._bottom = -0.5*self.wall_thickness
        self._top = self.full_bin_size[2] + self.handle_height - 0.5*self.wall_thickness
        max_half_width = 0.5*np.max(self.full_bin_size[0:2])
        self._radius = np.sqrt(2*max_half_width**2)


    def _get_object_attrs(self):
        """
        Creates object elements that will be passed to superclass CompositeBodyObject constructor
        Returns:
            dict: args to be used by CompositeBodyObject to generate geoms
        """
        # Initialize dict of obj args that we'll pass to the CompositeObject constructor
        base_args = dict(
            name=self.name,
        )

        # Material and RGBA
        if self.transparent_walls:
            wall_rgba = (1.0, 1.0, 1.0, 0.3)
            wall_mat = None
        else:
            wall_rgba = None if self.use_texture else self.rgba
            wall_mat = self.wood_mat if self.use_texture else None

        if self.use_texture:
            base_rgba = None
            base_mat = self.wood_mat
        else:
            base_rgba = self.rgba
            base_mat = None

        # 1) Bin Base:
        half_base_size = 0.5*np.array([
            self.full_bin_size[0] - 2*self.wall_thickness,
            self.full_bin_size[1] - 2*self.wall_thickness,
            self.wall_thickness
        ])

        base_quat = np.array([1, 0, 0, 0])
        base = BoxObject(
            name="base",
            size=half_base_size,
            rgba=base_rgba,
            material=base_mat,
            density=self.bin_density,
            friction=self.friction,
        )

        # 2) Bin Walls
        x_vals = np.array([
            0,
            half_base_size[0] + 0.5*self.wall_thickness,
            0,
            -(half_base_size[0] + 0.5*self.wall_thickness),
        ])
        y_vals = np.array([
            half_base_size[1] + 0.5*self.wall_thickness,
            0,
            -(half_base_size[1] + 0.5*self.wall_thickness),
            0,
        ])
        half_width_vals = np.array([
            half_base_size[1] + self.wall_thickness,
            half_base_size[0] + self.wall_thickness,
            half_base_size[1] + self.wall_thickness,
            half_base_size[0] + self.wall_thickness,
        ])
        r_vals = np.array([
            np.pi/2,
            0,
            np.pi/2,
            0,
        ])
        half_wall_height = 0.5*(self.full_bin_size[2] - self.wall_thickness)

        wall_list = []
        wall_pos_list = []
        wall_quat_list = []
        for i, (x, y, hw, r) in enumerate(zip(x_vals, y_vals, half_width_vals, r_vals)):
            half_wall_size = np.array([
                0.5*self.wall_thickness,
                hw,
                half_wall_height
            ])
            # position relative to base
            wall_pos = np.array([
                x,
                y,
                half_wall_height - 0.5*self.wall_thickness
            ])
            wall_quat = convert_quat(axisangle2quat(np.array([0, 0, r])), to="wxyz")
            wall = BoxObject(
                name=f"wall_{i}",
                size=half_wall_size,
                rgba=wall_rgba,
                material=wall_mat,
                density=self.bin_density,
                friction=self.friction,
            )

            wall_list.append(wall)
            wall_pos_list.append(wall_pos)
            wall_quat_list.append(wall_quat)

        # 3) Lid
        half_lid_size = 0.5*np.array([
            self.full_bin_size[0],
            self.full_bin_size[1],
            self.wall_thickness
        ])
        # position relative to left wall
        lid_pos = np.array([
            -(half_lid_size[0]-0.5*self.wall_thickness),
            0,
            half_wall_height + half_lid_size[2]
        ])
        lid_quat = np.array([1, 0, 0, 0])
        lid = BoxObject(
            name="lid",
            size=half_lid_size,
            rgba=wall_rgba,
            material=wall_mat,
            density=self.lid_density,
            friction=self.friction,
        )

        # 4) Hinge
        half_hinge_size = np.array([
            self.wall_thickness,
            half_lid_size[1]
        ])
        # position relative to lid
        hinge_pos = np.array([
            half_lid_size[0],
            0,
            -0.5*self.wall_thickness])
        hinge_quat = convert_quat(axisangle2quat(np.array([np.pi/2, 0, 0])), to="wxyz")
        hinge = CylinderObject(
            name="hinge",
            size=half_hinge_size,
            rgba=base_rgba,
            material=base_mat,
            density=self.lid_density,
            friction=self.friction,
            obj_type="visual",
        )

        hinge_joint = {
            "name": "bin_hinge",
            "type": "hinge",
            "axis": "0 1 0", # hinge around y-axis
            "pos": array_to_string(hinge_pos),
            "stiffness": "0.0001",
            "limited": "true",
            "range": "0 1.5708", # 0 to 90 degrees
        }

        # 5) Handle
        # the handle consists of three parts, to form a flipped u
        # the handle is attached to the lid
        handle_half_size = 0.5*np.array([
            self.handle_thickness,
            self.handle_width,
            self.handle_height
        ])
        handle_pos = np.array([
            -0.8*half_lid_size[0],
            0,
            0.5*self.handle_height + half_lid_size[2]
        ])
        handle_quat = np.array([1, 0, 0, 0])
        self.handle = BoxObject(
            name="handle",
            size=handle_half_size,
            rgba=base_rgba,
            material=base_mat,
            density=self.lid_density,
            friction=self.friction,
        )

        # build args
        objects = [
            base,
            *wall_list,
            lid,
            hinge,
            self.handle
        ]

        parents = [
            None, # Bin base attached to top-level body
            *[base.root_body for _ in range(len(wall_list))], # Bin walls attached to bin base
            wall_list[0].root_body, # Lid attached to first wall, which is the left one
            lid.root_body, # Hinge attached to lid
            lid.root_body, # Handle attached to lid
        ]

        joints = {
            lid.root_body: [hinge_joint],
        }

        positions = [
            np.zeros(3),  # First element is centered at top-level body anyway
            *wall_pos_list,
            lid_pos,
            hinge_pos,
            handle_pos,
        ]
        quats = [
            base_quat,
            *wall_quat_list,
            lid_quat,
            hinge_quat,
            handle_quat,
        ]

        obj_args = dict(
            objects=objects,
            object_locations=positions,
            object_quats=quats,
            object_parents=parents,
            body_joints=joints
        )

        obj_args.update(base_args)

        # Return this dict
        return obj_args

    @property
    def base_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to bin base
        """
        return [self.correct_naming(self._base_geom)]