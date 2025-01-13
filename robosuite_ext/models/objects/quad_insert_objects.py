import numpy as np

from robosuite.models.objects import MujocoXMLObject
from utils.paths import path_completion

class QuadBracket(MujocoXMLObject):
    def __init__(self, name: str):
        super().__init__(
            path_completion("objects/quad_insert/quad_peg/quad_bracket.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle_a'`: Name of first handle site
                :`'handle_b'`: Name of second handle site
                :`'center'`: Name of center site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle0": self.naming_prefix + "handle0_site"})
        dic.update({"handle1": self.naming_prefix + "handle1_site"})
        dic.update({"center": self.naming_prefix + "center_site"})
        dic.update({"flap_a_hole": self.naming_prefix + "flap_a_hole_site"})
        dic.update({"flap_b_hole": self.naming_prefix + "flap_b_hole_site"})
        dic.update({"flap_c_hole": self.naming_prefix + "flap_c_hole_site"})
        dic.update({"flap_d_hole": self.naming_prefix + "flap_d_hole_site"})
        return dic

    @property
    def center_to_handle0(self):
        """
        Returns:
            np.array: vector from center to handle_a
        """
        return np.array([0, -0.175, 0.055])

    @property
    def center_to_handle1(self):
        """
        Returns:
            np.array: vector from center to handle_b
        """
        return np.array([0, 0.325, 0.055])

class QuadPeg(MujocoXMLObject):
    def __init__(self, name: str):
        super().__init__(
            path_completion("objects/quad_insert/quad_peg/quad_peg.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'target'`: Name of target location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"target": self.naming_prefix + "target_site"})
        dic.update({"flap_a_peg": self.naming_prefix + "flap_a_peg_site"})
        dic.update({"flap_b_peg": self.naming_prefix + "flap_b_peg_site"})
        dic.update({"flap_c_peg": self.naming_prefix + "flap_c_peg_site"})
        dic.update({"flap_d_peg": self.naming_prefix + "flap_d_peg_site"})
        return dic