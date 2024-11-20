from robosuite.models.objects import MujocoXMLObject

from src.utils.paths import xml_path_completion

class QuadBracketObject(MujocoXMLObject):
    def __init__(self, name: str):
        super().__init__(
            xml_path_completion("quad_insert/quad_peg/quad_bracket.xml"),
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
        dic.update({"handle_a": self.naming_prefix + "handle_a_site"})
        dic.update({"handle_b": self.naming_prefix + "handle_b_site"})
        dic.update({"center": self.naming_prefix + "center_site"})
        return dic

class QuadPegObject(MujocoXMLObject):
    def __init__(self, name: str):
        super().__init__(
            xml_path_completion("quad_insert/quad_peg/quad_peg.xml"),
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
        return dic