from robosuite_ext.environments.base import register_env

from robosuite_ext.environments.manipulation.two_arm_pick_place import TwoArmPickPlace
from robosuite_ext.environments.manipulation.two_arm_ball_insert import TwoArmBallInsert
from robosuite_ext.environments.manipulation.two_arm_hinged_bin import TwoArmHingedBin
from robosuite_ext.environments.manipulation.two_arm_quad_insert import TwoArmQuadInsert

from robosuite_ext.environments.manipulation.flipped import TwoArmPegInHoleFlipped, TwoArmLiftFlipped, TwoArmHandoverFlipped, \
    TwoArmTransportFlipped, TwoArmBallInsertFlipped, TwoArmQuadInsertFlipped, TwoArmHingedBinFlipped, \
    TwoArmPickPlaceFlipped

ENVIRONMENTS = [
    TwoArmPickPlace, TwoArmBallInsert, TwoArmHingedBin, TwoArmQuadInsert,
    TwoArmPickPlaceFlipped, TwoArmBallInsertFlipped, TwoArmQuadInsertFlipped, TwoArmHingedBinFlipped,
    TwoArmLiftFlipped, TwoArmHandoverFlipped, TwoArmTransportFlipped, TwoArmPegInHoleFlipped
]

for env in ENVIRONMENTS:
    register_env(env)