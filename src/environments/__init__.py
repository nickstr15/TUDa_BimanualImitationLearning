from robosuite.environments.base import register_env

from src.environments.manipulation.two_arm_pick_place import TwoArmPickPlace
from src.environments.manipulation.two_arm_ball_insert import TwoArmBallInsert
from src.environments.manipulation.two_arm_hinged_bin import TwoArmHingedBin
from src.environments.manipulation.two_arm_quad_insert import TwoArmQuadInsert

from src.environments.manipulation.flipped import TwoArmPegInHoleFlipped, TwoArmLiftFlipped, TwoArmHandoverFlipped, \
    TwoArmTransportFlipped, TwoArmBallInsertFlipped, TwoArmQuadInsertFlipped, TwoArmHingedBinFlipped, \
    TwoArmPickPlaceFlipped

ENVIRONMENTS = [
    TwoArmPickPlace, TwoArmBallInsert, TwoArmHingedBin, TwoArmQuadInsert,
    TwoArmPickPlaceFlipped, TwoArmBallInsertFlipped, TwoArmQuadInsertFlipped, TwoArmHingedBinFlipped,
    TwoArmLiftFlipped, TwoArmHandoverFlipped, TwoArmTransportFlipped, TwoArmPegInHoleFlipped
]

for env in ENVIRONMENTS:
    register_env(env)