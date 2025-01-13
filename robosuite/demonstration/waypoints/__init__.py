from robosuite.demonstration.waypoints.two_arm_handover_wp_expert import TwoArmHandoverWaypointExpert
from robosuite.demonstration.waypoints.two_arm_lift_wp_expert import TwoArmLiftWaypointExpert
from robosuite.demonstration.waypoints.two_arm_peg_in_hole_wp_expert import TwoArmPegInHoleWaypointExpert
from robosuite.demonstration.waypoints.two_arm_pick_place_wp_expert import TwoArmPickPlaceWaypointExpert
from robosuite.demonstration.waypoints.two_arm_quad_insert_wp_expert import TwoArmQuadInsertWaypointExpert
from robosuite.demonstration.waypoints.two_arm_transport_wp_expert import TwoArmTransportWaypointExpert
from robosuite.demonstration.waypoints.two_arm_hinged_bin_wp_expert import TwoArmHingedBinWaypointExpert
from robosuite.demonstration.waypoints.two_arm_ball_insert_expert import TwoArmBallInsertWaypointExpert

ENV_TO_WAYPOINT_EXPERT = {
    "TwoArmHandover": TwoArmHandoverWaypointExpert,
    "TwoArmLift": TwoArmLiftWaypointExpert,
    "TwoArmPegInHole": TwoArmPegInHoleWaypointExpert,
    "TwoArmPickPlace": TwoArmPickPlaceWaypointExpert,
    "TwoArmQuadInsert": TwoArmQuadInsertWaypointExpert,
    "TwoArmTransport": TwoArmTransportWaypointExpert,
    "TwoArmHingedBin": TwoArmHingedBinWaypointExpert,
    "TwoArmBallInsert": TwoArmBallInsertWaypointExpert
}