from src.demonstration.waypoints.two_arm_handover_wp_expert import TwoArmHandoverWaypointExpert
from src.demonstration.waypoints.two_arm_lift_wp_expert import TwoArmLiftWaypointExpert
from src.demonstration.waypoints.two_arm_peg_in_hole_wp_expert import TwoArmPegInHoleWaypointExpert
from src.demonstration.waypoints.two_arm_pick_place_wp_expert import TwoArmPickPlaceWaypointExpert
from src.demonstration.waypoints.two_arm_quad_insert_wp_expert import TwoArmQuadInsertWaypointExpert
from src.demonstration.waypoints.two_arm_transport_wp_expert import TwoArmTransportWaypointExpert

ENV_TO_WAYPOINT_EXPERT = {
    "TwoArmHandover": TwoArmHandoverWaypointExpert,
    "TwoArmLift": TwoArmLiftWaypointExpert,
    "TwoArmPegInHole": TwoArmPegInHoleWaypointExpert,
    "TwoArmPickPlace": TwoArmPickPlaceWaypointExpert,
    "TwoArmQuadInsert": TwoArmQuadInsertWaypointExpert,
    "TwoArmTransport": TwoArmTransportWaypointExpert,
}