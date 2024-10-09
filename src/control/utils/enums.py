from enum import Enum, IntEnum

class GripperState(IntEnum):
    OPEN = 255
    CLOSED = 0

class DeviceState(Enum):
    Q = "Q"
    Q_ACTUATED = "Q_ACTUATED"
    DQ = "DQ"
    DQ_ACTUATED = "DQ_ACTUATED"
    DDQ = "DDQ"
    EE_XYZ = "EE_XYZ"
    EE_XYZ_VEL = "EE_XYZ_VEL"
    EE_QUAT = "EE_QUAT"
    FORCE = "FORCE"
    TORQUE = "TORQUE"
    J = "JACOBIAN"
    GRIPPER = "GRIPPER"

class RobotState(Enum):
    M = "INERTIA"
    DQ = "DQ"
    J = "JACOBIAN"
    G = "GRAVITY"