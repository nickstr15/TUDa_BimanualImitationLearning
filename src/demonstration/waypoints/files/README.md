# Waypoint-Files

This folder contains the waypoint files for the different tasks.

#### Table of Contents
- [Structure](#structure) 
- [Examples](#examples)

<a name="structure"></a>
## Structure of the waypoint files
```yaml
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

- id: <id>                           # unique identifier of the waypoint, starting with 0
  description: <description>         # name/description of the waypoint
  targets:
    - device: <device_name>          # mujoco name of the device
      
      # the end-effector target can be specified in different ways:
      # if ee_target is specified position, orientation are derived
      # from the respective method defined in the WaypointExpert class
      # otherwise, the position and orientation are used directly
      ee_target: <method_name>       # method to calculate the target position
      
      pos: [x, y, z]                 # desired position [unit: m]
        
      # orientation can be specified in different ways:
      # only one of the following fields should be used
      # if multiple fields are declared, the priority is as follows: 
      # quat > euler > ax_angle
      quat: [w, x, y, z]             # desired orientation as quaternion
      euler: [roll, pitch, yaw]      # desired orientation as euler angles in radians
      ax_angle: [vx, vy, vz, angle]  # desired orientation as axis-angle in radians
        
      grip: <gripper_state>          # desired gripper state (0: closed, 255: open)
      
      tolerance_position: <tol>      # position tolerance (default: 0.01)
      tolerance_orientation: <tol>   # orientation tolerance in degrees (default: 5°)
    
    - device: <device_name>
      # ...
      
  min_duration: <time>               # time [sec] before status "reached" is possible (default: 1.0)
  max_duration: <time>               # time [sec] to mark waypoint as "unreachable" (default: 30.0)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

- id: <id>
  # ...
```

### Details for the `ee_target`
If the `ee_target` value is specified, the tags `pos`, `quat`, `euler`, `ax_angle`, `grip` are ignored. 
The WaypointExpert class must provide a method with the name specified in `ee_target` that returns a dictionary with the keys `pos`, `quat`, `grip`. If for example the `ee_target` is set to `"pick_up_position"`, the WaypointExpert class must provide a method with the following signature:

```python
@property
def __pick_up_position(self):
    return {
        'pos': [0, 0, 0],
        'quat': [1, 0, 0, 0],
        'grip': 0
    }
```

A reserved method name is `"wp_<id>"`, where the id count starts with 0 for the first waypoint. 
This method returns the position, orientation, and gripper state of a **previous** waypoint with the id `<id>`.

### Details for the position `pos`

The position [`pos`] declaration can be a list of 3 elements `[x, y, z]` 
or a string with the following options:
- `"wp_<id>"`: the position is the same as in a **previous** waypoint with the id `<id>`.
- `"wp_<id>" + [dx, dy, dz]`: the position is the same as in a **previous** waypoint with the id `<id>` plus the offset `[dx, dy, dz]`.

### Details for the orientation `quat`/`euler`/`ax_angle`
The orientation can be specified in different ways:
- `quat`: a list of 4 elements `[w, x, y, z]` representing the quaternion
- `euler`: a list of 3 elements `[roll, pitch, yaw]` representing the euler angles in radians
- `ax_angle`: a list of 4 elements `[vx, vy, vz, angle]` representing the axis-angle in radians

Furthermore, the orientation can be a string with the following options:
- `"wp_<id>"`: the orientation is the same as in a **previous** waypoint with the id `<id>`
- `"wp_<id> * <rot>"`: the orientation is the same as in a **previous** waypoint with the id `<id>` plus the offset `<rot>`.
  In all three cases (`quat`, `euler`, `ax_angle`), the offset is converted to a quaternion and modifies the **previous** orientation by
  ```
  quat = qmult(quat_id, quat_offset)
  ```
- `<rot> * "wp_<id>"`: the orientation is the same as in a **previous** waypoint with the id `<id>` plus the offset `<rot>`.
  In all three cases (`quat`, `euler`, `ax_angle`), the offset is converted to a quaternion and modifies the **previous** orientation by
  ```
  quat = qmult(quat_offset, quat_id)
  ```

### Details for the gripper state `grip`
The gripper state can be specified as an integer value between 0 (closed) and 255 (open) or as a string with the following options:
- `"OPEN"`: the gripper is open (255)
- `"CLOSED"`: the gripper is closed (0)

<a name="examples"></a>
## Examples
```yaml
TODO
```