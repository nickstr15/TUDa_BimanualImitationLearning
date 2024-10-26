# Waypoint-Files

This folder contains the waypoint files for the different tasks.

#### Table of Contents
- [Structure](#structure) 
- [Examples](#examples)

<a name="structure"></a>
## Structure of the waypoint files
```yaml
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
- description: <description>         # name/description of the waypoint
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

- id: 1
  # ...
```

### Details for the `ee_target`
If the `ee_target` value is specified, the tags `pos`, `quat`, `euler`, `ax_angle`, `grip` are ignored. 
The WaypointExpert class must provide a method with the name specified in `ee_target` that returns a dictionary with the keys `pos`, `quat`, `grip`:

```python
def method_name(self):
    return {
        'pos': [0, 0, 0],
        'quat': [1, 0, 0, 0],
        'grip': 0
    }
```

### Details for the position `pos`

The position [`pos`] declaration can be a list of 3 elements `[x, y, z]` 
or a string with the following options:
- `"as_before"`: the position is the same as in the previous waypoint, can only be used from the second waypoint onwards
- `"as_before + [dx, dy, dz]"`: the position is the same as in the previous waypoint plus the offset `[dx, dy, dz]`, can only be used from the second waypoint onwards.
- `"wp_<id>"`: the position is the same as in the waypoint with the id `<id>`, where the id count starts with 0 for the first waypoint

### Details for the orientation `quat`/`euler`/`ax_angle`
The orientation can be specified in different ways:
- `quat`: a list of 4 elements `[w, x, y, z]` representing the quaternion
- `euler`: a list of 3 elements `[roll, pitch, yaw]` representing the euler angles in radians
- `ax_angle`: a list of 4 elements `[vx, vy, vz, angle]` representing the axis-angle in radians

Furthermore, the orientation can be a string with the following options:
- `"as_before"`: the orientation is the same as in the previous waypoint, can only be used from the second waypoint onwards
- `"as_before + <rot>"`: the orientation is the same as in the previous waypoint plus the offset `<offset>`, can only be used from the second waypoint onwards.
  In all three cases (`quat`, `euler`, `ax_angle`), the offset is converted to a quaternion and modifies the previous orientation by
  ```
  quat = qmult(quat_before, quat_offset)
  ```
- `<rot> + "as_before"`: the orientation is the same as in the previous waypoint plus the offset `<offset>`, can only be used from the second waypoint onwards.
  In all three cases (`quat`, `euler`, `ax_angle`), the offset is converted to a quaternion and modifies the previous orientation by
  ```
  quat = qmult(quat_offset, quat_before)
  ```
- `"wp_<id>"`: the orientation is the same as in the waypoint with the id `<id>`, where the id count starts with 0 for the first waypoint

### Details for the gripper state `grip`
The gripper state can be specified as an integer value between 0 (closed) and 255 (open) or as a string with the following options:
- `"OPEN"`: the gripper is open (255)
- `"CLOSED"`: the gripper is closed (0)

<a name="examples"></a>
## Examples
```yaml
TODO
```