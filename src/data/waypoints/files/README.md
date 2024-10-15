# Waypoint-Files

The structure of the waypoint files is as follows:

```
initial_configuration:
    ...
    
waypoints:
    ...
```

The `initial_configuration` key contains 
the initial configuration of the environment that is passed to 
`env.setup()` after `env.reset()` is called. The structure is as follows:

```
initial_configuration:
    seed: <seed>              # seed for the environment (default: 0)
    bodies:                   # if empty, the bodies are not moved
        - name: <name>        # name of the target body
          pos: [x, y, z]      # desired position
          quat: [w, x, y, z]  # desired orientation
        - name: <name>
          pos: [x, y, z]
          rot: [w, x, y, z]
        ...
    
```

The `waypoints` key contains the waypoints that the agent must visit. The structure is as follows:

```
waypoints:
  - id: 0                                 # id of the waypoint
    name: <name>                          # name/description of the waypoint
    targets:
      - device: <device_name>             # mujoco name of the device
        position: [x, y, z]               # desired position
        orientation: [x, y, z]            # desired orientation
        gripper: <gripper_state>          # desired gripper state (0: closed, 255: open)
        tolerance_position: <tol>         # position tolerance (default: 0.01)
        tolerance_orientation: <tol>      # orientation tolerance in degrees (default: 2°)
      - device: <device_name>
        ...
    min_duration: <time>                  # minimum time in seconds before moving to the next waypoint (default: 1.0)
    max_duration: 10.0                    # maximum time in seconds before moving to the next waypoint (default: 10.0)
  - id: 1
    ...
```