# Demonstration Data

This folder contains demonstration data as .hdf5-files.
Similar to [`robosuite`](https://github.com/ARISE-Initiative/robosuite) 
simulation states and not observations are stored. This allows for a more flexible usage.

The data is stored in the following format:

```html
data (group)
    date (attribute) - date of collection
    time (attribute) - time of collection
    env (attribute) - environment name on which demos were collected

    demo1 (group) - every demonstration has a group
        model_file (attribute) - model xml string for demonstration
        states (dataset) - flattened mujoco states
        actions (dataset) - actions applied during demonstration

    demo2 (group)
        ...
```