# TUDa - Bimanual Imitation Learning
Master Thesis about Bimanual Imitation Learning (Intelligent Autonomous Systems Group - TU Darmstadt)

-----

# Installation & Setup
Run
```bash
setup.sh
```

If you later want to use [`wandb`](https://wandb.ai/site/) for logging, please set the `WANDB_API_KEY` variable
to your API key by running the following command.
```bash
./config_wandb.sh <key>
```

-----

# Structure & Usage
- **robosuite_ext** extends the the [robosuite](https://robosuite.ai/) library with new bimanual environments and
additionally adds *flipped* versions for each environment.
- **robomimic_ext** extends the [robomimic](https://robomimic.github.io/) library.

Both libraries are part of the [ARISE Initiative](https://github.com/ARISE-Initiative).

---- 

## Robosuite Extension
### Different environments
- red: robosuite environments
- blue: new environments

![demo_envs](https://github.com/user-attachments/assets/54137f13-fce7-411b-a208-fe3fcc22db53)

### Support for various robots (by default)

![demo_robots](https://github.com/user-attachments/assets/db3df7d7-c4d5-4a82-9d10-4bcfa8994f03)

-----


## Robosuite Usage
### Create environment
TODO
### Setup new waypoints expert
TODO
### Run waypoint expert to collect data
TODO

-----

## Robomimic Extension
Coming soon...

---

## Robomimic Usage
### Prepare dataset
TODO
### Train policy
TODO (normal + on cluster)
### Evaluate/Run policy
TODO

-----





