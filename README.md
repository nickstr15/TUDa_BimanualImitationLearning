# TUDa - Bimanual Imitation Learning
Master Thesis about Bimanual Imitation Learning (Intelligent Autonomous Systems Group - TU Darmstadt)

-----

# Installation & Setup
Run
```bash
setup.sh --install-mujoco
```
to set up the environment. If you already have a working mujoco installation, running
```bash
setup.sh
```
is sufficient.

If you later want to use [`wandb`](https://wandb.ai/site/) for logging, please set the `WANDB_API_KEY` variable
to your API key by running the following command.
```bash
./config_wandb.sh <key>
```

-----

# Structure & Usage
- **robosuite_ext** extends the [robosuite](https://robosuite.ai/) library with new bimanual environments and
additionally adds *flipped* versions for each environment.
- **robomimic_ext** extends the [robomimic](https://robomimic.github.io/) library.

Both libraries are part of the [ARISE Initiative](https://github.com/ARISE-Initiative).

---- 

## Robosuite Extension
The robosuite extension adds new bimanual environments to the [robosuite](https://robosuite.ai/) library.
Additionally, custom waypoint-based experts are implemented to collect demonstrations for imitation learning.
### Different environments
- red: robosuite environments
- blue: new environments

![demo_envs](https://github.com/user-attachments/assets/54137f13-fce7-411b-a208-fe3fcc22db53)

### Support for various robots (by default)

![demo_robots](https://github.com/user-attachments/assets/db3df7d7-c4d5-4a82-9d10-4bcfa8994f03)

-----


## Robosuite Usage
### Create environment
1. A new environment has to implement the 
[`TwoArmEnv`](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/environments/manipulation/two_arm_env.py) of the `robosuite` package.
2. Place the class into a new file in the [`robosuite_ext/environments/manipulation`](robosuite_ext/environments/manipulation)
folder. There you can also find the existing environments and use them as a reference.
3. Optionally you can implement the *flipped* version of the environment by adding a new class to
[`robosuite_ext/environments/manipulation/flipped.py`](robosuite_ext/environments/manipulation/flipped.py).
4. Register the new environment(s) (default and flipped) in the 
[`robosuite_ext/environments/__init__.py`](robosuite_ext/environments/__init__.py) file.

### Setup new waypoints expert
1. A new waypoint expert must implement the 
[`TwoArmWaypointExpertBase`](robosuite_ext/demonstration/waypoints/core/waypoint_expert.py) interface.
Please refer to the already existing experts in the 
[`robosuite_ext/demonstration/waypoints`](robosuite_ext/demonstration/waypoints) folder.
2. For each expert a waypoint configuration must be created. These `yaml`-files should be placed in the
[`robosuite_ext/demonstration/waypoints/files`](robosuite_ext/demonstration/waypoints/files) folder.

A detailed explanation of the waypoint expert can be found in 
[`robosuite_ext/demonstration/waypoints/files/README.md`](robosuite_ext/demonstration/waypoints/files/README.md).

### Run waypoint expert to collect data
To run an expert and collect data, use the following command:
```bash
python -m robosuite_ext.scripts.collect_waypoint_demonstrations --env <env_name> --waypoints <file.yaml>
```
For further information on possible settings, please refer to the help message of the script:
```bash
python -m robosuite_ext.scripts.collect_waypoint_demonstrations
```

The collected data will be stored in the `data` folder. Each trajectory is stored as a set of environment joint values 
(which can be used to recreate the environment state) and a set of the corresponding actions.

#### Play back collected data
To play back the collected data, use the following command:
```bash
python -m robosuite_ext.scripts.playback_from_hdf5 -f <folder>
```
with `<folder>` being the folder containing the created `demo.hdf5` file.

-----

## Robomimic Extension
The robomimic extension brings [robomimic](https://robomimic.github.io/) library to work with the new bimanual 
environments. It also adds new imitation learning algorithms.

---

## Robomimic Usage
### Prepare dataset
The collected data from the waypoint expert can be used to create a dataset for imitation learning.
Just run
```bash
python -m robomimic_ext.scripts.dataset_states_to_obs.py -i <input.hdf5> -o <output.hdf5>
```
This will create a new dataset with the observations and actions from the collected data. Please refer to `--help` or 
the script ([`robomimic_ext/scripts/dataset_states_to_obs.py`](robomimic_ext/scripts/dataset_states_to_obs.py)) 
for further information on possible settings.

Alternatively, you can download existing datasets by running
```bash
python -m robomimic_ext.scripts.download_datasets --dataset <dataset_name>
```
Pass the `--all` flag to download all available datasets.

### Train policy
To run training of a policy run
```bash
python -m robomimic_ext.scripts.train --config <config.json> 
```
with `<config.json>` being the configuration file in the 
[`robomimic_ext/exp_configs`](robomimic_ext/exp_configs) folder. If you would like to run a quick debug test, you can 
pass the `--debug` flag.

To run the training on the IAS cluster, use the following command:
```bash
python -m robomimic_ext.scripts.cluster.launch_train_ias.py --config <config.json> -m <memory_single_job>
```
The `memory_single_job` parameter specifies the memory in MB that should be allocated for a single job 
(defaults to 2000).

### Evaluate/Run policy
To evaluate a trained policy, run
```bash
python -m robomimic_ext.scripts.run_trained_agent --agent <path/to/agent.pth>
```
For all available settings, please refer to the `--help` flag of the script.

-----





