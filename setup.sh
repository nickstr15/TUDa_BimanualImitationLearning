#!/bin/bash

# Create a new conda environment with Python 3.10
conda create -n bil-py310 python=3.10 -y
conda activate bil-py310
# Add conda activation to .bashrc
echo "conda activate bil-py310" >> ~/.bashrc

# Install required packages
# A) Main packages
pip install robosuite robomimic

# B) Additional packages
pip install wandb
pip3 install torch torchvision torchaudio
pip install experiment-launcher
pip install diffusers

pip install -e .

# C) Install MuJoCo (optional)
if [ "$1" == "--install-mujoco" ]; then
  echo "Installing MuJoCo..."
  cd ~ || exit
  mkdir -p .mujoco
  cd .mujoco || exit
  wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
  tar -xvzf mujoco210-linux-x86_64.tar.gz
  rm mujoco210-linux-x86_64.tar.gz
  echo "Installation done. Please add 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin' to your .bashrc file."
else
  echo "Skipping MuJoCo installation."
fi




