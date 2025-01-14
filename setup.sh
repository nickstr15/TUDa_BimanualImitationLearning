# Create a new conda environment with Python 3.10
conda create -n bil-py310 python=3.10
conda activate bil-py310
# add conda activation to .bashrc

# install required packages
# A) Main packages
pip install robosuite robomimic

# B) Additional packages
pip install wandb
pip3 install torch torchvision torchaudio
pip install experiment-launcher

# C) Install MuJoCo
cd ~ || exit
mkdir .mujoco
cd .mujoco || exit
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xvzf mujoco210-linux-x86_64.tar.gz
rm mujoco210-linux-x86_64.tar.gz




