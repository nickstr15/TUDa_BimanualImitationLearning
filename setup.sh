
# Create a new conda environment with Python 3.10
conda create -n bil-py310 python=3.10
conda activate bil-py310
# add conda activation to .bashrc
echo "conda activate bil-py310" >> ~/.bashrc

# install required packages
# A) Main packages
pip install robosuite robomimic

# B) Additional packages
pip install wandb
pip3 install torch torchvision torchaudio
pip install experiment-launcher



