# needed for automatic logging to wandb
try:
    from .wandb_private import WANDB_API_KEY
except ImportError:
    print("No WANDB_API_KEY found in wandb_private.py. If you want to use wandb, please run" + \
          " 'bash utils/config_wandb.sh <key>' to set the key.")
    WANDB_API_KEY = None