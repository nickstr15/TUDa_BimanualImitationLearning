import ctypes
import sys
import os

import yaml

import time
import numpy as np
import threading

######################################################################################
# Setup PSMove #######################################################################
######################################################################################

full_config_path = os.path.join(os.path.dirname(__file__), "psmove_config.yml")
with open(full_config_path, 'r') as f:
    PSMOVE_CONFIG = yaml.safe_load(f)

PSMOVE_API_PATH = PSMOVE_CONFIG["psmove_api_path"]
LEFT_ADDRESS = PSMOVE_CONFIG["left_controller_address"]
RIGHT_ADDRESS = PSMOVE_CONFIG["right_controller_address"]

lib_path = os.path.join(PSMOVE_API_PATH, "lib")
bindings_path = os.path.join(PSMOVE_API_PATH, 'bindings', 'python')
additional_path = os.path.join(PSMOVE_API_PATH, 'additional')

if 'PSMOVEAPI_LIBRARY_PATH' not in os.environ:
    os.environ['PSMOVEAPI_LIBRARY_PATH'] = lib_path

if 'LD_LIBRARY_PATH' not in os.environ:
    os.environ['LD_LIBRARY_PATH'] = lib_path + ":" + bindings_path + ":" + additional_path
else:
    os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ":" + lib_path + ":" + bindings_path + ":" + additional_path

sys.path.insert(0, lib_path)
sys.path.insert(0, bindings_path)

import psmove
######################################################################################



