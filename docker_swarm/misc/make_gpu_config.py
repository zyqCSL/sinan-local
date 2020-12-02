# assume docker version >= 1.13
import sys
import os
import argparse
import logging
from pathlib import Path
import json
import math
# from socket import SOCK_STREAM, socket, AF_INET, SOL_SOCKET, SO_REUSEADDR

# -----------------------------------------------------------------------
# parser args definition
# -----------------------------------------------------------------------
parser = argparse.ArgumentParser()
# parser.add_argument('--cpus', dest='cpus', type=int, required=True)
# parser.add_argument('--stack-name', dest='stack_name', type=str, required=True)
parser.add_argument('--gpu-config', dest='gpu_config', type=str, required=True)
# data collection parameters
# TODO: add argument parsing here

# -----------------------------------------------------------------------
# parse args
# -----------------------------------------------------------------------
args = parser.parse_args()
# todo: currently assumes all vm instances have the same #cpus
# MaxCpus = args.cpus
# StackName = args.stack_name
gpu_config_path = Path('..') / 'config' / args.gpu_config.strip()

gpu_config = {}
gpu_config['gpus'] = [0, 1]
gpu_config['ip_addr'] = 'gpu-server-ip'
gpu_config['host'] = 'tsg-gpu1.ece.cornell.edu'
gpu_config['working_dir'] = '/home/yz2297/sinan-local/ml_docker_swarm'
gpu_config['script'] = 'social_media_predictor.py'

with open(str(gpu_config_path), 'w+') as f:
	json.dump(gpu_config, f, indent=4, sort_keys=True)

