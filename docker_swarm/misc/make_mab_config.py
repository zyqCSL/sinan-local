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
parser.add_argument('--mab-config', dest='mab_config', type=str, required=True)
# data collection parameters
# TODO: add argument parsing here

# -----------------------------------------------------------------------
# parse args
# -----------------------------------------------------------------------
args = parser.parse_args()
# todo: currently assumes all vm instances have the same #cpus
# MaxCpus = args.cpus
# StackName = args.stack_name
mab_config_path = Path('..') / 'config' / args.mab_config.strip()

# clustering = [
#     ["user-memcached"],
#     ["user-mongodb"],
#     ["media-filter"],
#     ["user-mention-service"],
#     ["user-timeline-service"],
#     ["user-timeline-redis"],
#     ["user-timeline-mongodb"],
#     ["write-home-timeline-rabbitmq"],
#     ["write-user-timeline-service"],
#     ["write-user-timeline-rabbitmq"],
#     ["social-graph-mongodb"],
#     ["post-storage-mongodb", "user-service", "social-graph-redis", 
#      "unique-id-service", "compose-post-redis", "write-home-timeline-service", 
#      "text-filter", "social-graph-service", "compose-post-service", 
#      "media-service", "post-storage-memcached", "url-shorten-service", 
#      "text-service", "home-timeline-redis", "home-timeline-service", 
#      "nginx-thrift", "post-storage-service"],
# ]

operation = {}
operation['-1'] = {}
operation['-1']['bias'] = 4.0
operation['0'] = {}
operation['0']['bias'] = 2.0
operation['1'] = {}
operation['1']['bias'] = 0.05
operation['0.25x'] = {}
operation['0.25x']['bias'] = 0.05
operation['0.5x'] = {}
operation['0.5x']['bias'] = 0.025
operation['-0.1x'] = {}
operation['-0.1x']['bias'] = 4.1

mab_config = {}
mab_config['operation'] = operation
# mab_config['clustering'] = clustering

mab_config['violation_bias'] = 0.8
mab_config['explore_rate'] = 0.01
mab_config['hold_rate'] = 0.05

with open(str(mab_config_path), 'w+') as f:
	json.dump(mab_config, f, indent=4, sort_keys=True)

