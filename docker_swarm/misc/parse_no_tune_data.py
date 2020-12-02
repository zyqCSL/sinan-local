# assume docker version >= 1.13
import sys
import os
import argparse
import logging
from pathlib import Path
import json
import numpy as np
# from socket import SOCK_STREAM, socket, AF_INET, SOL_SOCKET, SO_REUSEADDR

# -----------------------------------------------------------------------
# parser args definition
# -----------------------------------------------------------------------
parser = argparse.ArgumentParser()
# parser.add_argument('--cpus', dest='cpus', type=int, required=True)
# parser.add_argument('--stack-name', dest='stack_name', type=str, required=True)
parser.add_argument('--data-dir', dest='data_dir', type=str, required=True)
parser.add_argument('--mean-file', dest='mean_file', type=str, required=True)
parser.add_argument('--max-file', dest='max_file', type=str, required=True)
# data collection parameters
# TODO: add argument parsing here

# -----------------------------------------------------------------------
# parse args
# -----------------------------------------------------------------------
args = parser.parse_args()
# todo: currently assumes all vm instances have the same #cpus
# MaxCpus = args.cpus
# StackName = args.stack_name
data_dir = Path('.') / 'logs' / 'collected_data' / args.data_dir.strip()
mean_file = Path('.') / 'config' / args.mean_file.strip()
max_file = Path('.') / 'config' / args.max_file.strip()

services = [
    "nginx-thrift",
    "compose-post-service",
    "compose-post-redis",
    "text-service",
    "text-filter",
    "user-service",
    "user-memcached",
    "user-mongodb",
    "media-service",
    "media-filter",
    "unique-id-service",
    "url-shorten-service",
    "user-mention-service",
    "post-storage-service",
    "post-storage-memcached",
    "post-storage-mongodb",
    "user-timeline-service",
    "user-timeline-redis",
    "user-timeline-mongodb",
    "write-home-timeline-service",
    "write-home-timeline-rabbitmq",
    "write-user-timeline-service",
    "write-user-timeline-rabbitmq",
    "home-timeline-service",
    "home-timeline-redis",
    "social-graph-service",
    "social-graph-redis",
    "social-graph-mongodb"
    # "jaeger": {"replica": 1, "node": ""}
]

mean_cpu = {}
max_cpu = {}
for service in services:
    cpu_file = data_dir / ('cpu_util_' + service + '.txt')
    data = []
    with open(str(cpu_file), 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line == '' or line == '\n':
                continue
            data.append(float(line))

    mean_cpu[service] = np.mean(data)
    max_cpu[service] = np.max(data)

with open(str(mean_file), 'w+') as f:
	json.dump(mean_cpu, f)

with open(str(max_file), 'w+') as f:
    json.dump(max_cpu, f)

