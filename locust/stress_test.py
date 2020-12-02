# test
# python3 ./profile_function.py --min-users 1 --max-users 2 --user-step 1 --exp-time 60s --profile-users 1 --profile-time 60s --warmup-time 30s --function mobilenet 
# python3 ./profile_function.py --min-users 5 --max-users 30 --user-step 5 --profile-users 10 --function mobilenet

# assume docker version >= 1.13
import sys
import os
import time
import numpy as np
import json
import math
import random
import argparse
import logging
import subprocess
from pathlib import Path
import copy
import shutil
import csv

from pathlib import Path
sys.path.append(str(Path.cwd() / 'util'))
from cpu_util import *

# from socket import SOCK_STREAM, socket, AF_INET, SOL_SOCKET, SO_REUSEADDR

random.seed(time.time())
# -----------------------------------------------------------------------
# miscs
# -----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
					format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser()
# parser.add_argument('--cpus', dest='cpus', type=int, required=True)
# parser.add_argument('--stack-name', dest='stack_name', type=str, required=True)
parser.add_argument('--min-users', dest='min_users', type=int, required=True)
parser.add_argument('--max-users', dest='max_users', type=int, required=True)
parser.add_argument('--user-step', dest='user_step', type=int, required=True)
parser.add_argument('--exp-time', dest='exp_time', type=str, default='20m')
parser.add_argument('--warmup-time', dest='warmup_time', type=str, default='1m')
args = parser.parse_args()

min_users = args.min_users
max_users = args.max_users
user_step = args.user_step
exp_time = args.exp_time
warmup_time = args.warmup_time

data_dir = Path.cwd() / 'stress_test_data'
locust_stats_dir = Path.home() / 'sinan_locust_log'

if not os.path.isdir(str(data_dir)):
	os.makedirs(str(data_dir))

script = Path.cwd() / 'scripts' / ('test_social_rps_1.sh')
assert os.path.isfile(str(script))

tested_users = range(min_users, max_users+1, user_step)
print('users')
print(tested_users)

def change_time(time_str):
	if 'm' in time_str:
		return int(time_str.replace('m', '')) * 60
	elif 's' in time_str:
		return int(time_str.replace('s', ''))
	else:
		return int(time_str)

def run_exp(test_time, user, quiet=False):
	cmd = str(script) + ' ' + str(test_time) + ' ' + str(user)
	if not quiet:
		p = subprocess.Popen(cmd, shell=True)
	else:
		p = subprocess.Popen(cmd, shell=True, 
			stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	return p

def copy_locust_stats(dir_name):
	full_path = data_dir / dir_name
	if os.path.isdir(str(full_path)):
		shutil.rmtree(str(full_path))
	shutil.copytree(str(locust_stats_dir), str(full_path))

def clear_locust_state():
	for fn in os.listdir(str(locust_stats_dir)):
		full_path = locust_stats_dir / fn
		os.remove(str(full_path))

clear_locust_state()
time.sleep(10)
# stress test
for u in tested_users:
	# warumup
	p = run_exp(test_time=warmup_time, user=u)
	p.wait()
	time.sleep(30)
	# time.sleep(10)
	# real exp

	p = run_exp(test_time=exp_time, user=u)
	p.wait()

	dir_name = 'locust_social_user_' + str(u)
	copy_locust_stats(dir_name)
	clear_locust_state()
	time.sleep(20)

