import sys
import os
import subprocess
import time
import logging
from pathlib import Path
import shutil

def run_locust_docker_compose(docker_compose_file, 
		duration, users, workers=0, quiet=False):
	# _stdout = subprocess.PIPE
	_stdout = sys.stdout
	_stderr = sys.stderr
	if quiet:
		_stdout = subprocess.DEVNULL
		_stderr = subprocess.DEVNULL

	locust_proc = None
	if workers == 0:
		workers = max(1, users // 2)
	# env variables
	cmd = 'USERS=' + str(users) + ' EXP_TIME=' + str(duration) + 's '
	cmd += 'docker-compose -f ' + str(docker_compose_file) + \
		' up --scale worker=' + str(workers)
	print(cmd)
	locust_proc = subprocess.Popen(cmd, shell=True,
		stdout=_stdout, stderr=_stderr)

	assert locust_proc != None
	return locust_proc

def run_locust(client_script, csv, nginx_ip, volumes,
		log_file, duration=10,
		users=10, quiet=False):
	# _stdout = subprocess.PIPE
	_stdout = sys.stdout
	_stderr = sys.stderr
	if quiet:
		_stdout = subprocess.DEVNULL
		_stderr = subprocess.DEVNULL

	locust_proc = None
	cmd = 'docker run --network host '
	if len(volumes) != 0:
		for vol in volumes:
			cmd += '-v '
			src = vol[0]
			targ = vol[1]
			cmd += src + ':' + targ + ' '
	cmd += 'yz2297/locust_openwhisk '	# locust docker image
	cmd += '-f ' + client_script + ' '
	cmd += '--csv ' + csv + ' '
	cmd += '--headless -t ' + str(duration) + 's '
	cmd += '--host ' + nginx_ip + ' '
	cmd += '--users ' + str(users) + ' '
	cmd += '--logfile ' + log_file
	print(cmd)
	locust_proc = subprocess.Popen(cmd, shell=True,
		stdout=_stdout, stderr=_stderr)

	assert locust_proc != None
	return locust_proc

def copy_locust_stats(locust_stats_dir, targ_dir):
	if os.path.isdir(str(targ_dir)):
		shutil.rmtree(str(targ_dir))
	shutil.copytree(str(locust_stats_dir), str(targ_dir))

def clear_locust_state(locust_stats_dir):
	for fn in os.listdir(str(locust_stats_dir)):
		full_path = locust_stats_dir / fn
		os.remove(str(full_path))

def _get_int_val(str_val):
	str_val = str_val.replace('\"', '')
	if 'N/A' in str_val:
		return 0
	else:
		return int(str_val)

def _get_float_val(str_val):
	str_val = str_val.replace('\"', '')
	if 'N/A' in str_val:
		return 0.0
	else:
		return float(str_val)

def get_locust_users(log_path):
	with open(str(log_path), 'r') as f:
		lines = f.readlines()
		assert len(lines) > 1
		pos = None
		fields = lines[0].split(',')
		for i, k in enumerate(fields):
			if 'User Count' in k:
				pos = i
				break
		assert pos != None

		data = lines[-1].split(',')
		# print(data[pos])
		try:
			users = _get_int_val(data[pos])
		except:
			users = 0
	return users

def get_locust_data(feature, log_path):
	lat = -1
	rps = -1
	failures = -1
	with open(str(log_path), 'r') as f:
		lines = f.readlines()
		assert len(lines) > 1
		fields = lines[0].split(',')

		# "Timestamp","User Count","Type","Name","Requests/s","Failures/s","50%","66%","75%","80%","90%","95%","98%","99%","99.9%","99.99%","99.999%","100%","Total Request Count","Total Failure Count"
		pos = {}
		pos['90%'] = None
		pos['95%'] = None
		pos['98%'] = None
		pos['99%'] = None
		pos['99.9%'] = None
		pos['rps'] = None
		pos['failure'] = None
		pos['fps'] = None
		for i, k in enumerate(fields):
			k = k.replace('\"', '')
			if k == '90%':
				pos['90%'] = i
			elif k == '95%':
				pos['95%'] = i
			elif k == '98%':
				pos['98%'] = i
			elif k == '99%':
				pos['99%'] = i
			elif k == '99.9%':
				pos['99.9%'] = i
			elif k == 'Requests/s':
				pos['rps'] = i
			elif k == 'Failures/s':
				pos['fps'] = i
			elif k == 'Total Failure Count':
				pos['failures'] = i

		data = lines[-1].split(',')
		try:
			feature.fps = _get_float_val(data[ pos['fps'] ])
			feature.rps = _get_float_val(data[ pos['rps'] ])
		except:
			feature.fps = 0
			feature.rps = 0

		try:
			feature.end_to_end_lat['90.0'] = _get_int_val(data[ pos['90%'] ])
			feature.end_to_end_lat['95.0'] = _get_int_val(data[ pos['95%'] ])
			feature.end_to_end_lat['98.0'] = _get_int_val(data[ pos['98%'] ])
			feature.end_to_end_lat['99.0'] = _get_int_val(data[ pos['99%'] ])
			feature.end_to_end_lat['99.9'] = _get_int_val(data[ pos['99.9%'] ])
		except:
			feature.end_to_end_lat['90.0'] = 0
			feature.end_to_end_lat['95.0'] = 0
			feature.end_to_end_lat['98.0'] = 0
			feature.end_to_end_lat['99.0'] = 0
			feature.end_to_end_lat['99.9'] = 0

		try:
			feature.failures = _get_int_val(data[ pos['failures'] ])
		except:
			feature.failures = 0

		rps = feature.rps
		failures = feature.failures
		lat = feature.end_to_end_lat['99.0']

	return lat, rps, failures