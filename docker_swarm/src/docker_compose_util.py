import os
import sys
import subprocess
import docker
import logging
import threading
import time

from pathlib import Path
sys.path.append(str(Path.cwd()))
from util import ssh

def docker_compose_rm(compose_file, quiet):
	_stdout = sys.stdout
	_stderr = sys.stderr
	if quiet:
		_stdout = subprocess.DEVNULL
		_stderr = subprocess.DEVNULL

	cmd = 'docker-compose --f ' + str(compose_file) + ' down'
	subprocess.run(cmd, shell=True, stdout=_stdout,
	               stderr=_stderr)
	time.sleep(5)

def docker_compose_deploy(benchmark, benchmark_dir, compose_file, quiet=False):
	_stdout = sys.stdout
	_stderr = sys.stderr
	if quiet:
		_stdout = subprocess.DEVNULL
		_stderr = subprocess.DEVNULL

	docker_compose_rm(compose_file, quiet)
	time.sleep(5)
	cmd = 'docker-compose -f ' + str(compose_file) + ' up -d'
	subprocess.run(cmd, shell=True, stdout=_stdout,
	               stderr=_stderr)
	logging.info('30s cool down after docker-compose...')
	time.sleep(30)
	logging.info('cool down complete')

	# set up social network topoloy and post storage
	if 'social' in benchmark:
		cmd = 'python3 ' + str(benchmark_dir / 'scripts' / 'setup_social_graph_init_data_sync.py') + \
		 	' ' + str(benchmark_dir / 'datasets' / 'social-graph' / 'socfb-Reed98' / 'socfb-Reed98.mtx')
		subprocess.call(cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr, preexec_fn=os.setsid,bufsize=-1)
		# print 'setup_social_graph_init_data.py out: ', out
		logging.info('social network set up done')
		time.sleep(30)