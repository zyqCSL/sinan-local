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

client = docker.DockerClient(base_url='unix://var/run/docker.sock')
api_client = docker.APIClient(base_url='unix://var/run/docker.sock')

# for replication state management
class ReplicaState:
	def __init__(self, replica):
		self._lock = threading.Lock()
		self._replica = replica
		self._next_replica = -1
		self._in_transit = False
		self._thread = None	# the thread that is doing docker scale
		# if the slave is informed of the replica change
		self._slave_informed = True 

	def reset(self):
		with self._lock:
			t = self._thread
		if t != None:
			t.join()
		with self._lock:
			self._in_transit = False
			self._thread = None	# the thread that is doing docker scale
			# if the slave is informed of the replica change
			self._slave_informed = True 

	def get_replica(self):
		r = 0 
		next_r = 0
		with self._lock:
			r = self._replica
			next_r = self._replica
		return r, next_r

	def is_in_transit(self):
		t = False
		with self._lock:
			t = self._in_transit
			if self._in_transit:
				assert not self._slave_informed
		return t

	def slave_need_informed(self):
		i = True
		with self._lock:
			# transit must be complete
			i = not self._slave_informed and not self._in_transit
		return i

	def set_slave_informed(self):
		with self._lock:
			assert not self._slave_informed
			assert not self._in_transit
			self._slave_informed = True

	def set_in_transit(self, next_replica):
		with self._lock:
			assert not self._in_transit
			assert self._slave_informed
			self._in_transit = True
			self._next_replica = next_replica
			self._slave_informed = False

	def set_thread(self, thread):
		with self._lock:
			# don't check here since the thread handling scale up might not finish
			# assert self._in_transit
			self._thread = thread

	def join_thread(self):
		with self._lock:
			if self._thread != None:
				self._thread.join()

	def unset_thread(self):
		with self._lock:
			assert not self._in_transit
			self._thread = None

	def update(self, replica):
		with self._lock:
			# print('replica state update r = %d' %replica, flush=True)
			assert self._in_transit
			assert not self._slave_informed
			self._replica = replica
			self._next_replica = -1
			self._in_transit = False

# called on master node
def setup_swarm(username, worker_nodes, quiet=False):
	_stdout = sys.stdout
	_stderr = sys.stderr
	if quiet:
		_stdout = subprocess.DEVNULL
		_stderr = subprocess.DEVNULL

	cmd = 'docker swarm init'
	subprocess.run(cmd, shell=True, stdout=_stdout)

	cmd = 'docker swarm join-token worker'
	worker_join_cmd = subprocess.check_output(cmd, shell=True, stderr=_stderr).decode(
	    'utf-8').strip().splitlines()[2].lstrip()

	for worker in worker_nodes:
		# cmd = 'ssh -q -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no ' + username + '@' + \
		# worker_nodes + ' \"' + worker_join_cmd + '\"'
		# subprocess.run(cmd, shell=True, stdout=_stdout, stderr=_stderr)
		p = ssh(username, host=worker, cmd=worker_join_cmd, quiet=quiet)
		p.wait()

def update_node_label(node, label, quiet=False):
	_stdout = sys.stdout
	_stderr = sys.stderr
	if quiet:
		_stdout = subprocess.DEVNULL
		_stderr = subprocess.DEVNULL

	cmd = 'docker node update --label-add ' + label + ' ' + node
	subprocess.run(cmd, shell=True, stdout=_stdout, stderr=_stderr)
	    
def docker_stack_rm(stack_name):
	docker_stack_rm = subprocess.Popen(
		['docker', 'stack', 'rm', stack_name],
		universal_newlines=True,
		stdout=subprocess.PIPE,
	)
	docker_stack_rm.wait()
	# outs, errs = docker_stack_deploy.communicate()
	# print(outs, errs)

	# docker stack ps social-network-swarm -q
	rm_finish = False
	while not rm_finish:
		docker_stack_ps = subprocess.Popen(
			['docker', 'stack', 'ps', stack_name, '-q'],
			universal_newlines=True,
			stdout=subprocess.PIPE,
		)
		outs, errs = docker_stack_ps.communicate()
		if not outs:
			rm_finish = True
		else:
			time.sleep(5)

def docker_stack_deploy(stack_name, benchmark, benchmark_dir, compose_file, quiet=False):
	global client
	_stdout = sys.stdout
	_stderr = sys.stderr
	if quiet:
		_stdout = subprocess.DEVNULL
		_stderr = subprocess.DEVNULL

	docker_stack_rm(stack_name)
	time.sleep(5)
	cmd = 'docker stack deploy --compose-file ' + str(compose_file) \
	    + ' ' + stack_name
	subprocess.run(cmd, shell=True, stdout=_stdout,
	               stderr=_stderr)

	# wait for services to converge
	logging.info('wait for services to converge')
	converged = False
	waits = 0
	while converged is not True:
		for service in client.services.list():
			cmd = 'docker service ls --format \'{{.Replicas}}\' --filter \'id=' + \
				service.id + '\''
			out = subprocess.check_output(
				cmd, stderr=subprocess.STDOUT, shell=True, universal_newlines=True).strip()
			print('service.id %s: %s' %(service.id, out))
			# remove "(max x per node)"
			if '(' in out:
				out = out.split('(')[0]
			actual = int(out.split('/')[0])
			desired = int(out.split('/')[1])
			converged = actual == desired
			if not converged:
				break
		time.sleep(5)
		waits += 1
		if waits > 30:
			logging.info('services failed to converge')
			return False
	logging.info('services converged')

	# set up social network topoloy and post storage
	if 'social' in benchmark:
		cmd = 'python3 ' + str(benchmark_dir / 'scripts' / 'setup_social_graph_init_data_sync.py') + \
		 	' ' + str(benchmark_dir / 'datasets' / 'social-graph' / 'socfb-Reed98' / 'socfb-Reed98.mtx')
		subprocess.call(cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr, preexec_fn=os.setsid,bufsize=-1)
		# print 'setup_social_graph_init_data.py out: ', out
		logging.info('social network set up done')
		time.sleep(30)
	
	return True

def docker_check_replica(stack_name, service):
	service_name = stack_name + '_' + service
	cmd = 'docker service ls --format \'{{.Replicas}}\' --filter \'name=' + \
		service_name + '\''
	out = subprocess.check_output(
		cmd, stderr=subprocess.STDOUT, shell=True, universal_newlines=True).strip()
	# remove "(max x per node)"
	if '(' in out:
		out = out.split('(')[0]
	actual = int(out.split('/')[0])
	desired = int(out.split('/')[1])
	return actual, desired

def docker_service_scale(stack_name, service, replica):
	service_name = stack_name + '_' + service
	cmd = 'docker service scale ' + service_name + '=' + str(replica)
	start = time.time()
	out = subprocess.check_output(
			cmd, stderr=subprocess.STDOUT, shell=True, universal_newlines=True)
	logging.info('docker scale %s out: %s, time: %s' %(
		service, out, round(time.time() - start, 2) ))

	# print('docker scale %s out: %s, time: %s' %(
	# 	service, out, round(time.time() - start, 2) ), flush=True)
