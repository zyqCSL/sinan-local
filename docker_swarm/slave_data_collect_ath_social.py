# Docker version 19.03.11, Ubuntu 18.04
import sys
import os
import socket
import subprocess
import threading
import time
import json
import argparse
import logging
from pathlib import Path
import copy

# slave is responsible for adjusting resources on each server
# collecting per-server information, including cpu, network & memory
# assume each vm can hold multiple containers

# -----------------------------------------------------------------------
# parser args definition
# -----------------------------------------------------------------------
parser = argparse.ArgumentParser()
# parser.add_argument('--instance-name', dest='instance_name', type=str, required=True)
parser.add_argument('--cpus', dest='cpus', type=int, required=True)
# parser.add_argument('--max-memory', dest='max_memory',type=str, required=True)	# in MB
parser.add_argument('--server-port', dest='server_port',type=int, default=40011)
parser.add_argument('--service-config', dest='service_config', type=str, required=True)
parser.add_argument('--stack-name', dest='stack_name', type=str, required=True)

# -----------------------------------------------------------------------
# parse args
# -----------------------------------------------------------------------
args = parser.parse_args()
# global variables
Cpus 	 = args.cpus
Stackname = args.stack_name
# MaxMemory 	 = args.max_memory
ServerPort   = args.server_port
MsgBuffer    = ''

Services = []
ServiceConfig = {}
# services deployed on the server
with open(args.service_config, 'r') as f:
	Services 	= json.load(f)['services']
	ServiceConfig = {}
	for s in Services:
		ServiceConfig[s] = {}
		ServiceConfig[s]['cpus'] = 0

Containers  = {}	# indexed by service name
ContainerList = []	# a list of all container names
ContainerStats = {}	# indexed by container names
ServiceReplicaUpdated = []	# services whose replica is just updated

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def clear_container_stats():
	global Containers
	global ContainerList
	global ContainerStats
	Containers = {}
	ContainerList = []
	ContainerStats = {}

def create_container_stats(service, container_name, container_id):
	global Containers
	global ContainerList
	global ContainerStats

	logging.info("Create stats for container %s of %s" %(container_name, service))
	assert service in container_name # make sure service matches container_name
	if service not in Containers:
		Containers[service] = []
	assert container_name not in Containers[service]
	assert container_name not in ContainerStats
	assert container_name not in ContainerList
	Containers[service].append(container_name)
	ContainerList.append(container_name)
	ContainerStats[container_name] = {}
	ContainerStats[container_name]['id']   = container_id
	ContainerStats[container_name]['pids'] = []
	# variables below are cummulative
	ContainerStats[container_name]['rx_packets'] = 0
	ContainerStats[container_name]['rx_bytes'] = 0
	ContainerStats[container_name]['tx_packets'] = 0
	ContainerStats[container_name]['tx_bytes'] = 0
	ContainerStats[container_name]['page_faults'] = 0
	ContainerStats[container_name]['cpu_time'] = 0
	ContainerStats[container_name]['io_bytes'] = 0
	ContainerStats[container_name]['io_serviced'] = 0

# used when previous container failed and a new one is rebooted
def reset_container_id_pids():
	logging.info('reset_container_id_pids')
	clear_container_stats()
	docker_ps()

def docker_ps():
	global Services
	global Stackname

	texts = subprocess.check_output('docker ps', shell=True, stderr=sys.stderr).decode(
			'utf-8').splitlines()
	for i in range(1, len(texts)):
		c_name = [s for s in texts[i].split(' ') if s][-1]
		if Stackname not in c_name:
			continue
		try:
			c_id = get_container_id(c_name)
		except:
			logging.warning('container %s disappears after docker ps' %c_name)
			continue
		service = ''
		sim_c_name = c_name.replace(Stackname, '')
		for s in Services:
			# choose the longest matching name
			if s in sim_c_name and len(s) > len(service):	
				service = s
		if service == '':
			logging.warning("docker ps container_name = %s, container_id = %s has no matching service" %(c_name, c_id))
			continue

		logging.info("docker ps container_name = %s, container_id = %s service = %s" %(c_name, c_id, service))
		create_container_stats(service, c_name, c_id)
		update_container_pids(c_name)

def get_container_id(container_name):
	cmd = "docker inspect --format=\"{{.Id}}\" " + container_name
	container_id = subprocess.check_output(cmd, shell=True, stderr=sys.stderr).decode(
		'utf-8').replace('\n', '')
	return str(container_id)

def update_container_pids(container_name):
	global ContainerStats
	assert container_name in ContainerStats
	cmd = "docker inspect -f \"{{ .State.Pid }}\" " + ContainerStats[container_name]['id']
	pid_strs = subprocess.check_output(cmd, shell=True, stderr=sys.stderr).decode(
		'utf-8').split('\n')
	for pid_str in pid_strs:
		if pid_str != '':
			ContainerStats[container_name]['pids'].append(pid_str)

def get_container_pids(container_name, pid_list):
	assert container_name in ContainerStats
	cmd = "docker inspect -f \"{{ .State.Pid }}\" " + ContainerStats[container_name]['id']
	pid_strs = subprocess.check_output(cmd, shell=True, stderr=sys.stderr).decode(
		'utf-8').split('\n')
	for pid_str in pid_strs:
		if pid_str != '':
			pid_list.append(pid_str)

def remove_stale_container(service, updated_containers):
	global Containers
	global ContainerList
	global ContainerStats

	if service in Containers:
		stale_containers = [c for c in Containers[service] if c not in updated_containers]
	else:
		stale_containers = []
	Containers[service] = list(updated_containers)
	for c in stale_containers:
		del ContainerStats[c]
	ContainerList = [c for c in ContainerList if c not in stale_containers]

# used when replica of a service is updated
def update_replica(updated_service_list):
	global Containers
	global ContainerList
	global ContainerStats
	global ServiceReplicaUpdated

	# todo: update Containers[service] & ContainerList & ContainerStats
	# delete records of removed containers

	new_cnames = []
	texts = subprocess.check_output('docker ps', shell=True, stderr=sys.stderr).decode(
			'utf-8').splitlines()
	updated_containers = {}	# indexed by service
	for i in range(1, len(texts)):
		c_name = [s for s in texts[i].split(' ') if s][-1]
		c_id = get_container_id(c_name)
		service = ''
		for s in Services:
			# choose the longest matching name
			if s in c_name and len(s) > len(service):	
				service = s
		if service == '':
			logging.warning("docker ps container_name = %s, container_id = %s has no matching service" %(c_name, c_id))
			continue
		if service not in updated_service_list:
			continue
		if service not in updated_containers:
			updated_containers[service] = []
		assert c_name not in updated_containers[service]
		updated_containers[service].append(c_name)
		# no need to create record for previously existing container
		if service in Containers and c_name in Containers[service]:
			continue
		if service not in ServiceReplicaUpdated:
			ServiceReplicaUpdated.append(service)
		assert c_name not in new_cnames
		new_cnames.append(c_name)
		logging.info("docker ps container_name = %s, container_id = %s service = %s" %(c_name, c_id, service))
		create_container_stats(service, c_name, c_id)

	for service in updated_containers:
		remove_stale_container(service, updated_containers[service])
	# get pids in parallel
	t_list = []
	for new_container in new_cnames:
		t = threading.Thread(target=get_container_pids, kwargs={
			'container_name': new_container,
			'pid_list': ContainerStats[new_container]['pids']
		})
		t_list.append(t)
		t.start()
	for t in t_list:
		t.join()

def compute_mean(stat_dict):
	global Containers

	return_state_dict = {}
	for service in Containers:
		s = 0
		for c in Containers[service]:
			assert c in stat_dict
			s += stat_dict[c]
		return_state_dict[service] = float(s)/len(Containers[service])

	return return_state_dict

def compute_sum(stat_dict):
	global Containers

	return_state_dict = {}
	for service in Containers:	# only count existing services
		s = 0
		for c in Containers[service]:
			assert c in stat_dict
			s += stat_dict[c]
		return_state_dict[service] = s

	return return_state_dict

def compute_max(stat_dict):
	global Containers

	return_state_dict = {}
	for service in Containers:	# only count existing services
		s = 0
		for c in Containers[service]:
			assert c in stat_dict
			s = max(s, stat_dict[c])
		return_state_dict[service] = s

	return return_state_dict

def concatenate(stat_dict):
	global Containers

	return_state_dict = {}
	for service in Containers:	# only count existing services
		return_state_dict[service] = []
		for c in Containers[service]:
			assert c in stat_dict
			return_state_dict[service].append(stat_dict[c])

	return return_state_dict

def get_replica():
	global Services
	global Containers

	replica_dict = {}
	for service in Services:
		if service not in Containers:
			replica_dict[service] = 0
		else:
			replica_dict[service] = len(Containers[service])
	return replica_dict

# Inter-|   Receive                                                |  Transmit
#  face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
#     lo:       0       0    0    0    0     0          0         0        0       0    0    0    0     0       0          0
#   eth0: 49916697477 44028473    0    0    0     0          0         0 84480565155 54746827    0    0    0     0       0          0

def get_network_usage():
	global Containers
	global ContainerList
	global ContainerStats

	rx_packets = {}
	rx_bytes   = {}
	tx_packets = {}
	tx_bytes   = {}

	ret_rx_packets = {}
	ret_rx_bytes   = {}
	ret_tx_packets = {}
	ret_tx_bytes   = {}

	while True:
		fail = False
		for container in ContainerList:
			rx_packets[container] = 0
			rx_bytes[container]   = 0
			tx_packets[container] = 0
			tx_bytes[container]   = 0

			for pid in ContainerStats[container]['pids']:
				# pseudo_file = '/proc/' + str(pid) + '/net/dev'
				pseudo_file = Path('/proc') / str(pid) / 'net' / 'dev'
				if not os.path.isfile(str(pseudo_file)):
					fail = True
					break
				with open(str(pseudo_file), 'r') as f:
					lines = f.readlines()
					for line in lines:
						if 'Inter-|   Receive' in line or 'face |bytes    packets errs' in line:
							continue
						else:
							data = line.split(' ')
							data = [d for d in data if (d != '' and '#' not in d and ":" not in d)]
							rx_packets[container] += int(data[1])
							rx_bytes[container]   += int(data[0])
							tx_packets[container] += int(data[9])
							tx_bytes[container]   += int(data[8])

			if fail:
				break
			ret_rx_packets[container] = rx_packets[container] - ContainerStats[container]['rx_packets']
			ret_rx_bytes[container]   = rx_bytes[container]	- ContainerStats[container]['rx_bytes']
			ret_tx_packets[container] = tx_packets[container] - ContainerStats[container]['tx_packets']
			ret_tx_bytes[container]   = tx_bytes[container]	- ContainerStats[container]['tx_bytes']

			if ret_rx_packets[container] < 0:
				ret_rx_packets[container] = rx_packets[container]
			if ret_rx_bytes[container] < 0:
				ret_rx_bytes[container] = rx_bytes[container]
			if ret_tx_packets[container] < 0:
				ret_tx_packets[container] = tx_packets[container]
			if ret_tx_bytes[container] < 0:
				ret_tx_bytes[container] = tx_bytes[container]

			ContainerStats[container]['rx_packets'] = rx_packets[container]
			ContainerStats[container]['rx_bytes']   = rx_bytes[container]
			ContainerStats[container]['tx_packets'] = tx_packets[container]
			ContainerStats[container]['tx_bytes']   = tx_bytes[container]

		if not fail:
			break

		else:
			reset_container_id_pids()

	# return compute_mean(ret_rx_packets), compute_mean(ret_rx_bytes), compute_mean(ret_tx_packets), compute_mean(ret_tx_bytes)
	# return compute_sum(ret_rx_packets), compute_sum(ret_rx_bytes), compute_sum(ret_tx_packets), compute_sum(ret_tx_bytes) 
	return concatenate(ret_rx_packets), concatenate(ret_rx_bytes), concatenate(ret_tx_packets), concatenate(ret_tx_bytes)

def get_memory_usage():
	global Containers
	global ContainerList
	global ContainerStats

	rss = {}	# resident set size, memory belonging to process, including heap & stack ...
	cache_memory = {}	# data stored on disk (like files) currently cached in memory
	page_faults  = {}

	for container in ContainerList:
		# pseudo_file = '/sys/fs/cgroup/memory/docker/' + str(ContainerIds[tier]) + '/memory.stat'
		pseudo_file = Path('/sys') / 'fs' / 'cgroup' / 'memory' / 'docker' / ContainerStats[container]['id'] / 'memory.stat'
		with open(str(pseudo_file), 'r') as f:
			lines = f.readlines()
			for line in lines:
				if 'total_cache' in line:
					cache_memory[container] = round(int(line.split(' ')[1])/(1024.0**2), 3)	# turn byte to mb
				elif 'total_rss' in line and 'total_rss_huge' not in line:
					rss[container] = round(int(line.split(' ')[1])/(1024.0**2), 3)
				elif 'total_pgfault' in line:
					pf = int(line.split(' ')[1])
					page_faults[container] = pf - ContainerStats[container]['page_faults']
					if page_faults[container] < 0:
						page_faults[container] = pf
					ContainerStats[container]['page_faults'] = pf

		assert rss[container] >= 0
		assert cache_memory[container] >= 0
		assert page_faults[container] >= 0

	# return compute_mean(rss), compute_mean(cache_memory), compute_mean(page_faults)
	# return compute_sum(rss), compute_sum(cache_memory), compute_sum(page_faults)
	return concatenate(rss), concatenate(cache_memory), concatenate(page_faults)

# cpu time percentages used on behalf on the container
# mpstat gets information of total cpu usage including colated workloads
def get_docker_cpu_usage():
	global Containers
	global ContainerList
	global ContainerStats

	docker_cpu_time = {}
	while True:
		fail = False
		for container in ContainerList:
			# pseudo_file = '/sys/fs/cgroup/cpuacct/docker/' + ContainerStats[container]['id'] + '/cpuacct.usage'
			pseudo_file = Path('/sys') / 'fs' / 'cgroup' / 'cpuacct' / 'docker' / ContainerStats[container]['id'] / 'cpuacct.usage'
			if not os.path.isfile(str(pseudo_file)):
				fail = True
				break
			with open(str(pseudo_file), 'r') as f:
				cum_cpu_time = int(f.readlines()[0])/1000000.0	# turn ns to ms
				docker_cpu_time[container] = max(cum_cpu_time - ContainerStats[container]['cpu_time'], 0)
				# logging.info(container + ' docker cummulative cpu time: ' + \
				# 	format(cum_cpu_time, '.1f') + ' interval cpu time: ' + \
				# 	format(docker_cpu_time[container], '.1f'))
				ContainerStats[container]['cpu_time'] = cum_cpu_time

		if not fail:
			break
		else:
			reset_container_id_pids()

	# return compute_mean(docker_cpu_time)
	# return compute_sum(docker_cpu_time) 
	return concatenate(docker_cpu_time)

def get_io_usage():
	global Containers
	global ContainerList
	global ContainerStats

	ret_io_bytes	= {}
	ret_io_serviced = {}

	for container in ContainerList:	
		# io sectors (512 bytes)
		# pseudo_file = '/sys/fs/cgroup/blkio/docker/' + str(ContainerIds[container])  + '/blkio.sectors_recursive'
		pseudo_file = Path('/sys') / 'fs' / 'cgroup' / 'blkio' / 'docker' / ContainerStats[container]['id']  / 'blkio.throttle.io_service_bytes_recursive'
		with open(str(pseudo_file), 'r') as f:
			lines = f.readlines()
			if len(lines) > 0:
				sector_num = int(lines[0].split(' ')[-1])
				ret_io_bytes[container] = sector_num - ContainerStats[container]['io_bytes']
				if ret_io_bytes[container] < 0:
					ret_io_bytes[container] = sector_num
			else:
				sector_num = 0
				ret_io_bytes[container] = 0

			ContainerStats[container]['io_bytes'] = sector_num

		# io services
		# pseudo_file = '/sys/fs/cgroup/blkio/docker/' + str(ContainerIds[container])  + '/blkio.io_serviced_recursive'
		pseudo_file = Path('/sys') / 'fs' / 'cgroup' / 'blkio' / 'docker' / ContainerStats[container]['id'] / 'blkio.throttle.io_serviced_recursive'
		with open(str(pseudo_file), 'r') as f:
			lines = f.readlines()
			for line in lines:
				if 'Total' in line:
					serv_num = int(line.split(' ')[-1])
					ret_io_serviced[container] = serv_num - ContainerStats[container]['io_serviced']
					if ret_io_serviced[container] < 0:
						ret_io_serviced[container] = serv_num
					ContainerStats[container]['io_serviced'] = serv_num

		assert container in ret_io_bytes
		assert container in ret_io_serviced

	# return compute_mean(ret_io_bytes), compute_mean(ret_io_serviced), compute_mean(ret_io_wait)
	# return compute_sum(ret_io_bytes), compute_sum(ret_io_serviced), compute_sum(ret_io_wait)
	return concatenate(ret_io_bytes), concatenate(ret_io_serviced)

# run before each experiment
# TODO: reimplement
# @service_restart: set to true if entire application is restarted
def init_data():
	global Services
	global ServiceConfig
	# reset container id pid every time, since we can't control placement with docker swarm
	reset_container_id_pids()

	# read initial values
	get_docker_cpu_usage()
	get_memory_usage()
	get_network_usage()
	get_io_usage()

# cpu cycle limit
def set_cpu_limit(cpu_config, quiet=False):
	global Services
	global ServiceConfig
	global Containers
	global Cpus
	global ServiceReplicaUpdated

	_stdout = sys.stdout
	_stderr = sys.stderr
	if quiet:
		_stdout = subprocess.DEVNULL
		_stderr = subprocess.DEVNULL

	p_list = []
	for service in Services:
		assert service in cpu_config
		assert 'cpus' in cpu_config[service]
		if ServiceConfig[service]['cpus'] == cpu_config[service]['cpus'] and \
			service not in ServiceReplicaUpdated:
			continue
		if service not in Containers:
			continue

		if cpu_config[service]['cpus'] == 0:
			per_container_cpu = Cpus
		else:
			assert cpu_config[service]['cpus'] <= Cpus
			# per_container_cpu = float(cpu_config[service]['cpus'])/len(Containers[service])
			per_container_cpu = float(cpu_config[service]['cpus'])	# cpus field here directly refers to per container cpu
		ServiceConfig[service]['cpus'] = cpu_config[service]['cpus']	
	
		for container in Containers[service]:
			cmd = 'docker update --cpus=%s %s' %(format(per_container_cpu, '.4f'), container)
			# logging.info(cmd)
			p = subprocess.Popen(cmd, shell=True, stdout=_stdout, stderr=_stderr)
			p_list.append(p)
	for p in p_list:
		p.communicate()

	ServiceReplicaUpdated = []	# clear replica update history

#--------- Resources not yet implemented -----------#
def set_freq(freq_config, quiet=False):
	pass

# physical cores
def set_core(core_config, quiet=False):
	# docker update --cpuset-cpus=29,31,33,35 gomicroserviceszipkinsample_rate_1
	pass

# return list of cores allocated for service
def allocate_core(service, core_config):
	return []
#---------------------------------------------------#

# TODO: experiment writing net fs or sending through network
def start_experiment(host_sock):
	global MsgBuffer
	global Services
	global ServiceConfig

	logging.info('experiment starts')
	prev_host_query_time = time.time()
	terminate = False

	exp_succ = True
	while True:
		data = host_sock.recv(1024).decode('utf-8')
		# print 'recv: ', data
		MsgBuffer += data

		if len(data) == 0:
			logging.error('host_sock reset during experiment')
			terminate = True
			exp_succ = False

		while '\n' in MsgBuffer:
			(cmd, rest) = MsgBuffer.split('\n', 1)
			MsgBuffer = rest

			logging.info('cmd: ' + cmd)

			if 'get_info' in cmd:
				cur_time = time.time()
				logging.info('time since last host query: ' + format(cur_time - prev_host_query_time, '.2f') + 's')
				replica_dict = get_replica()
				docker_cpu_time = get_docker_cpu_usage()
				rss, cache_memory, page_faults = get_memory_usage()
				rx_packets, rx_bytes, tx_packets, tx_bytes = get_network_usage()
				io_bytes, io_serviced = get_io_usage()

				ret_info = {}
				elapsed_time = (cur_time - prev_host_query_time)*1000	# in ms
				for service in Containers:
					ret_info[service] = {}
					ret_info[service]['replica']	 = replica_dict[service]
					# turn to virtual cpu number
					# ret_info[service]['cpu_docker']  = round(docker_cpu_time[service]/((cur_time - prev_host_query_time)*1000), 4)
					ret_info[service]['cpu_docker']  = [ round(c/elapsed_time, 4) for c in docker_cpu_time[service] ]
					ret_info[service]['rss'] 		 = rss[service]
					ret_info[service]['cache_mem']   = cache_memory[service]
					ret_info[service]['pgfault'] 	 = page_faults[service]
					ret_info[service]['rx_pkt'] 	 = rx_packets[service]
					ret_info[service]['rx_byte'] 	 = rx_bytes[service]
					ret_info[service]['tx_pkt'] 	 = tx_packets[service]
					ret_info[service]['tx_byte'] 	 = tx_bytes[service]
					ret_info[service]['io_bytes'] 	 = io_bytes[service]
					ret_info[service]['io_serv'] 	 = io_serviced[service]

				prev_host_query_time = cur_time
				ret_msg = json.dumps(ret_info) + '\n'
				host_sock.sendall(ret_msg.encode('utf-8'))

			elif 'set_rsc' in cmd:
				cpu_config = json.loads(cmd.split('----')[-1])
				set_cpu_limit(cpu_config, quiet=True)

			elif 'update_replica' in cmd:
				updated_service_list = json.loads(cmd.split('----')[-1])
				update_replica(updated_service_list)
				host_sock.sendall(('update_replica_done\n').encode('utf-8'))

			elif 'terminate_exp' in cmd:
				# host_sock.sendall('experiment_done\n')
				terminate = True

			elif len(cmd) == 0:
				continue

			else:
				logging.error('Error: undefined cmd: ' + cmd)
				exp_succ = False
				terminate = True

		if terminate:
			host_sock.sendall(('experiment_done\n').encode('utf-8'))
			return exp_succ

def main():
	global ServerPort
	global MsgBuffer
	global Services

	local_serv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	local_serv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	#---------------------------------
	# When application / server is configured for localhost or 127.0.0.1, 
	# which means accept connections only on the local machine. 
	# You need to bind with 0.0.0.0 which means listen on all available networks.
	#------------------------------------
	local_serv_sock.bind(('0.0.0.0', ServerPort))
	local_serv_sock.listen(1024)
	host_sock, addr = local_serv_sock.accept()
	host_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

	MsgBuffer = ''
	terminate = False
	while True:
		data = host_sock.recv(1024).decode('utf-8')
		if len(data) == 0:
			logging.warning('connection reset by host, exiting...')
			break

		MsgBuffer += data
		while '\n' in MsgBuffer:
			(cmd, rest) = MsgBuffer.split('\n', 1)
			MsgBuffer = rest
			logging.info('cmd = ' + cmd)

			if 'init_data' in cmd:
				init_data()
				host_sock.sendall(('init_data_done\n').encode('utf-8'))
			elif 'exp_start' in cmd:
				assert '\n' not in rest
				# docker_restart = (int(cmd.split(' ')[2]) == 1)
				stat = start_experiment(host_sock)
				if not stat:	# experiment failed
					terminate = True
					break
				if len(MsgBuffer) > 0:
					logging.info('Cmds left in MsgBuffer (after exp complete): ' + MsgBuffer)
					MsgBuffer = ''
			elif 'terminate_slave' in cmd:
				terminate = True
				break

		if terminate:
			break

if __name__ == '__main__':
	# reload_sched_states()
	main()