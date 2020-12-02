# test the performance impact of docker service scale
# assume docker version >= 1.13
import sys
import os
import subprocess
import multiprocessing
import threading
import time
import numpy as np
import json
import math
import random
import socket
import argparse
import logging
from pathlib import Path
import copy

from pathlib import Path
sys.path.append(str(Path.cwd() / 'src'))
from util import *
from wrk2_util import *
from master_slave_msg import *
from docker_swarm_util import *

DockerMetrics = [
		'cpu_usage', # cpu cpu usage from docker, in terms of virtual cpu
		'rss', 'cache_mem', 'page_faults', # memory rss: resident set size
		'rx_packets', 'rx_bytes', 'tx_packets', 'tx_bytes', # network
		'io_serviced', 'io_bytes' # disk io (disk io monitors can't be found on gce, might need change nn arch later)
		]

# from socket import SOCK_STREAM, socket, AF_INET, SOL_SOCKET, SO_REUSEADDR

random.seed(time.time())
# -----------------------------------------------------------------------
# miscs
# -----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
					filename=str(Path.cwd() / 'logs' / 'debug_log.txt'),
					filemode='w+',
                    format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# -----------------------------------------------------------------------
# parser args definition
# -----------------------------------------------------------------------
parser = argparse.ArgumentParser()
# parser.add_argument('--cpus', dest='cpus', type=int, required=True)
# parser.add_argument('--stack-name', dest='stack_name', type=str, required=True)
parser.add_argument('--user-name', dest='user_name', type=str, default='yz2297')
parser.add_argument('--setup_swarm', dest='setup_swarm', action='store_true')
parser.add_argument('--deploy', dest='deploy', action='store_true')
parser.add_argument('--stack-name', dest='stack_name', type=str, required=True)
parser.add_argument('--benchmark', dest='benchmark', type=str, default='socialNetwork-ml-swarm')
parser.add_argument('--compose-file', dest='compose_file', type=str, default='docker-compose-swarm.yml')
parser.add_argument('--min-rps', dest='min_rps', type=int, required=True)
parser.add_argument('--max-rps', dest='max_rps', type=int, required=True)
parser.add_argument('--rps-step', dest='rps_step', type=int, required=True)
parser.add_argument('--exp-time', dest='exp_time', type=int, required=True)
parser.add_argument('--measure-interval', dest='measure_interval', type=int, default=1)
parser.add_argument('--slave-port', dest='slave_port', type=int, required=True)
parser.add_argument('--deploy-config', dest='deploy_config', type=str, required=True)
# data collection parameters
parser.add_argument('--qos', dest='qos', type=int, default=300)
# max latency in roi (region of interest)
# after consecutive violations (of roi_latency) of viol_recover_delay cycles, starts to propose recover actions
parser.add_argument('--viol-timeout', dest='viol_timeout', type=int, default=30)
parser.add_argument('--hold-delay', dest='hold_delay', type=int, default=3,
					help='#cycles to hold after each violation')
# scale out/in
parser.add_argument('--scale-inertia', dest='scale_inertia', type=int, default=20)	# time to stop scaling in after each scale in is down

# -----------------------------------------------------------------------
# parse args
# -----------------------------------------------------------------------
args = parser.parse_args()
# todo: currently assumes all vm instances have the same #cpus
# MaxCpus = args.cpus
Username = args.user_name
Deploy = args.deploy
SetupSwarm = args.setup_swarm
Stackname = args.stack_name
Benchmark = args.benchmark
BenchmarkDir =  Path.cwd() / '..' / 'benchmarks' / args.benchmark
ComposeFile = BenchmarkDir / args.compose_file
MinRps = args.min_rps
MaxRps = args.max_rps
RpsStep = args.rps_step
ExpTime = args.exp_time	# in second
MeasureInterval = args.measure_interval	# in second
SlavePort = args.slave_port
DeployConfig = Path.cwd() / 'config' / args.deploy_config.strip()

# wrk2 = Path.home() / 'wrk2_archive' / 'change_load_wrk2_general' / 'wrk2_periodic_stats_sample_full_percentile' / 'wrk'
wrk2 = BenchmarkDir / 'wrk2' / 'wrk'
# Wrk2pt = Path('/filer-01') / 'yz2297' / 'wrk2_log' / 'pt.txt'
Wrk2pt = Path.cwd() / 'wrk2_log' / 'pt.txt'
SchedStateFile = Path.cwd() / 'logs' / 'sched_states.txt'
StateSummary = Path.cwd() / 'logs' / 'social_state_summary.txt'
DataDir =  Path.cwd() / 'logs' / 'test_docker_service_scale'

Wrk2LastTime		 = -1

if not os.path.isdir(str(DataDir)):
	os.makedirs(str(DataDir))

# -----------------------------------------------------------------------
# service & server configuration
# -----------------------------------------------------------------------
Servers = {}
HostServer = ''
with open('/proc/sys/kernel/hostname', 'r') as f:
	HostServer = f.read().replace('\n', '')
SlaveSocks = {}

ReplicaCpus = 0
Services = []
TestedServices = {}
ScalableServices = []
ServiceConfig = {}
ServiceReplicaStates = {} # states for controlling scaling out/in
ServiceInitConfig = {} # inital configuration of services
ServiceReplicaStates = {}
with open(str(DeployConfig), 'r') as f:
	config_info = json.load(f)
	ReplicaCpus = config_info['replica_cpus']
	Servers = config_info['nodes']
	for node in Servers:
		assert 'ip_addr' in Servers[node]
		assert 'cpus' in Servers[node]
		# assert 'label' in Servers[node]	# docker swarm tag of the node
	ServiceConfig = config_info['service']
	ScalableServices = config_info['scalable_service']
	Services = list(ServiceConfig.keys())
	for service in Services:
		# cpu cycle limit
		if 'cpus' not in ServiceConfig[service]:
			ServiceConfig[service]['cpus'] = 0
		if 'replica' not in ServiceConfig[service]:
			ServiceReplicaStates[service] = ReplicaState(1)
		else:
			# ServiceConfig[service]['replica'] only used for initialization
			ServiceReplicaStates[service] = ReplicaState(ServiceConfig[service]['replica'])
		if 'replica_cpus' not in ServiceConfig[service]:
			# cpu limit assigned to each replica
			ServiceConfig[service]['replica_cpus'] = ServiceConfig[service]['cpus']/ServiceConfig[service]['replica']
			assert ServiceConfig[service]['replica_cpus'] <= ReplicaCpus
		assert 'max_replica' in ServiceConfig[service]
		assert 'max_cpus' in ServiceConfig[service]

	ServiceInitConfig = copy.deepcopy(ServiceConfig)

for service in ScalableServices:
	if ServiceConfig[service]['max_replica'] > 1:
		TestedServices[service] = ServiceConfig[service]['max_replica']


# ------------------------------------------------------
# Scaling out/in control
# ------------------------------------------------------
ScaleIneratia = args.scale_inertia
ScaleInertiaCycles = 0

# -----------------------------------------------------------------------
# scheduling state parameters
# -----------------------------------------------------------------------
# qos
EndToEndQos	= args.qos		# ms to us

# -----------------------------------------------------------------------
# data collection
# -----------------------------------------------------------------------
TestRps  	 = range(MinRps, MaxRps+1, RpsStep)
StartTime			 = -1		# the time that the script started
ViolTimeout	 = args.viol_timeout	    # #cycles of continuous violations after which the application is unlikely to recover even at max rsc
Wrk2LastTime		 = -1	# last time wrk2 pt is written

# data features
StateLog			= []

# TODO: implement init & init_data
def init_data():
	global StateLog
	global ServiceConfig
	global ServiceInitConfig
	global Servers
	global SlaveSocks

	StateLog = []
	ServiceConfig = copy.deepcopy(ServiceInitConfig)
	send_init_data(servers=Servers, slave_socks=SlaveSocks)

class Action:
	def __init__(self):
		self.vic_service = []
		self.cpu = {}	# cpu cycle limit
		self.core = {}	# physical core allocation
		self.type = 'global'

	def show(self):
		global Services
		report = 'vic_service: ' + ', '.join(self.vic_service)
		for service in Services:
			report += ' ' + service + ': %d;' %(self.cpu[service])
		return report

	def derive_per_replica(self, actual_replica):
		global ServiceReplicaStates
		per_replica_action = Action()
		per_replica_action.type = 'replica'
		per_replica_action.vic_service = list(self.vic_service)
		for service in self.cpu:
			r, next_r = ServiceReplicaStates[service].get_replica()
			actual_r = actual_replica[service]
			if r != next_r or r != actual_r:
				logging.warning('Service %s sched_replica=%d, next_replica=%d, actual_replica=%d' %(
					service, r, next_r, actual_r))
			# enough cpu capacity for scaling down
			if next_r < 0:
				r = min(r, actual_r)
			else:
				r = min(min(r, next_r), actual_r)
			per_replica_action.cpu[service] = ceil_float(num=self.cpu[service]/r,
				precision=1)
			logging.info('derive_per_replica service %+20s total=%d, per_replica=%f' %(service, self.cpu[service], per_replica_action.cpu[service]))
			per_replica_action.core[service] = math.ceil(per_replica_action.cpu[service])
		return per_replica_action

class Feature:
	def __init__(self, time):
		global DockerMetrics

		self.time = time

		self.replica   = {}	# replica reported from slaves
		self.sched_replica = {} # replica anticipated by scheduler

		# rsc
		self.cpu_limit = {} # cpu cycle limit of all replicas
		self.replica_cpu_limit = {}	# cpu cycle limit of each replica

		# lat info
		self.end_to_end_lat = {}
		self.xput  			= 0

		self.docker_metrics = {}
		for m in DockerMetrics:
			self.docker_metrics[m] = {}	# indexed by server

	def total_cpu_usage(self, service):
		assert service in self.docker_metrics['cpu_usage']
		return np.sum(self.docker_metrics['cpu_usage'][service])

	def total_cpu_util(self, service):
		assert service in self.cpu_limit
		return self.total_cpu_usage(service)/self.cpu_limit[service]

	def show_docker_metric(self):
		global Services
		global DockerMetrics

		line = ''
		for service in Services:
			line += service + '-replica: ' + str(self.replica[service]) + ';\t'
			line += service + '-sched_replica: ' + str(self.sched_replica[service]) + ';\n'

		for metric in DockerMetrics:
			for service in Services:
				if metric == 'cpu_usage':
					line += service + '-' + metric + ': ' + \
						', '.join([format(s, '.1f') for s in self.docker_metrics[metric][service]]) + ';\n'
				else:
					line += service + '-' + metric + ': ' + \
						', '.join([str(s) for s in self.docker_metrics[metric][service]]) + ';\n'

		return line

def get_metric(feature):
	global Servers
	global ServiceConfig
	global ServiceReplicaStates
	global SlaveSocks
	global ReplicaCpus
	global DockerMetrics

	stats_accum = get_slave_metric(Servers, SlaveSocks)
	for service in ServiceConfig:
		if service == 'jaeger' or service == 'zipkin':
			continue
		if service not in stats_accum:
			logging.error("get_metric: service %s not in stats_accum" %service)
		assert service in stats_accum

		sched_replica, _ = ServiceReplicaStates[service].get_replica()
		if sched_replica != stats_accum[service]['replica']:
			logging.warning("get_metric: service %s replica disagree master=%d, slave=%d" %(service,
				sched_replica,
				stats_accum[service]['replica']))

		feature.sched_replica[service] = sched_replica
		feature.replica[service] = stats_accum[service]['replica']
		for metric in DockerMetrics:
			feature.docker_metrics[metric][service] = np.array(stats_accum[service][metric])

# -----------------------------------------------------------------------
# observed state
# -----------------------------------------------------------------------
class State:
	def __init__(self):
		self.feature = None
		self.action = None
		self.instant_observation = None
		self.delayed_observation = []
		self.rewards = None

	def update(self, instant_obs):
		self.instant_observation = instant_obs

def get_mean_tail():
	global StateLog
	global ScaleIneratia
	l = ScaleIneratia // 2

	if len(StateLog) == 0:
		return 1.0
	else:
		if len(StateLog) < l:
			s = 0
			for state in StateLog:
				s += state.feature.end_to_end_lat[99.0]
			return s/len(StateLog)
		else:
			s = 0
			for i in range(-l, 0, 1):
				s += StateLog[i].feature.end_to_end_lat[99.0]
			return s/l

def propose_replica(vic_service):
	global ServiceReplicaStates
	global ScalableServices
	global TestedServices
	global EndToEndQos

	tail_lat = get_mean_tail()

	replica_proposal = {}
	if vic_service != '':
		assert vic_service in TestedServices
		assert not ServiceReplicaStates[vic_service].is_in_transit()

		r, _ = ServiceReplicaStates[vic_service].get_replica()
		if tail_lat <= EndToEndQos:
			next_r = max(r-1, 1)
			if next_r != r:
				replica_proposal[vic_service] = next_r
		else:
			next_r = min(r+1, ServiceConfig[vic_service]['max_replica'])
			if r != next_r:
				replica_proposal[vic_service] = next_r
		return replica_proposal

	else:
		if tail_lat <= EndToEndQos:
			for s in TestedServices:
				if ServiceReplicaStates[s].is_in_transit():
					continue
				r, _ = ServiceReplicaStates[s].get_replica()
				next_r = max(r-1, 1)
				if r != next_r:
					replica_proposal[s] = next_r
				
		else:
			for s in TestedServices:
				if ServiceReplicaStates[s].is_in_transit():
					continue
				r, _ = ServiceReplicaStates[s].get_replica()
				next_r = min(r+1, ServiceConfig[s]['max_replica'])
				if next_r != r:
					replica_proposal[s] = next_r
		return replica_proposal


def do_service_docker_scale(stack_name, service, next_r, replica_state):
	# print('scale service %s=%d' %(service, next_r), flush=True)
	replica_state.set_in_transit(next_r)
	docker_service_scale(stack_name=stack_name, 
		service=service, replica=next_r)
	replica_state.update(next_r)

def do_docker_scale(replica_proposal):
	global ServiceReplicaStates
	global Stackname

	assert len(replica_proposal) != 0
	# print(replica_proposal)
	for service in replica_proposal:
		# do_service_docker_scale(Stackname, service, replica_proposal[service], ServiceReplicaStates[service])
		t = threading.Thread(target=do_service_docker_scale, kwargs={
			'stack_name': Stackname,
			'service': service,
			'next_r': replica_proposal[service],
			'replica_state': ServiceReplicaStates[service]
		})
		ServiceReplicaStates[service].set_thread(t)
		t.start()
		# t.join()

def do_docker_scale_init():
	global ServiceInitConfig
	global ServiceReplicaStates
	global Stackname

	time.sleep(15)

	for service in ServiceInitConfig:
		if service == 'jaeger' or service == 'zipkin':
			continue
		actual, desired = docker_check_replica(stack_name=Stackname,
			service=service)
		init_r = ServiceInitConfig[service]['replica']
		ServiceReplicaStates[service].reset()
		if actual != desired or actual != init_r:
			logging.warning('Service %s replica actual=%d, desired=%d, required_init=%d' %(service,
				actual, desired, init_r))
		if init_r != actual or init_r != desired:
			do_service_docker_scale(
				stack_name=Stackname, service=service, next_r=init_r, 
				replica_state=ServiceReplicaStates[service])

def reset_docker_scale_clock():
	global ScaleIneratia
	global ScaleInertiaCycles
	ScaleInertiaCycles = ScaleIneratia

def trigger_docker_scale(service):
	global ScaleInertiaCycles
	global ServiceReplicaStates

	if service != '':
		return ScaleInertiaCycles == 0 and not ServiceReplicaStates[service].is_in_transit()
	else:
		return ScaleInertiaCycles == 0

def inform_slaves_new_replica():
	global ServiceReplicaStates
	global Servers
	global SlaveSocks

	services_need_informed = []
	for s in ServiceReplicaStates:
		if ServiceReplicaStates[s].slave_need_informed():
			logging.info("service %s needs informed of replica update" %s)
			services_need_informed.append(s)
			ServiceReplicaStates[s].join_thread()
			ServiceReplicaStates[s].unset_thread()
	if len(services_need_informed) > 0:
		send_update_replica(servers=Servers, slave_socks=SlaveSocks,
			service_list=services_need_informed)
		for s in services_need_informed:
			ServiceReplicaStates[s].set_slave_informed()

# all clock counting related stats here
def proceed_clock():
	global ScaleInertiaCycles
	ScaleInertiaCycles -= 1
	ScaleInertiaCycles = max(0, ScaleInertiaCycles)

def set_rsc_config(cpu_config, replica_cpu_config):
	global Services
	global ServiceConfig
	global ReplicaCpus
	# global MaxCpus

	for service in Services:
		if service == 'jaeger' or service == 'zipkin':
			continue
		ServiceConfig[service]['cpus'] = cpu_config[service]
		ServiceConfig[service]['replica_cpus'] = ReplicaCpus
		# assert ServiceConfig[service]['cpus'] <= MaxCpus

def save_states(log_dir):
	global StateLog
	global Services
	global DockerMetrics

	data = {}
	data['e2e'] = {}
	data['qps'] = []
	for service in Services:
		data[service] = {}
		data[service]['replica'] = []
		data[service]['sched_replica'] = []
		data[service]['cpu_limit'] = []
		data[service]['replica_cpu_limit'] = []
		for metric in DockerMetrics:
			for t in ['mean', 'max', 'min', 'std']:
				metric_stat = metric + '_' + t
				data[service][metric_stat] = []
			if metric in ['cpu_usage', 'rx_packets', 'rx_bytes', 'tx_packets', 'tx_bytes']:
				data[service][metric + '_sum'] = []

	for state in StateLog:
		feature = state.feature
		data['qps'].append(feature.xput)
		for service in Services:
			data[service]['replica'].append(feature.replica[service])
			data[service]['sched_replica'].append(feature.sched_replica[service])

			data[service]['cpu_limit'].append(feature.cpu_limit[service])
			data[service]['replica_cpu_limit'].append(feature.replica_cpu_limit[service])

			for metric in DockerMetrics:
				data[service][metric + '_mean'].append(feature.docker_metrics[metric][service].mean())
				data[service][metric + '_min'].append(feature.docker_metrics[metric][service].min())
				data[service][metric + '_max'].append(feature.docker_metrics[metric][service].max())
				data[service][metric + '_std'].append(np.std(feature.docker_metrics[metric][service]))
				if metric in ['cpu_usage', 'rx_packets', 'rx_bytes', 'tx_packets', 'tx_bytes']:
					data[service][metric + '_sum'].append(np.sum(feature.docker_metrics[metric][service]))

		percent_keys = sorted(list(feature.end_to_end_lat.keys()))
		for p in percent_keys:
			if p not in data['e2e']:
				data['e2e'][p] = []
			data['e2e'][p].append(feature.end_to_end_lat[p])

	# write scheduler metric
	sched_metric = ['replica', 'sched_replica', 'cpu_limit', 'replica_cpu_limit']
	for service in Services:
		for s in sched_metric:
			with open(str(log_dir / (s + '_' + service + '.txt')), 'w+') as f:
				for val in data[service][s]:
					f.write(str(val) + '\n')

	# # write docker metric
	# for service in Services:
	# 	for metric in DockerMetrics:
	# 		for stat in ['mean', 'min', 'max', 'std']:
	# 			metric_stat = metric + '_' + stat
	# 			with open(str(log_dir / (metric_stat + '_' + service + '.txt')), 'w+') as f:
	# 				for val in data[service][metric_stat]:
	# 					f.write(str(val) + '\n')
	# 		if metric in  ['cpu_usage', 'rx_packets', 'rx_bytes', 'tx_packets', 'tx_bytes']:
	# 			metric_stat = metric + '_sum'
	# 			with open(str(log_dir / (metric_stat + '_' + service + '.txt')), 'w+') as f:
	# 				for val in data[service][metric_stat]:
	# 					f.write(str(val) + '\n')

	# write tail latency distributions
	for i in np.arange(95.0, 100.0, 0.5):
		with open(str(log_dir / ('e2e_lat_' + str(i) + '.txt')), 'w+') as f:
			assert i in data['e2e']
			for val in data['e2e'][i]:
				f.write(str(val) + '\n')

	# write rps
	with open(str(log_dir / 'qps.txt'), 'w+') as f:
		for val in data['qps']:
			f.write(str(val) + '\n')


# return true if application needs redeloyed
def run_exp(rps, load_script, log_dir, vic_service):
	global ViolTimeout
	global ExpTime
	global MeasureInterval
	global StateLog
	global EndToEndQos
	global Benchmark
	global BenchmarkDir

	global wrk2
	global Wrk2LastTime
	global Wrk2pt
	global Servers
	global Services
	global ScalableServices
	global TestedServices
	global ServiceReplicaStates
	global ServiceConfig
	global ServiceInitConfig
	global SlaveSocks

	global ReplicaCpus

	init_data()
	time.sleep(5)
	StartTime = time.time()
	if not os.path.isdir(str(log_dir)):
		os.makedirs(str(log_dir))

	# states_path 	= log_dir / 'states.txt'
	wrk2_exp_log	= log_dir / 'wrk2_log.txt'
	wrk2_summary	= open(str(log_dir / 'wrk2_summary.txt'), 'w+')
	logging.info('Test rps: ' + str(rps))

	# scale to init config
	do_docker_scale_init()
	send_exp_start(servers=Servers, slave_socks=SlaveSocks)
	inform_slaves_new_replica()

	# start wrk2
	logging.info("start wrk2")
	init_cpu_config = {}
	for s in ServiceInitConfig:
		if s != 'jaeger' and s != 'zipkin':
			init_cpu_config[s] = ServiceInitConfig[s]['replica_cpus']
	send_rsc_config(cpu_config=init_cpu_config,
		servers=Servers, slave_socks=SlaveSocks)
	time.sleep(5)

	warmup_app(wrk2=wrk2, benchmark_dir=BenchmarkDir, benchmark=Benchmark)

	wrk2_p = None
	if rps > 0:
		wrk2_p = run_wrk2(wrk2=wrk2,
			lua_script=str(
				BenchmarkDir / 'wrk2' / 'scripts' / 'social-network' / 'mixed-workload.lua'),
			nginx_ip='http://127.0.0.1:8080',
			dist='exp', tail=95, tail_resolution=0.5, stats_rate=0.2, tail_report_interval=1,
			num_threads=10, num_conns=300, duration=ExpTime+15, reqs_per_sec=rps,
			output=wrk2_summary, quiet=False)

	elif load_script != '':
		# duration decided by script
		wrk2_p = run_wrk2(wrk2=wrk2,
			lua_script=str(
				BenchmarkDir / 'wrk2' / 'scripts' / 'social-network' / load_script),
			nginx_ip='http://127.0.0.1:8080',
			dist='exp', tail=95, tail_resolution=0.5, stats_rate=0.2, tail_report_interval=1,
			num_threads=10, num_conns=300, duration=0, reqs_per_sec='script',
			output=wrk2_summary, quiet=False)

	assert(wrk2_p != None)

	time.sleep(15)	# wait for latencys to become stable

	StartTime = time.time()
	# start adjusting rsc
	prev_time = StartTime
	cur_time  = prev_time

	consecutive_viol = 0
	exp_terminated = False
	interval_idx = 1
	exp_terminated = False
	reset_docker_scale_clock()

	while time.time() - StartTime < ExpTime:
		cur_time = time.time()
		if cur_time - StartTime < MeasureInterval*interval_idx:
			time.sleep(MeasureInterval*interval_idx - (cur_time - StartTime))
			continue
		else:
			logging.info('current interval_idx: ' + str(interval_idx) + ', cur_time = ' + format(cur_time - StartTime, '.1f'))
			interval_idx = int((cur_time - StartTime)/MeasureInterval) + 1
			inform_slaves_new_replica()
			# prev_time  = cur_time
			timestamp = round(cur_time - StartTime, 1)
			cur_state = State()

			feature = Feature(timestamp)
			# save resources before update
			feature.cpu_limit = {}
			feature.replica_cpu_limit = {}
			for s in Services:
				feature.cpu_limit[s] = ServiceConfig[s]['cpus']
				feature.replica_cpu_limit[s] = ServiceConfig[s]['replica_cpus']

			tail, xput = get_wrk2_data(feature=feature, wrk2_last_time=Wrk2LastTime, wrk2_pt=Wrk2pt)
			get_metric(feature=feature)
			cur_state.feature = feature	

			logging.info('cur_time = ' +  format(cur_time - StartTime, '.1f') + \
				 			', tail = ' + str(tail) + ', xput = ' + str(xput))
			proceed_clock()

			if tail != 0:
				# print 'tail = ', tail, ', xput = ', xput
				if tail > EndToEndQos:
					# check qos violation timeout
					consecutive_viol += 1
					if consecutive_viol > ViolTimeout:
						exp_terminated = True
						logging.warning('consecutive viol of %d cycles, experiment aborted' %(consecutive_viol))
						break
				else:
					consecutive_viol = 0

				# check whether to trigger docker scale
				if trigger_docker_scale(vic_service):
					new_replica = propose_replica(vic_service)
					# print('trigger_docker_scale new_replica')
					# print(new_replica)
					if len(new_replica) != 0:
						do_docker_scale(new_replica)
						reset_docker_scale_clock()

				replica_cpu_config = {}
				cpu_config = {}
				for s in ServiceConfig:
					replica_cpu_config[s] = ReplicaCpus
					r, _ = ServiceReplicaStates[s].get_replica()
					cpu_config[s] = r*ReplicaCpus
				send_rsc_config(cpu_config=replica_cpu_config, servers=Servers,
					slave_socks=SlaveSocks)
				set_rsc_config(cpu_config=cpu_config, replica_cpu_config=replica_cpu_config)

			StateLog.append(cur_state)
			prev_time = time.time()

	if not exp_terminated:
		msg = wrk2_p.communicate()
	else:
		wrk2_p.kill()

	# terminate exp
	send_terminate_exp(servers=Servers, slave_socks=SlaveSocks)

	# save logs
	save_states(log_dir)
	cmd = 'cp ' + str(Wrk2pt) + ' ' + str(wrk2_exp_log)
	wrk2_summary.flush()
	wrk2_summary.close()
	subprocess.call(cmd, shell=True)

	# return unlike_to_recover
	return False	# always assume that experiemnt terminates successfully

def main():
	global TestRps
	global DataDir
	global Services
	global SetupSwarm
	global Deploy

	global Servers
	global HostServer
	global SlavePort
	global SlaveSocks

	global wrk2
	global BenchmarkDir
	global Benchmark
	global Stackname
	global ComposeFile

	global Stackname
	global Username

	global TestedServices

	victim_services = []
	tmp_ = sorted(TestedServices.items(), key=lambda x: x[1], reverse=True)
	for t in tmp_:
		victim_services.append(t[0])
	print('victim services:')
	print(victim_services)

	if SetupSwarm:
		# establish docker swarm
		worker_nodes = list(Servers.keys())
		worker_nodes.remove(HostServer)
		assert HostServer not in worker_nodes
		setup_swarm(username=Username, worker_nodes=worker_nodes)
		# label nodes
		for server in Servers:
			if 'label' in Servers[server]:
				update_node_label(server, Servers[server]['label'])
	if Deploy:
		docker_stack_rm(stack_name=Stackname)
		docker_stack_deploy(stack_name=Stackname, benchmark=Benchmark,
			benchmark_dir=BenchmarkDir, compose_file=ComposeFile)	# deploy benchmark
		
	slave_service_config = {}
	slave_service_config['services'] = list(Services)
	slaves = setup_slaves(stack_name=Stackname, username=Username, 
		servers=Servers, 
		slave_port=SlavePort, slave_script_dir=Path.cwd(), 
		service_config=slave_service_config,
		quiet=True)
	time.sleep(5)
	# set up connections with slaves and clear data structures
	connect_slave(servers=Servers, slave_port=SlavePort, slave_socks=SlaveSocks)	
	
	# data collection
	for service in victim_services:
		if 'nginx' in service:
			continue
		for rps in TestRps:
			rps_dir = DataDir / (service + '_rps_' + str(rps))
			run_exp(rps=rps, load_script='', log_dir=rps_dir, vic_service=service)
			time.sleep(20)

	# scale all tiers
	for rps in TestRps:
		rps_dir = DataDir / ('all_rps_' + str(rps))
		run_exp(rps=rps, load_script='', log_dir=rps_dir, vic_service='')
		time.sleep(20)

	send_terminate_slave(servers=Servers, slave_socks=SlaveSocks)
	for slave in slaves:
		slave.wait()

	logging.info('total samples: %d, sat: %d, viol: %d, non_roi: %d' %(TotalSamples, 
					SatSamples, ViolSamples, NonRoiSamples))

if __name__ == '__main__':
	main()
