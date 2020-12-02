# Docker version 19.03.11, Ubuntu 18.04
import sys
import os
import subprocess
import threading
import time
import numpy as np
import json
import math
import random
import argparse
import logging
from pathlib import Path
import copy

sys.path.append(str(Path.cwd() / 'src'))
from util import *
# from wrk2_util import *
from locust_util import *
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
					format='%(asctime)s %(levelname)s: %(message)s', 
					datefmt='%Y-%m-%d %H:%M:%S')

# -----------------------------------------------------------------------
# parser args definition
# -----------------------------------------------------------------------
parser = argparse.ArgumentParser()
# parser.add_argument('--cpus', dest='cpus', type=int, required=True)
# parser.add_argument('--stack-name', dest='stack_name', type=str, required=True)
parser.add_argument('--user-name', dest='user_name', type=str, default='yz2297')
parser.add_argument('--setup-swarm', dest='setup_swarm', action='store_true')
parser.add_argument('--deploy', dest='deploy', action='store_true')
parser.add_argument('--stack-name', dest='stack_name', type=str, required=True)
parser.add_argument('--benchmark', dest='benchmark', type=str, default='hotelReservation')
parser.add_argument('--compose-file', dest='compose_file', type=str, default='docker-compose-swarm.yml')
parser.add_argument('--min-users', dest='min_users', type=int, required=True)
parser.add_argument('--max-users', dest='max_users', type=int, required=True)
parser.add_argument('--users-step', dest='users_step', type=int, required=True)
parser.add_argument('--exp-time', dest='exp_time', type=int, required=True)
parser.add_argument('--measure-interval', dest='measure_interval', type=int, default=1)
parser.add_argument('--slave-port', dest='slave_port', type=int, required=True)
parser.add_argument('--deploy-config', dest='deploy_config', type=str, required=True)
parser.add_argument('--mab-config', dest='mab_config', type=str, required=True)	# config of multi-arm bandit
# data collection parameters
parser.add_argument('--qos', dest='qos', type=int, default=200)
# max latency in roi (region of interest)
parser.add_argument('--roi-latency', dest='roi_latency', type=int, default=250)
parser.add_argument('--quantize-latency-step', dest='quant_latency_step', type=int, default=25)
parser.add_argument('--quantize-rps-step', dest='quant_rps_step', type=int, default=50)
# after consecutive violations (of roi_latency) of viol_recover_delay cycles, starts to propose recover actions
parser.add_argument('--viol-recover-delay', dest='viol_recover_delay', type=int, default=1)	
parser.add_argument('--viol-timeout', dest='viol_timeout', type=int, default=30)
parser.add_argument('--max-cpu-util', dest='max_cpu_util', type=float, default=0.7)
parser.add_argument('--queueing-delay', dest='queueing_delay', type=int, default=5)
parser.add_argument('--hold-delay', dest='hold_delay', type=int, default=3,
					help='#cycles to hold after each violation')
# scale out/in
parser.add_argument('--scale-out-ratio', dest='scale_out_ratio', type=float, default=0.8)	# assigned cpus/total cpus across replicas
parser.add_argument('--scale-in-ratio', dest='scale_in_ratio', type=float, default=0.1)
parser.add_argument('--scale-inertia', dest='scale_inertia', type=int, default=30)	# time to stop scaling in after each scale in is down

# locust
parser.add_argument('--locust-docker-compose', dest='locust_docker_compose', type=str, 
	default='docker-compose-hotel.yml')
parser.add_argument('--locust-stats', dest='locust_stats', type=str,
	default='hotel_stats_history.csv')

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
MinUsers = args.min_users
MaxUsers = args.max_users
UsersStep = args.users_step
ExpTime = args.exp_time	# in second
MeasureInterval = args.measure_interval	# in second
SlavePort = args.slave_port
DeployConfig = Path.cwd() / 'config' / args.deploy_config.strip()
MultiArmBanditConfig = Path.cwd() / 'config' / args.mab_config.strip()

DataDir =  Path.cwd() / 'logs' / 'hotel_collected_data'

# keep track of collected data distribution
TotalSamples = 0
ViolSamples = 0
SatSamples = 0
NonRoiSamples = 0

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
		assert 'max_replica' in ServiceConfig[service]
		assert 'max_cpus' in ServiceConfig[service]
		# cpu cycle limit
		if 'cpus' not in ServiceConfig[service]:
			ServiceConfig[service]['cpus'] = ServiceConfig[service]['max_cpus']
		if 'replica' not in ServiceConfig[service]:
			ServiceReplicaStates[service] = ReplicaState(1)
		else:
			# ServiceConfig[service]['replica'] only used for initialization
			ServiceReplicaStates[service] = ReplicaState(ServiceConfig[service]['replica'])
		# cpu limit assigned to each replica
		ServiceConfig[service]['replica_cpus'] = ServiceConfig[service]['cpus']/ServiceConfig[service]['replica']
		# assert ServiceConfig[service]['replica_cpus'] <= ReplicaCpus
	ServiceInitConfig = copy.deepcopy(ServiceConfig)

# ------------------------------------------------------
# Scaling out/in control
# ------------------------------------------------------
ScaleOutRatio = args.scale_out_ratio
ScaleInRatio = args.scale_in_ratio
ScaleIneratia = args.scale_inertia
ScaleInertiaCycles = 0

# ------------------------------------------------------
# locust
# ------------------------------------------------------
LocustStatsDir = Path.home() / 'sinan_locust_log'
LocustDir = Path.cwd() / '..' / 'locust'
LocustStats = LocustStatsDir / args.locust_stats
LocustDockerCompose = LocustDir / args.locust_docker_compose

# -----------------------------------------------------------------------
# scheduling state parameters
# -----------------------------------------------------------------------
# qos
EndToEndQos	= args.qos		# ms to us
RoiLatency = args.roi_latency
QuantLatencyStep = args.quant_latency_step
QuantRpsStep = args.quant_rps_step
HoldDelay = args.hold_delay

# -----------------------------------------------------------------------
# data collection
# -----------------------------------------------------------------------
TestUsers  	 = range(MinUsers, MaxUsers+1, UsersStep)
StartTime			 = -1		# the time that the script started
ViolTimeout	 = args.viol_timeout	    # #cycles of continuous violations after which the application is unlikely to recover even at max rsc
ViolRecoverDelay = args.viol_recover_delay

# data features
StateLog			= []
ScaleDecLog 		= []

# -----------------------------------------------------------------------
# multi-arm bandit configurations
# -----------------------------------------------------------------------
OperationConfig = {}	# bias added to reward for each action
ViolaionBias = 0.9	# for two rsc config whose p_viol_a = (1-p_viol_b), bias towards the one that doesn't cause violation
ExplorationRate = 0.01
HoldRate = 0.05
# logs for previous Hidden states and Operations
QueueingDelay = args.queueing_delay
MaxCpuUtil = args.max_cpu_util
ActionLog = []	# past actions within queueing delay
HiddenStateLog = []	# past hidden states within queueing delay
HiddenStates = {}	# indexed by hidden state name

with open(str(MultiArmBanditConfig), 'r') as f:
	config_info = json.load(f)
	OperationConfig = config_info['operation']
	for operation in OperationConfig:
		assert 'bias' in OperationConfig[operation]
		try:
			if 'x' in operation:
				temp = float(operation.replace('x',''))
			else:
				temp = float(operation)
		except:
			print('non-interpretable op: %s' %(operation))
	ViolationBias = config_info['violation_bias']
	ExplorationRate = config_info['explore_rate']
	HoldRate = config_info['hold_rate']

# called after getting latency of current cycle
def update_hidden_state(viol):
	global HiddenStates
	global HiddenStateLog
	global ActionLog
	global QueueingDelay

	if viol:
		# blame all hidden states within the window for the actions they took
		for i in range(0, len(HiddenStateLog)):
			HiddenStates[HiddenStateLog[i]].update(viol, ActionLog[i])
		clear_mab_log()
	elif len(HiddenStateLog) == QueueingDelay:
		# consider an action not violating qos only only after QueueingDelay has passed
		HiddenStates[HiddenStateLog[0]].update(viol, ActionLog[0])

# called after update_hidden_state
def update_mab_log(hidden_state, action):
	global HiddenStateLog
	global ActionLog
	global QueueingDelay

	assert action.type == 'global'
	HiddenStateLog.append(hidden_state)
	ActionLog.append(action)
	if len(HiddenStateLog) > QueueingDelay:
		HiddenStateLog = HiddenStateLog[1:]
		ActionLog = ActionLog[1:]

def clear_mab_log():
	global HiddenStateLog
	global ActionLog
	HiddenStateLog = []
	ActionLog = []

def update_scale_dec_log(state):
	global ScaleDecLog
	global ScaleIneratia

	ScaleDecLog.append(state)
	if len(ScaleDecLog) > ScaleIneratia:
		ScaleDecLog = ScaleDecLog[1:]

def is_scale_dec_log_full():
	global ScaleDecLog
	global ScaleIneratia
	return len(ScaleDecLog) == ScaleIneratia

def clear_scale_dec_log():
	global ScaleDecLog
	ScaleDecLog = []

def is_recover_op(op_name):
	if 'x' in op_name:
		op = op_name.replace('x', '')
		return float(op) > 0
	else:
		return float(op_name) > 0

def is_reduce_op(op_name):
	if 'x' in op_name:
		op = op_name.replace('x', '')
		return float(op) < 0
	else:
		return float(op_name) < 0

def get_divided_cpus(cur_cpus):
	return 0.5
	# if cur_cpus >= 1:
	# 	return 0.8
	# else:
	# 	return max(cur_cpus-0.2, 0.2)

def get_op_cpus(service, op_name, cur_cpu, max_cpu):
	cpus = -1
	min_cpu = get_min_cpus(cur_cpu, get_cpu_util(service), max_cpu)
	assert min_cpu >= 0.2
	if 'x' in op_name:
		r = float(op_name.replace('x', ''))
		if r > 0:
			cpus = math.ceil(cur_cpu + r*max_cpu)
		else:
			cpus = cur_cpu + r*max_cpu
			if cpus < 1:
				cpus = get_divided_cpus(cur_cpu)
	else:
		cpus = math.ceil(cur_cpu + float(op_name))
		if cpus < 1:
			cpus = get_divided_cpus(cur_cpu)
	cpus = min(max(min_cpu, cpus), max_cpu)
	return cpus

def get_recover_op_cpus(service, op_name, cur_cpu, max_cpu):
	# print(op_name)
	# print('cur ', cur_cpu)
	# print('max ', max_cpu)
	assert is_recover_op(op_name)
	cpus = -1
	min_cpu = get_min_cpus(cur_cpu, get_cpu_util(service), max_cpu)
	# print('min ', min_cpu)
	if 'x' in op_name:
		r = float(op_name.replace('x', ''))
		cpus = math.ceil(cur_cpu + r*max_cpu)
	else:
		cpus = math.ceil(cur_cpu + float(op_name))
	assert cpus > 0
	cpus = min(cpus, max_cpu)
	# print('after min ', cpus)
	return max(cpus, min_cpu)

def get_min_cpus(cur_cpu, cpu_util, max_cpu):
	global ServiceConfig
	global MaxCpuUtil
	if cpu_util >= MaxCpuUtil:
		return min(math.ceil(cur_cpu*cpu_util/MaxCpuUtil), max_cpu)
	else:
		return 0.5

# compute k: when cpu_util >= 0.7, make downscaling as likely as holding, assuming holding & downscaling having same viol distribution
# compute k: e(-k*0.7) = 1/2 (set in mab config)
def cpu_util_scaling(cpu_util, op, cur_cpu, next_cpu, k=0.99):
	if not is_reduce_op(op) >= 0:
		return 1.0
	else:
		est_cpu_util = cpu_util*cur_cpu/next_cpu
		return math.exp(-k * est_cpu_util)

def bernoulli_std_dev(sat, viol):
	sat_rate = sat*1.0/(sat + viol)
	return (sat_rate*(1.0 - sat_rate))**0.5

def compute_reward(sat, viol, bias, op, cpu_util, cur_cpu, next_cpu):
	global ViolationBias
	sat_rate = sat*1.0/(sat + viol)
	cur_ci = bernoulli_std_dev(sat, viol) / (sat + viol)**0.5
	exp_ci = sat_rate * bernoulli_std_dev(sat + 1, viol) / (sat + viol + 1)**0.5 + \
			 (1 - sat_rate) * bernoulli_std_dev(sat, viol + 1) / (sat + viol + 1)**0.5
	# print('cur_ci = %f, exp_ci = %f' %(cur_ci, exp_ci))
	reward = (cur_ci - exp_ci)*bias
	if sat < viol:
		reward *= ViolationBias
	# print('reward (viol_bias) = %.f' %reward)
	reward *= cpu_util_scaling(cpu_util, op, cur_cpu, next_cpu)
	# print('reward final=%f' %reward)
	return reward

# TODO: implement init & init_data
def init_data():
	global StateLog
	global ServiceConfig
	global ServiceInitConfig
	global Servers
	global SlaveSocks

	StateLog = []
	ServiceConfig = copy.deepcopy(ServiceInitConfig)
	clear_mab_log()
	clear_scale_dec_log()
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
				precision=4)
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
		self.rps  			= 0
		self.fps			= 0
		self.failures		= 0

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
			return False
		# assert service in stats_accum

		sched_replica, _ = ServiceReplicaStates[service].get_replica()
		if sched_replica != stats_accum[service]['replica']:
			logging.warning("get_metric: service %s replica disagree master=%d, slave=%d" %(service,
				sched_replica,
				stats_accum[service]['replica']))

		feature.sched_replica[service] = sched_replica
		feature.replica[service] = stats_accum[service]['replica']
		for metric in DockerMetrics:
			feature.docker_metrics[metric][service] = np.array(stats_accum[service][metric])

	return True

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

# -----------------------------------------------------------------------
# hidden state for scheduling
# -----------------------------------------------------------------------
def get_cpu_util(service):
	global StateLog

	if len(StateLog) == 0:
		return 1.0
	else:
		if len(StateLog) < 3:
			s = 0
			for state in StateLog:
				s += state.feature.total_cpu_util(service)
			return s/len(StateLog)
		else:
			s = 0
			for i in range(-3, 0, 1):
				s += StateLog[i].feature.total_cpu_util(service)
			return s/3

def quantize_latency(lat):
	global RoiLatency
	global QuantLatencyStep
	lat = int(lat)
	if lat > RoiLatency:
		return 'INF'
	elif lat % QuantLatencyStep == 0:
		return lat
	else:
		return int(lat/QuantLatencyStep+1)*QuantLatencyStep

def quantize_rps(rps):
	global QuantRpsStep
	rps = int(rps)
	if rps % QuantRpsStep == 0:
		return rps
	else:
		return int(rps/QuantRpsStep+1)*QuantRpsStep

def name_hidden_state(prev_lat, lat, rps):
	return str(prev_lat) + 'ms_' + str(lat) + 'ms_' + str(rps) + 'rps'

class HiddenState:
	def __init__(self, name):
		global Services
		self.name = name
		self.activations = 0
		# multi-arm bandit with each service as an arm
		# history of previous resource assignments within this state
		self.rsc_history = {}
		for service in Services:
			self.rsc_history[service] = {}

	def update(self, viol, action):
		global Services
		self.activations += 1
		for service in Services:
			assert service in action.cpu
			cpu = action.cpu[service]
			# assert cpu in self.rsc_history[service]
			if cpu not in self.rsc_history[service]:
				self.rsc_history[service][cpu] = {}
				self.rsc_history[service][cpu]['sat'] = 0.1
				self.rsc_history[service][cpu]['viol'] = 0.1
			if viol:
				self.rsc_history[service][cpu]['viol'] += 1
			else:
				self.rsc_history[service][cpu]['sat'] += 1

	# todo: where to add exploration?
	def propose_action(self):
		global ServiceConfig
		global OperationConfig
		global ExplorationRate
		global HoldRate

		action = Action()
		coin = random.random()
		if coin < ExplorationRate:
			# explore a random rsc config
			for service in ServiceConfig:
				min_cpu = max(1, ServiceConfig[service]['cpus'] - 1)
				max_cpu = ServiceConfig[service]['max_cpus']
				cpu = math.ceil(random.uniform(min_cpu, max_cpu))
				action.cpu[service] = cpu
				action.core[service] = cpu
				logging.info('propose_action----explore %s, cpus: %d' %(service, cpu))
			return action

		elif coin < ExplorationRate + HoldRate:
			# hold to current rsc
			for service in ServiceConfig:
				action.cpu[service] = ServiceConfig[service]['cpus']
				action.core[service] = ServiceConfig[service]['cpus']
				logging.info('propose_action----hold %s, cpus: %d' %(service, ServiceConfig[service]['cpus']))
			return action

		for service in ServiceConfig:
			# choose the service with highest score
			chosen_op = ''
			chosen_cpu = -1
			expected_reward = 0
			service_cpu_util = get_cpu_util(service)
			print('propose_action service %s cpu_util=%f' %(service, service_cpu_util))
			for op in OperationConfig:
				op_cpus = get_op_cpus(service, op, ServiceConfig[service]['cpus'], 
									ServiceConfig[service]['max_cpus'])
				if op_cpus <= 0:
					continue
				if op_cpus not in self.rsc_history[service]:
					self.rsc_history[service][op_cpus] = {}
					# smoothing
					self.rsc_history[service][op_cpus]['sat'] = 0.1
					self.rsc_history[service][op_cpus]['viol'] = 0.1

				reward = compute_reward(sat=self.rsc_history[service][op_cpus]['sat'], 
										viol=self.rsc_history[service][op_cpus]['viol'],
										bias=OperationConfig[op]['bias'],
										op=op, 
										cpu_util=service_cpu_util,
										cur_cpu=ServiceConfig[service]['cpus'],
										next_cpu=op_cpus)

				if reward > expected_reward:
					chosen_op = op
					expected_reward = reward
					chosen_cpu = op_cpus
				elif reward == expected_reward:
					coin = random.random()
					if coin < 0.5:
						chosen_op = op
						expected_reward = reward
						chosen_cpu = op_cpus

				assert chosen_op != ''
				if chosen_cpu < ServiceConfig[service]['cpus']:
					action.vic_service.append(service)
				action.cpu[service] = chosen_cpu
				action.core[service] = math.ceil(chosen_cpu)
			logging.info('propose_action----op:%s, %+20s, cpus: %.2f, reward: %.8f, cpu_util: %.4f' %(chosen_op,
						service, chosen_cpu, expected_reward, service_cpu_util))

		return action

	# only core increase is allowed
	def propose_recover_action(self):
		global ServiceConfig
		global OperationConfig

		action = Action()
		for service in ServiceConfig:
			# choose the service with highest score
			chosen_op = ''
			chosen_cpu = -1
			expected_reward = 0
			service_cpu_util = get_cpu_util(service)

			for op in OperationConfig:
				if not is_recover_op(op):
					continue
				op_cpus = get_recover_op_cpus(service, op, ServiceConfig[service]['cpus'], 
									ServiceConfig[service]['max_cpus'])
				assert(op_cpus >= ServiceConfig[service]['cpus'])
				if op_cpus not in self.rsc_history[service]:
					self.rsc_history[service][op_cpus] = {}
					# smoothing
					self.rsc_history[service][op_cpus]['sat'] = 0.1
					self.rsc_history[service][op_cpus]['viol'] = 0.1
				op_reward = compute_reward(sat=self.rsc_history[service][op_cpus]['sat'], 
										   viol=self.rsc_history[service][op_cpus]['viol'],
										   bias=OperationConfig[op]['bias'], 
										   op=op,
										   cpu_util=service_cpu_util,
										   cur_cpu=ServiceConfig[service]['cpus'],
										   next_cpu=op_cpus)

				if op_reward > expected_reward:
					expected_reward = op_reward
					chosen_cpu = op_cpus
					chosen_op = op
				elif op_reward == expected_reward:
					coin = random.random()
					if coin < 0.5:
						expected_reward = op_reward
						chosen_cpu = op_cpus
						chosen_op = op

			assert chosen_op != ''
			assert chosen_cpu > 0
			# no victim since monotonically increasing
			# if chosen_cpu < ServiceConfig[service]['cpus']:
			# 	action.vic_service.append(service)
			action.cpu[service] = chosen_cpu
			action.core[service] = math.ceil(chosen_cpu)
			logging.info('propose_action----op:%s, %s, cpus: %d, reward: %.8f' %(chosen_op,
							service, chosen_cpu, expected_reward))

		return action

	# called during hold_delay after qos violation is resolved
	def propose_hold_action(self):
		global ServiceConfig
		action = Action()
		for service in ServiceConfig:
			action.cpu[service] = ServiceConfig[service]['cpus']
			action.core[service] = math.ceil(ServiceConfig[service]['cpus'])
		return action

def do_docker_scale(replica_proposal):
	return 

def do_docker_scale_init():
	return 

def reset_docker_scale_clock():
	return 

def trigger_docker_scale():
	# global ScaleInertiaCycles
	# return ScaleInertiaCycles == 0
	return False

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
	# global MaxCpus

	for service in Services:
		if service == 'jaeger' or service == 'zipkin':
			continue
		ServiceConfig[service]['cpus'] = cpu_config[service]
		ServiceConfig[service]['replica_cpus'] = replica_cpu_config[service]
		# assert ServiceConfig[service]['cpus'] <= MaxCpus

def save_states(log_dir):
	global StateLog
	global Services
	global DockerMetrics

	data = {}
	data['e2e'] = {}
	data['rps'] = []
	data['fps'] = []
	data['failures'] = []
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
		data['rps'].append(feature.rps)
		data['fps'].append(feature.fps)
		data['failures'].append(feature.failures)
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

	# write docker metric
	for service in Services:
		for metric in DockerMetrics:
			for stat in ['mean', 'min', 'max', 'std']:
				metric_stat = metric + '_' + stat
				with open(str(log_dir / (metric_stat + '_' + service + '.txt')), 'w+') as f:
					for val in data[service][metric_stat]:
						f.write(str(val) + '\n')
			if metric in  ['cpu_usage', 'rx_packets', 'rx_bytes', 'tx_packets', 'tx_bytes']:
				metric_stat = metric + '_sum'
				with open(str(log_dir / (metric_stat + '_' + service + '.txt')), 'w+') as f:
					for val in data[service][metric_stat]:
						f.write(str(val) + '\n')

	# write tail latency distributions
	for i in ['90.0', '95.0', '98.0', '99.0', '99.9']:
		with open(str(log_dir / ('e2e_lat_' + i + '.txt')), 'w+') as f:
			assert i in data['e2e']
			for val in data['e2e'][i]:
				f.write(str(val) + '\n')

	# write rps
	with open(str(log_dir / 'rps.txt'), 'w+') as f:
		for val in data['rps']:
			f.write(str(val) + '\n')

	with open(str(log_dir / 'fps.txt'), 'w+') as f:
		for val in data['fps']:
			f.write(str(val) + '\n')

	with open(str(log_dir / 'failures.txt'), 'w+') as f:
		for val in data['failures']:
			f.write(str(val) + '\n')


# return true if application needs redeloyed
def run_exp(users, log_dir):
	global ViolTimeout
	global ViolRecoverDelay
	global HoldDelay
	global ExpTime
	global MeasureInterval
	global StateLog
	global EndToEndQos
	global RoiLatency
	global Benchmark
	global BenchmarkDir
	global HiddenStates
	global HiddenStateLog
	global ActionLog

	# track distribution of collected data
	global TotalSamples
	global ViolSamples
	global SatSamples
	global NonRoiSamples

	global Servers
	global Services
	global ServiceReplicaStates
	global ServiceConfig
	global ServiceInitConfig
	global SlaveSocks

	global LocustDockerCompose
	global LocustStats

	init_data()
	time.sleep(5)
	StartTime = time.time()
	if not os.path.isdir(str(log_dir)):
		os.makedirs(str(log_dir))

	# states_path 	= log_dir / 'states.txt'
	logging.info('\nTest users: ' + str(users))

	# scale to init config
	do_docker_scale_init()
	send_exp_start(servers=Servers, slave_socks=SlaveSocks)
	inform_slaves_new_replica()

	# start wrk2
	logging.info("start locust")
	init_cpu_config = {}
	for s in ServiceInitConfig:
		if s != 'jaeger' and s != 'zipkin':
			init_cpu_config[s] = ServiceInitConfig[s]['replica_cpus']
	send_rsc_config(cpu_config=init_cpu_config,
		servers=Servers, slave_socks=SlaveSocks)
	time.sleep(5)

	locust_p = run_locust_docker_compose(
		docker_compose_file=LocustDockerCompose, 
		duration=ExpTime+60, users=users, 
		workers=0, quiet=True)

	assert(locust_p != None)

	time.sleep(60)	# wait for latencys to become stable

	StartTime = time.time()
	# start adjusting rsc
	prev_time = StartTime
	cur_time  = prev_time

	consecutive_viol = 0
	recovery_hold = 0
	interval_idx = 1
	exp_terminated = False
	service_fail = False
	in_recovery = False
	in_hold = False
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

			tail, rps, failures = get_locust_data(feature=feature, log_path=str(LocustStats))
			success = get_metric(feature=feature)
			if not success:
				exp_terminated = True
				service_fail = True
				break
			cur_state.feature = feature	

			TotalSamples += 1
			if tail <= EndToEndQos:
				SatSamples += 1
			elif tail <= RoiLatency:
				ViolSamples += 1
			else:
				NonRoiSamples += 1

			# update hidden state before appending to state log
			qos_violated = tail > EndToEndQos
			if not in_recovery:
				update_hidden_state(qos_violated)
			prev_tail = 0
			if len(StateLog) >= 1:
				prev_tail = StateLog[-1].feature.end_to_end_lat['99.0']
			hidden_state_name = name_hidden_state(quantize_latency(prev_tail),
										quantize_latency(tail),
										quantize_rps(rps))
			logging.info('reach hidden state: %s' %hidden_state_name)
			if hidden_state_name not in HiddenStates:
				HiddenStates[hidden_state_name] = HiddenState(hidden_state_name)

			logging.info('cur_time = %.1f, tail = %d, rps = %.1f, failures = %d' %(
				cur_time - StartTime, tail, rps, failures))

			proceed_clock()

			if tail != 0:
				action = None
				# print 'tail = ', tail, ', rps = ', rps
				if tail > RoiLatency:
					# check qos violation timeout
					consecutive_viol += 1
					in_hold = False
					recovery_hold = 0
					if consecutive_viol > ViolTimeout:
						exp_terminated = True
						logging.warning('consecutive viol of %d cycles, experiment aborted' %(consecutive_viol))
						break
					if consecutive_viol >= ViolRecoverDelay:
						action = HiddenStates[hidden_state_name].propose_recover_action()
						in_recovery = True
					else:
						action = HiddenStates[hidden_state_name].propose_action()
						in_recovery = False
				else:
					if consecutive_viol > 0:
						in_hold = True
						recovery_hold = 0
					consecutive_viol = 0

					if not in_hold:
						action = HiddenStates[hidden_state_name].propose_action()
					else:
						recovery_hold += 1
						action = HiddenStates[hidden_state_name].propose_hold_action()
						if recovery_hold >= HoldDelay:
							in_recovery = False
							in_hold = False
							recovery_hold = 0

				assert action != None
				per_replica_action = action.derive_per_replica(feature.replica)
				# add hidden state & action to log
				update_mab_log(hidden_state_name, action)
				send_rsc_config(cpu_config=per_replica_action.cpu, servers=Servers,
					slave_socks=SlaveSocks)
				set_rsc_config(cpu_config=action.cpu, replica_cpu_config=per_replica_action.cpu)

			StateLog.append(cur_state)
			update_scale_dec_log(cur_state)
			prev_time = time.time()

	if not exp_terminated:
		msg = locust_p.communicate()
	else:
		locust_p.kill()

	# terminate exp
	send_terminate_exp(servers=Servers, slave_socks=SlaveSocks)

	# save logs
	if not service_fail:
		save_states(log_dir)
		copy_locust_stats(LocustStatsDir, log_dir / 'locust_logs')
	clear_locust_state(LocustStatsDir)

	# return unlike_to_recover
	return service_fail	# always assume that experiemnt terminates successfully

def main():
	global TestUsers
	global DataDir
	global Services
	global SetupSwarm
	global Deploy

	global TotalSamples
	global ViolSamples
	global SatSamples
	global NonRoiSamples

	global Servers
	global HostServer
	global SlavePort
	global SlaveSocks

	global BenchmarkDir
	global Benchmark
	global Stackname
	global ComposeFile

	global Stackname
	global Username

	# if SetupSwarm:
	# 	# establish docker swarm
	# 	worker_nodes = list(Servers.keys())
	# 	worker_nodes.remove(HostServer)
	# 	assert HostServer not in worker_nodes
	# 	setup_swarm(username=Username, worker_nodes=worker_nodes)
	# 	# label nodes
	# 	for server in Servers:
	# 		if 'label' in Servers[server]:
	# 			update_node_label(server, Servers[server]['label'])
		
	slave_service_config = {}
	slave_service_config['services'] = list(Services)
	slaves = setup_slaves(stack_name=Stackname, username=Username, 
		servers=Servers, 
		slave_port=SlavePort, slave_script_dir=Path.cwd(), 
		service_config=slave_service_config)
	time.sleep(5)
	# set up connections with slaves and clear data structures
	connect_slave(servers=Servers, slave_port=SlavePort, slave_socks=SlaveSocks)	
	
	# data collection
	i = 0
	while i < len(TestUsers):
		users = TestUsers[i]
		if Deploy:
			ath_9_cmd = 'docker-compose -f ' + str( BenchmarkDir / 'docker-compose-ath9.yml') + ' down'
			p = ssh(Username, 'ath-9', ath_9_cmd, quiet=False)
			p.wait()

			ath_8_cmd = 'docker-compose -f ' + str( BenchmarkDir / 'docker-compose-ath8.yml') + ' down'
			p = ssh(Username, 'ath-8', ath_8_cmd, quiet=False)
			p.wait()

			ath_9_cmd = 'docker-compose -f ' + str( BenchmarkDir / 'docker-compose-ath9.yml') + ' up -d'
			p = ssh(Username, 'ath-9', ath_9_cmd, quiet=False)
			p.wait()

			ath_8_cmd = 'docker-compose -f ' + str( BenchmarkDir / 'docker-compose-ath8.yml') + ' up -d'
			p = ssh(Username, 'ath-8', ath_8_cmd, quiet=False)
			p.wait()

			time.sleep(10)

		users_dir = DataDir / ('users_' + str(users))
		service_fail = run_exp(users=users, log_dir=users_dir)
		if not service_fail:
			i += 1
		time.sleep(20)

	send_terminate_slave(servers=Servers, slave_socks=SlaveSocks)
	for slave in slaves:
		slave.wait()

	logging.info('total samples: %d, sat: %d, viol: %d, non_roi: %d' %(TotalSamples, 
					SatSamples, ViolSamples, NonRoiSamples))

if __name__ == '__main__':
	main()
