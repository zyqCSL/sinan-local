# assume docker version >= 1.13
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
import operator

sys.path.append(str(Path.cwd() / 'src'))
from util import *
# from wrk2_util import *
from locust_util import *
from master_slave_msg import *
from master_predictor_msg import *
from docker_swarm_util import *

# DockerMetrics = [
# 		'cpu_usage', # cpu cpu usage from docker, in terms of virtual cpu
# 		'rss', 'cache_mem', 'page_faults', # memory rss: resident set size
# 		'rx_packets', 'rx_bytes', 'tx_packets', 'tx_bytes', # network
# 		'io_serviced', 'io_bytes' # disk io (disk io monitors can't be found on gce, might need change nn arch later)
# 		]

DockerMetrics = [
		'cpu_usage', # cpu cpu usage from docker, in terms of virtual cpu
		'rss', 'cache_mem' # memory rss: resident set size
		]

random.seed(time.time())
# -----------------------------------------------------------------------
# miscs
# -----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
					filename=str(Path.cwd() / 'logs' / 'hotel_deploy_log.txt'),
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

# ml info
parser.add_argument('--gpu-config', dest='gpu_config', type=str, required=True)
parser.add_argument('--gpu-port', dest='gpu_port', type=int, required=True)
parser.add_argument('--cnn-time-steps', dest='cnn_time_steps', type=int, default=5)
parser.add_argument('--xgb-look-forward', dest='xgb_look_forward', type=int, default=4)
parser.add_argument('--cnn-valid-err', dest='cnn_valid_err', type=int, default=25)
parser.add_argument('--xgb-scale-down-threshold', dest='xgb_scale_down_threshold', 
					type=float, default=0.055,
					help='reduce rsc if xgb hold prediction <= threshold')
parser.add_argument('--xgb-scale-up-threshold', dest='xgb_scale_up_threshold', 
					type=float, default=0.10,
					help='increase rsc if xgb prediction >= threshold')

# scheduler specification
parser.add_argument('--qos', dest='qos', type=int, default=200)
parser.add_argument('--rps-provision-rate', dest='rps_provision_rate', type=float, default=1.3,
					help='predicted next rps = rps_provision_rate * current rps')
parser.add_argument('--rps-spike', dest='rps_spike', type=int, default=15,
					help='random rps spike that system should tolerate')
parser.add_argument('--viol-timeout', dest='viol_timeout', type=int, default=30)
parser.add_argument('--max-cpu-util', dest='max_cpu_util', type=float, default=0.7)
parser.add_argument('--queueing-delay', dest='queueing_delay', type=int, default=5)

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
# StackName = args.stack_name
Username = args.user_name
Deploy = args.deploy
SetupSwarm = args.setup_swarm
Stackname = args.stack_name
Benchmark = args.benchmark
BenchmarkDir =  Path.cwd() / '..' / 'benchmarks' / args.benchmark
ComposeFile = BenchmarkDir / args.compose_file
ExpTime = args.exp_time	# in second
MeasureInterval = args.measure_interval	# in second
SlavePort = args.slave_port
DeployConfig = Path.cwd() / 'config' / args.deploy_config.strip()
MultiArmBanditConfig = Path('.') / 'config' / args.mab_config.strip()
GpuServerConfig = Path('.') / 'config' / args.gpu_config.strip()

DataDir =  Path('.') / 'logs' / 'hotel_deploy_data'

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
			ServiceConfig[service]['cpus'] = 0
		if 'replica' not in ServiceConfig[service]:
			ServiceReplicaStates[service] = ReplicaState(1)
		else:
			# ServiceConfig[service]['replica'] only used for initialization
			ServiceReplicaStates[service] = ReplicaState(ServiceConfig[service]['replica'])
		if 'replica_cpus' not in ServiceConfig[service]:
			# cpu limit assigned to each replica
			ServiceConfig[service]['replica_cpus'] = ServiceConfig[service]['cpus']/ServiceConfig[service]['replica']
			# assert ServiceConfig[service]['replica_cpus'] <= ReplicaCpus
		
	ServiceInitConfig = copy.deepcopy(ServiceConfig)

# ------------------------------------------------------
# locust
# ------------------------------------------------------
LocustStatsDir = Path.home() / 'sinan_locust_log'
LocustDir = Path.cwd() / '..' / 'locust'
LocustStats = LocustStatsDir / args.locust_stats
LocustDockerCompose = LocustDir / args.locust_docker_compose

# ------------------------------------------------------
# Scaling out/in control
# ------------------------------------------------------
ScaleOutRatio = args.scale_out_ratio
ScaleInRatio = args.scale_in_ratio
ScaleIneratia = args.scale_inertia
ScaleInertiaCycles = 0

# -----------------------------------------------------------------------
# scheduler specifications
# -----------------------------------------------------------------------
EndToEndQos	= args.qos		# ms to us
RpsProvisionRate = args.rps_provision_rate
RpsSpike = args.rps_spike
PrevScaleDown = False

# -----------------------------------------------------------------------
# ml parameters
# -----------------------------------------------------------------------
GpuSock = None
GpuPort = args.gpu_port
CnnTimeSteps = args.cnn_time_steps
CnnValidError = args.cnn_valid_err
XgbScaleDownThreshold = args.xgb_scale_down_threshold
XgbScaleUpThreshold = args.xgb_scale_up_threshold
XgbLookForward = args.xgb_look_forward

GpuConfig = {}
with open(str(GpuServerConfig), 'r') as f:
	config_info = json.load(f)
	GpuConfig = config_info
	# assert 'ip_addr' in GpuConfig
	assert 'gpus' in GpuConfig
	assert 'host' in GpuConfig
	assert 'working_dir' in GpuConfig
	assert 'script' in GpuConfig

# -----------------------------------------------------------------------
# experiment parameters
# -----------------------------------------------------------------------
MinUsers = args.min_users
MaxUsers = args.max_users
UsersStep = args.users_step
TestUsers  	 = range(MinUsers, MaxUsers+1, UsersStep)
StartTime			 = -1		# the time that the script started
ViolTimeout	 = args.viol_timeout	    # #cycles of continuous violations after which the application is unlikely to recover even at max rsc

# data features
StateLog			= []
ScaleDecLog 		= []

# -----------------------------------------------------------------------
# multi-arm bandit config (service clustering)
# -----------------------------------------------------------------------
OperationConfig = {}	# bias added to reward for each action
# logs for previous Hidden states and Operations
QueueingDelay = args.queueing_delay
MaxCpuUtil = args.max_cpu_util
ActionLog = []	# past actions within queueing delay

with open(str(MultiArmBanditConfig), 'r') as f:
	config_info = json.load(f)
	OperationConfig = config_info['operation']

def update_action_log(action):
	global ActionLog
	global QueueingDelay
	ActionLog.append(action)
	if len(ActionLog) > QueueingDelay:
		ActionLog = ActionLog[1:]

def clear_action_log():
	global ActionLog
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

def get_op_cpus(service, op_name, cur_cpu, max_cpu):
	cpus = -1
	min_cpu = get_min_cpus(cur_cpu, get_cpu_util(service), max_cpu)
	assert min_cpu >= 0.2
	if 'x' in op_name:
		r = float(op_name.replace('x', ''))
		if r > 0:
			cpus = math.ceil(cur_cpu + r*max_cpu)
		else:
			cpus = cur_cpu + r*cur_cpu
			if cpus < 1:
				cpus = get_divided_cpus(cur_cpu)
			else:
				cpus = math.ceil(cpus)
	else:
		cpus = math.ceil(cur_cpu + float(op_name))
		if cpus < 1:
			cpus = get_divided_cpus(cur_cpu)
	cpus = min(max(min_cpu, cpus), max_cpu)
	return cpus

def get_min_cpus(cur_cpu, cpu_util, max_cpu):
	global ServiceConfig
	global MaxCpuUtil
	if cpu_util > MaxCpuUtil:
		return min(math.ceil(cur_cpu*cpu_util/MaxCpuUtil), max_cpu)
	else:
		return 0.5

def get_divided_cpus(cur_cpus):
	return 0.5
	# if cur_cpus >= 1:
	# 	return 0.8
	# else:
	# 	return max(cur_cpus-0.2, 0.2)

# TODO: implement init & init_data
def init_data():
	global StateLog
	global ServiceConfig
	global ServiceInitConfig
	global Servers
	global SlaveSocks

	StateLog = []
	ServiceConfig = copy.deepcopy(ServiceInitConfig)
	clear_action_log()
	clear_scale_dec_log()
	send_init_data(servers=Servers, slave_socks=SlaveSocks)

#---------------------------------------
# State & Action
#---------------------------------------
class Action:
	def __init__(self):
		self.beneficiary_service = []	# beneficiary
		self.victim_service = []	# victims
		self.cpu = {}	# cpu cycle limit
		self.core = {}	# physical core allocation
		self.pred_lat = 0
		self.pred_viol = 0
		self.type = 'global'

	def show(self):
		global Services
		report = 'victim_service: ' + ', '.join(self.victim_service)
		report += ' beneficiary_service: ' + ', '.join(self.beneficiary_service) 
		for service in Services:
			report += ' ' + service + ' %.3f;' %(self.cpu[service])
		report += ' pred_99_tail_lat ' + format(self.pred_lat, '.1f') + \
				  ' pred_viol_prob ' + format(self.pred_viol, '.4f')
		return report

	def derive_per_replica(self, actual_replica):
		global ServiceReplicaStates
		per_replica_action = Action()
		per_replica_action.type = 'replica'
		per_replica_action.beneficiary_service = list(self.beneficiary_service)
		per_replica_action.victim_service = list(self.victim_service)
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
			logging.info('derive_per_replica service %+20s total=%.4f, per_replica=%.4f' %(service, self.cpu[service], per_replica_action.cpu[service]))
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

		# ml prediction
		self.pred_lat = 0
		self.pred_viol = 0

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
# state must be updated before action is proposed
class State:
	def __init__(self):
		self.feature = None 	# instant observation

# def tune_xgb_threshold(rps):
# 	global XgbScaleDownThreshold
# 	if rps <= 160:
# 		XgbScaleDownThreshold = 0.15
# 		XgbScaleUpThreshold = 0.25
# 	elif rps <= 200:
# 		XgbScaleDownThreshold = 0.1
# 		XgbScaleUpThreshold = 0.2
# 	else:
# 		XgbScaleDownThreshold = 0.075
# 		XgbScaleUpThreshold = 0.175

def propose_action(rps_fluct):
	global ServiceConfig
	global OperationConfig
	global EndToEndQos
	global MaxCpuUtil

	global CnnTimeSteps
	global CnnValidError
	global XgbScaleDownThreshold
	global XgbScaleUpThreshold
	global RpsProvisionRate
	global RpsSpike
	global DockerMetrics
	global PrevScaleDown

	global StateLog
	global ActionLog

	global GpuSock

	# # tune xgb threshold
	# tune_xgb_threshold(rps=get_rps())

	# hold when history info is inadequate
	if len(StateLog) < CnnTimeSteps:
		action = Action()
		for service in ServiceConfig:
			action.cpu[service] = ServiceConfig[service]['max_cpus']
			action.core[service] = math.ceil(ServiceConfig[service]['max_cpus'])
		return action

	info = {}
	#------------------------------------------		
	# sys data
	#------------------------------------------	
	sys_data = {}
	assert len(StateLog) >= CnnTimeSteps
	# e2e_lat
	sys_data['e2e_lat'] = {}
	for key in ['90.0', '95.0', '98.0', '99.0', '99.9']:
		sys_data['e2e_lat'][key] = []
		for i in range(-CnnTimeSteps, 0):
			sys_data['e2e_lat'][key].append(StateLog[i].feature.end_to_end_lat[key])
	
	# resource usage
	for service in ServiceConfig:
		sys_data[service] = {}
		sys_data[service]['rps'] 		  = []
		sys_data[service]['replica'] 	  = []
		sys_data[service]['cpu_limit']    = []
		for metric in DockerMetrics:
			# for t in ['mean', 'min', 'max', 'std']:
			for t in ['mean']:
				metric_stat = metric + '_' + t
				sys_data[service][metric_stat] = []

		for i in range(-CnnTimeSteps, 0):
			sys_data[service]['replica'].append(StateLog[i].feature.replica[service])
			sys_data[service]['rps'].append(StateLog[i].feature.rps)
			sys_data[service]['cpu_limit'].append(StateLog[i].feature.cpu_limit[service])

			for metric in DockerMetrics:
				sys_data[service][metric + '_mean'].append(float(StateLog[i].feature.docker_metrics[metric][service].mean()))
				# sys_data[service][metric + '_min'].append(float(StateLog[i].feature.docker_metrics[metric][service].min()))
				# sys_data[service][metric + '_max'].append(float(StateLog[i].feature.docker_metrics[metric][service].max()))
				# sys_data[service][metric + '_std'].append(float(np.std(StateLog[i].feature.docker_metrics[metric][service])))

	info['sys_data'] = sys_data

	#------------------------------------------	
	# next_info/next_k_info 
	#------------------------------------------	
	next_info = []
	# next_rps = StateLog[-1].feature.rps * RpsProvisionRate + RpsSpike

	# propose next resource assignment
	cur_index  = 0
	hold_index = 0
	max_index = 0
	scale_up_index = []
	scale_down_index = []
	scale_up_op = []
	scale_down_op = []
	for op in OperationConfig:
		if is_recover_op(op):
			scale_up_op.append(op)
		elif is_reduce_op(op):
			scale_down_op.append(op)

	# check saturated services
	saturated_services = {}
	for service in ServiceConfig:
		cpu_util = get_cpu_util(service)
		if cpu_util > MaxCpuUtil:
			saturated_services[service] = math.ceil(get_min_cpus(ServiceConfig[service]['cpus'], 
				cpu_util, ServiceConfig[service]['max_cpus']))

	# hold
	proposal = {}
	for service in ServiceConfig:
		proposal[service] = {}
		if service in saturated_services:
			proposal[service]['cpus'] = max(ServiceConfig[service]['cpus'], saturated_services[service])
		else:
			proposal[service]['cpus'] = ServiceConfig[service]['cpus']
		# proposal[service]['rps'] = next_rps
	next_info.append(proposal) 

	#----------- maximize all services ------------#
	proposal = {}
	for service in ServiceConfig:
		proposal[service] = {}
		proposal[service]['cpus'] = ServiceConfig[service]['max_cpus']
		# proposal[service]['rps'] = next_rps
	cur_index += 1
	max_index = cur_index
	next_info.append(proposal)
	scale_up_index.append(cur_index)

	#----------- scale up 1 service ------------#
	for ben_service in ServiceConfig:
		if ServiceConfig[ben_service]['cpus'] < ServiceConfig[ben_service]['max_cpus']:
			service_cpu_limits = []
			for op in scale_up_op:
				proposal = {}
				op_cpu = get_op_cpus(ben_service, op, ServiceConfig[ben_service]['cpus'], 
						ServiceConfig[ben_service]['max_cpus'])
				assert op_cpu >= ServiceConfig[ben_service]['cpus']
				if op_cpu not in service_cpu_limits:
					service_cpu_limits.append(op_cpu)
				else:
					continue
				for service in ServiceConfig:
					proposal[service] = {}
					# proposal[service]['rps'] = next_rps
					if service != ben_service:
						if service not in saturated_services:
							proposal[service]['cpus'] = ServiceConfig[service]['cpus']
						else:
							proposal[service]['cpus'] = max(ServiceConfig[service]['cpus'], saturated_services[service])
					else:
						proposal[service]['cpus'] = op_cpu
						assert proposal[service]['cpus'] >= ServiceConfig[service]['cpus']
				cur_index += 1
				next_info.append(proposal)
				scale_up_index.append(cur_index)

	#----------- scale up all services ------------#
	for op in scale_up_op:
		proposal = {}
		for service in ServiceConfig:
			proposal[service] = {}
			proposal[service]['cpus'] = get_op_cpus(service, op, ServiceConfig[service]['cpus'], 
					ServiceConfig[service]['max_cpus'])
			# print(proposal[service]['cpus'])
			# print(ServiceConfig[service]['cpus'])
			assert proposal[service]['cpus'] >= ServiceConfig[service]['cpus']
			# proposal[service]['rps'] = next_rps

		cur_index += 1
		next_info.append(proposal)
		scale_up_index.append(cur_index)

	#----------- scale up victim services in ActionLog ------------#
	# original order
	victim_services = []
	num_victim_services = 0
	for action in ActionLog:
		for vs in action.victim_service:
			if vs not in victim_services:
				victim_services.append(vs)
		# no new victim services
		if len(victim_services) == num_victim_services:
			continue
		else:
			num_victim_services = len(victim_services)

		for op in scale_up_op:
			proposal = {}
			for service in ServiceConfig:
				proposal[service] = {}
				# proposal[service]['rps'] = next_rps
				if service not in victim_services:
					if service not in saturated_services:
						proposal[service]['cpus'] = ServiceConfig[service]['cpus']
					else:
						proposal[service]['cpus'] = max(ServiceConfig[service]['cpus'], saturated_services[service])
				else:
					proposal[service]['cpus'] = get_op_cpus(service, op, ServiceConfig[service]['cpus'], 
						ServiceConfig[service]['max_cpus'])
					assert proposal[service]['cpus'] >= ServiceConfig[service]['cpus']
			cur_index += 1
			next_info.append(proposal)
			scale_up_index.append(cur_index)

	# reverse order
	victim_services = []
	num_victim_services = 0
	# no need to travel to the last index since it's already included in original order
	for i in range(len(ActionLog) - 1, 0, -1):
		for vs in ActionLog[i].victim_service:
			if vs not in victim_services:
				victim_services.append(vs)
		# no new victim services
		if len(victim_services) == num_victim_services:
			continue
		else:
			num_victim_services = len(victim_services)

		for op in scale_up_op:
			proposal = {}
			for service in ServiceConfig:
				proposal[service] = {}
				# proposal[service]['rps'] = next_rps
				if service not in victim_services:
					if service not in saturated_services:
						proposal[service]['cpus'] = ServiceConfig[service]['cpus']
					else:
						proposal[service]['cpus'] = max(ServiceConfig[service]['cpus'], saturated_services[service])
				else:
					proposal[service]['cpus'] = get_op_cpus(service, op, ServiceConfig[service]['cpus'], 
						ServiceConfig[service]['max_cpus'])
					assert proposal[service]['cpus'] >= ServiceConfig[service]['cpus']
			cur_index += 1
			next_info.append(proposal)
			scale_up_index.append(cur_index)

	#----------- scale down all services ------------#
	for op in scale_down_op:
		proposal = {}
		for service in ServiceConfig:
			proposal[service] = {}
			proposal[service]['cpus'] = get_op_cpus(service, op, ServiceConfig[service]['cpus'], 
					ServiceConfig[service]['max_cpus'])
			# if service not in saturated_services:
			# 	assert proposal[service]['cpus'] <= ServiceConfig[service]['cpus']
			# proposal[service]['rps'] = next_rps

		cur_index += 1
		next_info.append(proposal)
		scale_down_index.append(cur_index)

	#----------- scale down 1 service ------------#
	for vic_service in ServiceConfig:
		service_cpu_limits = []
		for op in scale_down_op:
			op_cpu = get_op_cpus(vic_service, op, ServiceConfig[vic_service]['cpus'], 
						ServiceConfig[vic_service]['max_cpus'])

			# if vic_service not in saturated_services:
			# 	assert op_cpu <= ServiceConfig[vic_service]['cpus']
			if op_cpu not in service_cpu_limits:
				service_cpu_limits.append(op_cpu)
				proposal = {}
				for service in ServiceConfig:
					proposal[service] = {}
					# proposal[service]['rps'] = next_rps
					if service == vic_service:
						proposal[service]['cpus'] = op_cpu
					else:
						if service not in saturated_services:
							proposal[service]['cpus'] = ServiceConfig[service]['cpus']
						else:
							proposal[service]['cpus'] = max(ServiceConfig[service]['cpus'], saturated_services[service])
				cur_index += 1
				next_info.append(proposal)
				scale_down_index.append(cur_index)

	#----------- scale down multiple services by sorted cpu utilization ------------#
	services_cpu_util = {}
	for service in ServiceConfig:
		if ServiceConfig[service]['cpus'] > 1:
			cpu_util = get_cpu_util(service)
			if cpu_util < MaxCpuUtil:
				services_cpu_util[service] = cpu_util
	services_cpu_sorted = sorted(services_cpu_util.items(), key=operator.itemgetter(1))
	for l in range(1, len(services_cpu_sorted) - 1):
		# scaling down 1 service or all service have already been included
		victim_services = []
		for i in range(0, l + 1):
			victim_services.append(services_cpu_sorted[i][0])
		for op in scale_down_op:
			proposal = {}
			for service in ServiceConfig:
				if service in victim_services:
					proposal[service] = {}
					# proposal[service]['rps'] = next_rps
					proposal[service]['cpus'] = get_op_cpus(service, op, 
													ServiceConfig[service]['cpus'], 
													ServiceConfig[service]['max_cpus'])
					# if service not in saturated_services:
					# 	assert proposal[service]['cpus'] <= ServiceConfig[service]['cpus']
				else:
					proposal[service] = {}
					# proposal[service]['rps'] = next_rps
					if service not in saturated_services:
						proposal[service]['cpus'] = ServiceConfig[service]['cpus']
					else:
						proposal[service]['cpus'] = max(ServiceConfig[service]['cpus'], saturated_services[service])
			cur_index += 1
			next_info.append(proposal)
			scale_down_index.append(cur_index)

	info['next_info'] = next_info

	# get prediction from gpu
	prediction = get_ml_prediction(info=info, gpu_sock=GpuSock)
	# logging.info(str(prediction))

	scale_up_pred = []
	scale_down_pred = []

	# all_down_pred = []
	# for k in scale_down_index:
	# 	all_down_pred.append([k, prediction[k], next_info[k]])
	# logging.info('all_down_pred ' + str(all_down_pred))

	assert len(prediction) == len(next_info)

	action = Action()
	scale_down_doable = True
	# check if scale down is acceptable
	least_core = -1
	least_index = -1
	least_lat = -1
	for index in scale_down_index:
		if prediction[index][0] < EndToEndQos - CnnValidError and \
			prediction[index][1] <= XgbScaleDownThreshold:
			total_core = 0
			for service in ServiceConfig:
				total_core += next_info[index][service]['cpus']
			if least_core < 0 or least_core > total_core or \
				(least_core == total_core and least_lat > prediction[index][0]):
				least_core = total_core
				least_index = index
				least_lat = prediction[index][0]
				scale_down_pred.append(prediction[index])

	# if least_index < 0 or PrevScaleDown or abs(rps_fluct) >= 0.02:
	if least_index < 0 or abs(rps_fluct) >= 0.02:
	# if least_index < 0:
		# # no scale down satisfies and we just hold
		# for service in ServiceConfig:
		# 	action.pred_lat  = prediction[hold_index][0]
		# 	action.pred_viol = prediction[hold_index][1]
		# 	action.cpu[service] = ServiceConfig[service]['cpus']
		# 	action.core[service] = math.ceil(ServiceConfig[service]['cpus'])
		PrevScaleDown = False
		logging.info('scale down not acceptable')
		scale_down_doable = False
	else:
		PrevScaleDown = True
		assert least_index != 0 # 0 is hold
		logging.info('scale_down acceptable')
		action.pred_lat  = prediction[least_index][0]
		action.pred_viol = prediction[least_index][1]
		for service in ServiceConfig:
			action.cpu[service] = next_info[least_index][service]['cpus']
			action.core[service] = math.ceil(next_info[least_index][service]['cpus'])
			if next_info[least_index][service]['cpus'] > ServiceConfig[service]['cpus']:
				action.beneficiary_service.append(service)
			elif next_info[least_index][service]['cpus'] < ServiceConfig[service]['cpus']:
				action.victim_service.append(service)

	if not scale_down_doable and prediction[hold_index][0] < EndToEndQos - CnnValidError \
		and prediction[hold_index][1] < XgbScaleUpThreshold:
		PrevScaleDown = False
		# safe to hold current resource
		logging.info('hold acceptable')
		action.pred_lat  = prediction[hold_index][0]
		action.pred_viol = prediction[hold_index][1]
		for service in ServiceConfig:
			action.cpu[service] = ServiceConfig[service]['cpus']
			action.core[service] = math.ceil(ServiceConfig[service]['cpus'])

	elif not scale_down_doable:
		PrevScaleDown = False
		# scale up needed
		logging.info('scale up needed')
		least_core = -1
		least_index = -1
		least_lat = -1
		for index in scale_up_index:
			if prediction[index][0] < EndToEndQos - CnnValidError and \
				prediction[index][1] < XgbScaleUpThreshold:
				total_core = 0
				for service in ServiceConfig:
					total_core += next_info[index][service]['cpus']
				if least_core < 0 or least_core > total_core or \
					(least_core == total_core and least_lat > prediction[index][0]):
					least_core = total_core
					least_index = index
					least_lat = prediction[index][0]
		if least_index < 0:
			# no scale up satisfies and we maxmize rsc
			for service in ServiceConfig:
				action.pred_lat  = prediction[max_index][0]
				action.pred_viol = prediction[max_index][1]
				action.cpu[service] = ServiceConfig[service]['max_cpus']
				action.core[service] = math.ceil(ServiceConfig[service]['max_cpus'])
				if action.cpu[service] > ServiceConfig[service]['cpus']:
					action.beneficiary_service.append(service)
		else:
			assert least_index != 0 # 0 is hold
			action.pred_lat  = prediction[least_index][0]
			action.pred_viol = prediction[least_index][1]
			for service in ServiceConfig:
				action.cpu[service] = next_info[least_index][service]['cpus']
				action.core[service] = math.ceil(next_info[least_index][service]['cpus'])
				if next_info[least_index][service]['cpus'] > ServiceConfig[service]['cpus']:
					action.beneficiary_service.append(service)
				elif next_info[least_index][service]['cpus'] < ServiceConfig[service]['cpus']:
					action.victim_service.append(service)

	logging.info('scale_down_doable: ' + str(scale_down_pred))
	return action

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

def get_rps():
	global StateLog
	
	if len(StateLog) == 0:
		return 1.0
	else:
		if len(StateLog) < 3:
			s = 0
			for state in StateLog:
				s += state.feature.rps
			return s/len(StateLog)
		else:
			s = 0
			for i in range(-3, 0, 1):
				s += StateLog[i].feature.rps
			return s/3

def get_rps_fluct(cur_rps):
	global StateLog

	if len(StateLog) == 0:
		return 0
	else:
		return (cur_rps - StateLog[-1].feature.rps)/cur_rps


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

# -----------------------------------------------------------------------
# scale in/out
# -----------------------------------------------------------------------
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

def save_states(log_dir):
	global StateLog
	global Services
	global DockerMetrics

	data = {}
	data['e2e'] = {}
	data['rps'] = []
	data['pred_lat'] = []
	data['pred_viol'] = []
	for service in Services:
		data[service] = {}
		data[service]['replica'] = []
		data[service]['sched_replica'] = []
		data[service]['cpu_limit'] = []
		data[service]['replica_cpu_limit'] = []
		for metric in DockerMetrics:
			# for t in ['mean', 'max', 'min', 'std']:
			for t in ['mean']:
				metric_stat = metric + '_' + t
				data[service][metric_stat] = []
			# if metric in ['cpu_usage', 'rx_packets', 'rx_bytes', 'tx_packets', 'tx_bytes']:
			if metric in ['cpu_usage']:
				data[service][metric + '_sum'] = []

	for state in StateLog:
		feature = state.feature
		data['rps'].append(feature.rps)
		data['pred_lat'].append(feature.pred_lat)
		data['pred_viol'].append(feature.pred_viol)
		for service in Services:
			data[service]['replica'].append(feature.replica[service])
			data[service]['sched_replica'].append(feature.sched_replica[service])

			data[service]['cpu_limit'].append(feature.cpu_limit[service])
			data[service]['replica_cpu_limit'].append(feature.replica_cpu_limit[service])

			for metric in DockerMetrics:
				data[service][metric + '_mean'].append(feature.docker_metrics[metric][service].mean())
				# data[service][metric + '_min'].append(feature.docker_metrics[metric][service].min())
				# data[service][metric + '_max'].append(feature.docker_metrics[metric][service].max())
				# data[service][metric + '_std'].append(np.std(feature.docker_metrics[metric][service]))
				# if metric in ['cpu_usage', 'rx_packets', 'rx_bytes', 'tx_packets', 'tx_bytes']:
				if metric in ['cpu_usage']:
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
			# for stat in ['mean', 'min', 'max', 'std']:
			for stat in ['mean']:
				metric_stat = metric + '_' + stat
				with open(str(log_dir / (metric_stat + '_' + service + '.txt')), 'w+') as f:
					for val in data[service][metric_stat]:
						f.write(str(val) + '\n')
			# if metric in  ['cpu_usage', 'rx_packets', 'rx_bytes', 'tx_packets', 'tx_bytes']:
			if metric in  ['cpu_usage']:
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

	# write predictions
	with open(str(log_dir / 'pred_lat.txt'), 'w+') as f:
		for val in data['pred_lat']:
			f.write(str(val) + '\n')

	with open(str(log_dir / 'pred_viol.txt'), 'w+') as f:
		for val in data['pred_viol']:
			f.write(str(val) + '\n')


# return true if application needs redeloyed
def run_exp(users, log_dir):
	global ViolTimeout
	global ExpTime
	global MeasureInterval
	global StateLog
	global ActionLog
	global EndToEndQos
	global Benchmark
	global BenchmarkDir

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
	logging.info("start wrk2")
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
	interval_idx = 1
	exp_terminated = False
	in_recovery = False
	service_fail = False
	rps_fluct = 0
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
			rps_fluct = get_rps_fluct(cur_rps=rps)
			success = get_metric(feature=feature)
			if not success:
				exp_terminated = True
				service_fail = True
				break
			cur_state.feature = feature
			if len(ActionLog) > 0:
				feature.pred_lat = ActionLog[-1].pred_lat
				feature.pred_viol = ActionLog[-1].pred_viol
			cur_state.feature = feature	

			# update hidden state before appending to state log
			qos_violated = tail > EndToEndQos
			logging.info('cur_time = %.1f, tail = %d, rps = %.1f, failures = %d' %(
				cur_time - StartTime, tail, rps, failures))
			StateLog.append(cur_state)

			proceed_clock()

			if tail != 0:
				# print 'tail = ', tail, ', rps = ', rps
				if tail > EndToEndQos:
					# check qos violation timeout
					consecutive_viol += 1
					if consecutive_viol > ViolTimeout:
						exp_terminated = True
						logging.warning('consecutive viol of %d cycles, experiment aborted' %(consecutive_viol))
						break
				else:
					consecutive_viol = 0

				tp_0 = time.time()
				action = propose_action(rps_fluct=rps_fluct)
				tp_1 = time.time()
				logging.info('propose_action time = %.3fs' %(tp_1 - tp_0))
				assert action != None
				per_replica_action = action.derive_per_replica(feature.replica)
				print(action.show())
				logging.info(action.show())
				# add hidden state & action to log
				update_action_log(action)
				send_rsc_config(cpu_config=per_replica_action.cpu, servers=Servers,
					slave_socks=SlaveSocks)
				set_rsc_config(cpu_config=action.cpu, replica_cpu_config=per_replica_action.cpu)

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

	# new 
	global GpuConfig
	global GpuSock

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
	#---- connect slaves -----#
	slave_service_config = {}
	slave_service_config['services'] = list(Services)
	slaves = setup_slaves(stack_name=Stackname, username=Username, 
		servers=Servers, 
		slave_port=SlavePort, slave_script_dir=Path.cwd(), 
		service_config=slave_service_config)
	time.sleep(5)
	# set up connections with slaves and clear data structures
	connect_slave(servers=Servers, slave_port=SlavePort, slave_socks=SlaveSocks)

	#---- connect gpu -----#
	gpu = setup_gpu(username=Username, gpu_host=GpuConfig['host'],
		work_dir=GpuConfig['working_dir'], script=GpuConfig['script'],
		gpus=GpuConfig['gpus'])
	time.sleep(5)
	GpuSock = connect_gpu(gpu_host=GpuConfig['host'], gpu_port=GpuPort)
	
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

	send_terminate_gpu(gpu_sock=GpuSock)
	gpu.wait()

# There is no Seer in the world. You'd have to rely on yourself :)
if __name__ == '__main__':
	main()
