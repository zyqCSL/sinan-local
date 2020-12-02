import numpy as np
import sys
import os
import re
import scipy.misc
import math
import argparse

CnnTimeSteps = 5

LookForward = 5	 # including the immediate future
Upsample = False
QoS = 200
TargetViolatioRatio = 0.5

parser = argparse.ArgumentParser()
parser.add_argument('--log-dir', type=str, dest='log_dir', required=True)
parser.add_argument('--save-dir', type=str, dest='save_dir', default='')
parser.add_argument('--look-forward', type=int, default=5, dest='look_forward')
parser.add_argument('--upsample', action='store_true', dest='upsample')
args = parser.parse_args()

LookForward = args.look_forward
LogDir = args.log_dir
SaveDir = args.save_dir
Upsample = args.upsample

# directories

# sorted order for services
Services	= ["frontend",
			   "profile",
			   "search",
			   "geo",
			   "rate",
			   "recommendation",
			   "user",
			   "reservation",
			   "memcached-rate",
			   "memcached-profile",
			   "memcached-reserve",
			   "mongodb-geo",
			   "mongodb-profile",
			   "mongodb-rate",
			   "mongodb-recommendation",
			   "mongodb-reservation",
			   "mongodb-user"]

DockerMetrics = [
		'cpu_usage', # cpu cpu usage from docker, in terms of virtual cpu
		'rss', 'cache_mem', 'page_faults', # memory rss: resident set size
		'rx_packets', 'rx_bytes', 'tx_packets', 'tx_bytes', # network
		'io_serviced', 'io_bytes' # disk io (disk io monitors can't be found on gce, might need change nn arch later)
		]

Stats = ['mean', 'max', 'min', 'std']

def upsample(sys_data, lat_data, next_k_info, lat_next_k_label):
	global QoS 
	global LookForward

	# classify data to do upsampling
	label_nxt_k = np.squeeze(lat_next_k_label[:, -2, :])	 # only keep 99% percentile

	# print label_nxt_t.shape
	label_nxt_k = np.greater_equal(label_nxt_k, QoS)
	# return

	# print label_nxt_t.shape
	if LookForward > 1:
		label_nxt_k = np.sum(label_nxt_k, axis = 1)
	# print label_nxt_t.shape
	final_label_k = np.greater_equal(label_nxt_k, 1)

	print('final_label_k.shape = '), 
	print(final_label_k.shape)

	sat_idx  = np.where(final_label_k == 0)[0]
	viol_idx = np.where(final_label_k == 1)[0]

	# print 'sat_idx ', sat_idx
	# print 'viol_idx ', viol_idx

	print('sat_idx.shape = '), 
	print(sat_idx.shape)
	print('viol_idx.shape = '),
	print(viol_idx.shape)

	viol_sat_ratio = len(viol_idx)*1.0/(len(sat_idx) + len(viol_idx))
	if len(viol_idx) == 0:
		sample_time = 0
	else:
		sample_time = int(TargetViolatioRatio/(1 - TargetViolatioRatio)*len(sat_idx)*1.0/len(viol_idx))
	print('#viol/#total = %.4f' %(viol_sat_ratio))

	if len(viol_idx) == 0:
		print('no viol in this run')

	elif sample_time <= 1:
		print('sample_time = %d, skipped' %sample_time)

	elif viol_sat_ratio < TargetViolatioRatio:
		sys_data_sat  = np.take(sys_data, indices = sat_idx, axis = 0)
		sys_data_viol = np.take(sys_data, indices = viol_idx, axis = 0)

		lat_data_sat  = np.take(lat_data, indices = sat_idx, axis = 0)
		lat_data_viol = np.take(lat_data, indices = viol_idx, axis = 0)

		next_k_info_sat  = np.take(next_k_info, indices = sat_idx, axis = 0)
		next_k_info_viol = np.take(next_k_info, indices = viol_idx, axis = 0)

		lat_next_k_label_sat  = np.take(lat_next_k_label, indices = sat_idx, axis = 0)
		lat_next_k_label_viol = np.take(lat_next_k_label, indices = viol_idx, axis = 0)

		# sample_time = int(math.ceil(TargetViolatioRatio/(1 - TargetViolatioRatio)*len(sat_idx)*1.0/len(viol_idx)))
		sample_time = int(TargetViolatioRatio/(1 - TargetViolatioRatio)*len(sat_idx)*1.0/len(viol_idx))
		print('sample_time = %d' %sample_time)
		print('after upsample #total = %d, #viol = %d' %(sample_time * len(viol_idx) + len(sat_idx), 
			sample_time * len(viol_idx)))

		sys_data = sys_data_sat
		lat_data = lat_data_sat
		next_k_info = next_k_info_sat
		lat_next_k_label = lat_next_k_label_sat

		for i in range(0, sample_time):
			sys_data = np.concatenate((sys_data, sys_data_viol), axis = 0)
			next_k_info = np.concatenate((next_k_info, next_k_info_viol), axis = 0)
			lat_data = np.concatenate((lat_data, lat_data_viol), axis = 0)
			lat_next_k_label	= np.concatenate((lat_next_k_label, lat_next_k_label_viol), axis = 0)

		print('sys_data.shape = '), 
		print(sys_data.shape)
		print('lat_data.shape = '), 
		print(lat_data.shape)
		print('next_k_info.shape = '), 
		print(next_k_info.shape)
		print('lat_next_k_label.shape = '), 
		print(lat_next_k_label.shape)

	else:
		print('Upsample not needed')

	return sys_data, lat_data, next_k_info, lat_next_k_label

def get_metric_stat(file_name):
	global DockerMetrics
	global Stats
	metric = ''
	stat = ''
	for m in DockerMetrics:
		if m in file_name and len(m) > len(metric):
			metric = m

	for s in Stats:
		if s in file_name.replace(metric, '') and len(s) > len(stat):
			stat = s

	if metric == '' or stat == '':
		return None
	else:
		return metric + '_' + stat

def compose_sys_data_channel(raw_data, metric):
	global Services
	global CnnTimeSteps
	global LookForward

	if metric != 'rps':
		for i, service in enumerate(Services):
			if i == 0:
				metric_data = np.array(raw_data[metric][service])
			else:
				metric_data = np.vstack((metric_data, raw_data[metric][service]))
	else:
		# mock rps to be per-service
		for i, service in enumerate(Services):
			if i == 0:
				metric_data = np.array(raw_data[metric])
			else:
				metric_data = np.vstack((metric_data, raw_data[metric]))

	# the last sys_data point is, in terms of timestamps, 
	# from (n - CnnTimeSteps - LookForward) to (n - LookForward - 1) (two ends included)
	# the left (n - LookForward) to (n - 1) time steps (LookForward in total) are reserved for labels
	# which includes LookForward time steps
	for i in range(0, metric_data.shape[1] - CnnTimeSteps - LookForward + 1):
		if i == 0:
			channel_data = metric_data[:, i:i+CnnTimeSteps].reshape([1, metric_data.shape[0], CnnTimeSteps])
		else:
			channel_data = np.vstack((channel_data, 
				metric_data[:, i:i+CnnTimeSteps].reshape([1, metric_data.shape[0], CnnTimeSteps])))

	# shape from (batch_size, #services, #time steps) to (batch_size, 1, #services, #CnnTimeSteps)
	# where 1 is reserved for channel concatenation
	channel_data = channel_data.reshape(
		[channel_data.shape[0], 1, channel_data.shape[1], channel_data.shape[2]])

	return channel_data

def compose_next_k_data_channel(raw_data, metric):
	global Services
	global CnnTimeSteps
	global LookForward

	if metric != 'rps':
		for i, service in enumerate(Services):
			if i == 0:
				metric_data = np.array(raw_data[metric][service])
			else:
				metric_data = np.vstack((metric_data, raw_data[metric][service]))
	else:
		# mock rps to be per-service
		for i, service in enumerate(Services):
			if i == 0:
				metric_data = np.array(raw_data[metric])
			else:
				metric_data = np.vstack((metric_data, raw_data[metric]))

	# the last sample consists of time steps from (n-LookForward) to (n-1), (two ends included)
	for i in range(CnnTimeSteps, metric_data.shape[1]-LookForward+1):
		if i == CnnTimeSteps:
			next_k_channel = metric_data[:, i: i + LookForward].reshape(
				[1, metric_data.shape[0], LookForward])
		else:
			next_k_channel = np.vstack((next_k_channel, 
				metric_data[:, i: i + LookForward].reshape([1, metric_data.shape[0], LookForward])))

	# shape from (batch_size, #services, #LookForward) to (batch_size, 1,  #services, LookForward)
	# where 1 is reserved for channel concatenation
	next_k_channel = next_k_channel.reshape(
		[next_k_channel.shape[0], 1, next_k_channel.shape[1], next_k_channel.shape[2]])

	return next_k_channel


#service_names = ['frontend', 'rate', 'geo', 'search', 'profile']
def parse_subdir(log_dir):
	global LookForward
	global CnnTimeSteps
	global QoS
	global Upsample

	global TargetViolatioRatio
	global Services
	global DockerMetrics
	global Stats

	print('\nprocessing %s' %log_dir)

	raw_data = {}	# indexed by [metric][service] (per-service metric)
	raw_data['latency'] = {}	# indexed by percentile
	raw_data['replica'] = {}	# assuming no dynamic scale in/out
	raw_data['cpu_limit'] = {}  # total cpu limit
	for metric in DockerMetrics:
		for stat in Stats:
			metric_stat = metric + '_' + stat
			raw_data[metric_stat] = {}

	# read docker metrics
	for file in os.listdir(log_dir):
		# system & sched metric
		if file == 'rps.txt':
			raw_data['rps'] = np.loadtxt(log_dir+'/'+file, dtype=np.float)
			continue
		elif file.startswith('e2e'):
			# latency data
			percent = re.sub('e2e_lat_', '', file)
			percent = re.sub('.txt', '', percent)
			raw_data['latency'][percent] = np.loadtxt(log_dir+'/'+file, dtype=np.float)
			continue
		elif file.startswith('cpu_limit'):
			service = file.replace('cpu_limit_', '').replace('.txt', '')
			assert service in Services
			raw_data['cpu_limit'][service] = np.loadtxt(log_dir+'/'+file, dtype=np.float)
			continue
		elif file.startswith('replica') and 'replica_cpu_limit' not in file:
			service = file.replace('replica_', '').replace('.txt', '')
			assert service in Services
			raw_data['replica'][service] = np.loadtxt(log_dir+'/'+file, dtype=np.float)
			continue

		# per service docker metrics
		metric_stat = get_metric_stat(file)
		if metric_stat == None:
			# print('Unuseful file %s/%s' %(log_dir, file))
			continue
		service = file.replace(metric_stat + '_', '').replace('.txt', '')
		assert service in Services
		# print('Metric stat: %s, service: %s' %(metric_stat, service))
		raw_data[metric_stat][service] = np.loadtxt(log_dir+'/'+file, dtype=np.float)

	assert 'rps' in raw_data

	#-------------------- system data ------------------#
	# rps
	rps_data = compose_sys_data_channel(raw_data, 'rps')

	# replica
	replica_data = compose_sys_data_channel(raw_data, 'replica')

	# cpu limit
	cpu_limit_data = compose_sys_data_channel(raw_data, 'cpu_limit')

	# cpu usage
	cpu_usage_mean_data = compose_sys_data_channel(raw_data, 'cpu_usage_mean')
	# memory
	rss_mean_data = compose_sys_data_channel(raw_data, 'rss_mean')

	cache_mem_mean_data = compose_sys_data_channel(raw_data, 'cache_mem_mean')

	page_faults_mean_data = compose_sys_data_channel(raw_data, 'page_faults_mean')

	# network
	rx_packets_mean_data = compose_sys_data_channel(raw_data, 'rx_packets_mean')

	rx_bytes_mean_data = compose_sys_data_channel(raw_data, 'rx_bytes_mean')

	tx_packets_mean_data = compose_sys_data_channel(raw_data, 'tx_packets_mean')

	tx_bytes_mean_data = compose_sys_data_channel(raw_data, 'tx_bytes_mean')
	# io
	io_bytes_mean_data = compose_sys_data_channel(raw_data, 'io_bytes_mean')

	io_serviced_mean_data = compose_sys_data_channel(raw_data, 'io_serviced_mean')

	# # shape: (batch_size, channel width, #servers, CnnTimeSteps)
	# sys_data = np.concatenate(
	# 	(rps_data, replica_data, cpu_limit_data,
	# 	 cpu_usage_mean_data,
	# 	 # network
	# 	 rx_packets_mean_data,
	# 	 rx_bytes_mean_data,	
	# 	 tx_packets_mean_data, 
	# 	 tx_bytes_mean_data,	
	# 	 # memory
	# 	 rss_mean_data, 
	# 	 cache_mem_mean_data, 
	# 	 page_faults_mean_data,
	# 	 # io
	# 	 io_serviced_mean_data, 
	# 	 io_bytes_mean_data), 
	# 	axis=1)

	sys_data = np.concatenate(
		(rps_data, 
		 replica_data, 
		 cpu_limit_data,
		 cpu_usage_mean_data,
		 # # network
		 # rx_packets_mean_data,
		 # rx_bytes_mean_data,	
		 # tx_packets_mean_data, 
		 # tx_bytes_mean_data,	
		 # memory
		 rss_mean_data, 
		 cache_mem_mean_data
		 # page_faults_mean_data,
		 # # io
		 # io_serviced_mean_data, 
		 # io_bytes_mean_data
		 ), 
		axis=1)


	#----------------------------- next_k data -------------------------------#
	# rps_next_k = compose_next_k_data_channel(raw_data, 'rps')
	cpu_limit_next_k = compose_next_k_data_channel(raw_data, 'cpu_limit')

	# # shape (batch_size, channel width(=2), #servers, LookForward)
	# next_k_info = np.concatenate((cpu_limit_next_k, rps_next_k), axis=1)

	# shape (batch_size, #servers, LookForward)
	next_k_info = np.squeeze(cpu_limit_next_k)

	#----------------------------- latency -------------------------------#
	# lat shape (5, batch_size)
	for i, p in enumerate(['90.0', '95.0', '98.0', '99.0', '99.9']):
		if i == 0:
			lat = np.array(raw_data['latency'][p])
		else:
			lat = np.vstack((lat, raw_data['latency'][p]))

	# lat_data shape (batch_size, 5, CnnTimeSteps)
	for i in range(0, lat.shape[1] - CnnTimeSteps - LookForward + 1):
		if i == 0:
			lat_data = lat[:, i:i+CnnTimeSteps].reshape([1, lat.shape[0], CnnTimeSteps])
		else:
			lat_data = np.vstack((lat_data, lat[:, i:i+CnnTimeSteps].reshape([1, lat.shape[0], CnnTimeSteps])))
	# # lat_data shape (batch_size, 5, CnnTimeSteps) to (batch_size, 5, 1, CnnTimeSteps)
	# # the 3rd dimension 1 is reserved for concatenating per-service latency (not used)
	# lat_data = lat_data.reshape([lat_data.shape[0], lat_data.shape[1], 1, lat_data.shape[2]])

	#------------------------------ label (latency) --------------------------------#
	# the last sample is in terms of time steps from (n - LookForward) to (n - 1), 
	# including the last LookForward - 1 time steps
	# lat_next_k_label shape (batch_size, 5, LookForward)
	for i in range(CnnTimeSteps, lat.shape[1]-LookForward+1):
		if i == CnnTimeSteps:
			lat_next_k_label = lat[:, i:i+LookForward].reshape([1, lat.shape[0], LookForward])
		else:
			lat_next_k_label = np.vstack((lat_next_k_label, 
				lat[:, i:i+LookForward].reshape([1, lat.shape[0], LookForward])))

	# shuffle and split into training & validation set
	shuffle_in_unison([sys_data, lat_data, next_k_info, lat_next_k_label])
	print('sys_data.shape = '), 
	print(sys_data.shape)
	print('lat_data.shape = '),
	print(lat_data.shape)
	print('next_k_info.shape = '),
	print(next_k_info.shape)
	print('lat_next_k_label.shape = '),
	print(lat_next_k_label.shape)

	num_val = int(lat_next_k_label.shape[0] * 0.1)
	sys_data_v = sys_data[:num_val,:,:,:]
	sys_data_t = sys_data[num_val:,:,:,:]

	lat_data_v = lat_data[:num_val,:,:]
	lat_data_t = lat_data[num_val:,:,:]

	next_k_info_v = next_k_info[:num_val,:,:]
	next_k_info_t = next_k_info[num_val:,:,:]

	lat_next_k_label_v = lat_next_k_label[:num_val,:]
	lat_next_k_label_t = lat_next_k_label[num_val:,:]

	if Upsample:
		sys_data_t, lat_data_t, next_k_info_t, lat_next_k_label_t = upsample(
			sys_data_t, lat_data_t, next_k_info_t, lat_next_k_label_t)
		sys_data_v, lat_data_v, next_k_info_v, lat_next_k_label_v = upsample(
			sys_data_v, lat_data_v, next_k_info_v, lat_next_k_label_v)

	return [sys_data_t, lat_data_t, next_k_info_t, lat_next_k_label_t, 
			sys_data_v, lat_data_v, next_k_info_v, lat_next_k_label_v]

def shuffle_in_unison(arr):
	rnd_state = np.random.get_state()
	for a in arr:
		np.random.set_state(rnd_state)
		np.random.shuffle(a)
		np.random.set_state(rnd_state)

def main():
	global LookForward
	global Upsample
	global LogDir
	global SaveDir

	count = 0
	for subdir in os.listdir(LogDir):
		if ("diurnal" in subdir) or ("users" in subdir):
			if len(os.listdir(LogDir+'/'+subdir)) == 0:
				continue
			[sys_data_t, lat_data_t, next_k_info_t, lat_next_k_label_t, 
			 sys_data_v, lat_data_v, next_k_info_v, lat_next_k_label_v] = parse_subdir(LogDir+'/'+subdir+'/')
			if count == 0:
				glob_sys_data_train = sys_data_t
				glob_lat_data_train = lat_data_t
				glob_next_k_info_train	= next_k_info_t
				glob_lat_next_k_label_train  = lat_next_k_label_t

				glob_sys_data_valid = sys_data_v
				glob_lat_data_valid = lat_data_v
				glob_next_k_info_valid	= next_k_info_v
				glob_lat_next_k_label_valid  = lat_next_k_label_v
			else:
				glob_sys_data_train = np.concatenate((glob_sys_data_train,sys_data_t),axis = 0)
				glob_lat_data_train = np.concatenate((glob_lat_data_train,lat_data_t),axis = 0)
				glob_next_k_info_train	= np.concatenate((glob_next_k_info_train, next_k_info_t), axis = 0)
				glob_lat_next_k_label_train	= np.concatenate((glob_lat_next_k_label_train,lat_next_k_label_t), axis = 0)

				glob_sys_data_valid = np.concatenate((glob_sys_data_valid,sys_data_v),axis = 0)
				glob_lat_data_valid = np.concatenate((glob_lat_data_valid,lat_data_v),axis = 0)
				glob_next_k_info_valid	= np.concatenate((glob_next_k_info_valid, next_k_info_v), axis = 0)
				glob_lat_next_k_label_valid	= np.concatenate((glob_lat_next_k_label_valid,lat_next_k_label_v), axis = 0)
			count = count + 1

	print('glob_sys_data_train.shape = '),  
	print(glob_sys_data_train.shape)
	print('glob_lat_data_train.shape = '),  
	print(glob_lat_data_train.shape)
	print('glob_next_k_info_train.shape = '), 
	print(glob_next_k_info_train.shape)
	print('glob_lat_next_k_label_train.shape = '),	
	print(glob_lat_next_k_label_train.shape)

	if SaveDir == '':
		SaveDir = './hotel_data_next_' + str(LookForward) + 's/'
	if not os.path.isdir(SaveDir):
		os.makedirs(SaveDir)

	np.save(SaveDir + "/sys_data_train", glob_sys_data_train)
	np.save(SaveDir + "/sys_data_valid", glob_sys_data_valid)

	np.save(SaveDir + "/lat_data_train", glob_lat_data_train)
	np.save(SaveDir + "/lat_data_valid", glob_lat_data_valid)

	np.save(SaveDir + "/nxt_k_data_train", glob_next_k_info_train)
	np.save(SaveDir + "/nxt_k_data_valid", glob_next_k_info_valid)

	np.save(SaveDir + "/nxt_k_train_label",  glob_lat_next_k_label_train)
	np.save(SaveDir + "/nxt_k_valid_label",  glob_lat_next_k_label_valid)

if __name__ == '__main__':
	main()
