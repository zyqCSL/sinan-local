import matplotlib
matplotlib.use('Agg')
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import random

services   = ['compose-post-redis',
              'compose-post-service',
              'home-timeline-redis',
              'home-timeline-service',
              # 'jaeger',
              'nginx-thrift',
              'post-storage-memcached',
              'post-storage-mongodb',
              'post-storage-service',
              'social-graph-mongodb',
              'social-graph-redis',
              'social-graph-service',
              'text-service',
              'text-filter-service',
              'unique-id-service',
              'url-shorten-service',
              'media-service',
              'media-filter-service',
              'user-mention-service',
              'user-memcached',
              'user-mongodb',
              'user-service',
              'user-timeline-mongodb',
              'user-timeline-redis',
              'user-timeline-service',
              'write-home-timeline-service',
              'write-home-timeline-rabbitmq',
              'write-user-timeline-service',
              'write-user-timeline-rabbitmq'] 

QoS   = 500

def count_dir(dir_name, user, stats):
	global services
	global QoS

	stats[user] = {}

	cpu_limit = {}
	cpu_limit_sum = 0
	cpu_usage = {}
	cpu_util = {}
	exp_time = 0
	for t in services:
		cpu_limit[t] = np.loadtxt(dir_name + '/cpu_limit_' + t + '.txt')
		cpu_usage[t] = np.loadtxt(dir_name + '/cpu_usage_sum_' + t + '.txt')
		cpu_limit_sum += np.sum(cpu_limit[t])
		cpu_util[t] = []
		for i in range(0, cpu_limit[t].shape[0]):
			cpu_util[t].append(cpu_usage[t][i]/cpu_limit[t][i])

		if exp_time == 0:
			exp_time = len(cpu_limit[t])
		else:
			exp_time = min(exp_time, len(cpu_limit[t]))

	rps = np.loadtxt(dir_name + '/rps.txt')
	true_lat = np.loadtxt(dir_name + '/e2e_lat_99.0.txt')
	# pred_lat = np.loadtxt(dir_name + '/pred_lat.txt')
	# pred_viol = np.loadtxt(dir_name + '/pred_viol.txt')

	exp_t = len(true_lat)
	stats[user]['viol_rate'] = float(np.sum(true_lat > QoS))/exp_t
	stats[user]['lat_mean'] = true_lat.mean()
	stats[user]['lat_max'] = true_lat.max()
	stats[user]['lat_min'] = true_lat.min()
	stats[user]['lat_std'] = np.std(true_lat)
	stats[user]['cpu_mean'] = cpu_limit_sum/exp_t

	cpu_max = 0
	for i in range(60, exp_time-60):
		total = 0
		for t in services:
			total += cpu_limit[t][i]
		cpu_max = max(total, cpu_max)

	stats[user]['cpu_max'] = cpu_max

	stats[user]['cpu_util'] = {}
	for t in services:
		stats[user]['cpu_util'][t] = np.mean(cpu_util[t])


def test():
	dir_name = sys.argv[1]
	count_dir(dir_name)

def main():
	top_dir_name = sys.argv[1]
	stats = {}
	if 'diurnal' in top_dir_name:
		count_dir(top_dir_name, 0, stats)
	else:
		for dir_n in os.listdir(top_dir_name):
			if 'user' in dir_n:
				user = int(dir_n.split('users_')[-1])
			else:
				continue
			full_path = top_dir_name + '/' + dir_n + '/'
			count_dir(full_path, user, stats)

	print('user\tviol_rate\tlat_mean\tlat_max\tlat_min\tlat_std\tcpu_mean\tcpu_max')
	for u in sorted(list(stats.keys())):
		print('%d\t%.3f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f' %(u ,
			stats[u]['viol_rate'],
			stats[u]['lat_mean'],
			stats[u]['lat_max'],
			stats[u]['lat_min'],
			stats[u]['lat_std'],
			stats[u]['cpu_mean'],
			stats[u]['cpu_max']))

	print('\n')
	for u in sorted(list(stats.keys())):
		print('user ', u)
		for t in services:
			print('%s: %.3f' %(t, stats[u]['cpu_util'][t]))

if __name__ == '__main__':
	#test()
	main()



			





