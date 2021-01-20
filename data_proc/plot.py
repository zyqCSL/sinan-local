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
colors = ['orange', 'yellow', 'darkcyan', 'violet', 'pink', 'aqua', 'purple', 
	'deepskyblue', 'teal', 'navy', 'darkorchid', 'crimson', 'olive', 'slateblue']

colors = ['orange', 'darkcyan', 'violet', 'pink', 'aqua', 'purple', 
	'deepskyblue', 'teal', 'navy', 'darkorchid', 'crimson', 'olive', 'slateblue']
# random.shuffle(colors)


def plot_dir(dir_name):
	global services
	global QoS
	global colors

	dir_levels =[i for i in dir_name.split('/') if i != '' and i != ' ']
	last_level_dir = dir_levels[-1]

	fig_name = './' + last_level_dir + '.png'

	cpu_limit = {}
	for t in services:
		cpu_limit[t] = []

	rps = np.loadtxt(dir_name + '/rps.txt')
	true_lat = np.loadtxt(dir_name + '/e2e_lat_99.0.txt')
	pred_lat = np.loadtxt(dir_name + '/pred_lat.txt')
	pred_viol = np.loadtxt(dir_name + '/pred_viol.txt')

	for t in services:
		cpu_limit[t] = np.loadtxt(dir_name + '/cpu_limit_' + t + '.txt')
	qos = [QoS] * rps.shape[0]
	time = list(range(1, rps.shape[0] + 1))

	my_dpi = 144
	plt.figure(figsize=(3000/my_dpi, 2100/my_dpi), dpi=my_dpi)

	ax = plt.subplot(2, 2, 1)
	title_str = 'Load(RPS)'
	# plt.title(title_str, fontsize=18, y = 0.9)
	plt.title(title_str, fontsize=18)
	# plt.xlabel('time(s)', fontsize=18)
	# ax.xaxis.set_label_coords(0.50, 0.08)
	plt.ylabel('rps', fontsize=18)
	# plt.yticks(np.arange(20, 100, 10))
	plt.plot(time, rps, linestyle = '-', color = 'red', linewidth = 1.0)
	plt.tick_params(
    	axis='x',          # changes apply to the x-axis
    	which='both',      # both major and minor ticks are affected
    	bottom=False,      # ticks along the bottom edge are off
    	top=False,         # ticks along the top edge are off
    	labelbottom=False) # labels along the bottom edge are off

	plt.tick_params(axis='y', which='major', labelsize=18)
	plt.grid(color='lavender')

	ax = plt.subplot(2, 2, 3)
	title_str = 'True vs Predicted 99% Tail Latency'
	# plt.title(title_str, fontsize=18, y = 0.9)
	plt.title(title_str, fontsize=18)
	plt.xlabel('time(s)', fontsize=18)
	ax.xaxis.set_label_coords(0.50, 0.08)
	plt.ylabel('ms', fontsize=18)
	# plt.ylim(0, target * 2)
	# print pred_lat
	plt.step(time, qos,  linestyle = '-', label = "qos", color = 'green', linewidth = 1.0)
	plt.step(time, true_lat,  linestyle = '-', label = "true", color = 'blue', linewidth = 1.5)
	plt.step(time, pred_lat,  linestyle = '-', label = "pred", color = 'red', linewidth = 1.5)
	plt.tick_params(
    	axis='x',          # changes apply to the x-axis
    	which='both',      # both major and minor ticks are affected
    	bottom=False,      # ticks along the bottom edge are off
    	top=False,         # ticks along the top edge are off
    	labelbottom=False) # labels along the bottom edge are off

	plt.tick_params(axis='y', which='major', labelsize=18)
	plt.grid()
	# ax.set_facecolor('xkcd:white')
	plt.legend(prop={'size': 10})
	# ax.legend(loc='lower right', bbox_to_anchor=(0.0, 0.0), fancybox=True, fontsize=15, ncol=2, frameon=True)

	# cpu_limit
	ax = plt.subplot(2, 2, 2)
	title_str = 'cpu_limit'
	# plt.title(title_str, fontsize=18, y = 0.9)
	plt.title(title_str, fontsize=18)
	# plt.xlabel('time(s)', fontsize=18)
	# ax.xaxis.set_label_coords(0.50, 0.08)
	plt.ylabel('cpu_limit', fontsize=18)
	for i, t in enumerate(services):
		# print t
		# print cpu_limit
		# print colors
		if i % 3 == 0:
			plt.plot(time, cpu_limit[t], linestyle = '-', label = t, color = colors[i/3], linewidth = 1.0)
		elif i % 3 == 1:
			plt.plot(time, cpu_limit[t], linestyle = '-.', label = t, color = colors[i/3], linewidth = 1.0)
		else:
			plt.plot(time, cpu_limit[t], linestyle = ':', label = t, color = colors[i/3], linewidth = 1.0)

	plt.legend(prop={'size': 10})
	# ax.legend(loc='upper right', bbox_to_anchor=(0.0, 0.0), fancybox=True, fontsize=15, ncol=2, frameon=True)
	# plt.tick_params(
 #    	axis='x',          # changes apply to the x-axis
 #    	which='both',      # both major and minor ticks are affected
 #    	bottom=False,      # ticks along the bottom edge are off
 #    	top=False,         # ticks along the top edge are off
 #    	labelbottom=False) # labels along the bottom edge are off
 	plt.tick_params(axis='both', which='major', labelsize=18)
 	plt.grid(color='lavender')


 	# pred_viol
	ax = plt.subplot(2, 2, 4)
	title_str = 'pred_viol'
	# plt.title(title_str, fontsize=18, y = 0.9)
	plt.title(title_str, fontsize=18)
	plt.xlabel('time(s)', fontsize=18)
	ax.xaxis.set_label_coords(0.50, 0.08)
	plt.step(time, pred_viol,  linestyle = '-', label = "pred_viol", color = 'red', linewidth = 1.0)

	plt.legend(prop={'size': 10})
	# ax.legend(loc='upper right', bbox_to_anchor=(0.0, 0.0), fancybox=True, fontsize=15, ncol=2, frameon=True)
	# plt.tick_params(
 #    	axis='x',          # changes apply to the x-axis
 #    	which='both',      # both major and minor ticks are affected
 #    	bottom=False,      # ticks along the bottom edge are off
 #    	top=False,         # ticks along the top edge are off
 #    	labelbottom=False) # labels along the bottom edge are off
 	plt.tick_params(axis='both', which='major', labelsize=18)
 	plt.grid(color='lavender')

 	plt.savefig(fig_name, dpi=my_dpi, bbox_inches='tight', pad_inches = 0)
 	plt.close()


def test():
	dir_name = sys.argv[1]
	plot_dir(dir_name)

def main():
	top_dir_name = sys.argv[1]
	if 'diurnal' not in top_dir_name:
		for dir_n in os.listdir(top_dir_name):
			if 'users' not in dir_n:
				continue
			full_path = top_dir_name + '/' + dir_n + '/'
			plot_dir(full_path)
	else:
		plot_dir(top_dir_name)

if __name__ == '__main__':
	#test()
	main()



			





