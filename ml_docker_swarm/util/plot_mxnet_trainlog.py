import matplotlib
matplotlib.use('Agg')
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import random


def plot_log(log_file):
	# INFO:root:Epoch[145] Validation-RMSE=37.798640
	# INFO:root:Epoch[146] Train-RMSE=31.338707
	# INFO:root:Epoch[146] Time cost=0.953
	# INFO:root:Saved checkpoint to "./model/cnv-0147.params"
	# INFO:root:Epoch[146] Validation-RMSE=37.891041
	# INFO:root:Update[7501]: Change learning rate to 1.00000e-05

	epochs = []
	train_rmse = []
	valid_rmse = []
	epoch_change_rate = []
	cur_epoch = 0
	with open(log_file, 'r') as f:
		lines = f.readlines()
		for line in lines:
			if 'Validation-RMSE' in line:
				epoch = int(line.split('[')[-1].split(']')[0])
				cur_epoch = epoch
				if cur_epoch not in epochs:
					epochs.append(cur_epoch)
				rmse = float(line.split('Validation-RMSE=')[-1])
				valid_rmse.append(rmse)
			elif 'Train-RMSE' in line:
				epoch = int(line.split('[')[-1].split(']')[0])
				cur_epoch = epoch
				if cur_epoch not in epochs:
					epochs.append(cur_epoch)
				rmse = float(line.split('Train-RMSE=')[-1])
				train_rmse.append(rmse)

			elif 'Change learning rate to' in line:
				epoch_change_rate.append(cur_epoch)

	fig_name = './mxnet_train_log.png'

	my_dpi = 144
	plt.figure(figsize=(3000/my_dpi, 2100/my_dpi), dpi=my_dpi)

	ax = plt.subplot(1, 1, 1)
	title_str = 'train-rmse vs valid-rmse'
	# plt.title(title_str, fontsize=18, y = 0.9)
	plt.title(title_str, fontsize=18)
	# plt.xlabel('time(s)', fontsize=18)
	# ax.xaxis.set_label_coords(0.50, 0.08)
	plt.ylabel('rmse', fontsize=18)
	# plt.yticks(np.arange(20, 100, 10))
	plt.plot(epochs, train_rmse, linestyle = '-', color = 'red', linewidth = 1.0)
	plt.plot(epochs, valid_rmse, linestyle = '-', color = 'blue', linewidth = 1.0)
	for e in epoch_change_rate:
		plt.axvline(x=e, color='green')
	plt.tick_params(axis='y', which='major', labelsize=18)
	plt.tick_params(axis='x', which='major', labelsize=18)
	plt.grid(color='lavender')

 	plt.savefig(fig_name, dpi=my_dpi, bbox_inches='tight', pad_inches = 0)
 	plt.close()

if __name__ == '__main__':
	log_file = sys.argv[1]
	plot_log(log_file)



			





