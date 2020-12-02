import sys
import time
import socket
import threading
import json
import logging

from pathlib import Path
sys.path.append(str(Path.cwd()))
from util import *

def setup_gpu(username, gpu_host, work_dir, script, gpus):
	slave_cmd = 'cd ' + work_dir + ';'
	if len(gpus) > 0:
		slave_cmd += 'python ' + script + ' --gpus ' + ','.join([str(g) for g in gpus])
	else:
		slave_cmd += 'python ' + script
	# slave_cmd += 'python social_media_predictor.py --gpus ' + ','.join([str(g) for g in gpus])
	print(slave_cmd)
	p = ssh(username=username, host=gpu_host, cmd=slave_cmd)
	print('--------waiting for gpu setup---------')
	time.sleep(20)
	print('--------gpu setup---------')
	return p

def connect_gpu(gpu_host, gpu_port):
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	sock.connect((gpu_host, gpu_port))
	sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
	logging.info('gpu connected')
	return sock

def get_ml_prediction(info, gpu_sock):
	cmd = 'pred----' + json.dumps(info) + '\n'
	gpu_sock.sendall(cmd.encode('utf-8'))

	msg = ''
	while True:
		data = gpu_sock.recv(2048).decode('utf-8')
		msg += data
		while '\n' in msg:
			(pred, rest) = msg.split('\n', 1)
			msg = rest
			# logging.info('recv %s from gpu' %(cmd))
			if 'pred----' in pred:
				pred = pred.split('pred----')[-1]
				return json.loads(pred)

def send_terminate_gpu(gpu_sock):
	cmd = 'terminate\n'
	gpu_sock.sendall(cmd.encode('utf-8'))
	msg = ''
	exp_done = False
	while True:
		data = gpu_sock.recv(1024).decode('utf-8')
		msg += data
		while '\n' in msg:
			(cmd, rest) = msg.split('\n', 1)
			msg = rest
			if cmd == 'experiment_done':
				exp_done = True
				break
		if exp_done:
			break