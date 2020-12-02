import sys
import time
import socket
import threading
import json
import logging

from pathlib import Path
sys.path.append(str(Path.cwd()))
from util import *

def setup_server_slave(stack_name, username, server, server_cpus, slave_port, 
		slave_script_dir, service_config_path, quiet):
	cwd = str(Path.cwd())

	# scp to remote nodes
	scp(source=str(service_config_path), target=username+'@'+server+':'+ str(service_config_path),
		identity_file='')

	slave_cmd = 'cd ' + cwd + ';'
	if stack_name == '':
		stack_name = '\'\''
	slave_cmd += 'python3 ' + str(slave_script_dir / 'slave_data_collect_ath_social.py') + \
				' --stack-name ' + str(stack_name) + \
				' --cpus ' + str(server_cpus) + \
				' --server-port ' + str(slave_port) + \
				' --service-config ' + str(service_config_path)
	p = ssh(username=username, host=server, cmd=slave_cmd, quiet=quiet)
	return p
	# cmd = 'ssh -q -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no ' + \
	# 	Username + '@' + server + ' \"' + slave_cmd + '\"'
	# logging.info(cmd)

	# subprocess.call(cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr)

def setup_slaves(stack_name, username, servers, 
		slave_port, slave_script_dir, service_config, quiet=False):
	config_path = Path.cwd() / 'config' / 'slave_config.json'
	with open(str(config_path), 'w+') as f:
		json.dump(service_config, f)

	p_list = []
	for server in servers:
		# ip_addr = servers[server]['ip_addr']
		server_cpus = servers[server]['cpus']
		p = setup_server_slave(
			stack_name=stack_name,
			username=username, 
			server=server,
			server_cpus=server_cpus,
			slave_port=slave_port,
			slave_script_dir=slave_script_dir, 
			service_config_path=config_path,
			quiet=quiet)
		p_list.append(p)
	return p_list

def connect_slave(servers, slave_port, slave_socks):
	for server in servers:
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.connect((servers[server]['ip_addr'], slave_port))
		sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
		slave_socks[server] = sock
		logging.info('%s connected' %server)
	time.sleep(5)

def send_server_terminate_exp(server, sock):
	cmd = 'terminate_exp\n'
	sock.sendall(cmd.encode('utf-8'))
	msg = ''
	exp_done = False
	while True:
		data = sock.recv(1024).decode('utf-8')
		msg += data
		while '\n' in msg:
			(cmd, rest) = msg.split('\n', 1)
			msg = rest
			logging.info('recv %s from %s' %(cmd, server))
			if cmd == 'experiment_done':
				exp_done = True
				break
		if exp_done:
			break

def send_terminate_exp(servers, slave_socks):
	t_list = []
	for server in servers:
		t = threading.Thread(target=send_server_terminate_exp, kwargs={
			'server': server, 
			'sock':slave_socks[server]
		})
		t_list.append(t)
		t.start()

	for t in t_list:
		t.join()
	logging.info('experiment fully terminated')

def send_terminate_slave(servers, slave_socks):
	cmd = 'terminate_slave\n'
	for server in servers:
		slave_socks[server].sendall(cmd.encode('utf-8'))

def send_server_init_data(server, sock):
	cmd = 'init_data\n'
	sock.sendall(cmd.encode('utf-8'))

	msg = ''
	init_data_done = False
	while True:
		data = sock.recv(1024).decode('utf-8')
		msg += data
		while '\n' in msg:
			(cmd, rest) = msg.split('\n', 1)
			msg = rest
			logging.info('recv %s from %s' %(cmd, server))
			if cmd == 'init_data_done':
				init_data_done = True
				break
		if init_data_done:
			break

# send init_data request to all slaves
def send_init_data(servers, slave_socks):
	t_list = []
	for server in servers:
		t = threading.Thread(target=send_server_init_data, kwargs={
			'server': server,
			'sock': slave_socks[server]
		})
		t_list.append(t)
		t.start()

	for t in t_list:
		t.join()
	logging.info('send_init_data done')

def send_exp_start(servers, slave_socks):
	cmd = 'exp_start\n'
	for server in servers:
		slave_socks[server].sendall(cmd.encode('utf-8'))

def send_server_rsc_config(cpu_config_json, sock):
	cmd = 'set_rsc----' + cpu_config_json + '\n'
	sock.sendall(cmd.encode('utf-8'))

def send_rsc_config(cpu_config, servers, slave_socks):
	t_list = []
	# compose cpu config json
	rsc_config = {}
	# in swarm node we don't know which node service is placed on
	for service in cpu_config:
		if service == 'jaeger' or service == 'zipkin':
			continue
		rsc_config[service] = {}
		# rsc_config[service]['core'] = ServiceConfig[service]['core']
		rsc_config[service]['cpus']  = cpu_config[service]
		# rsc_config[service]['freq'] = ServiceConfig[service]['freq']
	rsc_json = json.dumps(rsc_config)

	for server in servers:
		t = threading.Thread(target=send_server_rsc_config, kwargs={
			'cpu_config_json': rsc_json,
			'sock': slave_socks[server]
		})
		t_list.append(t)
		t.start()

	for t in t_list:
		t.join()

def _init_record(record, service):
	assert service not in record
	record[service] = {}
	record[service]['replica']    = 0

	record[service]['cpu_usage']   =  []
	record[service]['rss'] 		  =  []
	record[service]['cache_mem']  =  []
	record[service]['page_faults'] = []
	record[service]['rx_packets']  = []
	record[service]['rx_bytes']    = []
	record[service]['tx_packets']  = []
	record[service]['tx_bytes']    = []
	record[service]['io_bytes']  = []
	record[service]['io_serviced'] = []

def get_server_slave_metric(server, record, sock):
	sock.sendall(('get_info\n').encode('utf-8'))
	msg = ''
	while True:
		msg += (sock.recv(1024)).decode('utf-8')
		if '\n' not in msg:
			continue
		else:
			metric = json.loads(msg.split('\n')[0])
			# debug
			logging.info('recv metric from %s' %server)
			for service in metric:
				if service == 'jaeger':
					continue
				if service not in record:
					_init_record(record, service)

				record[service]['replica']     += metric[service]['replica']
				record[service]['cpu_usage']   += metric[service]['cpu_docker']
				record[service]['rss'] 		   += metric[service]['rss']
				record[service]['cache_mem']   += metric[service]['cache_mem']
				record[service]['page_faults'] += metric[service]['pgfault']
				record[service]['rx_packets']  += metric[service]['rx_pkt']
				record[service]['rx_bytes']    += metric[service]['rx_byte']
				record[service]['tx_packets']  += metric[service]['tx_pkt']
				record[service]['tx_bytes']    += metric[service]['tx_byte']
				record[service]['io_bytes']    += metric[service]['io_bytes']
				record[service]['io_serviced'] += metric[service]['io_serv']
			break

def get_slave_metric(servers, slave_socks):
	records = {}
	for server in servers:
		records[server] = {} 

	t_list = []
	for server in servers:
		t = threading.Thread(target=get_server_slave_metric, kwargs={
			'server': server,
			'record': records[server],
			'sock': slave_socks[server]
		})
		t_list.append(t)
		t.start()

	for t in t_list:
		t.join()

	stats_accum = {}
	# merge stats from multiple servers into one
	for server in servers:
		record = records[server]
		for service in record:
			if record[service]['replica'] == 0:
				logging.warning("get_slave_metric: server %s service %s replica=0" %(server, service))
				continue
			if service not in stats_accum:
				_init_record(stats_accum, service)
			stats_accum[service]['replica']     += record[service]['replica']
			stats_accum[service]['cpu_usage']   += record[service]['cpu_usage']
			stats_accum[service]['rss'] 		+= record[service]['rss']
			stats_accum[service]['cache_mem']   += record[service]['cache_mem']
			stats_accum[service]['page_faults'] += record[service]['page_faults']
			stats_accum[service]['rx_packets']  += record[service]['rx_packets']
			stats_accum[service]['rx_bytes']    += record[service]['rx_bytes']
			stats_accum[service]['tx_packets']  += record[service]['tx_packets']
			stats_accum[service]['tx_bytes']    += record[service]['tx_bytes']
			stats_accum[service]['io_bytes']  += record[service]['io_bytes']
			stats_accum[service]['io_serviced'] += record[service]['io_serviced']

	return stats_accum

def send_server_update_replica(server, sock, service_list):
	cmd = 'update_replica----' + json.dumps(service_list) + '\n'
	sock.sendall(cmd.encode('utf-8'))

	msg = ''
	update_replica_done = False
	while True:
		data = sock.recv(1024).decode('utf-8')
		msg += data
		while '\n' in msg:
			(cmd, rest) = msg.split('\n', 1)
			msg = rest
			logging.info('recv %s from %s' %(cmd, server))
			if cmd == 'update_replica_done':
				update_replica_done = True
				break
		if update_replica_done:
			break

def send_update_replica(servers, slave_socks, service_list):
	assert len(service_list) > 0
	t_list = []
	for server in servers:
		t = threading.Thread(target=send_server_update_replica, kwargs={
			'server': server,
			'sock': slave_socks[server],
			'service_list': service_list
		})
		t_list.append(t)
		t.start()

	for t in t_list:
		t.join()
	logging.info('send_update_replica done')