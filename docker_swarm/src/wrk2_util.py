import sys
import os
import subprocess
import time
import logging

def run_wrk2(wrk2,  lua_script, nginx_ip,
		dist='exp', tail=95, tail_resolution=0.5, stats_rate=0.2, tail_report_interval=1,
		num_threads=10, num_conns=300, duration=0, reqs_per_sec=6000, output=None,
		quiet=False):
	# _stdout = subprocess.PIPE
	_stdout = sys.stdout
	_stderr = sys.stderr
	if quiet:
		_stdout = subprocess.DEVNULL
		_stderr = subprocess.DEVNULL

	if output != None:
		_stdout=output

	wrk2_proc = None
	if duration == 0 or reqs_per_sec == 'script':
		# duration decided by script
		wrk2_proc = subprocess.Popen([str(wrk2),
			'-L', 
			'-D', str(dist), 
			'-p', str(tail),
			'-r', str(tail_resolution),
			'-S', str(stats_rate),
			'-i', str(tail_report_interval),
			'-t', str(num_threads),
			'-c', str(num_conns),
			# '-d', str(duration) + 's',
			'-s', str(lua_script),
			nginx_ip,
			'-R', 'script'],
			stdout=_stdout,
			stderr=_stderr,
			bufsize=1,
			universal_newlines=True)
	else:
		# duration decided by argument
		wrk2_proc = subprocess.Popen([str(wrk2),
			'-L', 
			'-D', str(dist), 
			'-p', str(tail),
			'-r', str(tail_resolution),
			'-S', str(stats_rate),
			'-i', str(tail_report_interval),
			'-t', str(num_threads),
			'-c', str(num_conns),
			'-d', str(duration) + 's',
			'-s', str(lua_script),
			nginx_ip,
			'-R', str(reqs_per_sec)],
			stdout=_stdout,
			stderr=_stderr,
			bufsize=1,
			universal_newlines=True)
	assert(wrk2_proc != None)
	return wrk2_proc

def get_wrk2_data(feature, wrk2_last_time, wrk2_pt):
	# wait for wrk2 log to update
	# while True:
	# 	mtime = os.path.getmtime(Wrk2LogPath)
	# 	if mtime == wrk2_last_time:
	# 		time.sleep(0.05)
	# 		print 'Wrk2 mtime = ', mtime
	# 		continue
	# 	else:
	# 		wrk2_last_time = os.path.getmtime(Wrk2LogPath)
	# 		break

	while True:
		first_time = False
		with open(str(wrk2_pt), 'r') as f:
			lines = f.readlines()
			if len(lines) == 0:
				first_time = True
				time.sleep(0.1)
				continue
			else:
				if first_time:
					wrk2_last_time = os.path.getmtime(str(wrk2_pt))
				line = lines[-1]
				data = line.split(';')

				lat  = 0
				load = 0
				xput = 0
				for item in data:
					t = item.split(':')
					if t[0] != 'xput':
						percent_key = round(float(t[0]), 1)
						lat = round(int(t[1])/1000.0, 2)
						feature.end_to_end_lat[percent_key] = lat
					else:
						feature.xput = int(t[1])

					if '99.00' in t[0]:
						lat = round(int(t[1])/1000.0, 2)	# turn us to ms
					if 'xput' in t[0]:
						xput = int(t[1])

				# file_handle.write('time:' + str(timestamp) +'s: ' + line + '\n')
				return lat, xput

def warmup_app(wrk2, benchmark_dir, benchmark):
	if 'social' in benchmark:
		wrk2_proc = run_wrk2(wrk2=wrk2,
			lua_script=str(
				benchmark_dir / 'wrk2' / 'scripts' / 'social-network' / 'mixed-workload.lua'),
			nginx_ip='http://127.0.0.1:8080',
			dist='exp', tail=95, tail_resolution=0.5, stats_rate=0.2, tail_report_interval=1,
			num_threads=10, num_conns=300, duration=30, reqs_per_sec=100,
			quiet=False)
		wrk2_proc.wait()
		time.sleep(10)