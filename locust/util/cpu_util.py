import sys
import os
from pathlib import Path
import time

stat = Path('/proc/stat')
interval = 1

total_time = 0
idle_time = 0
prev_record = []

def check_proc_stat():
	global idle_time
	global total_time
	with open(str(stat), 'r') as f:
		lines = f.readlines()
		data = [int(k) for k in lines[0].split(' ') if k != '' and k != '\n' and k != 'cpu']
		total = sum(data)
		idle = data[3]
		
		return total, idle

def compute_cpu_util(prev_idle, prev_total, idle, total):
	return 1.0 - (idle - prev_idle) / (total - prev_total)