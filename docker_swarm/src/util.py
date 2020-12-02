import sys
import subprocess
import logging
import math

#-------------- system ----------------#
def ssh(username, host, cmd, quiet=False):
	if quiet:
		ssh_cmd = 'ssh -q -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no ' + \
			username + '@' + host + ' \"' + cmd + '\"'
	else:
		ssh_cmd = 'ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no ' + \
			username + '@' + host + ' \"' + cmd + '\"'
	logging.info(ssh_cmd)
	p = subprocess.Popen(ssh_cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr)
	return p

def scp(source, target, identity_file, quiet=False):
	_stdout = None
	_stderr = None
	if quiet:
		_stdout = subprocess.DEVNULL
		_stderr = subprocess.DEVNULL

	if identity_file != '':
		cmd = 'scp -r -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ' + \
			'-i ' + str(identity_file) + ' ' + str(source) + ' ' + str(target)
		subprocess.run(cmd, shell=True, stdout=_stdout, stderr=_stderr)
	else:
		cmd = 'scp -r -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ' + \
			str(source) + ' ' + str(target)
		subprocess.run(cmd, shell=True, stdout=_stdout, stderr=_stderr)


def rsync(source, target, identity_file, quiet=False):
	_stdout = None
	_stderr = None
	if quiet:
		_stdout = subprocess.DEVNULL
		_stderr = subprocess.DEVNULL
	cmd = 'rsync -arz --info=progress2 -e ' + \
		'\"ssh -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ' + \
		'-i ' + str(identity_file) + '\" ' + \
		str(source) + ' ' + str(target)
	subprocess.run(cmd, shell=True, stdout=_stdout, stderr=_stderr)

#--------------------- math -----------------------#
def ceil_float(num, precision):
	num *= 10**precision
	num = int(math.ceil(num))
	return num/(10**precision)