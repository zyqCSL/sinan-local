import sys
fn = sys.argv[1]

with open(fn, 'r') as f:
	lines = f.readlines()
	pos_99 = None
	pos_fps = None
	pos_rps = None
	pos_fail = None
	for i, l in enumerate(lines):
		if i == 0:
			items = l.split(',')
			for i, k in enumerate(items):
				k = k.replace('\"', '')
				if k == '99%':
					pos_99 = i
				elif k == 'Requests/s':
					pos_rps = i
				elif k == 'Failures/s':
					pos_fps = i
				elif k == 'Total Failure Count':
					pos_fail = i

			print('rps\tfps\ttail_99\ttotal_failure')
			assert pos_99 != None
			assert pos_fps != None
			assert pos_rps != None
			assert pos_fail != None
		else:
			items = l.split(',')
			print('%s\t%s\t%s\t%s\t' %(items[pos_rps],
				items[pos_fps], items[pos_99], items[pos_fail]))
