load_arr_len = 51
load_intervals = []
load_rates = []

if __name__ == '__main__':
	i = 0
	time = 0
	cur_val = 2000
	while cur_val <= 14000:
		load_rates.append(str(cur_val))
		load_intervals.append('10s')
		cur_val += 1000

	load_rates.append('14000')
	load_intervals.append('10s')

	while cur_val >= 2000:
		load_rates.append(str(cur_val))
		load_intervals.append('10s')
		cur_val -= 1000

	print load_rates
	print load_intervals

	print len(load_rates)
	print len(load_intervals)
	print time

	time = 0
	i = 0 
	while i < len(load_rates):
		time += int(load_intervals[i].split('s')[0])
		print load_rates[i] + ', end time: ' + str(time)
		i += 1