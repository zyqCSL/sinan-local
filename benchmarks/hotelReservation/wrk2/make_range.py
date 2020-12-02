load_arr_len = 51
load_intervals = []
load_rates = []

if __name__ == '__main__':
	i = 0
	time = 0
	cur_val = 25
	while i <= 24:
		load_rates.append(str(cur_val) + 'k')
		if cur_val < 27:
			load_intervals.append('45s')
			time += 45
			cur_val +=2
		elif cur_val >= 66:
			load_intervals.append('20s')
			time += 20
			cur_val += 2
		else:
			load_intervals.append('4s')
			cur_val += 2
			time += 4
		i += 1

	i = 0
	cur_val -= 4
	print cur_val
	while i < 24:
		load_rates.append(str(cur_val) + 'k')
		if cur_val < 29:
			load_intervals.append('45s')
			time += 45
			cur_val -= 2
		elif cur_val >= 66:
			load_intervals.append('20s')
			time += 20
			cur_val -= 2 
		else:
			load_intervals.append('4s')
			cur_val -= 2
			time += 4
		i += 1

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