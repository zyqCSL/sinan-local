load_arr_len = 51
load_intervals = []
load_rates = []

if __name__ == '__main__':
	i = 0
	time = 0
	cur_val = 25
	while cur_val <= 90:
		load_rates.append(str(cur_val) + 'k')
		if cur_val < 26:
			load_intervals.append('30s')
			time += 30
			cur_val +=4
		elif cur_val >= 80:
			load_intervals.append('15s')
			time += 15
			cur_val += 2
		else:
			load_intervals.append('4s')
			cur_val += 3
			time += 4
		i += 1

	i = 0
	cur_val -= 4
	print cur_val
	while cur_val > 15:
		load_rates.append(str(cur_val) + 'k')
		if cur_val < 30:
			load_intervals.append('30s')
			time += 30
			cur_val -= 4
		elif cur_val >= 80:
			load_intervals.append('15s')
			time += 15
			cur_val -= 2 
		else:
			load_intervals.append('4s')
			cur_val -= 3
			time += 4
		i += 1

	print load_rates
	print load_intervals

	print len(load_rates)
	print len(load_intervals)
	print time