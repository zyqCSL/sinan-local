load_arr_len = 51
load_intervals = []
load_rates = []

if __name__ == '__main__':
	i = 0
	time = 0
	cur_val = 1000
	while cur_val <= 6000:
		load_rates.append(str(cur_val))
		if cur_val <= 1500:
			load_intervals.append('45s')
			time += 45
			cur_val +=250
		elif cur_val >= 5500:
			load_intervals.append('20s')
			time += 20
			cur_val += 250
		else:
			load_intervals.append('2s')
			cur_val += 250
			time += 2
		i += 1

	i = 0
	cur_val -= 500
	print cur_val
	while cur_val >= 1500:
		load_rates.append(str(cur_val))
		if cur_val <= 1800:
			load_intervals.append('45s')
			time += 45
			cur_val -= 250
		elif cur_val >= 5000:
			load_intervals.append('20s')
			time += 20
			cur_val -= 250
		else:
			load_intervals.append('2s')
			cur_val -= 250
			time += 2
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