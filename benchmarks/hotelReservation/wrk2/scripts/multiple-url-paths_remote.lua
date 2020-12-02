
require "socket"
time = socket.gettime()*1000
math.randomseed(time)
math.random(); math.random(); math.random()

-- diurnal pattern all on ath8
-- load_arr_len = 49
-- load_intervals = {'45s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '20s', '20s', '20s', '20s', '20s', '20s', '20s', '20s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '45s'}
-- load_rates = {'25k', '28k', '32k', '36k', '40k', '44k', '48k', '52k', '56k', '60k', '64k', '68k', '72k', '76k', '80k', '84k', '88k', '92k', '96k', '100k', '104k', '108k', '110k', '112k', '114k', '112k', '110k', '108k', '106k', '104k', '100k', '96k', '92k', '88k', '84k', '80k', '76k', '72k', '68k', '64k', '60k', '56k', '52k', '48k', '44k', '40k', '36k', '32k', '28k'}
-- load_intervals = {'60s', '60s', '60s', '20s', '60s', '10s', '10s'}
-- load_rates = {'50k', '80k', '100k', '120k', '110k', '60k', '90k'}

-- diurnal pattern for remote

load_arr_len = 49
load_intervals = {'30s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '15s', '15s', '15s', '15s', '15s', '15s', '15s', '15s', '15s', '15s', '15s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '4s', '30s', '30s', '30s'}
load_rates = {'25k', '29k', '32k', '35k', '38k', '41k', '44k', '47k', '50k', '53k', '56k', '59k', '62k', '65k', '68k', '71k', '74k', '77k', '80k', '82k', '84k', '86k', '88k', '90k', '88k', '86k', '84k', '82k', '80k', '78k', '75k', '72k', '69k', '66k', '63k', '60k', '57k', '54k', '51k', '48k', '45k', '42k', '39k', '36k', '33k', '30k', '27k', '23k', '19k'}

counter = 0

request = function()  
    -- url_path = math.random(131072)
	-- keep all data in memcached, effectively 2tier
	url_path = math.random(1000)
	-- url_path = math.random(1400000)
	-- url_path = math.random(100)
	-- url_path = zipf(1.01, 10000000)
	-- url_path = zipf(1.01, 1000000)

    -- url_path = zipf(1.2, 100000)

    counter = counter + 1
    -- print("req", url_path)
    -- return wrk.format(nil, "http://localhost:8088/test/" .. tostring(url_path))
    return wrk.format(nil, "http://128.253.128.66:8088/test/" .. tostring(url_path))
end

function zipf (s, N)
	p = math.random()
	local tolerance = 0.01
	local x = N / 2;
	
	local D = p * (12 * (math.pow(N, 1 - s) - 1) / (1 - s) + 6 - 6 * math.pow(N, -s) + s - math.pow(N, -1 - s) * s)
	
	while true do 
		local m    = math.pow(x, -2 - s);
		local mx   = m   * x;
		local mxx  = mx  * x;
		local mxxx = mxx * x;

		local a = 12 * (mxxx - 1) / (1 - s) + 6 * (1 - mxx) + (s - (mx * s)) - D
		local b = 12 * mxx + 6 * (s * mx) + (m * s * (s + 1))
		local newx = math.max(1, x - a / b)
		if math.abs(newx - x) <= tolerance then
			return math.floor(newx)
		end
		x = newx
	end
end


response = function(status, headers, body)
	-- print("ngx-end")
	-- print(headers["ngx-end"])
	if headers["X-cache"] == "MISS" then
		return 0, headers["request-latency"], headers["get-latency"], headers["find-latency"], headers["set-latency"], headers['ngx-end']
	else
		return 1, headers["request-latency"], headers["get-latency"], 0, 0, headers['ngx-end']
	end

    -- return headers["X-cache"]
-- -- end
--     if headers["X-cache"] == "MISS" then
--         print("MISS", headers["mmc-get-start"], headers["mmc-get-end"], headers["mongo-find-start"], headers["mongo-find-end"], headers["mmc-set-start"], headers["mmc-set-end"])
--     else
--         print("HIT", headers["mmc-get-start"], headers["mmc-get-end"])
--     end
end

  
