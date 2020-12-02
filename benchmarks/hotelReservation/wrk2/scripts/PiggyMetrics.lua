-- example script demonstrating HTTP pipelining

init = function(args)
   local r = {}
   r[1] = wrk.format(nil, "http://localhost:4000/accounts/demo")
   r[2] = wrk.format(nil, "http://localhost:4000/statistics/demo")

   req = table.concat(r)
end

request = function()
   return req
end
