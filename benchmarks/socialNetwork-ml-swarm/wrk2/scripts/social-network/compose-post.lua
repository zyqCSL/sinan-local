require "socket"
math.randomseed(socket.gettime()*1000)
math.random(); math.random(); math.random()

local charset = {'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's',
  'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm', 'Q',
  'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'A', 'S', 'D', 'F', 'G', 'H',
  'J', 'K', 'L', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', '1', '2', '3', '4', '5',
  '6', '7', '8', '9', '0'}

local decset = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}

------ media data ---------
local b='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/' -- You will need this for encoding/decoding
-- encoding
function enc(data)
    return ((data:gsub('.', function(x) 
        local r,b='',x:byte()
        for i=8,1,-1 do r=r..(b%2^i-b%2^(i-1)>0 and '1' or '0') end
        return r;
    end)..'0000'):gsub('%d%d%d?%d?%d?%d?', function(x)
        if (#x < 6) then return '' end
        local c=0
        for i=1,6 do c=c+(x:sub(i,i)=='1' and 2^(6-i) or 0) end
        return b:sub(c+1,c+1)
    end)..({ '', '==', '=' })[#data%3+1])
end

-- decoding
function dec(data)
    data = string.gsub(data, '[^'..b..'=]', '')
    return (data:gsub('.', function(x)
        if (x == '=') then return '' end
        local r,f='',(b:find(x)-1)
        for i=6,1,-1 do r=r..(f%2^i-f%2^(i-1)>0 and '1' or '0') end
        return r;
    end):gsub('%d%d%d?%d?%d?%d?%d?%d?', function(x)
        if (#x ~= 8) then return '' end
        local c=0
        for i=1,8 do c=c+(x:sub(i,i)=='1' and 2^(8-i) or 0) end
            return string.char(c)
    end))
end

benchmark_dir = "/home/yz2297/Software/deathstar_suite/socialNetwork-ml-swarm/DeathStarBench/socialNetwork-ml-swarm"
media_dir = benchmark_dir .. "/wrk2/scripts/social-network/images/"
media_jpg = {}
media_jpg_num = 17
media_png  = {}
media_png_num = 15

for i = 1, media_jpg_num do
  f = io.open(media_dir .. tostring(i) .. ".jpg", "rb")
  if f then 
    -- local temp = f:read("*all")
    -- ocr_img_jpg[i] = mime.b64(temp)
    media_jpg[i] = enc(f:read("*all"))
    f:close()
    print(media_dir .. tostring(i) .. ".jpg cached")
  else
    print(media_dir .. tostring(i) .. ".jpg doesn't exist")
  end
end 

for i = 1, media_png_num do 
  f = io.open(media_dir .. tostring(i) .. ".png", "rb")
  if f then 
    -- local temp = f:read("*all")
    -- ocr_img_png[i] = mime.b64(temp)
    media_png[i] = enc(f:read("*all"))
    f:close()
    print(media_dir .. tostring(i) .. ".png cached")
  else
    print(media_dir .. tostring(i) .. ".png doesn't exist")
  end
end
--------------------------

local function stringRandom(length)
  if length > 0 then
    return stringRandom(length - 1) .. charset[math.random(1, #charset)]
  else
    return ""
  end
end

local function decRandom(length)
  if length > 0 then
    return decRandom(length - 1) .. decset[math.random(1, #decset)]
  else
    return ""
  end
end

-- read:write = 83:17
request = function()
  local user_index = math.random(1, 962)
  local username = "username_" .. tostring(user_index)
  local user_id = tostring(user_index)
  local text = stringRandom(256)
  local num_user_mentions = math.random(0, 5)
  local num_urls = math.random(0, 5)
  -- local num_media = math.random(0, 4)
  local num_media = 0
  local medium = '['
  local media_types = '['

  -- decide if image is included
  if math.random() < 0.2 then
    num_media = math.random(1, 3)
  end

  for i = 0, num_user_mentions, 1 do
    local user_mention_id
    while (true) do
      user_mention_id = math.random(1, 962)
      if user_index ~= user_mention_id then
        break
      end
    end
    text = text .. " @username_" .. tostring(user_mention_id)
  end

  for i = 0, num_urls, 1 do
    text = text .. " http://" .. stringRandom(64)
  end

  for i = 0, num_media, 1 do
    coin = math.random(1, media_jpg_num + media_png_num)
    if coin <= media_jpg_num then
      local media_id = math.random(1, media_jpg_num)
      medium = medium .. "\"" .. media_jpg[media_id] .. "\","
      media_types = media_types .. "\"jpg\","
    else
      local media_id = math.random(1, media_png_num)
      medium = medium .. "\"" .. media_png[media_id] .. "\","
      media_types = media_types .. "\"png\","
    end
  end

  medium = medium:sub(1, #medium - 1) .. "]"
  media_types = media_types:sub(1, #media_types - 1) .. "]"

  local method = "POST"
  local path = "http://localhost:8080/wrk2-api/post/compose"
  local headers = {}
  local body
  headers["Content-Type"] = "application/x-www-form-urlencoded"
  if num_media then
    body   = "username=" .. username .. "&user_id=" .. user_id ..
        "&text=" .. text .. "&medium=" .. medium ..
        "&media_types=" .. media_types .. "&post_type=0"
  else
    body   = "username=" .. username .. "&user_id=" .. user_id ..
        "&text=" .. text .. "&medium=" .. "&post_type=0"
  end

  return wrk.format(method, path, headers, body)
end