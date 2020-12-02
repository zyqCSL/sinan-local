import random
from locust import HttpUser, task, tag, between
import base64
import os
from pathlib import Path
import logging
import numpy as np
import time
import json

import locust.stats
locust.stats.CSV_STATS_INTERVAL_SEC = 1 # second

random.seed(time.time())

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# logging.basicConfig(level=logging.INFO,
#                     # filename='/mnt/locust_log/locust_openwhisk_log.txt',
#                     # filemode='w+',
#                     format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

logging.basicConfig(level=logging.INFO)

def get_user():
    user_id = random.randint(0, 500)
    user_name = 'Cornell_' + str(user_id)
    password = ""
    for i in range(0, 10):
        password = password + str(user_id)
    return user_name, password

mean_iat = 0.005  # seconds

class SocialMediaUser(HttpUser):
    # wait_time = between(5, 9)
    # return wait time in second
    def wait_time(self):
        global intervals
        global mean_iat
        return np.random.exponential(scale=mean_iat)
        # return random.choice(intervals)
        # self.last_wait_time += 1
        # return self.last_wait_time

    @task(600)
    @tag('search_hotel')
    def search_hotel(self):
        in_date = random.randint(9, 23)
        out_date = random.randint(in_date+1, 24)
        if in_date <= 9:
            in_date = "2015-04-0" + str(in_date)
        else:
            in_date = "2015-04-" + str(in_date)

        if out_date <= 9:
            out_date = "2015-04-0" + str(out_date)
        else:
            out_date = "2015-04-" + str(out_date)


        lat = 38.0235 + (random.randint(0, 481) - 240.5)/1000.0
        lon = -122.095 + (random.randint(0, 325) - 157.0)/1000.0

        url = '/hotels?inDate=' + in_date + '&outDate=' + out_date + \
            '&lat=' + str(lat) + "&lon=" + str(lon)

        r = self.client.get(url, name='search_hotel', timeout=10)
        if r.status_code > 202:
            logging.warning('search_hotel resp.status = %d, text=%s' %(r.status_code,
                r.text))

    @task(390)
    @tag('recommend')
    def recommend(self):
        coin = random.random()
        if coin < 0.33:
            req = 'dis'
        elif coin < 0.66:
            req = 'rate'
        else:
            req = 'price'

        lat = 38.0235 + (random.randint(0, 481) - 240.5)/1000.0
        lon = -122.095 + (random.randint(0, 325) - 157.0)/1000.0

        url = '/recommendations?require=' + req + \
            "&lat=" + str(lat) + "&lon=" + str(lon)

        r = self.client.get(url, name='recommend', timeout=10)
        if r.status_code > 202:
            logging.warning('recommend resp.status = %d, text=%s' %(r.status_code,
                r.text))

    @task(5)
    @tag('reserve')
    def reserve(self):
        in_date = random.randint(9, 23)
        out_date = random.randint(in_date+1, 24)

        if in_date <= 9:
            in_date = "2015-04-0" + str(in_date)
        else:
            in_date = "2015-04-" + str(in_date)

        if out_date <= 9:
            out_date = "2015-04-0" + str(out_date)
        else:
            out_date = "2015-04-" + str(out_date)

        lat = 38.0235 + (random.randint(0, 481) - 240.5)/1000.0
        lon = -122.095 + (random.randint(0, 325) - 157.0)/1000.0

        hotel_id = str(random.randint(1, 80))
        user_name, password = get_user()

        num_room = 1

        url = '/reservation?inDate=' + in_date + "&outDate=" + out_date + \
            "&lat=" + str(lat) + "&lon=" + str(lon) + "&hotelId=" + hotel_id + \
            "&customerName=" + user_name + "&username=" + user_name + \
            "&password=" + password + "&number=" + str(num_room)

        r = self.client.post(url, name='reserve', timeout=10)

    @task(5)
    @tag('user_login')
    def read_user_timeline(self):
        user_name, password = get_user()
        url = '/user?username=' + user_name + "&password=" + password

        r = self.client.get(url, name='user_login', timeout=10)

        if r.status_code > 202:
            logging.warning('read_user_timeline resp.status = %d, text=%s' %(r.status_code,
                r.text))