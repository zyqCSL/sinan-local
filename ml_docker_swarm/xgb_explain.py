# Note: must run from microservices directory!
import sys
import os
import socket
import subprocess
import time
import json
import math
import random
import argparse,logging

import mxnet as mx
import xgboost as xgb
import numpy as np
from importlib import import_module

# ml parameters
Model = None
InternalSysState = None
BoostTree = None
Services   = ['compose-post-redis',
			  'compose-post-service',
			  'home-timeline-redis',
			  'home-timeline-service',
			  # 'jaeger',
			  'nginx-thrift',
			  'post-storage-memcached',
			  'post-storage-mongodb',
			  'post-storage-service',
			  'social-graph-mongodb',
			  'social-graph-redis',
			  'social-graph-service',
			  'text-service',
			  'text-filter-service',
			  'unique-id-service',
			  'url-shorten-service',
			  'media-service',
			  'media-filter-service',
			  'user-mention-service',
			  'user-memcached',
			  'user-mongodb',
			  'user-service',
			  'user-timeline-mongodb',
			  'user-timeline-redis',
			  'user-timeline-service',
			  'write-home-timeline-service',
			  'write-home-timeline-rabbitmq',
			  'write-user-timeline-service',
			  'write-user-timeline-rabbitmq']

# -----------------------------------------------------------------------
# parser args definition
# -----------------------------------------------------------------------
parser = argparse.ArgumentParser()
# parser.add_argument('--cpus', dest='cpus', type=int, required=True)
# parser.add_argument('--stack-name', dest='stack_name', type=str, required=True)
# xgb-look-forward doesn't include immediate future (next time step)
parser.add_argument('--xgb-look-forward', dest='xgb_look_forward', type=int, default=4)
parser.add_argument('--xgb-prefix', dest='xgb_prefix', type=str, 
	default='./xgb_model/social_nn_sys_state_look_forward_')
args = parser.parse_args()


BoostTree = xgb.Booster()  # init model
print 'load ', args.xgb_prefix + str(args.xgb_look_forward) + '.model'
BoostTree.load_model(args.xgb_prefix + str(args.xgb_look_forward) + '.model')  # load data

feature_score = BoostTree.get_score(importance_type='gain')
print len(feature_score)
sorted_features = {}
for f in feature_score:
	f_int = int(f.replace('f', ''))
	assert f_int not in sorted_features
	sorted_features[f_int] = feature_score[f]
feature_array = sorted(sorted_features, key=sorted_features.get, reverse=True)
for f in feature_array:
	print f, ' ', sorted_features[f]

# first 64 elements are extracted from ml
mock_feature_array = list(range(0, 64 + len(Services)*5 ))
print 'mock_feature array ', mock_feature_array
# feature_array = np.array(feature_array)
mock_feature_array = np.array(mock_feature_array)

raw_feature_array = mock_feature_array[64:].reshape([len(Services), -1])
print raw_feature_array.shape

def find_in_feature_array(val):
	for i in range(0, raw_feature_array.shape[0]):
		for j in range(0, raw_feature_array.shape[1]):
			if raw_feature_array[i][j] == val:
				return i, j
	return None, None

for f in feature_array:
	i, j = find_in_feature_array(f)
	if i == None:
		print('%s\tnn latent state\t%s' %(f, sorted_features[f]))
	else:
		print('%s\t%s\ttime%d\t%s' %(f, Services[i], j, sorted_features[f]))
