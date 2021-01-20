# SINAN-LOCAL

## Publication
If you use Sinan in your research, please cite our ASPLOS'21 paper.
```
@inproceedings{sinan-asplos2021,
author = {Yanqi, Zhang and Weizhe, Hua and Zhuangzhuang, Zhou and G. Edward, Suh and Christina, Delimitrou
},
title = {Sinan: ML-Based & QoS-Aware Resource Management for Cloud Microservices},
booktitle = {Proceedings of the Twenty-Sixth International Conference on Architectural Support for Programming Languages and Operating Systems},
series = {ASPLOS '21}
}
```

## Prerequisites
- Ubuntu 18.04
- Docker 19.03
- Docker swarm latest
- Python 3.5+
- MXNet
- XGBoost

## Code structure
Similar to https://github.com/zyqCSL/sinan-gcp

## Usage
### Generating cluster configurations
Please make sure to generate the configuration files specific to your local cluster. 
The config directory in this repo contains an example configuration with 2 serveres each with 88 cpus, and a dedicated gpu server for inference. In order to reproduce experiments in the paper, cpu resoruces in the cluster are expected to be no less than the provided configuration.

#### Service cluster configuration (`docker_swarm/misc/make_cluster_config.py`)
For exmaple, in `docker_swarm/misc`, `python3 make_cluster_config.py --nodes ath-8 ath-9 --cluster-config test_cluster.json --replica-cpus 4` generates the cluster configuration for deploying SocialNetwork using two servers ath-8 and ath-9, and saves it as `docker_swarm/config/test_cluster.json`. 

For adapting to your own local cluster, please check the following:
* modify https://github.com/zyqCSL/sinan-local/blob/master/docker_swarm/misc/make_cluster_config.py#L40-L47 to include the ips of your servers. You probably need to remove the DROP rule in INPUT & FORWARD from iptables for these servers to be connected, and make sure servers can be sshed from each other

* modify https://github.com/zyqCSL/sinan-local/blob/master/docker_swarm/misc/make_cluster_config.py#L108-L121 to change the virtual cpu number and tags of the servers included in the cluster (the tags are used to control service placement. Please check the placement constraints in the benchmarks/socialNetwork-ml-swarm/docker-compose-swarm.yml, for example in https://github.com/zyqCSL/sinan-local/blob/master/benchmarks/socialNetwork-ml-swarm/docker-compose-swarm.yml#L17-L19. The compose file assumes two types of servers, tagged with 'data' and 'compute' correspondingly.)

#### Inference engine configuration (`docker_swarm/misc/make_gpu_config.py`)
In `docker_swarm/misc`, `python3 make_gpu_config.py --gpu-config gpu.json` generates the predictor configuration and saves it in `docker_swarm/config/gpu.json`. 

Modify https://github.com/zyqCSL/sinan-local/blob/master/docker_swarm/misc/make_gpu_config.py#L31-L34 to adapt to your own cluster 
* `gpu_config['gpus']` to the list of gpus to use, `[0]` if you only have one 
* `gpu_config['host']` to the ip of your own gpu server
* `gpu_config['working_dir']` to your own working directory of predictor (the path of  `ml_docker_swarm` of the cloned repo). 

#### Setting up docker swarm cluster
The following steps assume you are in `docker_swarm`.

On the master node, in `docker_swarm` directory, execute `python3 setup_swarm_cluster.py --user-name USERNAME --deploy-config test_cluster.json`, in which `test_cluster.json` is the generated cluster configuration. You can use `docker node ls` to make sure that the required servers are addeded and `docker node inspect SERVERNAME` to inspect they are tagged properly (check "Spec"::"Label" in the output json).

### Data collection
Short cut scripts for data collection & deployment can be found in `docker_swarm/scripts`. For example,  within `docker_swarm/scripts`, executing `run_data_collect.sh` will collect training samples of SocialNetwork for concurrent user number from 2 to 48. Generated data are stored in `docker_swarm/logs/collected_data`.

For adapting to your own local cluster, please modify:
* `--deploy-config` to your own cluster configuration (generated before)
* `--user-name` Your own username
* `--exp-time` the running time for each user number
* `--min-users`, `--max-users`, `--users-step`. Generated training will include \[min_users, max_users, users_step\]. If you are using a smaller cluster, you might want to scale down the max_users. This repo uses 2 88-vcpu servers, and you can scale down the max-users proportionally, or you can start from short experiments and check when your cluster saturates.

### Modeling training
GPU is required for model training. The following steps assume you are in `ml_docker_swarm`.

* Execute `python3 data_parser_socialml_next_k.py --log-dir LOGDIR --save-dir DATADIR` to format the training dataset. `LOGDIR` is the directory of the raw training set (`ocker_swarm/logs/collected_data`), and `DATADIR` points to the path to save the formatted training samples. Please record the first dimension of output `glob_sys_data_train.shape`, which is your training data set size and you will need it later.

* Execute `python train_cnvnet.py --num-examples NUMSAMPLES --lr 0.001 --gpus 0,1 --data-dir DATADIR --wd 0.001` to train the CNN. `NUMSAMPLES` is your training dataset size. If you only have 1 gpu, please change to `--gpus 0`. Generated models are saved in `ml_docker_swarm/model`. You can also read the accuracy from the output log, whose name is `test_single_qps_upsample` by default.

* Execute `python xgb_train_latent.py --gpus 0,1 --data-dir DATADIR` to train the XGBoost model. Generated models are saved in `ml_docker_swarm/xgb_model`.

### Deployment
GPU is required for deployment. The shortcut scripts for deployment are in `docker_swarm/scripts`. 

#### Static load
`test_deploy.sh` tests the deployment situation with concurrent users in \[5, 45, 5\]. To adapt to your own cluster, please modify the following:

* Add `--deploy` flag if the application is not already deployed.

* `--slave-port` & `--gpus-port` are the ports master connects to slave and gpu server, please make sure there's no conflict.

* `--deploy-config` & `--gpu-config` should be changed to your own configuration previously generated

* `--min-users`, `--max-users`, `--users-step` should be scaled proportionally to your cluster size

* `--user-name` to your own.

#### Diurnal load
`test_deploy.sh` tests the  diurnal pattern where concurrent users start as 4, gradually rises to 36 and then gradually drops back to 4, with each period lasting 120s. The instructions are the same as static load.
