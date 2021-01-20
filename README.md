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
- Docker swarm
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

For adapting to your own local cluster, please modify https://github.com/zyqCSL/sinan-local/blob/master/docker_swarm/misc/make_cluster_config.py#L40-L47 to include the ips of your servers (you probably need to remove the DROP rule in INPUT & FORWARD for these servers to be connected, and make sure servers can be sshed from each other), and modify https://github.com/zyqCSL/sinan-local/blob/master/docker_swarm/misc/make_cluster_config.py#L108-L121 to change the virtual cpu number and tags of the servers included in the cluster (the tags are used to control service placement. Please check the placement constraints in the benchmarks/socialNetwork-ml-swarm/docker-compose-swarm.yml, for example in https://github.com/zyqCSL/sinan-local/blob/master/benchmarks/socialNetwork-ml-swarm/docker-compose-swarm.yml#L17-L19. The compose file assumes two types of servers, tagged with 'data' and 'compute' correspondingly.)

#### Inference engine configuration (`docker_swarm/misc/make_gpu_config.py`)
In `docker_swarm/misc`, `python3 make_gpu_config.py --gpu-config gpu.json` generates the predictor configuration and saves it in `docker_swarm/config/gpu.json`. 

Modify https://github.com/zyqCSL/sinan-local/blob/master/docker_swarm/misc/make_gpu_config.py#L31-L34 to adapt to your own cluster: `gpu_config['gpus']` to the list of gpus to use, `[0]` if you only have one, `gpu_config['host']` to the ip of your own gpu server, and `gpu_config['working_dir']` to your own working directory of predictor (the path of  `ml_docker_swarm` of the cloned repo). 

#### Setting up docker swarm cluster
On the master node, in `docker_swarm` directory, execute `python3 setup_swarm_cluster.py --user-name USERNAME --deploy-config test_cluster.json`, in which `test_cluster.json` is the generated cluster configuration. You can use `docker node ls` to make sure that the required servers are addeded and `docker node inspect SERVERNAME` to inspect they are tagged properly (check "Spec"::"Label" in the output json).

### Data collection
Short cut scripts for data collection & deployment can be found in `docker_swarm/scripts`. For example,  within `docker_swarm/scripts`, executing ``
The data collection phase does not require an inference server, while the deployment phase does.

### Modeling training

### Deployment 
