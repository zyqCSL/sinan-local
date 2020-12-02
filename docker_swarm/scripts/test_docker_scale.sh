cd ../
python3 master_test_docker_service_scale.py --user-name yz2297 \
	--stack-name sinan-socialnet \
	--min-rps 20 --max-rps 125 --rps-step 20 \
	--exp-time 300 --measure-interval 1 --slave-port 40011 --deploy-config swarm_ath-8.json