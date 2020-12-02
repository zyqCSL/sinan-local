cd ../
python3 master_deploy_ath_social.py --user-name yz2297 \
	--stack-name sinan-socialnet \
	--min-users 5 --max-users 45 --users-step 5 \
	--exp-time 300 --measure-interval 1 --slave-port 40011 \
	--deploy-config swarm_ath.json \
	--gpu-config gpu.json --gpu-port 40010 \
	--mab-config social_mab.json