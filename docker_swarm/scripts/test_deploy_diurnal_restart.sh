cd ../
python3 master_deploy_diurnal_ath_social.py --user-name yz2297 \
	--stack-name sinan-socialnet \
	--min-users 4 --max-users 36 --users-step 4 \
	--exp-time 120 --measure-interval 1 --slave-port 40011 \
	--deploy-config swarm_ath.json \
	--gpu-config gpu.json --gpu-port 40010 \
	--mab-config social_mab.json --deploy