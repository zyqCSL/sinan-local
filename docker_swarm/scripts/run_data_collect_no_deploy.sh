cd ../
python3 master_data_collect_ath_social.py --user-name yz2297 \
	--stack-name sinan-socialnet \
	--min-users 4 --max-users 48 --users-step 2 \
	--exp-time 1200 --measure-interval 1 --slave-port 40011 --deploy-config swarm_ath.json \
	--mab-config social_mab.json