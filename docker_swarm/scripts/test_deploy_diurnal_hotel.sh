cd ../
python3 master_deploy_diurnal_ath_hotel.py --user-name yz2297 \
	--stack-name hotelreservation \
	--min-users 6 --max-users 16 --users-step 1 \
	--exp-time 120 --measure-interval 1 --slave-port 40011 \
	--deploy-config hotel_swarm_ath.json \
	--gpu-config hotel_gpu.json --gpu-port 40010 \
	--mab-config social_mab.json --deploy