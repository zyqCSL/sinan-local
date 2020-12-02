cd ../
python3 master_deploy_ath_hotel.py --user-name yz2297 \
	--stack-name hotelreservation \
	--min-users 10 --max-users 40 --users-step 3 \
	--exp-time 300 --measure-interval 1 --slave-port 40011 \
	--deploy-config hotel_swarm_ath.json \
	--gpu-config hotel_gpu.json --gpu-port 40010 \
	--mab-config hotel_mab.json --deploy