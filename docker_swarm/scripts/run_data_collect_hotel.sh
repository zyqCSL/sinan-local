cd ../
python3 master_data_collect_ath_hotel.py --user-name yz2297 \
	--stack-name hotelreservation \
	--min-users 2 --max-users 48 --users-step 1 \
	--exp-time 750 --measure-interval 1 --slave-port 40011 --deploy-config hotel_swarm_ath.json \
	--mab-config hotel_mab.json --deploy