docker run --network host -v $PWD/src:/mnt/locust -v $HOME/sinan_locust_log:/mnt/locust_log yz2297/locust_openwhisk \
	-f /mnt/locust/hotel_rps_100.py \
	--csv=/mnt/locust_log/hotel --headless -t $1 \
	--host http://127.0.0.1:5000 --users $2 \
	--logfile /mnt/locust_log/hotel_locust_log.txt