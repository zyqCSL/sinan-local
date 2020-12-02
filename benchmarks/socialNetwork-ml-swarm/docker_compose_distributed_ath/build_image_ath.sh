docker build -f ./TextFilterDockerfile -t yz2297/social-network-text-filter .
docker build -f ./MediaFilterDockerfile -t yz2297/social-network-media-filter .
docker build --no-cache -f ./Dockerfile -t yz2297/social-network-ml-swarm .
docker push yz2297/social-network-text-filter
docker push yz2297/social-network-media-filter
docker push yz2297/social-network-ml-swarm
# docker build -f ./MediaFilterDebugDockerfile -t yz2297/social-network-media-filter-debug .
# docker build --no-cache -f ./TextFilterDockerfile -t yz2297/social-network-text-filter .
# docker build -f ./TextFilterDockerfile -t yz2297/social-network-text-filter .
# docker build --no-cache -f ./MediaFilterDockerfile -t yz2297/social-network-media-filter .


# sudo ~/wrk2_archive/change_load_wrk2_general/wrk2_periodic_stats_sample_full_percentile/wrk -p 95 -r 0.5 -t 10 -S 0.2 -i 1.0 -D exp -t 10 -c 200 -d 180s -s ./wrk2/scripts/social-network/mixed-workload.lua http://ath-9-ip:8080 -R 1000
# ./wrk -p 95 -r 0.5 -t 10 -S 0.2 -i 1.0 -D exp -t 10 -c 200 -d 300s -s ./scripts/social-network/mixed-workload.lua http://ath-9-ip:8080 -R 500