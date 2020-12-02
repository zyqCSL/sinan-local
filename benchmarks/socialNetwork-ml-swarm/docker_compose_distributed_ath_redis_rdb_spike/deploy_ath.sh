ssh -t yz2297@ath-9 'docker-compose -f /home/yz2297/Software/deathstar_suite/socialNetwork-distributed-ath-write-user-tl/docker_compose_distributed_ath/docker-compose-ath9-1.yml up -d'
sleep 5
ssh -t yz2297@ath-8 'docker-compose -f /home/yz2297/Software/deathstar_suite/socialNetwork-distributed-ath-write-user-tl/docker_compose_distributed_ath/docker-compose-ath8-2.yml up -d'
sleep 5
#ssh -t yz2297@ath-2 'docker-compose -f /home/yz2297/Software/deathstar_suite/socialNetwork-distributed-ath-write-user-tl/docker_compose_distributed_ath/docker-compose-ath2-3.yml up -d'
#sleep 5
# ssh -t yz2297@ath-5 'docker-compose -f /home/yz2297/Software/deathstar_suite/socialNetwork-distributed-ath-write-user-tl/docker_compose_distributed_ath/docker-compose-ath5-3.yml up -d'
ssh -t yz2297@ath-1 'docker-compose -f /home/yz2297/Software/deathstar_suite/socialNetwork-distributed-ath-write-user-tl/docker_compose_distributed_ath/docker-compose-ath1-3.yml up -d'
sleep 5
ssh -t yz2297@ath-3 'docker-compose -f /home/yz2297/Software/deathstar_suite/socialNetwork-distributed-ath-write-user-tl/docker_compose_distributed_ath/docker-compose-ath3-4.yml up -d'
