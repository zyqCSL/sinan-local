#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TThreadedServer.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TBufferTransports.h>
#include <nlohmann/json.hpp>
#include <signal.h>

#include "HomeTimelineHandler.h"
#include "../ClientPool.h"
#include "../RedisClient.h"
#include "../logger.h"
#include "../tracing.h"
#include "../utils.h"

#include <iostream>

using apache::thrift::server::TThreadedServer;
using apache::thrift::transport::TServerSocket;
using apache::thrift::transport::TFramedTransportFactory;
using apache::thrift::protocol::TBinaryProtocolFactory;
using namespace social_network;

void sigintHandler(int sig) {
  exit(EXIT_SUCCESS);
}

int main(int argc, char *argv[]) {
  std::cout << "in main" << std::endl;

  signal(SIGINT, sigintHandler);
  init_logger();
  SetUpTracer("config/jaeger-config.yml", "home-timeline-service");

  std::cout << "after setting up jaeger" << std::endl;

  json config_json;
  if (load_config_file("config/service-config.json", &config_json) != 0) {
    exit(EXIT_FAILURE);
  }

  int port = config_json["home-timeline-service"]["port"];
  std::string redis_addr =
      config_json["home-timeline-redis"]["addr"];
  int redis_port = config_json["home-timeline-redis"]["port"];
  int redis_conns = config_json["home-timeline-redis"]["connections"];
  int redis_timeout = config_json["home-timeline-redis"]["timeout_ms"];

  int post_storage_port = config_json["post-storage-service"]["port"];
  std::string post_storage_addr = config_json["post-storage-service"]["addr"];
  int post_storage_conns = config_json["post-storage-service"]["connections"];
  int post_storage_timeout = config_json["post-storage-service"]["timeout_ms"];

  ClientPool<RedisClient> redis_client_pool("home-timeline-redis",
      redis_addr, redis_port, 0, redis_conns, redis_timeout);

  std::cout << "after setting up redis client pool" << std::endl;

  ClientPool<ThriftClient<PostStorageServiceClient>>
      post_storage_client_pool("post-storage-client", post_storage_addr,
                               post_storage_port, 0, post_storage_conns, post_storage_timeout);

  std::cout << "after setting up post-storage-client client pool" << std::endl;

  TThreadedServer server (
      std::make_shared<HomeTimelineServiceProcessor>(
          std::make_shared<ReadHomeTimelineHandler>(
              &redis_client_pool,
              &post_storage_client_pool)),
      std::make_shared<TServerSocket>("0.0.0.0", port),
      std::make_shared<TFramedTransportFactory>(),
      std::make_shared<TBinaryProtocolFactory>()
  );

  std::cout << "Starting the home-timeline-service server..." << std::endl;
  server.serve();
}