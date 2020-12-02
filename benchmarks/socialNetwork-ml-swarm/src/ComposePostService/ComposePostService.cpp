#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TThreadedServer.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TBufferTransports.h>
#include <signal.h>

#include "../utils.h"
#include "ComposePostHandler.h"

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
  signal(SIGINT, sigintHandler);
  init_logger();
  SetUpTracer("config/jaeger-config.yml", "compose-post-service");

  std::cout << "jaeger set up succeeded" << std::endl;


  json config_json;
  if (load_config_file("config/service-config.json", &config_json) != 0) {
    exit(EXIT_FAILURE);
  }

  int port = config_json["compose-post-service"]["port"];
  int redis_port = config_json["compose-post-redis"]["port"];
  std::string redis_addr = config_json["compose-post-redis"]["addr"];
  int redis_conns = config_json["compose-post-redis"]["connections"];
  int redis_timeout = config_json["compose-post-redis"]["timeout_ms"];

  int home_tl_rabbitmq_port = config_json["write-home-timeline-rabbitmq"]["port"];
  std::string home_tl_rabbitmq_addr =
      config_json["write-home-timeline-rabbitmq"]["addr"];
  int home_tl_rabbitmq_conns = config_json["write-home-timeline-rabbitmq"]["connections"];
  int home_tl_rabbitmq_timeout = config_json["write-home-timeline-rabbitmq"]["timeout_ms"];

  int post_storage_port = config_json["post-storage-service"]["port"];
  std::string post_storage_addr = config_json["post-storage-service"]["addr"];
  int post_storage_conns = config_json["post-storage-service"]["connections"];
  int post_storage_timeout = config_json["post-storage-service"]["timeout_ms"];

  // int user_timeline_port = config_json["user-timeline-service"]["port"];
  // std::string user_timeline_addr = config_json["user-timeline-service"]["addr"];

  int user_tl_rabbitmq_port = config_json["write-user-timeline-rabbitmq"]["port"];
  std::string user_tl_rabbitmq_addr =
      config_json["write-user-timeline-rabbitmq"]["addr"];
  int user_tl_rabbitmq_conns = config_json["write-user-timeline-rabbitmq"]["connections"];
  int user_tl_rabbitmq_timeout = config_json["write-user-timeline-rabbitmq"]["timeout_ms"];

  ClientPool<RedisClient> redis_client_pool("redis", redis_addr, redis_port,
                                            0, redis_conns, redis_timeout);

  std::cout << "redis client created" << std::endl;

  ClientPool<ThriftClient<PostStorageServiceClient>>
      post_storage_client_pool("post-storage-client", post_storage_addr,
                               post_storage_port, 0, post_storage_conns, post_storage_timeout);

  std::cout << "post-storage-client created" << std::endl;

  ClientPool<RabbitmqClient> home_tl_rabbitmq_client_pool("rabbitmq", home_tl_rabbitmq_addr,
      home_tl_rabbitmq_port, 0, home_tl_rabbitmq_conns, home_tl_rabbitmq_timeout);

  ClientPool<RabbitmqClient> user_tl_rabbitmq_client_pool("rabbitmq", user_tl_rabbitmq_addr,
      user_tl_rabbitmq_port, 0, user_tl_rabbitmq_conns, user_tl_rabbitmq_timeout);

  std::cout << "both rabbitmqs created" << std::endl;

  TThreadedServer server(
      std::make_shared<ComposePostServiceProcessor>(
          std::make_shared<ComposePostHandler>(
              &redis_client_pool,
              &post_storage_client_pool,
              // &user_timeline_client_pool,
              &home_tl_rabbitmq_client_pool,
              &user_tl_rabbitmq_client_pool)),
      std::make_shared<TServerSocket>("0.0.0.0", port),
      std::make_shared<TFramedTransportFactory>(),
      std::make_shared<TBinaryProtocolFactory>()
  );
  std::cout << "Starting the compose-post-service server ..." << std::endl;
  server.serve();

}