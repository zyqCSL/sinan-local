
#include <csignal>
#include <cpp_redis/cpp_redis>
#include <mutex>
#include <thread>
#include <sstream>
#include <set>

#include <mongoc.h>
#include <bson/bson.h>

#include "../AmqpLibeventHandler.h"
#include "../ClientPool.h"
#include "../RedisClient.h"
#include "../ThriftClient.h"
#include "../logger.h"
#include "../tracing.h"
#include "../utils.h"
#include "../utils_mongodb.h"
#include "../../gen-cpp/social_network_types.h"

#define NUM_WORKERS 32

using namespace social_network;

static std::exception_ptr _teptr;
static ClientPool<RedisClient> *_redis_client_pool;
static mongoc_client_pool_t *_mongodb_client_pool;

#include <iostream>

void sigintHandler(int sig) {
  exit(EXIT_SUCCESS);
}

void OnReceivedWorker(const AMQP::Message &msg) {
  // std::cout << "On Handler" << std::endl;
  try {
    json msg_json = json::parse(std::string(msg.body(), msg.bodySize()));

    std::map<std::string, std::string> carrier;
    for (auto it = msg_json["carrier"].begin();
        it != msg_json["carrier"].end(); ++it) {
      carrier.emplace(std::make_pair(it.key(), it.value()));
    }

    // Jaeger tracing
    TextMapReader span_reader(carrier);
    auto parent_span = opentracing::Tracer::Global()->Extract(span_reader);
    auto span = opentracing::Tracer::Global()->StartSpan(
        "InsertUserTimelines",
        {opentracing::ChildOf(parent_span->get())});
    std::map<std::string, std::string> writer_text_map;
    TextMapWriter writer(writer_text_map);
    opentracing::Tracer::Global()->Inject(span->context(), writer);

    // Extract information from rabbitmq messages
    int64_t user_id = msg_json["user_id"];
    int64_t req_id = msg_json["req_id"];
    int64_t post_id = msg_json["post_id"];
    int64_t timestamp = msg_json["timestamp"];

    // Update Redis ZSet
    auto redis_span = opentracing::Tracer::Global()->StartSpan(
        "RedisUpdate", {opentracing::ChildOf(&span->context())});
    auto redis_client_wrapper = _redis_client_pool->Pop();
    if (!redis_client_wrapper) {
      ServiceException se;
      se.errorCode = ErrorCode::SE_REDIS_ERROR;
      se.message = "Cannot connected to Redis server";
      throw se;
    }
    auto redis_client = redis_client_wrapper->GetClient();
    std::vector<std::string> options{"NX"};
    std::string post_id_str = std::to_string(post_id);
    std::string timestamp_str = std::to_string(timestamp);
    std::multimap<std::string, std::string> value =
        {{timestamp_str, post_id_str}};

    redis_client->zadd(std::to_string(user_id), options, value);

    redis_client->sync_commit();
    redis_span->Finish();
    _redis_client_pool->Push(redis_client_wrapper);

    // std::cout << "After updating redis" << std::endl;

    // // do not record mongodb insert time here since its asynchronous
    // std::future<void> store_post_future = std::async(
    //   std::launch::async, [&]() {
    mongoc_client_t *mongodb_client = mongoc_client_pool_pop(
      _mongodb_client_pool);
    if (!mongodb_client) {
      ServiceException se;
      se.errorCode = ErrorCode::SE_MONGODB_ERROR;
      se.message = "Failed to pop a client from MongoDB pool";
      throw se;
    }
    auto collection = mongoc_client_get_collection(
        mongodb_client, "user-timeline", "user-timeline");
    if (!collection) {
      ServiceException se;
      se.errorCode = ErrorCode::SE_MONGODB_ERROR;
      se.message = "Failed to create collection user-timeline from MongoDB";
      mongoc_client_pool_push(_mongodb_client_pool, mongodb_client);
      throw se;
    }
    bson_t *query = bson_new();

    BSON_APPEND_INT64(query, "user_id", user_id);

    bson_t *update = BCON_NEW(
        "$push", "{",
            "posts", "{",
                "$each", "[", "{",
                    "post_id", BCON_INT64(post_id),
                    "timestamp", BCON_INT64(timestamp),
                "}", "]",
                "$position", BCON_INT32(0),
            "}",
        "}"
    );
    bson_error_t error;
    bson_t reply;
    auto update_span = opentracing::Tracer::Global()->StartSpan(
        "MongoInsert", {opentracing::ChildOf(&span->context())});
    // If no document matches the query criteria,
    // insert a single document (upsert: true)
    bool updated = mongoc_collection_find_and_modify(
        collection, query, nullptr, update, nullptr, false, true,
        true, &reply, &error);
    update_span->Finish();

    if (!updated) {
      // update the newly inserted document (upsert: false)
      updated = mongoc_collection_find_and_modify(
        collection, query, nullptr, update, nullptr, false, false,
        true, &reply, &error);
      if (!updated) {
        LOG(error) << "Failed to update user-timeline for user " << user_id
                    << " to MongoDB: " << error.message;
        ServiceException se;
        se.errorCode = ErrorCode::SE_MONGODB_ERROR;
        se.message = error.message;
        bson_destroy(update);
        bson_destroy(query);
        bson_destroy(&reply);
        mongoc_collection_destroy(collection);
        mongoc_client_pool_push(_mongodb_client_pool, mongodb_client);
        throw se;
      }
    }

    bson_destroy(update);
    bson_destroy(&reply);
    bson_destroy(query);
    mongoc_collection_destroy(collection);
    mongoc_client_pool_push(_mongodb_client_pool, mongodb_client);

    span->Finish();

  } catch (...) {
    LOG(error) << "OnReveived worker error";
    throw;
  }
}

void HeartbeatSend(AmqpLibeventHandler &handler,
    AMQP::TcpConnection &connection, int interval){
  while(handler.GetIsRunning()){
    LOG(debug) << "Heartbeat sent";
    connection.heartbeat();
    sleep(interval);
  }
}

void WorkerThread(std::string &addr, int port) {
  // std::cout << "in WorkerThread" << std::endl;
  // std::cout << "addr = " << addr << ", port = " << port << std::endl;

  AmqpLibeventHandler handler;
  AMQP::TcpConnection connection(handler, AMQP::Address(
      addr, port, AMQP::Login("guest", "guest"), "/"));

  // std::cout << "After creating connection" << std::endl;

  AMQP::TcpChannel channel(&connection);
  channel.onError(
      [&handler](const char *message) {
        LOG(error) << "Channel error: " << message;
        handler.Stop();
      });
  channel.declareQueue("write-timeline", AMQP::durable).onSuccess(
      [&connection](const std::string &name, uint32_t messagecount,
                    uint32_t consumercount) {
        LOG(debug) << "Created queue: " << name;
        // std::cout << "Created queue: " << name << std::endl;
      });
  channel.consume("write-timeline", AMQP::noack).onReceived(
      [](const AMQP::Message &msg, uint64_t tag, bool redelivered) {
        LOG(debug) << "Received: " << std::string(msg.body(), msg.bodySize());
        OnReceivedWorker(msg);
      });


  std::thread heartbeat_thread(HeartbeatSend, std::ref(handler),
      std::ref(connection), 30);
  heartbeat_thread.detach();
  handler.Start();
  LOG(debug) << "Closing connection.";
  connection.close();
}

int main(int argc, char *argv[]) {
  signal(SIGINT, sigintHandler);
  init_logger();

  SetUpTracer("config/jaeger-config.yml", "write-user-timeline-service");

  json config_json;
  if (load_config_file("config/service-config.json", &config_json) != 0) {
    exit(EXIT_FAILURE);
  }

  std::cout << "load_config_file done" << std::endl;

  // int port = config_json["write-user-timeline-service"]["port"];

  std::string rabbitmq_addr =
      config_json["write-user-timeline-rabbitmq"]["addr"];
  int rabbitmq_port = config_json["write-user-timeline-rabbitmq"]["port"];

  std::string redis_addr =
      config_json["user-timeline-redis"]["addr"];
  int redis_port = config_json["user-timeline-redis"]["port"];
  int redis_conns = config_json["user-timeline-redis"]["connections"];
  int redis_timeout = config_json["user-timeline-redis"]["timeout_ms"];

  ClientPool<RedisClient> redis_client_pool("redis", redis_addr, redis_port,
                                            0, redis_conns, redis_timeout);

  std::cout << "redis client created" << std::endl;

  _redis_client_pool = &redis_client_pool;

  // monogodb
  int mongodb_conns = config_json["user-timeline-mongodb"]["connections"];
  int mongodb_timeout = config_json["user-timeline-mongodb"]["timeout_ms"];

  _mongodb_client_pool = init_mongodb_client_pool(
      config_json, "user-timeline", mongodb_conns);

  if (_mongodb_client_pool == nullptr) {
    LOG(error) << "Failed to create mongodb client pool";
    return EXIT_FAILURE;
  }

  std::cout << "mongodb client created" << std::endl;

  std::unique_ptr<std::thread> threads_ptr[NUM_WORKERS];
  for (auto & thread_ptr : threads_ptr) {
    thread_ptr = std::make_unique<std::thread>(
        WorkerThread, std::ref(rabbitmq_addr), rabbitmq_port);
  }
  for (auto &thread_ptr : threads_ptr) {
    thread_ptr->join();
    if (_teptr) {
      try{
        std::rethrow_exception(_teptr);
      }
      catch(const std::exception &ex)
      {
        LOG(error) << "Thread exited with exception: " << ex.what();
      }
    }
  }



  return 0;
}