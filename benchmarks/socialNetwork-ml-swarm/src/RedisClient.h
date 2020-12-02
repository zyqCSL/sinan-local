#ifndef SOCIAL_NETWORK_MICROSERVICES_REDISCLIENT_H
#define SOCIAL_NETWORK_MICROSERVICES_REDISCLIENT_H

#include <string>
#include <cpp_redis/cpp_redis>

#include "logger.h"
#include "GenericClient.h"

namespace social_network {

class RedisClient : public GenericClient {
 public:
  RedisClient(const std::string &addr, int port, int keep_alive = 0);
  RedisClient(const RedisClient &) = delete;
  RedisClient & operator=(const RedisClient &) = delete;
  RedisClient(RedisClient &&) = default;
  RedisClient & operator=(RedisClient &&) = default;

  ~RedisClient() override ;

  cpp_redis::client *GetClient() const;

  void Connect() override ;
  void Disconnect() override ;
  void KeepAlive(int keep_alive = 0) override ;
  bool IsConnected() override ;
  bool IsAlive() override ;

 private:
  cpp_redis::client * _client;
};

RedisClient::RedisClient(const std::string &addr, int port, int keep_alive) {
  _addr = addr;
  _port = port;
  _keep_alive = keep_alive;
  if (_keep_alive)  KeepAlive(_keep_alive);
  _client = new cpp_redis::client();
}

RedisClient::~RedisClient() {
  Disconnect();
  delete _client;
}

cpp_redis::client* RedisClient::GetClient() const {
  return _client;
}

void RedisClient::Connect() {
  if (!IsConnected()) {
    _client->connect(_addr, _port, [](const std::string& host, std::size_t port,
        cpp_redis::client::connect_state status) {
      if (status == cpp_redis::client::connect_state::dropped) {
        LOG(error) << "Failed to connect " << host << ":" << port;
        throw status;
      }
    });
  }
}

void RedisClient::Disconnect() {
  if (IsConnected()) {
    _client->disconnect();
  }
}

void RedisClient::KeepAlive(int keep_alive) {
  _alive_until = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count() + keep_alive;
}

bool RedisClient::IsConnected() {
  return _client->is_connected();
}

bool RedisClient::IsAlive() {
  // if (_keep_alive) {
  //   long long now = std::chrono::duration_cast<std::chrono::milliseconds>(
  //     std::chrono::system_clock::now().time_since_epoch()).count();
  //   return now < _alive_until;
  // }
  // else
  //   return true;
  return true;
}

} // social_network

#endif //SOCIAL_NETWORK_MICROSERVICES_REDISCLIENT_H
