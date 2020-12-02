#ifndef SOCIAL_NETWORK_MICROSERVICES_SRC_COMPOSEPOSTSERVICE_RABBITMQCLIENT_H_
#define SOCIAL_NETWORK_MICROSERVICES_SRC_COMPOSEPOSTSERVICE_RABBITMQCLIENT_H_

#include <SimpleAmqpClient/SimpleAmqpClient.h>

#include "../GenericClient.h"

namespace social_network {


class RabbitmqClient : public GenericClient {
 public:
  RabbitmqClient(const std::string &addr, int port, int keep_alive = 1000);
  RabbitmqClient(const RabbitmqClient &) = delete;
  RabbitmqClient & operator=(const RabbitmqClient &) = delete;
  RabbitmqClient(RabbitmqClient &&) = default;
  RabbitmqClient & operator=(RabbitmqClient &&) = default;

  ~RabbitmqClient() override ;

  void Connect() override ;
  void Disconnect() override ;
  void KeepAlive(int timeout_ms=1000) override ;
  bool IsConnected() override ;
  bool IsAlive() override ;

  AmqpClient::Channel::ptr_t GetChannel();

 private:
  std::string _addr;
  int _port;
  AmqpClient::Channel::ptr_t _channel;
  bool _is_connected;
};

RabbitmqClient::RabbitmqClient(const std::string &addr, int port, int keep_alive) {
  _addr = addr;
  _port = port;
  _keep_alive = keep_alive;
  if (_keep_alive)  KeepAlive(_keep_alive);
  _channel = AmqpClient::Channel::Create(addr, port);
  _is_connected = false;
}

RabbitmqClient::~RabbitmqClient() {
  Disconnect();
}

void RabbitmqClient::Connect() {
  if (!IsConnected()) {
    try {
      _channel->DeclareQueue("write-timeline", false, true, false, false);
    } catch (...) {
      throw;
    }
    _is_connected = true;
  }
}

void RabbitmqClient::Disconnect() {
  if (IsConnected()) {
    try {
      _channel->DeleteQueue("write-timeline");
      _is_connected = false;
    } catch (...) {
      throw;
    }
  }
}

void RabbitmqClient::KeepAlive(int keep_alive) {
  _alive_until = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count() + keep_alive;
}

bool RabbitmqClient::IsConnected() {
  return _is_connected;
}

bool RabbitmqClient::IsAlive() {
  // if (_keep_alive) {
  //   long long now = std::chrono::duration_cast<std::chrono::milliseconds>(
  //     std::chrono::system_clock::now().time_since_epoch()).count();
  //   return now < _alive_until;
  // }
  // else
  //   return true;
  return true;
}

AmqpClient::Channel::ptr_t RabbitmqClient::GetChannel() {
  return _channel;
}

} // namespace social_network



#endif //SOCIAL_NETWORK_MICROSERVICES_SRC_COMPOSEPOSTSERVICE_RABBITMQCLIENT_H_
