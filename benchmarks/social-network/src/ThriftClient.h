#ifndef SOCIAL_NETWORK_MICROSERVICES_THRIFTCLIENT_H
#define SOCIAL_NETWORK_MICROSERVICES_THRIFTCLIENT_H

#include <string>
#include <thread>
#include <iostream>
#include <boost/log/trivial.hpp>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>
#include <thrift/stdcxx.h>
#include "logger.h"
#include "GenericClient.h"

namespace social_network {

using apache::thrift::protocol::TProtocol;
using apache::thrift::protocol::TBinaryProtocol;
using apache::thrift::transport::TFramedTransport;
using apache::thrift::transport::TSocket;
using apache::thrift::transport::TTransport;
using apache::thrift::TException;

template<class TThriftClient>
class ThriftClient : public GenericClient {
 public:
  ThriftClient(const std::string &addr, int port, int keep_alive = 1000);

  ThriftClient(const ThriftClient &) = delete;
  ThriftClient &operator=(const ThriftClient &) = delete;
  ThriftClient(ThriftClient<TThriftClient> &&) = default;
  ThriftClient &operator=(ThriftClient &&) = default;

  ~ThriftClient() override;

  TThriftClient *GetClient() const;

  void Connect() override;
  void Disconnect() override;
  void KeepAlive(int keep_alive = 1000) override;
  bool IsConnected() override;
  bool IsAlive() override;

 private:
  TThriftClient *_client;

  std::shared_ptr<TTransport> _socket;
  std::shared_ptr<TTransport> _transport;
  std::shared_ptr<TProtocol> _protocol;
};

template<class TThriftClient>
ThriftClient<TThriftClient>::ThriftClient(
    const std::string &addr, int port, int keep_alive) {
  _addr = addr;
  _port = port;
  _keep_alive = keep_alive;
  if (_keep_alive)  KeepAlive(_keep_alive);
  _socket = std::shared_ptr<TTransport>(new TSocket(addr, port));
  _transport = std::shared_ptr<TTransport>(new TFramedTransport(_socket));
  _protocol = std::shared_ptr<TProtocol>(new TBinaryProtocol(_transport));
  _client = new TThriftClient(_protocol);
}

template<class TThriftClient>
ThriftClient<TThriftClient>::~ThriftClient() {
  Disconnect();
  delete _client;
}

template<class TThriftClient>
TThriftClient *ThriftClient<TThriftClient>::GetClient() const {
  return _client;
}

template<class TThriftClient>
bool ThriftClient<TThriftClient>::IsConnected() {
  return _transport->isOpen();
}

template<class TThriftClient>
void ThriftClient<TThriftClient>::Connect() {
  if (!IsConnected()) {
    try {
      _transport->open();
    } catch (TException &tx) {
      throw tx;
    }
  }
}

template<class TThriftClient>
void ThriftClient<TThriftClient>::Disconnect() {
  if (IsConnected()) {
    try {
      _transport->close();
    } catch (TException &tx) {
      throw tx;
    }
  }
}

// TODO: Implement KeepAlive Timeout
template<class TThriftClient>
void ThriftClient<TThriftClient>::KeepAlive(int keep_alive) {
  _alive_until = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count() + keep_alive;
}

template<class TThriftClient>
bool ThriftClient<TThriftClient>::IsAlive() {
  if (_keep_alive) {
    long long now = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
    return now < _alive_until;
  }
  else
    return true;
}

} // namespace social_network


#endif //SOCIAL_NETWORK_MICROSERVICES_THRIFTCLIENT_H
