#ifndef SOCIAL_NETWORK_MICROSERVICES_CLIENTPOOL_H
#define SOCIAL_NETWORK_MICROSERVICES_CLIENTPOOL_H

#include <vector>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <chrono>
#include <string>

#include "logger.h"

namespace social_network {

template<class TClient>
class ClientPool {
 public:
  ClientPool(const std::string &client_type, const std::string &addr,
      int port, int min_size, int max_size, int timeout_ms, int keep_alive=1000);
  ~ClientPool();

  ClientPool(const ClientPool&) = delete;
  ClientPool& operator=(const ClientPool&) = delete;
  ClientPool(ClientPool&&) = default;
  ClientPool& operator=(ClientPool&&) = default;

  TClient * Pop();
  void Push(TClient *);
  void Remove(TClient *);

 private:
  std::deque<TClient *> _pool;
  std::string _addr;
  std::string _client_type;
  int _port;
  int _min_pool_size{};
  int _max_pool_size{};
  int _curr_pool_size{};
  int _timeout_ms;
  int _keep_alive;
  std::mutex _mtx;
  std::condition_variable _cv;

  TClient * GetClientFromPool();
  TClient * ProduceClient();

};

template<class TClient>
ClientPool<TClient>::ClientPool(const std::string &client_type,
    const std::string &addr, int port, int min_pool_size,
    int max_pool_size, int timeout_ms, int keep_alive) {
  _addr = addr;
  _port = port;
  _min_pool_size = min_pool_size;
  _max_pool_size = max_pool_size;
  _timeout_ms = timeout_ms;
  _keep_alive = keep_alive;
  _client_type = client_type;

  for (int i = 0; i < min_pool_size; ++i) {
    TClient *client = new TClient(addr, port, _keep_alive);
    _pool.emplace_back(client);
  }
  _curr_pool_size = min_pool_size;
}

template<class TClient>
ClientPool<TClient>::~ClientPool() {
  while (!_pool.empty()) {
    delete _pool.front();
    _pool.pop_front();
  }
}

template<class TClient>
TClient * ClientPool<TClient>::Pop() {
  TClient * client = nullptr;
  client = GetClientFromPool();
  if (!client)
    client = ProduceClient();
  if (client) {
    try {
      client->Connect();
    } catch (...) {
      LOG(error) << "Failed to connect " + _client_type;
      _pool.push_back(client);
      throw;
    }
  }
  return client;
}

template<class TClient>
void ClientPool<TClient>::Push(TClient *client) {
  std::unique_lock<std::mutex> cv_lock(_mtx);
  _pool.push_back(client);
  cv_lock.unlock();
  _cv.notify_one();
}

template<class TClient>
void ClientPool<TClient>::Remove(TClient *client) {
  std::unique_lock<std::mutex> lock(_mtx);
  delete client;
  client = nullptr;
  _curr_pool_size--;
  lock.unlock();
}

template<class TClient>
TClient * ClientPool<TClient>::GetClientFromPool() {
  std::unique_lock<std::mutex> cv_lock(_mtx);
  TClient * client = nullptr;
  if (!_pool.empty()) {
    client = _pool.front();
    _pool.pop_front();
  } else if (_curr_pool_size == _max_pool_size) {
    auto wait_time = std::chrono::system_clock::now() +
        std::chrono::milliseconds(_timeout_ms);
    bool wait_success = _cv.wait_until(cv_lock, wait_time,
        [this] { return _pool.size() > 0; });
    if (!wait_success) {
      LOG(warning) << "ClientPool pop timeout";
      cv_lock.unlock();
      return nullptr;
    }
    client = _pool.front();
    _pool.pop_front();
  }
  if (client && !client->IsAlive()) {
    delete client;
    client = nullptr;
    _curr_pool_size--;
  }
  cv_lock.unlock();
  return client;
}

template<class TClient>
TClient * ClientPool<TClient>::ProduceClient() {
  std::unique_lock<std::mutex> lock(_mtx);
  TClient * client = nullptr;
  if (_curr_pool_size < _max_pool_size) {
    try {
      client = new TClient(_addr, _port, _keep_alive);
      _curr_pool_size++;
      lock.unlock();
      return client;
    } catch (...) {
      lock.unlock();
      return nullptr;
    }
  }
}

} // namespace social_network

#endif //SOCIAL_NETWORK_MICROSERVICES_CLIENTPOOL_H
