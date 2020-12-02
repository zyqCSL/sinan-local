#ifndef SOCIAL_NETWORK_MICROSERVICES_SRC_MEDIASERVICE_MEDIAHANDLER_H_
#define SOCIAL_NETWORK_MICROSERVICES_SRC_MEDIASERVICE_MEDIAHANDLER_H_

#include <iostream>
#include <string>
#include <chrono>
#include <mutex>
#include <sstream>
#include <iomanip>
#include <arpa/inet.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <thread>
#include <future>
#include <cstdlib>
#include <ctime>

#include "../../gen-cpp/MediaService.h"
#include "../../gen-cpp/MediaFilterService.h"
#include "../../gen-cpp/ComposePostService.h"
#include "../ClientPool.h"
#include "../ThriftClient.h"
#include "../logger.h"
#include "../tracing.h"
#include "../gzip.h"
#include "../base64.h"

#include <iostream>

#define CUSTOM_EPOCH 1514764800000

namespace social_network {

using std::chrono::milliseconds;
using std::chrono::duration_cast;
using std::chrono::system_clock;


class MediaHandler : public MediaServiceIf {
 public:
  explicit MediaHandler(ClientPool<ThriftClient<ComposePostServiceClient>> *, 
                        ClientPool<ThriftClient<MediaFilterServiceClient>> *);
  ~MediaHandler() override = default;

  void UploadMedia(int64_t, const std::vector<std::string> &,
      const std::vector<std::string> &, const std::map<std::string,
      std::string> &) override;

 private:
  ClientPool<ThriftClient<ComposePostServiceClient>> *_compose_client_pool;
  ClientPool<ThriftClient<MediaFilterServiceClient>> *_media_filter_client_pool;
  void _findAndReplaceAll(std::string & data, std::string toSearch, std::string replaceStr);
};

MediaHandler::MediaHandler(
    ClientPool<ThriftClient<ComposePostServiceClient>> *compose_client_pool,
    ClientPool<ThriftClient<MediaFilterServiceClient>> *media_filter_client_pool) {
  _compose_client_pool = compose_client_pool;
  _media_filter_client_pool = media_filter_client_pool;
}

void MediaHandler::_findAndReplaceAll(std::string & data, std::string toSearch, std::string replaceStr) {
  // Get the first occurrence
  size_t pos = data.find(toSearch);
  // Repeat till end is reached
  while( pos != std::string::npos) {
    // Replace this occurrence of Sub String
    data.replace(pos, toSearch.size(), replaceStr);
    // Get the next occurrence from the current position
    pos =data.find(toSearch, pos + replaceStr.size());
  }
}

void MediaHandler::UploadMedia(
    int64_t req_id,
    const std::vector<std::string> &_media_types,
    const std::vector<std::string> &_medium,
    const std::map<std::string, std::string> &carrier) {

  // Initialize a span
  TextMapReader reader(carrier);
  std::map<std::string, std::string> writer_text_map;
  TextMapWriter writer(writer_text_map);
  auto parent_span = opentracing::Tracer::Global()->Extract(reader);
  auto span = opentracing::Tracer::Global()->StartSpan(
      "UploadMedia",
      { opentracing::ChildOf(parent_span->get()) });
  opentracing::Tracer::Global()->Inject(span->context(), writer);

  // copy strings and correct errors of base64 encodings
  std::vector<std::string> media_types = _media_types;
  std::vector<std::string> medium = _medium;
  for(int i = 0; i < medium.size(); ++i)
    _findAndReplaceAll(medium[i], " ", "+");

  if (media_types.size() != medium.size()) {
    ServiceException se;
    se.errorCode = ErrorCode::SE_THRIFT_HANDLER_ERROR;
    se.message = "The lengths of medium list and media_types list are not equal";
    throw se;
  }

  // std::cout << "UploadMedia before thread" << std::endl;
  // if(medium.size() > 0) {
  //   std::cout << "images:" << std::endl;
  //   for(const std::string& img: medium)
  //     std::cout << img << ";" << std::endl;
  // }

  // spawn a new thread so that we don't wait on time-consuming image filtering
  std::thread([=] 
  {
    // media-filter-service
    std::vector<bool> media_filter;
    // bool _filter_flag = (std::rand() % 10 <= 5);
    bool _filter_flag = true;
    // std::cout << "medium.size() = " << medium.size() << std::endl;
    if(_filter_flag && medium.size() > 0) {
      /******** debug starts *********/
      // std::cout << "images:" << std::endl;
      // for(const std::string& img: medium)
      //   std::cout << img << ";" << std::endl;
      /*****************/
      std::future<std::vector<bool>> media_filter_future = std::async(
          std::launch::async, [&](){
            auto media_filter_client_wrapper = _media_filter_client_pool->Pop();
            if (!media_filter_client_wrapper) {
              ServiceException se;
              se.errorCode = ErrorCode::SE_THRIFT_CONN_ERROR;
              se.message = "Failed to connect to media-filter-service";
              throw se;
            }
            std::vector<bool> return_filter;
            auto media_filter_client = media_filter_client_wrapper->GetClient();
            try {
              media_filter_client->UploadMedia(return_filter, req_id, media_types, medium, writer_text_map);
            } catch (...) {
              LOG(error) << "Failed to upload media to media-filter-service";
              _media_filter_client_pool->Push(media_filter_client_wrapper);
              throw;
            }    
            
            _media_filter_client_pool->Push(media_filter_client_wrapper);
            return return_filter;
          });

      try {
        media_filter = media_filter_future.get();
      } catch (...) {
        LOG(error) << "Failed to get media-filter from media-filter-service";
        throw;
      }
    } 
    // else 
    //   std::cout << "No images needs filtering" << std::endl;

    /********** debug ***********/
    std::string debug_filer_str = "image_filter: ";
    for(int i = 0; i < media_filter.size(); ++i) {
      if(media_filter[i])
        debug_filer_str += media_types[i] + " true; ";
      else
        debug_filer_str += media_types[i] + " false; ";
    }
    // std::cout << debug_filer_str << std::endl;
    /**************/

    std::vector<Media> media;
    for (int i = 0; i < medium.size(); ++i) {
      // nsfw images are removed
      if(_filter_flag && !media_filter[i])
        continue;

      Media new_media;

      // /********** debug ***********/
      // std::cout << "before compression" << std::endl;
      // std::cout << new_media.media_type << std::endl;
      // std::cout << new_media.media << std::endl << std::endl;
      // /**************/

      // new_media.media = medium[i];
      std::string compressed_media = Gzip::compress(Base64::decode(medium[i]));
      new_media.media = Base64::encode(reinterpret_cast<const unsigned char*>(compressed_media.c_str()), compressed_media.length());
      new_media.media_type = media_types[i];

      // /********** debug ***********/
      // std::cout << "after compression" << std::endl;
      // std::cout << new_media.media_type << std::endl;
      // std::cout << new_media.media << std::endl << std::endl;
      // /**************/

      media.emplace_back(new_media);
    }

    // Upload to compose post service
    auto compose_post_client_wrapper = _compose_client_pool->Pop();
    if (!compose_post_client_wrapper) {
      ServiceException se;
      se.errorCode = ErrorCode::SE_THRIFT_CONN_ERROR;
      se.message = "Failed to connected to compose-post-service";
      throw se;
    }
    auto compose_post_client = compose_post_client_wrapper->GetClient();
    try {
      compose_post_client->UploadMedia(req_id, media, writer_text_map);
    } catch (...) {
      _compose_client_pool->Push(compose_post_client_wrapper);
      LOG(error) << "Failed to upload media to compose-post-service";
      throw;
    }
    _compose_client_pool->Push(compose_post_client_wrapper);
  }).detach();

  span->Finish();

}

} //namespace social_network

#endif //SOCIAL_NETWORK_MICROSERVICES_SRC_MEDIASERVICE_MEDIAHANDLER_H_
