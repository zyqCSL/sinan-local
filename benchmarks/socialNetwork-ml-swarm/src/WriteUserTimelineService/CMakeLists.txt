add_executable(
    WriteUserTimelineService
    WriteUserTimelineService.cpp
    ${THRIFT_GEN_CPP_DIR}/social_network_types.cpp
)

target_include_directories(
    WriteUserTimelineService PRIVATE
    ${MONGOC_INCLUDE_DIRS}
    /usr/local/include/jaegertracing
    /usr/local/include/cpp_redis
    ${LIBEVENT_INCLUDE_DIRS}
)


target_link_libraries(
    WriteUserTimelineService
    ${MONGOC_LIBRARIES}
    ${THRIFT_LIB}
    ${CMAKE_THREAD_LIBS_INIT}
    ${Boost_LIBRARIES}
    nlohmann_json::nlohmann_json
    Boost::log
    Boost::log_setup
    OpenSSL::SSL
    /usr/local/lib/libjaegertracing.so
    /usr/local/lib/libamqpcpp.so
    ${LIBEVENT_LIBRARIES}
    /usr/local/lib/libcpp_redis.a
    /usr/local/lib/libtacopie.a
)

install(TARGETS WriteUserTimelineService DESTINATION ./)