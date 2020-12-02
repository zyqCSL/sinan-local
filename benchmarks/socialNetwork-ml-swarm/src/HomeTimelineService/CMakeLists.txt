add_executable(
    HomeTimelineService
    HomeTimelineService.cpp
    ${THRIFT_GEN_CPP_DIR}/HomeTimelineService.cpp
    ${THRIFT_GEN_CPP_DIR}/PostStorageService.cpp
    ${THRIFT_GEN_CPP_DIR}/social_network_types.cpp
)

target_include_directories(
    HomeTimelineService PRIVATE
    /usr/local/include/jaegertracing
    /usr/local/include/cpp_redis
)

target_link_libraries(
    HomeTimelineService
    nlohmann_json::nlohmann_json
    ${THRIFT_LIB}
    ${CMAKE_THREAD_LIBS_INIT}
    ${Boost_LIBRARIES}
    Boost::log
    Boost::log_setup
    Boost::iostreams
    jaegertracing
    /usr/local/lib/libcpp_redis.a
    /usr/local/lib/libtacopie.a
    ${ZLIB_LIBRARY}
)

install(TARGETS HomeTimelineService DESTINATION ./)