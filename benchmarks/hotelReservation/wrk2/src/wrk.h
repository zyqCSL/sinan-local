#ifndef WRK_H
#define WRK_H

#include "config.h"
#include <pthread.h>
#include <inttypes.h>
#include <sys/types.h>
#include <netdb.h>
#include <sys/socket.h>

#include <openssl/ssl.h>
#include <openssl/err.h>
#include <lua.h>

#include "stats.h"
#include "ae.h"
#include "http_parser.h"
#include "hdr_histogram.h"

#define VERSION  "4.0.0"
#define RECVBUF  8192
#define SAMPLES  100000000

#define SOCKET_TIMEOUT_MS   2000
#define CALIBRATE_DELAY_MS  10000
#define TIMEOUT_INTERVAL_MS 2000

#define MAXL 1000000
#define MAXO 16383
#define MAXTHREADS 40

typedef struct {
    uint64_t latency;
    uint64_t time;  // time that this response is received  
} resp_record;

typedef struct {
    pthread_t thread;
    aeEventLoop *loop;
    struct addrinfo *addr;
    uint64_t connections;
    int interval;
    uint64_t tid;
    uint64_t stop_at;
    uint64_t complete;
    uint64_t requests;
    uint64_t monitored;
    uint64_t target;
    uint64_t accum_latency;
    uint64_t bytes;
    uint64_t start;
    double throughput;
    uint64_t mean;
    struct hdr_histogram *latency_histogram;
    struct hdr_histogram *real_latency_histogram;

    tinymt64_t rand;
    lua_State *L;
    errors errors;
    struct connection *cs;
    FILE* ff;  
    uint64_t monitor_event_cnt; 

    int throughput_arr_len;
    double* throughput_arr;
    uint64_t* throughput_change_us;
    int throughput_arr_ptr;

    bool print_realtime_latency;
    uint64_t stats_report_us;

    uint64_t base_start_time;    // for debug
    /* verify load change*/
    uint64_t resp_received;
    uint64_t last_stats_time;

    bool record_per_request;
    resp_record* per_req_records;
    uint64_t per_req_limit;
    uint64_t per_req_start;
    uint64_t per_req_end;
    uint64_t per_req_ctr;
} thread;

typedef struct {
    char  *buffer;
    size_t length;
    char  *cursor;
} buffer;

typedef struct connection {
    thread *thread;
    http_parser parser;
    enum {
        FIELD, VALUE
    } state;
    int fd;
    SSL *ssl;
    double throughput;
    uint64_t interval;
    uint64_t sent;
    uint64_t estimate;
    uint64_t complete;
    uint64_t thread_start;
    uint64_t thread_next;
    uint64_t start;
    char *request;
    size_t length;
    size_t written;
    uint64_t pending;
    buffer headers;
    buffer body;
    char buf[RECVBUF];
    uint64_t actual_latency_start[MAXO+1];
    // Internal tracking numbers (used purely for debugging):
} connection;

// typedef struct latencies {
//     int if_hit;
//     int nginx_lua;
//     int set;
//     int get;
//     int find;

//     uint64_t nginx_end;
    
// } latencies;

#endif /* WRK_H */
