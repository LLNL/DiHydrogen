#pragma once

#include <sys/time.h>
#include <time.h>

namespace distconv {
namespace util {

struct stopwatch_t {
  struct timeval tv;
};

static inline void stopwatch_query(stopwatch_t *w) {
  gettimeofday(&(w->tv), nullptr);
  return;
}

// returns mili seconds
static inline float stopwatch_diff(const stopwatch_t *begin,
                                   const stopwatch_t *end) {
  return (end->tv.tv_sec - begin->tv.tv_sec) * 1000.0f
      + (end->tv.tv_usec - begin->tv.tv_usec) / 1000.0f;
}

static inline void stopwatch_start(stopwatch_t *w) {
  stopwatch_query(w);
  return;
}
    
static inline float stopwatch_stop(stopwatch_t *w) {
  stopwatch_t now;
  stopwatch_query(&now);
  return stopwatch_diff(w, &now);
}

} // namespace util
} // namespace distconv
