#include "intfer/common/logger.hpp"

#include <cstdarg>
#include <cstdio>
#include <string>

namespace intfer {

namespace {

const char* level_str(LogLevel l) {
  switch (l) {
    case LogLevel::TRACE: return "TRACE";
    case LogLevel::INFO:  return "INFO ";
    case LogLevel::WARN:  return "WARN ";
    case LogLevel::ERROR: return "ERROR";
  }
  return "?";
}

}  // namespace

void log_impl(LogLevel level, const char* file, int line, const char* fmt, ...) {
  char buf[2048];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  fprintf(stdout, "[intfer][%s] %s (%s:%d)\n", level_str(level), buf, file, line);
  fflush(stdout);
}

}  // namespace intfer
