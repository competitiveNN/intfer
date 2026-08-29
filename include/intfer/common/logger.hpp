#pragma once

#include <cstdio>

namespace intfer {

enum class LogLevel { TRACE, INFO, WARN, ERROR };

void log_impl(LogLevel level, const char* file, int line, const char* fmt, ...);

#define INTFER_LOG_TRACE(...) ::intfer::log_impl(::intfer::LogLevel::TRACE, __FILE__, __LINE__, __VA_ARGS__)
#define INTFER_LOG_INFO(...)  ::intfer::log_impl(::intfer::LogLevel::INFO,  __FILE__, __LINE__, __VA_ARGS__)
#define INTFER_LOG_WARN(...)  ::intfer::log_impl(::intfer::LogLevel::WARN,  __FILE__, __LINE__, __VA_ARGS__)
#define INTFER_LOG_ERROR(...) ::intfer::log_impl(::intfer::LogLevel::ERROR, __FILE__, __LINE__, __VA_ARGS__)

}  // namespace intfer
