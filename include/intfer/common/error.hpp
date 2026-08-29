#pragma once

#include <stdexcept>
#include <string>

namespace intfer {

[[noreturn]] inline void throw_error(const char* file, int line, const std::string& msg) {
  throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + msg);
}

}  // namespace intfer

#define INTFER_THROW(msg) ::intfer::throw_error(__FILE__, __LINE__, (msg))
