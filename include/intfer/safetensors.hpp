// include/intfer/safetensors.hpp
// Minimal host-only safetensors reader (no external deps). Supports the subset
// of dtypes emitted by HF transformers: BF16, FP16, FP32 (decoded to fp32).
// Memory-maps the file; tensor reads decode on demand.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace intfer {

struct STensor {
  std::string name;
  std::string dtype;
  std::vector<size_t> shape;
  size_t begin = 0;  // byte offset into the data section
  size_t end = 0;
};

class SafetensorsReader {
public:
  ~SafetensorsReader() { close(); }
  bool open(const std::string& path);
  void close();

  const std::vector<STensor>& tensors() const { return tensors_; }
  bool has(const std::string& name) const;

  // Decode the named tensor to a flat fp32 buffer (row-major). Returns false if
  // the dtype is unsupported or the name is missing.
  bool read(const std::string& name, std::vector<float>& out) const;

  size_t file_size() const { return size_; }

private:
  int fd_ = -1;
  void* map_ = nullptr;
  size_t size_ = 0;
  const uint8_t* data_base_ = nullptr;  // start of the tensor data section
  std::vector<STensor> tensors_;
};

} // namespace intfer
