// include/intfer/safetensors.hpp
// Minimal host-only safetensors reader (no external deps). Supports the subset
// of dtypes emitted by HF transformers: BF16, FP16, FP32 (decoded to fp32).
// Memory-maps the file; tensor reads decode on demand.
//
// Supports sharded checkpoints: if the path ends in ".safetensors.index.json"
// (HF transformers multi-file format), the reader opens all referenced shards
// and reads the index's weight_map to resolve each tensor to its shard.
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
  int shard_idx = 0;  // which shard file this tensor lives in
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
  // Directory containing the safetensors files (for sharded lookup).
  const std::string& base_dir() const { return base_dir_; }

private:
  // Per-shard state (for sharded multi-file checkpoints).
  struct Shard {
    int fd = -1;
    void* map = nullptr;
    size_t size = 0;
    const uint8_t* data_base = nullptr;  // start of the tensor data section
  };
  int fd_ = -1;
  void* map_ = nullptr;
  size_t size_ = 0;
  const uint8_t* data_base_ = nullptr;  // start of the tensor data section
  std::vector<STensor> tensors_;
  std::vector<Shard> shards_;            // size 1 for single-file, N for sharded
  std::string base_dir_;                 // directory containing the file(s)
  int num_shards_ = 0;
};

} // namespace intfer
