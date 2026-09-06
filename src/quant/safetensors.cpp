// src/quant/safetensors.cpp
// Minimal safetensors reader. Format:
//   [u64 LE header_len][JSON header][tensor data...]
// JSON: { "metadata": {...}, "<name>": {"dtype","shape","data_offsets":[b,e]} }
// data_offsets are relative to the start of the tensor data section.
//
// Supports both single-file and sharded multi-file checkpoints:
//   * Single file: path ends with ".safetensors". One mmap, one shard.
//   * Sharded:     path ends with ".safetensors.index.json". The index JSON
//     has {"metadata":{"total_size":N},"weight_map":{name: shard_file, ...}}.
//     We open every referenced shard file (memory-mapped), parse each one's
//     header, and store all STensor entries tagged with their shard index.
#include "intfer/safetensors.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace intfer {

static inline float bf16_to_f32(uint16_t b) {
  uint32_t u = ((uint32_t)b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

static inline float fp16_to_f32(uint16_t h) {
  uint32_t sign = (h >> 15) & 1;
  uint32_t exp  = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t u;
  if (exp == 0) {
    if (mant == 0) u = sign << 31;
    else { // subnormal
      float f = std::ldexp((float)mant, -24);
      f = sign ? -f : f;
      return f;
    }
  } else if (exp == 0x1F) {
    u = (sign << 31) | (0xFFu << 23) | (mant << 13);
    float f; std::memcpy(&f, &u, 4); return f;
  } else {
    u = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f; std::memcpy(&f, &u, 4);
  return f;
}

// Parse one safetensors file header. Appends STensor entries to `out`,
// each tagged with `shard_idx`. Stores the data section base pointer at
// `*out_data_base` so the caller can keep it in its shard table.
static void parse_header_into(const uint8_t* base, size_t fsize, int shard_idx,
                              const uint8_t** out_data_base,
                              std::vector<STensor>& out) {
  uint64_t hlen = 0;
  std::memcpy(&hlen, base, 8);
  if (8 + hlen > fsize) return;
  std::string header((const char*)base + 8, (size_t)hlen);
  *out_data_base = base + 8 + hlen;

  size_t pos = 0;
  while (true) {
    size_t q = header.find('"', pos);
    if (q == std::string::npos) break;
    size_t q2 = header.find('"', q + 1);
    if (q2 == std::string::npos) break;
    std::string key = header.substr(q + 1, q2 - q - 1);
    size_t ob = header.find('{', q2);
    if (ob == std::string::npos) break;
    int depth = 0; size_t i = ob;
    for (; i < header.size(); ++i) {
      if (header[i] == '{') ++depth;
      else if (header[i] == '}') { --depth; if (depth == 0) break; }
    }
    if (depth != 0) break;
    std::string obj = header.substr(ob, i - ob + 1);

    if (key == "metadata") { pos = i + 1; continue; }
    if (obj.find("\"dtype\"") == std::string::npos ||
        obj.find("\"shape\"") == std::string::npos ||
        obj.find("\"data_offsets\"") == std::string::npos) { pos = i + 1; continue; }

    STensor t;
    t.name = key;
    t.shard_idx = shard_idx;
    size_t d0 = obj.find("\"dtype\"");
    size_t dq = obj.find('"', d0 + 7);
    size_t dq2 = obj.find('"', dq + 1);
    t.dtype = obj.substr(dq + 1, dq2 - dq - 1);
    size_t s0 = obj.find("\"shape\"");
    size_t sb = obj.find('[', s0);
    size_t se = obj.find(']', sb);
    std::string shp = obj.substr(sb + 1, se - sb - 1);
    size_t p = 0;
    while (p < shp.size()) {
      size_t comma = shp.find(',', p);
      std::string tok = (comma == std::string::npos) ? shp.substr(p) : shp.substr(p, comma - p);
      size_t a = tok.find_first_not_of(" \t");
      size_t b = tok.find_last_not_of(" \t");
      if (a == std::string::npos) { if (comma == std::string::npos) break; p = comma + 1; continue; }
      tok = tok.substr(a, b - a + 1);
      if (!tok.empty() && tok.find_first_not_of("0123456789") == std::string::npos)
        t.shape.push_back((size_t)std::stoull(tok));
      if (comma == std::string::npos) break;
      p = comma + 1;
    }
    size_t o0 = obj.find("\"data_offsets\"");
    size_t obr = obj.find('[', o0);
    size_t obr2 = obj.find(',', obr);
    size_t oend = obj.find(']', obr);
    t.begin = (size_t)std::stoull(obj.substr(obr + 1, obr2 - obr - 1));
    t.end   = (size_t)std::stoull(obj.substr(obr2 + 1, oend - obr2 - 1));
    out.push_back(std::move(t));
    pos = i + 1;
  }
}

bool SafetensorsReader::open(const std::string& path) {
  close();

  // Detect sharded format: HF index file ends in ".safetensors.index.json".
  static constexpr size_t kIndexSuffixLen = 23;  // ".safetensors.index.json"
  bool sharded = (path.size() >= kIndexSuffixLen &&
                  path.compare(path.size() - kIndexSuffixLen,
                               kIndexSuffixLen, ".safetensors.index.json") == 0);

  if (!sharded) {
    // ---- Single-file path ----
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) return false;
    struct stat st;
    if (::fstat(fd_, &st) != 0) { close(); return false; }
    size_ = (size_t)st.st_size;
    if (size_ < 8) { close(); return false; }
    map_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (map_ == MAP_FAILED) { close(); return false; }

    shards_.resize(1);
    shards_[0].fd   = fd_;
    shards_[0].map  = map_;
    shards_[0].size = size_;
    num_shards_ = 1;
    parse_header_into((const uint8_t*)map_, size_, 0,
                      &shards_[0].data_base, tensors_);
    data_base_ = shards_[0].data_base;  // legacy single-file accessor

    // Derive base_dir for callers that want to do relative lookups.
    size_t p = path.rfind('/');
    base_dir_ = (p == std::string::npos) ? std::string(".") : path.substr(0, p);
    return !tensors_.empty();
  }

  // ---- Sharded path ----
  // mmap the index file, parse weight_map to enumerate shards.
  int idx_fd = ::open(path.c_str(), O_RDONLY);
  if (idx_fd < 0) return false;
  struct stat st;
  if (::fstat(idx_fd, &st) != 0) { ::close(idx_fd); return false; }
  size_t idx_size = (size_t)st.st_size;
  if (idx_size < 8) { ::close(idx_fd); return false; }
  void* idx_map = ::mmap(nullptr, idx_size, PROT_READ, MAP_PRIVATE, idx_fd, 0);
  if (idx_map == MAP_FAILED) { ::close(idx_fd); return false; }
  std::string idx_json((const char*)idx_map, idx_size);
  ::munmap(idx_map, idx_size);
  ::close(idx_fd);

  // base_dir = directory containing the .index.json
  size_t p = path.rfind('/');
  base_dir_ = (p == std::string::npos) ? std::string(".") : path.substr(0, p);

  // Extract the weight_map object body.
  std::string wm_block;
  {
    size_t w = idx_json.find("\"weight_map\"");
    if (w != std::string::npos) {
      size_t ob = idx_json.find('{', w);
      if (ob != std::string::npos) {
        int depth = 0; size_t i = ob;
        for (; i < idx_json.size(); ++i) {
          if (idx_json[i] == '{') ++depth;
          else if (idx_json[i] == '}') { --depth; if (depth == 0) break; }
        }
        wm_block = idx_json.substr(ob, i - ob + 1);
      }
    }
  }
  if (wm_block.empty()) return false;

  // Iterate "name": "shard_file" pairs.
  std::unordered_map<std::string, std::string> name_to_shard;
  {
    size_t pos = 0;
    while (true) {
      size_t q = wm_block.find('"', pos);
      if (q == std::string::npos) break;
      size_t q2 = wm_block.find('"', q + 1);
      if (q2 == std::string::npos) break;
      std::string key = wm_block.substr(q + 1, q2 - q - 1);
      size_t colon = wm_block.find(':', q2);
      if (colon == std::string::npos) break;
      size_t vq = wm_block.find('"', colon);
      if (vq == std::string::npos) break;
      size_t vq2 = wm_block.find('"', vq + 1);
      if (vq2 == std::string::npos) break;
      std::string val = wm_block.substr(vq + 1, vq2 - vq - 1);
      name_to_shard[key] = val;
      pos = vq2 + 1;
    }
  }

  // Enumerate unique shard files and assign each an index.
  std::unordered_map<std::string, int> shard_to_idx;
  std::vector<std::string> shard_files;
  for (const auto& kv : name_to_shard) {
    if (shard_to_idx.count(kv.second)) continue;
    int idx = (int)shard_files.size();
    shard_to_idx[kv.second] = idx;
    shard_files.push_back(kv.second);
  }

  // mmap each shard.
  shards_.resize(shard_files.size());
  num_shards_ = (int)shard_files.size();
  for (size_t i = 0; i < shard_files.size(); ++i) {
    std::string full = base_dir_ + "/" + shard_files[i];
    int fd = ::open(full.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat ss;
    if (::fstat(fd, &ss) != 0) { ::close(fd); return false; }
    size_t fsize = (size_t)ss.st_size;
    if (fsize < 8) { ::close(fd); return false; }
    void* m = ::mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) { ::close(fd); return false; }
    shards_[i].fd = fd;
    shards_[i].map = m;
    shards_[i].size = fsize;
    parse_header_into((const uint8_t*)m, fsize, (int)i,
                      &shards_[i].data_base, tensors_);
  }

  // In sharded mode, the legacy single-file fields (fd_, map_, size_,
  // data_base_) are unused — callers must route through shards_[t.shard_idx].
  fd_ = -1; map_ = nullptr; size_ = 0; data_base_ = nullptr;
  return !tensors_.empty();
}

void SafetensorsReader::close() {
  for (auto& s : shards_) {
    if (s.map && s.map != MAP_FAILED) ::munmap(s.map, s.size);
    if (s.fd >= 0) ::close(s.fd);
    s.fd = -1; s.map = nullptr; s.size = 0; s.data_base = nullptr;
  }
  shards_.clear();
  num_shards_ = 0;
  if (map_ && map_ != MAP_FAILED) ::munmap(map_, size_);
  if (fd_ >= 0) ::close(fd_);
  map_ = nullptr; fd_ = -1; size_ = 0; data_base_ = nullptr;
  tensors_.clear();
}

bool SafetensorsReader::has(const std::string& name) const {
  for (const auto& t : tensors_) if (t.name == name) return true;
  return false;
}

bool SafetensorsReader::read(const std::string& name, std::vector<float>& out) const {
  const STensor* t = nullptr;
  for (const auto& x : tensors_) if (x.name == name) { t = &x; break; }
  if (!t) return false;
  size_t n = 1;
  for (size_t d : t->shape) n *= d;
  // Bounds-check the slice against the mapped shard before decoding.
  // A corrupt header (or a short/truncated shard file) must fail loudly
  // here instead of segfaulting a 1.2B-element BF16 loop downstream.
  const uint8_t* base = nullptr;
  size_t avail = 0;
  if (t->shard_idx >= 0 && t->shard_idx < (int)shards_.size()) {
    const Shard& s = shards_[t->shard_idx];
    if (!s.data_base || !s.map) return false;
    base = s.data_base;
    avail = (s.map && s.size > (size_t)(s.data_base - (const uint8_t*)s.map))
              ? s.size - (size_t)(s.data_base - (const uint8_t*)s.map) : 0;
  } else {
    if (!data_base_ || !map_) return false;
    base = data_base_;
    avail = (size_ > (size_t)(data_base_ - (const uint8_t*)map_))
              ? size_ - (size_t)(data_base_ - (const uint8_t*)map_) : 0;
  }
  size_t elem_bytes = (t->dtype == "F32" || t->dtype == "FP32") ? 4 : 2;
  if (t->dtype != "F32" && t->dtype != "FP32" &&
      t->dtype != "BF16" && t->dtype != "F16" && t->dtype != "FP16")
    return false;
  if (t->end < t->begin || t->end > avail) return false;
  if (t->end - t->begin < n * elem_bytes) return false;
  out.resize(n);
  const uint8_t* src = base + t->begin;
  if (t->dtype == "F32" || t->dtype == "FP32") {
    std::memcpy(out.data(), src, n * 4);
  } else if (t->dtype == "BF16") {
    const uint16_t* p = (const uint16_t*)src;
    for (size_t i = 0; i < n; ++i) out[i] = bf16_to_f32(p[i]);
  } else if (t->dtype == "F16" || t->dtype == "FP16") {
    const uint16_t* p = (const uint16_t*)src;
    for (size_t i = 0; i < n; ++i) out[i] = fp16_to_f32(p[i]);
  } else {
    return false;
  }
  return true;
}

} // namespace intfer
