// src/quant/safetensors.cpp
// Minimal safetensors reader. Format:
//   [u64 LE header_len][JSON header][tensor data...]
// JSON: { "metadata": {...}, "<name>": {"dtype","shape","data_offsets":[b,e]} }
// data_offsets are relative to the start of the tensor data section.
#include "intfer/safetensors.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

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

bool SafetensorsReader::open(const std::string& path) {
  close();
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) return false;
  struct stat st;
  if (::fstat(fd_, &st) != 0) { close(); return false; }
  size_ = (size_t)st.st_size;
  if (size_ < 8) { close(); return false; }
  map_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (map_ == MAP_FAILED) { close(); return false; }

  const uint8_t* base = (const uint8_t*)map_;
  uint64_t hlen = 0;
  std::memcpy(&hlen, base, 8);
  if (8 + hlen > size_) { close(); return false; }
  std::string header((const char*)base + 8, (size_t)hlen);

  // Tiny JSON parse: extract tensor entries by name keys.
  size_t pos = 0;
  data_base_ = base + 8 + hlen;
  while (true) {
    size_t q = header.find('"', pos);
    if (q == std::string::npos) break;
    size_t q2 = header.find('"', q + 1);
    if (q2 == std::string::npos) break;
    std::string key = header.substr(q + 1, q2 - q - 1);
    // find the matching '{' for this key's object
    size_t ob = header.find('{', q2);
    if (ob == std::string::npos) break;
    // naive brace match
    int depth = 0; size_t i = ob;
    for (; i < header.size(); ++i) {
      if (header[i] == '{') ++depth;
      else if (header[i] == '}') { --depth; if (depth == 0) break; }
    }
    if (depth != 0) break;
    std::string obj = header.substr(ob, i - ob + 1);

    // Skip the metadata block.
    if (key == "metadata") { pos = i + 1; continue; }
    // Must have dtype + shape + data_offsets to be a tensor.
    if (obj.find("\"dtype\"") == std::string::npos ||
        obj.find("\"shape\"") == std::string::npos ||
        obj.find("\"data_offsets\"") == std::string::npos) { pos = i + 1; continue; }

    STensor t;
    t.name = key;
    // dtype
    size_t d0 = obj.find("\"dtype\"");
    size_t dq = obj.find('"', d0 + 7);
    size_t dq2 = obj.find('"', dq + 1);
    t.dtype = obj.substr(dq + 1, dq2 - dq - 1);
    // shape: numbers inside the shape array
    size_t s0 = obj.find("\"shape\"");
    size_t sb = obj.find('[', s0);
    size_t se = obj.find(']', sb);
    std::string shp = obj.substr(sb + 1, se - sb - 1);
    size_t p = 0;
    while (p < shp.size()) {
      size_t comma = shp.find(',', p);
      std::string tok = (comma == std::string::npos) ? shp.substr(p) : shp.substr(p, comma - p);
      // strip whitespace
      size_t a = tok.find_first_not_of(" \t");
      size_t b = tok.find_last_not_of(" \t");
      if (a == std::string::npos) { if (comma == std::string::npos) break; p = comma + 1; continue; }
      tok = tok.substr(a, b - a + 1);
      if (!tok.empty() && tok.find_first_not_of("0123456789") == std::string::npos)
        t.shape.push_back((size_t)std::stoull(tok));
      if (comma == std::string::npos) break;
      p = comma + 1;
    }
    // data_offsets [begin, end]
    size_t o0 = obj.find("\"data_offsets\"");
    size_t obr = obj.find('[', o0);
    size_t obr2 = obj.find(',', obr);
    size_t oend = obj.find(']', obr);
    t.begin = (size_t)std::stoull(obj.substr(obr + 1, obr2 - obr - 1));
    t.end   = (size_t)std::stoull(obj.substr(obr2 + 1, oend - obr2 - 1));
    tensors_.push_back(std::move(t));
    pos = i + 1;
  }
  return true;
}

void SafetensorsReader::close() {
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
  out.resize(n);
  const uint8_t* src = data_base_ + t->begin;
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
