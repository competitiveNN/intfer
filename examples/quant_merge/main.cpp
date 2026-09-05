// quant_merge — merge multiple partial .intifer files (produced by
// quant_model_real --layer-start/--layer-count) into a single
// native-SYCL .intifer artifact containing all tensors concatenated
// with correct offsets and deduplicated rotations / tensors.
#include "intfer/output.hpp"
#include "intfer/quant_spec.h"
#include "intfer/spinquant.hpp"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

using namespace intfer;

// ---- helpers ---------------------------------------------------------------
static const char* scheme_name(QuantScheme s) {
  switch (s) {
    case QuantScheme::INT2: return "INT2";
    case QuantScheme::INT4: return "INT4";
    case QuantScheme::INT8: return "INT8";
    default: break;
  }
  return "FP16";
}
static const char* scale_dtype_name(ScaleDtype d) {
  switch (d) { case ScaleDtype::FP8_E4M3: return "FP8_E4M3"; default: break; }
  return "FP16";
}
static const char* act_name(ActScheme a) {
  switch (a) { case ActScheme::BF16: return "BF16"; case ActScheme::INT8_G64: return "INT8_G64"; default: break; }
  return "FP16";
}
static const char* role_name(TensorRole r) {
  switch (r) {
    case TensorRole::EMBED: return "EMBED";
    case TensorRole::LM_HEAD: return "LM_HEAD";
    case TensorRole::ATTN_Q: return "ATTN_Q";
    case TensorRole::ATTN_K: return "ATTN_K";
    case TensorRole::ATTN_V: return "ATTN_V";
    case TensorRole::ATTN_O: return "ATTN_O";
    case TensorRole::MLP_GATE: return "MLP_GATE";
    case TensorRole::MLP_UP: return "MLP_UP";
    case TensorRole::MLP_DOWN: return "MLP_DOWN";
    case TensorRole::NORM: return "NORM";
    case TensorRole::ROPE: return "ROPE";
    default: break;
  }
  return "OTHER";
}
static const char* site_name(RotationSite s) {
  switch (s) {
    case RotationSite::QKV: return "QKV";
    case RotationSite::O: return "O";
    case RotationSite::FFN_IN: return "FFN_IN";
    case RotationSite::FFN_OUT: return "FFN_OUT";
    default: break;
  }
  return "NONE";
}

static void fput_u32(FILE* f, uint32_t v) {
  uint8_t buf[4];
  buf[0] = (uint8_t)(v >> 0); buf[1] = (uint8_t)(v >> 8);
  buf[2] = (uint8_t)(v >> 16); buf[3] = (uint8_t)(v >> 24);
  fwrite(buf, 1, 4, f);
}
static void fput_u64(FILE* f, uint64_t v) {
  uint8_t buf[8];
  for (int i = 0; i < 8; ++i) buf[i] = (uint8_t)(v >> (i * 8));
  fwrite(buf, 1, 8, f);
}

// ---- JSON builder (mirrors src/quant/output.cpp) --------------------------
static std::string build_json(const std::string& model_id,
                              uint32_t n_layers,
                              const std::vector<TensorIR>& tensors,
                              const std::vector<Rotation>& rots) {
  std::string js;
  js.reserve(1 << 20);
  js += "{\"model_id\":"; js += "\""; js += model_id; js += "\"";
  js += ",\"n_layers\":"; js += std::to_string(n_layers);
  js += ",\"version\":1";
  js += ",\"tensors\":[";
  bool first = true;
  for (const auto& t : tensors) {
    if (!first) js += ','; first = false;
    js += "{";
    js += "\"name\":"; js += "\""; js += t.name; js += "\",";
    js += "\"role\":"; js += "\""; js += role_name(t.role); js += "\",";
    js += "\"N\":"; js += std::to_string(t.N); js += ",";
    js += "\"K\":"; js += std::to_string(t.K); js += ",";
    js += "\"qcfg\":{"
         "\"scheme\":\""; js += scheme_name(t.qcfg.scheme); js += "\","
         "\"group_size\":"; js += std::to_string(t.qcfg.group_size); js += ","
         "\"scale_dtype\":\""; js += scale_dtype_name(t.qcfg.scale_dtype); js += "\","
         "\"has_global_divisor\":"; js += t.qcfg.has_global_divisor ? "true" : "false"; js += ","
         "\"act\":\""; js += act_name(t.qcfg.act); js += "\","
         "\"rotation_id\":"; js += std::to_string(t.qcfg.rotation_id);
    js += "},";
    js += "\"code_offset\":"; js += std::to_string(t.code_offset); js += ",";
    js += "\"code_bytes\":"; js += std::to_string(t.code_bytes); js += ",";
    js += "\"scale_offset\":"; js += std::to_string(t.scale_offset); js += ",";
    js += "\"scale_bytes\":"; js += std::to_string(t.scale_bytes); js += ",";
    js += "\"global_offset\":"; js += std::to_string(t.global_offset); js += ",";
    js += "\"global_bytes\":"; js += std::to_string(t.global_bytes); js += ",";
    js += "\"est_error\":"; js += std::to_string(t.est_error);
    js += "}";
  }
  js += "],\"rotations\":[";
  first = true;
  uint64_t rot_off = 0;
  for (const auto& r : rots) {
    if (!first) js += ','; first = false;
    js += "{";
    js += "\"id\":"; js += std::to_string(r.id); js += ",";
    js += "\"site\":\""; js += site_name(r.site); js += "\",";
    js += "\"dim\":"; js += std::to_string(r.dim); js += ",";
    js += "\"offset\":"; js += std::to_string(rot_off);
    js += "}";
    rot_off += (uint64_t)r.dim * r.dim * 4;
  }
  js += "]}";
  return js;
}

// ---- header helpers ---------------------------------------------------------
static void write_placeholder_header(FILE* f) {
  uint8_t hdr[54] = {};
  const char magic[8] = {'I','N','T','F','E','R','1','\0'};
  std::memcpy(hdr, magic, 8);
  fwrite(hdr, 1, 54, f);
}

static void patch_header(FILE* f, long hdr_pos, uint32_t n_tensors,
                         uint64_t code_len, uint64_t scale_len, uint64_t global_len,
                         uint32_t n_rot, uint64_t json_len) {
  uint8_t hdr[54] = {};
  const char magic[8] = {'I','N','T','F','E','R','1','\0'};
  std::memcpy(hdr, magic, 8);
  hdr[8] = 1;
  hdr[12] = (uint8_t)(n_tensors >> 0); hdr[13] = (uint8_t)(n_tensors >> 8);
  hdr[14] = (uint8_t)(n_tensors >> 16); hdr[15] = (uint8_t)(n_tensors >> 24);
  for (int i = 0; i < 8; ++i) hdr[16 + i] = (uint8_t)(code_len >> (i * 8));
  for (int i = 0; i < 8; ++i) hdr[24 + i] = (uint8_t)(scale_len >> (i * 8));
  for (int i = 0; i < 8; ++i) hdr[32 + i] = (uint8_t)(global_len >> (i * 8));
  hdr[40] = (uint8_t)(n_rot >> 0); hdr[41] = (uint8_t)(n_rot >> 8);
  hdr[42] = (uint8_t)(n_rot >> 16); hdr[43] = (uint8_t)(n_rot >> 24);
  hdr[44] = 0; hdr[45] = 0;
  for (int i = 0; i < 8; ++i) hdr[46 + i] = (uint8_t)(json_len >> (i * 8));
  fseek(f, hdr_pos, SEEK_SET);
  fwrite(hdr, 1, 54, f);
  fseek(f, 0, SEEK_END);
}

// ---- main ------------------------------------------------------------------
struct TensorOffset {
  uint64_t code_off = 0, scale_off = 0, global_off = 0;
};

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
      "Usage: quant_merge <out.intifer> <chunk1.intifer> [chunk2 ...]\n");
    return 1;
  }
  std::string out_path = argv[1];
  std::string payload_path = out_path + ".payload";
  std::string json_path = out_path + ".json.tmp";

  std::unordered_map<std::string, size_t> name_idx;
  std::vector<Rotation> merged_rs;
  std::unordered_map<uint8_t, int> rot_id_map;

  uint32_t n_layers = 0;
  std::string model_id = "Qwen3-32B";
  size_t total_dropped = 0;
  uint64_t cp = 0, sp = 0, gp = 0;
  uint32_t n_tensors = 0;
  std::string js;

  FILE* fhdr = fopen(out_path.c_str(), "wb");
  FILE* fpld = fopen(payload_path.c_str(), "wb");
  if (!fhdr || !fpld) {
    std::fprintf(stderr, "FAIL: cannot open output files\n");
    if (fhdr) fclose(fhdr); if (fpld) fclose(fpld);
    return 1;
  }
  write_placeholder_header(fhdr);

  struct TensorMeta {
    TensorIR t;
    TensorOffset off;
  };
  std::vector<TensorMeta> metas;

  for (int a = 2; a < argc; ++a) {
    std::string path = argv[a];
    std::vector<uint8_t> blob;
    if (!read_file(path, blob)) {
      std::fprintf(stderr, "FAIL: cannot read %s\n", path.c_str()); return 1;
    }
    std::fprintf(stderr, "chunk %d: read %zu bytes from %s\n", a - 1, blob.size(), path.c_str());
    ModelIR chunk_m; RotationSet chunk_rs;
    if (!deserialize_intfer(blob.data(), blob.size(), chunk_m, chunk_rs)) {
      std::fprintf(stderr, "FAIL: deserialize failed for %s\n", path.c_str());
      return 1;
    }
    std::fprintf(stderr, "chunk %d: deserialized %zu tensors, %zu rotations\n",
                 a - 1, chunk_m.tensors.size(), chunk_rs.rotations.size());
    if (!chunk_m.tensors.empty() && chunk_m.model_id != "Qwen3-32B")
      model_id = chunk_m.model_id;

    for (auto& t : chunk_m.tensors) {
      int L = parse_layer_index(t.name);
      uint32_t L1 = (L >= 0) ? (uint32_t)(L + 1) : 0;
      if (L1 > n_layers) n_layers = L1;

      if (t.qcfg.rotation_id != 0) {
        uint8_t old_id = t.qcfg.rotation_id;
        auto it = rot_id_map.find(old_id);
        if (it == rot_id_map.end()) {
          const Rotation* found = nullptr;
          for (const auto& r : chunk_rs.rotations) {
            if (r.id == old_id) { found = &r; break; }
          }
          if (!found) {
            std::fprintf(stderr, "FAIL: rotation_id %d not found in %s\n",
                         old_id, path.c_str());
            return 1;
          }
          // Rotation ids are 1-based (0 = none) in both chunk files and the
          // merged file, and identical shared rotations (same site+dim+matrix)
          // must map to ONE merged id. Dedupe by (site, dim, matrix) instead
          // of blindly appending per chunk (the old code appended a copy per
          // chunk AND assigned a 0-based index, so merged rid 0 silently
          // disabled rotations and later chunks pointed at the wrong matrix).
          int new_id = -1;
          for (size_t ri = 0; ri < merged_rs.size(); ++ri) {
            const Rotation& e = merged_rs[ri];
            if (e.id != 0 && e.site == found->site && e.dim == found->dim &&
                e.mat.size() == found->mat.size() &&
                std::memcmp(e.mat.data(), found->mat.data(),
                            e.mat.size() * sizeof(float)) == 0) {
              new_id = (int)e.id;
              break;
            }
          }
          if (new_id < 0) {
            // Fresh id: max existing id + 1 (keeps 1-based numbering).
            uint8_t mx = 0;
            for (const auto& e : merged_rs) mx = std::max(mx, e.id);
            new_id = (int)mx + 1;
            Rotation rc = *found;
            rc.id = (uint8_t)new_id;
            merged_rs.push_back(std::move(rc));
          }
          rot_id_map[old_id] = new_id;
        }
        t.qcfg.rotation_id = (uint8_t)rot_id_map[old_id];
      }

      if (name_idx.count(t.name)) { ++total_dropped; continue; }

      // Save the LOCAL chunk offsets before we overwrite them.
      uint64_t local_code_off = t.code_offset;
      uint64_t local_scale_off = t.scale_offset;
      uint64_t local_global_off = t.global_offset;

      TensorOffset off;
      off.code_off  = cp;
      off.scale_off = sp;
      off.global_off = gp;
      t.code_offset  = off.code_off;
      t.scale_offset = off.scale_off;
      t.global_offset = off.global_off;

      if (t.code_bytes) {
        fwrite(chunk_m.code_buf.data() + local_code_off, 1, t.code_bytes, fpld);
        cp += t.code_bytes;
      }
      if (t.scale_bytes) {
        fwrite(chunk_m.scale_buf.data() + local_scale_off, 1, t.scale_bytes, fpld);
        sp += t.scale_bytes;
      }
      if (t.global_bytes) {
        fwrite(chunk_m.global_buf.data() + local_global_off, 1, t.global_bytes, fpld);
        gp += t.global_bytes;
      }

      name_idx[t.name] = metas.size();
      TensorMeta m;
      m.t = std::move(t);
      m.off = off;
      metas.push_back(std::move(m));
    }
    std::vector<uint8_t>().swap(blob);
    std::fprintf(stderr, "chunk %d done: %zu tensors total, payload=%.1f MiB\n",
                 a - 1, metas.size(), (double)cp / (1024.0 * 1024.0));
  }

  // Append rotations to payload
  for (const auto& r : merged_rs) {
    for (uint32_t k = 0; k < r.dim * r.dim; ++k) {
      uint32_t u; std::memcpy(&u, &r.mat[k], 4);
      fput_u32(fpld, u);
    }
  }

  fclose(fpld);
  fclose(fhdr);

  {
    std::vector<TensorIR> all_tensors;
    all_tensors.reserve(metas.size());
    for (auto& m : metas) all_tensors.push_back(std::move(m.t));
    js = build_json(model_id, n_layers, all_tensors, merged_rs);
    n_tensors = (uint32_t)all_tensors.size();
    std::vector<TensorIR>().swap(all_tensors);
    std::vector<TensorMeta>().swap(metas);
  }

  {
    FILE* jf = fopen(json_path.c_str(), "wb");
    if (!jf) { std::fprintf(stderr, "FAIL: cannot write json tmp\n"); return 1; }
    fwrite(js.data(), 1, js.size(), jf);
    fclose(jf);
  }

  std::string final_path = out_path + ".final";
  {
    FILE* fhdr2 = fopen(out_path.c_str(), "rb");
    FILE* fpld2 = fopen(payload_path.c_str(), "rb");
    FILE* fdst = fopen(final_path.c_str(), "wb");
    if (!fhdr2 || !fpld2 || !fdst) {
      std::fprintf(stderr, "FAIL: cannot open files for final assembly\n");
      if (fhdr2) fclose(fhdr2); if (fpld2) fclose(fpld2); if (fdst) fclose(fdst);
      return 1;
    }
    uint8_t old_hdr[54] = {};
    fread(old_hdr, 1, 54, fhdr2);
    uint8_t new_hdr[54] = {};
    std::memcpy(new_hdr, old_hdr, 54);
    new_hdr[8] = 1;
    std::memcpy(new_hdr + 12, &n_tensors, 4);
    std::memcpy(new_hdr + 16, &cp, 8);
    std::memcpy(new_hdr + 24, &sp, 8);
    std::memcpy(new_hdr + 32, &gp, 8);
    uint32_t _nr = (uint32_t)merged_rs.size();
    std::memcpy(new_hdr + 40, &_nr, 4);
    uint64_t json_len = js.size();
    std::memcpy(new_hdr + 46, &json_len, 8);
    fwrite(new_hdr, 1, 54, fdst);
    FILE* jf = fopen(json_path.c_str(), "rb");
    char buf[65536];
    size_t nr;
    while ((nr = fread(buf, 1, sizeof(buf), jf)) > 0) fwrite(buf, 1, nr, fdst);
    fclose(jf);
    std::remove(json_path.c_str());
    while ((nr = fread(buf, 1, sizeof(buf), fpld2)) > 0) fwrite(buf, 1, nr, fdst);
    fclose(fhdr2); fclose(fpld2); fclose(fdst);
  }
  std::remove(out_path.c_str());
  std::remove(payload_path.c_str());
  std::rename(final_path.c_str(), out_path.c_str());

  {
    FILE* vf = fopen(out_path.c_str(), "rb");
    uint8_t vhdr[100] = {};
    fread(vhdr, 1, 100, vf);
    fclose(vf);
    std::fprintf(stderr, "FINAL verify: magic=%c%c%c%c%c%c%c%c hdr[8]=%u n_tensors=%u\n",
                 vhdr[0], vhdr[1], vhdr[2], vhdr[3], vhdr[4], vhdr[5], vhdr[6], vhdr[7],
                 vhdr[8], n_tensors);
    uint64_t _jl = *(uint64_t*)(vhdr + 46);
    std::fprintf(stderr, "FINAL json_len=%llu\n", (unsigned long long)_jl);
  }

  std::printf("Merged %d chunks -> %s\n", argc - 2, out_path.c_str());
  std::printf("  tensors: %u (dropped %zu duplicate names)\n",
              n_tensors, total_dropped);
  std::printf("  rotations: %zu\n", merged_rs.size());
  std::printf("  size: %.1f MiB (code %.1f MiB, scale %.1f MiB, global %.1f MiB)\n",
              ((double)js.size() + (double)cp + (double)sp + (double)gp) / (1024 * 1024),
              (double)cp / (1024 * 1024),
              (double)sp / (1024 * 1024),
              (double)gp / (1024 * 1024));
  return 0;
}
