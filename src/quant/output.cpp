// src/quant/output.cpp
// t6 — final Stage-1 serialization: Native SYCL loader (INTFER1) + GGUF interop.
// Pure host C++ (no SYCL), so it builds and round-trips anywhere (mirrors the
// other Stage-1 modules). Consumes the *finalized* ModelIR (+ RotationSet) that
// t3→t4→t5 produce and writes it to disk in two formats; both are parseable back
// into an equivalent ModelIR + RotationSet for Stage-2 (t7) dequant validation.
#include "intfer/output.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace intfer {

// ===========================================================================
// Little-endian primitives
// ===========================================================================
namespace {
void put_u16(std::vector<uint8_t>& o, uint16_t v) {
  for (int i = 0; i < 2; ++i) o.push_back((uint8_t)(v >> (8 * i)));
}
void put_u32(std::vector<uint8_t>& o, uint32_t v) {
  for (int i = 0; i < 4; ++i) o.push_back((uint8_t)(v >> (8 * i)));
}
void put_u64(std::vector<uint8_t>& o, uint64_t v) {
  for (int i = 0; i < 8; ++i) o.push_back((uint8_t)(v >> (8 * i)));
}
uint16_t get_u16(const uint8_t* p) {
  return (uint16_t)(p[0] | (p[1] << 8));
}
uint32_t get_u32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
uint64_t get_u64(const uint8_t* p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= (uint64_t)p[i] << (8 * i);
  return v;
}
float get_f32(const uint8_t* p) {
  uint32_t u = get_u32(p);
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// ===========================================================================
// Minimal JSON writer + parser (used only for the native INTFER1 metadata)
// ===========================================================================
struct Json {
  enum T { Null, Bool, Int, Flt, Str, Arr, Obj } t = Null;
  bool b = false;
  int64_t i = 0;
  double f = 0.0;
  std::string s;
  std::vector<Json> arr;
  std::vector<std::pair<std::string, Json>> obj;
  Json() = default;
  Json(const std::string& v) : t(Str), s(v) {}
  Json(const char* v) : t(Str), s(v) {}
  Json(bool v) : t(Bool), b(v) {}
  Json(int64_t v) : t(Int), i(v) {}
  Json(double v) : t(Flt), f(v) {}
};

void jwrite(std::string& o, const Json& v) {
  switch (v.t) {
    case Json::Null: o += "null"; break;
    case Json::Bool: o += v.b ? "true" : "false"; break;
    case Json::Int: o += std::to_string(v.i); break;
    case Json::Flt: {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%.9g", v.f);
      o += buf;
      break;
    }
    case Json::Str: {
      o += '"';
      for (char c : v.s) {
        if (c == '"' || c == '\\') {
          o += '\\';
          o += c;
        } else if (c == '\n')
          o += "\\n";
        else if (c == '\t')
          o += "\\t";
        else if (c == '\r')
          o += "\\r";
        else if ((unsigned char)c < 0x20) {
          char b[8];
          std::snprintf(b, sizeof(b), "\\u%04x", (unsigned char)c);
          o += b;
        } else
          o += c;
      }
      o += '"';
      break;
    }
    case Json::Arr: {
      o += '[';
      for (size_t k = 0; k < v.arr.size(); ++k) {
        if (k) o += ',';
        jwrite(o, v.arr[k]);
      }
      o += ']';
      break;
    }
    case Json::Obj: {
      o += '{';
      for (size_t k = 0; k < v.obj.size(); ++k) {
        if (k) o += ',';
        o += '"';
        o += v.obj[k].first;
        o += "\":";
        jwrite(o, v.obj[k].second);
      }
      o += '}';
      break;
    }
  }
}

struct JsonParser {
  const char* p;
  const char* end;
  bool ok = true;
  void skipws() {
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
  }
  Json parse_value() {
    skipws();
    if (p >= end) {
      ok = false;
      return Json();
    }
    if (*p == '{') return parse_object();
    if (*p == '[') return parse_array();
    if (*p == '"') return parse_string();
    if (*p == 't' || *p == 'f') return parse_bool();
    if (*p == 'n') {
      p += 4;
      return Json();
    }
    return parse_number();
  }
  Json parse_object() {
    Json v;
    v.t = Json::Obj;
    ++p;  // consume '{'
    skipws();
    if (p < end && *p == '}') {
      ++p;
      return v;
    }
    while (p < end) {
      skipws();
      Json key = parse_string();
      if (!ok) return v;
      skipws();
      if (p >= end || *p != ':') {
        ok = false;
        return v;
      }
      ++p;  // consume ':'
      Json val = parse_value();
      if (!ok) return v;
      v.obj.push_back({key.s, std::move(val)});
      skipws();
      if (p < end && *p == ',') {
        ++p;
        continue;
      }
      if (p < end && *p == '}') {
        ++p;
        break;
      }
      ok = false;
      return v;
    }
    return v;
  }
  Json parse_array() {
    Json v;
    v.t = Json::Arr;
    ++p;  // consume '['
    skipws();
    if (p < end && *p == ']') {
      ++p;
      return v;
    }
    while (p < end) {
      Json val = parse_value();
      if (!ok) return v;
      v.arr.push_back(std::move(val));
      skipws();
      if (p < end && *p == ',') {
        ++p;
        continue;
      }
      if (p < end && *p == ']') {
        ++p;
        break;
      }
      ok = false;
      return v;
    }
    return v;
  }
  Json parse_string() {
    Json v;
    v.t = Json::Str;
    if (p >= end || *p != '"') {
      ok = false;
      return v;
    }
    ++p;
    std::string s;
    while (p < end && *p != '"') {
      if (*p == '\\') {
        ++p;
        if (p >= end) {
          ok = false;
          return v;
        }
        char c = *p++;
        switch (c) {
          case '"': s += '"'; break;
          case '\\': s += '\\'; break;
          case '/': s += '/'; break;
          case 'b': s += '\b'; break;
          case 'f': s += '\f'; break;
          case 'n': s += '\n'; break;
          case 'r': s += '\r'; break;
          case 't': s += '\t'; break;
          case 'u':
            p += 4;  // skip \uXXXX (our data is ASCII)
            s += '?';
            break;
          default: s += c; break;
        }
      } else {
        s += *p++;
      }
    }
    if (p < end && *p == '"') ++p;
    v.s = std::move(s);
    return v;
  }
  Json parse_bool() {
    Json v;
    v.t = Json::Bool;
    if (p < end && *p == 't') {
      p += 4;
      v.b = true;
    } else if (p < end && *p == 'f') {
      p += 5;
      v.b = false;
    } else
      ok = false;
    return v;
  }
  Json parse_number() {
    const char* start = p;
    while (p < end && (std::isdigit((unsigned char)*p) || *p == '-' || *p == '+' ||
                       *p == '.' || *p == 'e' || *p == 'E'))
      ++p;
    std::string tok(start, p);
    Json v;
    bool is_float = tok.find('.') != std::string::npos || tok.find('e') != std::string::npos ||
                    tok.find('E') != std::string::npos;
    if (tok.empty()) {
      ok = false;
      return v;
    }
    if (is_float) {
      v.t = Json::Flt;
      v.f = std::strtod(tok.c_str(), nullptr);
    } else {
      v.t = Json::Int;
      v.i = std::strtoll(tok.c_str(), nullptr, 10);
    }
    return v;
  }
};

const Json* jobj_find(const Json& o, const std::string& k) {
  if (o.t != Json::Obj) return nullptr;
  for (const auto& kv : o.obj)
    if (kv.first == k) return &kv.second;
  return nullptr;
}

// ===========================================================================
// Enum <-> name helpers (shared by native + GGUF writers)
// ===========================================================================
const char* role_name(TensorRole r) {
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
    case TensorRole::LINEAR_ATTN_QKV: return "LINEAR_ATTN_QKV";
    case TensorRole::LINEAR_ATTN_Z:   return "LINEAR_ATTN_Z";
    case TensorRole::LINEAR_ATTN_B:   return "LINEAR_ATTN_B";
    case TensorRole::LINEAR_ATTN_A:   return "LINEAR_ATTN_A";
    case TensorRole::LINEAR_ATTN_CONV:return "LINEAR_ATTN_CONV";
    case TensorRole::LINEAR_ATTN_OUT: return "LINEAR_ATTN_OUT";
    default: return "OTHER";
  }
}
TensorRole role_from_name(const std::string& s) {
  if (s == "EMBED") return TensorRole::EMBED;
  if (s == "LM_HEAD") return TensorRole::LM_HEAD;
  if (s == "ATTN_Q") return TensorRole::ATTN_Q;
  if (s == "ATTN_K") return TensorRole::ATTN_K;
  if (s == "ATTN_V") return TensorRole::ATTN_V;
  if (s == "ATTN_O") return TensorRole::ATTN_O;
  if (s == "MLP_GATE") return TensorRole::MLP_GATE;
  if (s == "MLP_UP") return TensorRole::MLP_UP;
  if (s == "MLP_DOWN") return TensorRole::MLP_DOWN;
  if (s == "NORM") return TensorRole::NORM;
  if (s == "ROPE") return TensorRole::ROPE;
  if (s == "LINEAR_ATTN_QKV") return TensorRole::LINEAR_ATTN_QKV;
  if (s == "LINEAR_ATTN_Z")   return TensorRole::LINEAR_ATTN_Z;
  if (s == "LINEAR_ATTN_B")   return TensorRole::LINEAR_ATTN_B;
  if (s == "LINEAR_ATTN_A")   return TensorRole::LINEAR_ATTN_A;
  if (s == "LINEAR_ATTN_CONV")return TensorRole::LINEAR_ATTN_CONV;
  if (s == "LINEAR_ATTN_OUT") return TensorRole::LINEAR_ATTN_OUT;
  return TensorRole::OTHER;
}
const char* scheme_name(QuantScheme s) {
  switch (s) {
    case QuantScheme::INT2: return "INT2";
    case QuantScheme::INT4: return "INT4";
    case QuantScheme::INT8: return "INT8";
    default: return "FP16";
  }
}
QuantScheme scheme_from_name(const std::string& s) {
  if (s == "INT2") return QuantScheme::INT2;
  if (s == "INT4") return QuantScheme::INT4;
  if (s == "INT8") return QuantScheme::INT8;
  return QuantScheme::FP16;
}
const char* scale_dtype_name(ScaleDtype s) {
  return s == ScaleDtype::FP8_E4M3 ? "FP8_E4M3" : "FP16";
}
ScaleDtype scale_dtype_from_name(const std::string& s) {
  return s == "FP8_E4M3" ? ScaleDtype::FP8_E4M3 : ScaleDtype::FP16;
}
const char* act_name(ActScheme a) {
  switch (a) {
    case ActScheme::BF16: return "BF16";
    case ActScheme::INT8_G64: return "INT8_G64";
    default: return "FP16";
  }
}
ActScheme act_from_name(const std::string& s) {
  if (s == "BF16") return ActScheme::BF16;
  if (s == "INT8_G64") return ActScheme::INT8_G64;
  return ActScheme::FP16;
}
const char* site_name(RotationSite s) {
  switch (s) {
    case RotationSite::QKV: return "QKV";
    case RotationSite::O: return "O";
    case RotationSite::FFN_IN: return "FFN_IN";
    case RotationSite::FFN_OUT: return "FFN_OUT";
    default: return "NONE";
  }
}
RotationSite site_from_name(const std::string& s) {
  if (s == "QKV") return RotationSite::QKV;
  if (s == "O") return RotationSite::O;
  if (s == "FFN_IN") return RotationSite::FFN_IN;
  if (s == "FFN_OUT") return RotationSite::FFN_OUT;
  return RotationSite::NONE;
}

// ===========================================================================
}  // anonymous helpers namespace

// Native SYCL loader (magic "INTFER1")
// ===========================================================================
std::vector<uint8_t> serialize_intfer(const ModelIR& model, const RotationSet& rs) {
  // ---- JSON metadata ----
  Json root;
  root.t = Json::Obj;
  root.obj.push_back({"model_id", Json(model.model_id)});
  root.obj.push_back({"n_layers", Json((int64_t)model.n_layers)});
  root.obj.push_back({"version", Json((int64_t)1)});

  Json tensors;
  tensors.t = Json::Arr;
  for (const auto& t : model.tensors) {
    Json o;
    o.t = Json::Obj;
    o.obj.push_back({"name", Json(t.name)});
    o.obj.push_back({"role", Json(role_name(t.role))});
    o.obj.push_back({"N", Json((int64_t)t.N)});
    o.obj.push_back({"K", Json((int64_t)t.K)});
    Json q;
    q.t = Json::Obj;
    q.obj.push_back({"scheme", Json(scheme_name(t.qcfg.scheme))});
    q.obj.push_back({"group_size", Json((int64_t)t.qcfg.group_size)});
    q.obj.push_back({"scale_dtype", Json(scale_dtype_name(t.qcfg.scale_dtype))});
    q.obj.push_back({"has_global_divisor", Json(t.qcfg.has_global_divisor)});
    q.obj.push_back({"act", Json(act_name(t.qcfg.act))});
    q.obj.push_back({"rotation_id", Json((int64_t)t.qcfg.rotation_id)});
    o.obj.push_back({"qcfg", q});
    o.obj.push_back({"code_offset", Json((int64_t)t.code_offset)});
    o.obj.push_back({"code_bytes", Json((int64_t)t.code_bytes)});
    o.obj.push_back({"scale_offset", Json((int64_t)t.scale_offset)});
    o.obj.push_back({"scale_bytes", Json((int64_t)t.scale_bytes)});
    o.obj.push_back({"global_offset", Json((int64_t)t.global_offset)});
    o.obj.push_back({"global_bytes", Json((int64_t)t.global_bytes)});
    o.obj.push_back({"est_error", Json((double)t.est_error)});
    tensors.arr.push_back(std::move(o));
  }
  root.obj.push_back({"tensors", tensors});

  // rotation region offsets
  uint64_t rot_region = 0;
  Json rots;
  rots.t = Json::Arr;
  for (const auto& r : rs.rotations) {
    Json o;
    o.t = Json::Obj;
    o.obj.push_back({"id", Json((int64_t)r.id)});
    o.obj.push_back({"site", Json(site_name(r.site))});
    o.obj.push_back({"dim", Json((int64_t)r.dim)});
    o.obj.push_back({"offset", Json((int64_t)rot_region)});
    rots.arr.push_back(std::move(o));
    rot_region += (uint64_t)r.dim * r.dim * 4;
  }
  root.obj.push_back({"rotations", rots});

  std::string js;
  jwrite(js, root);

  // ---- header + payload ----
  std::vector<uint8_t> out;
  const char magic[8] = {'I', 'N', 'T', 'F', 'E', 'R', '1', '\0'};
  for (int i = 0; i < 8; ++i) out.push_back((uint8_t)magic[i]);
  put_u32(out, 1);                                        // version
  put_u32(out, (uint32_t)model.tensors.size());           // n_tensors
  put_u64(out, model.code_buf.size());                    // code_len
  put_u64(out, model.scale_buf.size());                   // scale_len
  put_u64(out, model.global_buf.size());                  // global_len
  put_u32(out, (uint32_t)rs.rotations.size());            // n_rotations
  put_u16(out, 0);                                        // reserved
  put_u64(out, js.size());                                // json_len
  out.insert(out.end(), js.begin(), js.end());
  out.insert(out.end(), model.code_buf.begin(), model.code_buf.end());
  out.insert(out.end(), model.scale_buf.begin(), model.scale_buf.end());
  out.insert(out.end(), model.global_buf.begin(), model.global_buf.end());
  for (const auto& r : rs.rotations) {
    for (uint32_t k = 0; k < r.dim * r.dim; ++k) {
      uint32_t u;
      float f = r.mat[k];
      std::memcpy(&u, &f, 4);
      put_u32(out, u);
    }
  }
  return out;
}

bool deserialize_intfer(const uint8_t* data, size_t n, ModelIR& out_model, RotationSet& out_rs) {
  out_model = ModelIR{};
  out_rs = RotationSet{};
  if (n < 8 + 4 + 4 + 24 + 4 + 2 + 8) return false;
  if (std::memcmp(data, "INTFER1", 7) != 0) return false;
  size_t pos = 8;
  /*version*/ pos += 4;
  uint32_t n_tensors = get_u32(data + pos);
  pos += 4;
  uint64_t code_len = get_u64(data + pos);
  pos += 8;
  uint64_t scale_len = get_u64(data + pos);
  pos += 8;
  uint64_t global_len = get_u64(data + pos);
  pos += 8;
  uint32_t n_rot = get_u32(data + pos);
  pos += 4;
  pos += 2;  // reserved
  uint64_t json_len = get_u64(data + pos);
  pos += 8;
  if (pos + json_len > n) return false;
  JsonParser p{reinterpret_cast<const char*>(data + pos),
               reinterpret_cast<const char*>(data + pos + json_len), true};
  Json root = p.parse_value();
  if (!p.ok || root.t != Json::Obj) return false;
  pos += json_len;

  const Json* jid = jobj_find(root, "model_id");
  const Json* jnl = jobj_find(root, "n_layers");
  if (jid && jid->t == Json::Str) out_model.model_id = jid->s;
  if (jnl && jnl->t == Json::Int) out_model.n_layers = (uint32_t)jnl->i;

  const Json* jt = jobj_find(root, "tensors");
  if (!jt || jt->t != Json::Arr) return false;
  uint64_t expected = (uint64_t)jt->arr.size();
  (void)expected;
  if (jt->arr.size() != n_tensors) return false;

  for (const auto& to : jt->arr) {
    if (to.t != Json::Obj) return false;
    TensorIR t;
    const Json* v;
    if ((v = jobj_find(to, "name")) && v->t == Json::Str) t.name = v->s;
    if ((v = jobj_find(to, "role")) && v->t == Json::Str) t.role = role_from_name(v->s);
    if ((v = jobj_find(to, "N")) && v->t == Json::Int) t.N = (uint32_t)v->i;
    if ((v = jobj_find(to, "K")) && v->t == Json::Int) t.K = (uint32_t)v->i;
    const Json* q = jobj_find(to, "qcfg");
    if (q && q->t == Json::Obj) {
      if ((v = jobj_find(*q, "scheme")) && v->t == Json::Str) t.qcfg.scheme = scheme_from_name(v->s);
      if ((v = jobj_find(*q, "group_size")) && v->t == Json::Int) t.qcfg.group_size = (uint16_t)v->i;
      if ((v = jobj_find(*q, "scale_dtype")) && v->t == Json::Str) t.qcfg.scale_dtype = scale_dtype_from_name(v->s);
      if ((v = jobj_find(*q, "has_global_divisor")) && v->t == Json::Bool) t.qcfg.has_global_divisor = v->b;
      if ((v = jobj_find(*q, "act")) && v->t == Json::Str) t.qcfg.act = act_from_name(v->s);
      if ((v = jobj_find(*q, "rotation_id")) && v->t == Json::Int) t.qcfg.rotation_id = (uint8_t)v->i;
    }
    auto u64 = [&](const char* k) -> uint64_t {
      const Json* x = jobj_find(to, k);
      return (x && x->t == Json::Int) ? (uint64_t)x->i : 0;
    };
    t.code_offset = u64("code_offset");
    t.code_bytes = u64("code_bytes");
    t.scale_offset = u64("scale_offset");
    t.scale_bytes = u64("scale_bytes");
    t.global_offset = u64("global_offset");
    t.global_bytes = u64("global_bytes");
    if ((v = jobj_find(to, "est_error")) && v->t == Json::Flt) t.est_error = (float)v->f;
    out_model.tensors.push_back(std::move(t));
  }

  // buffers
  if (pos + code_len + scale_len + global_len > n) return false;
  out_model.code_buf.assign(data + pos, data + pos + code_len);
  pos += code_len;
  out_model.scale_buf.assign(data + pos, data + pos + scale_len);
  pos += scale_len;
  out_model.global_buf.assign(data + pos, data + pos + global_len);
  pos += global_len;

  // rotations
  const Json* jr = jobj_find(root, "rotations");
  if (!jr || jr->t != Json::Arr) return false;
  if (jr->arr.size() != n_rot) return false;
  for (const auto& ro : jr->arr) {
    if (ro.t != Json::Obj) return false;
    Rotation r;
    const Json* v;
    if ((v = jobj_find(ro, "id")) && v->t == Json::Int) r.id = (uint8_t)v->i;
    if ((v = jobj_find(ro, "site")) && v->t == Json::Str) r.site = site_from_name(v->s);
    if ((v = jobj_find(ro, "dim")) && v->t == Json::Int) r.dim = (uint32_t)v->i;
    uint64_t off = 0;
    if ((v = jobj_find(ro, "offset")) && v->t == Json::Int) off = (uint64_t)v->i;
    if (pos + off + (uint64_t)r.dim * r.dim * 4 > n) return false;
    const uint8_t* rp = data + pos + off;
    r.mat.resize((size_t)r.dim * r.dim);
    for (uint32_t k = 0; k < r.dim * r.dim; ++k) r.mat[k] = get_f32(rp + k * 4);
    out_rs.rotations.push_back(std::move(r));
  }
  return true;
}

// ===========================================================================
// GGUF interop (ggml-compatible container)
// ===========================================================================
namespace {

// Top-bit marker for intfer custom ggml types (so they never alias real ggml
// enums F16=1, F32=0, ...). Field positions in the lower 31 bits never overlap
// this bit.
static const uint32_t kIntferTag = 0x80000000u;

// Composite custom ggml_type encoding scheme|group|scale_dtype|rotation_id|act.
inline uint32_t intfer_ggml_type(QuantScheme s, uint16_t g, ScaleDtype sd, uint8_t rot_id,
                                 ActScheme act) {
  uint32_t v = kIntferTag;
  v |= ((uint32_t)s & 0x7u);            // bits 0..2  scheme
  v |= ((uint32_t)g & 0xFFu) << 4;      // bits 4..11 group_size
  v |= ((uint32_t)sd & 0x1u) << 12;    // bit 12     scale_dtype
  v |= ((uint32_t)rot_id & 0xFFu) << 16; // bits 16..23 rotation_id
  v |= ((uint32_t)act & 0x3u) << 24;    // bits 24..25 act
  return v;
}
inline void decode_intfer_ggml_type(uint32_t v, QuantScheme& s, uint16_t& g, ScaleDtype& sd,
                                     uint8_t& rot_id, ActScheme& act) {
  uint32_t b = v & ~kIntferTag;         // lower 31 bits
  s = (QuantScheme)(b & 0x7u);
  g = (uint16_t)((b >> 4) & 0xFFu);
  sd = (ScaleDtype)((b >> 12) & 0x1u);
  rot_id = (uint8_t)((b >> 16) & 0xFFu);
  act = (ActScheme)((b >> 24) & 0x3u);
}

struct GgufTensor {
  std::string name;
  uint32_t ggml_type = 0;
  std::vector<uint64_t> dims;
  std::vector<uint8_t> data;
};

void gguf_put_str(std::vector<uint8_t>& o, const std::string& s) {
  put_u64(o, s.size());
  o.insert(o.end(), s.begin(), s.end());
}
// GGUF metadata value-type enum (subset we read/write).
enum GvType : uint32_t {
  GV_UINT32 = 4,
  GV_STRING = 8,
  GV_ARRAY = 9,
  GV_UINT64 = 10,
};
// Byte length of a tensor's data given its ggml_type + dims. Mirrors how the
// writer packed it (INT2=4/byte, INT4=2/byte, INT8=1/byte, F16=2B, F32=4B).
uint64_t gguf_tensor_bytes(uint32_t type, const std::vector<uint64_t>& dims) {
  uint64_t elems = 1;
  for (auto d : dims) elems *= d;
  if (type == 0) return elems * 4;            // F32
  if (type == 1) return elems * 2;            // F16
  if (type & kIntferTag) {                    // intfer composite (low-bit codes)
    QuantScheme s;
    uint16_t g;
    ScaleDtype sd;
    uint8_t rid;
    ActScheme act;
    decode_intfer_ggml_type(type, s, g, sd, rid, act);
    switch (s) {
      case QuantScheme::INT2: return (elems + 3) / 4;
      case QuantScheme::INT4: return (elems + 1) / 2;
      case QuantScheme::INT8: return elems;
      default: return elems * 2;  // FP16 fallback
    }
  }
  return elems * 2;  // conservative fallback
}

}  // anonymous helpers namespace

std::vector<uint8_t> serialize_gguf(const ModelIR& model, const RotationSet& rs) {
  std::vector<GgufTensor> tensors;
  for (const auto& t : model.tensors) {
    GgufTensor gt;
    gt.name = t.name;
    gt.dims = {(uint64_t)t.N, (uint64_t)t.K};
    if (t.qcfg.scheme == QuantScheme::FP16) {
      gt.ggml_type = 1;  // GGML_TYPE_F16
    } else {
      gt.ggml_type = intfer_ggml_type(t.qcfg.scheme, t.qcfg.group_size, t.qcfg.scale_dtype,
                                      t.qcfg.rotation_id, t.qcfg.act);
      if (t.scale_bytes) {
        GgufTensor st;
        st.name = t.name + ".scale";
        st.dims = {t.scale_bytes / 2};
        st.ggml_type = 1;  // F16 scale
        st.data.assign(model.scale_buf.begin() + t.scale_offset,
                       model.scale_buf.begin() + t.scale_offset + t.scale_bytes);
        tensors.push_back(std::move(st));
      }
      if (t.global_bytes) {
        GgufTensor gt2;
        gt2.name = t.name + ".global";
        gt2.dims = {1};
        gt2.ggml_type = 0;  // F32 alpha
        gt2.data.assign(model.global_buf.begin() + t.global_offset,
                        model.global_buf.begin() + t.global_offset + t.global_bytes);
        tensors.push_back(std::move(gt2));
      }
    }
    gt.data.assign(model.code_buf.begin() + t.code_offset,
                   model.code_buf.begin() + t.code_offset + t.code_bytes);
    tensors.push_back(std::move(gt));
  }
  // rotation matrices as F32 tensors
  for (const auto& r : rs.rotations) {
    GgufTensor rt;
    rt.name = "intfer.rotation." + std::to_string(r.id) + "." + site_name(r.site);
    rt.dims = {(uint64_t)r.dim, (uint64_t)r.dim};
    rt.ggml_type = 0;  // F32
    rt.data.resize((size_t)r.dim * r.dim * 4);
    for (uint32_t k = 0; k < r.dim * r.dim; ++k) {
      uint32_t u;
      float f = r.mat[k];
      std::memcpy(&u, &f, 4);
      rt.data[k * 4 + 0] = (uint8_t)(u & 0xFF);
      rt.data[k * 4 + 1] = (uint8_t)((u >> 8) & 0xFF);
      rt.data[k * 4 + 2] = (uint8_t)((u >> 16) & 0xFF);
      rt.data[k * 4 + 3] = (uint8_t)((u >> 24) & 0xFF);
    }
    tensors.push_back(std::move(rt));
  }

  // ---- metadata KV ----
  struct MetaKV {
    std::string key;
    uint32_t type;
    std::vector<uint8_t> val;
  };
  std::vector<MetaKV> meta;
  auto add_str = [&](const std::string& k, const std::string& v) {
    MetaKV m;
    m.key = k;
    m.type = GV_STRING;
    gguf_put_str(m.val, v);
    meta.push_back(std::move(m));
  };
  auto add_u32 = [&](const std::string& k, uint32_t v) {
    MetaKV m;
    m.key = k;
    m.type = GV_UINT32;
    put_u32(m.val, v);
    meta.push_back(std::move(m));
  };
  add_str("general.architecture", "intfer");
  add_str("intfer.model_id", model.model_id);
  add_u32("intfer.n_layers", model.n_layers);

  // ---- sizes ----
  size_t meta_bytes = 0;
  for (const auto& m : meta) meta_bytes += (8 + m.key.size()) + 4 + m.val.size();
  size_t ti_bytes = 0;
  for (const auto& t : tensors) ti_bytes += (8 + t.name.size()) + 4 + 8 * t.dims.size() + 4 + 8;
  size_t header_size = 4 + 4 + 8 + 8 + meta_bytes + ti_bytes;
  size_t data_start = (header_size + 31) & ~(size_t)31;

  // assign 32-byte-aligned offsets within the data section
  std::vector<uint64_t> offsets(tensors.size());
  uint64_t off = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    off = (off + 31) & ~(uint64_t)31;
    offsets[i] = off;
    off += tensors[i].data.size();
  }

  // ---- write ----
  std::vector<uint8_t> out;
  out.push_back('G');
  out.push_back('G');
  out.push_back('U');
  out.push_back('F');  // magic
  put_u32(out, 3);     // version
  put_u64(out, tensors.size());
  put_u64(out, meta.size());
  for (const auto& m : meta) {
    gguf_put_str(out, m.key);
    put_u32(out, m.type);
    out.insert(out.end(), m.val.begin(), m.val.end());
  }
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto& t = tensors[i];
    gguf_put_str(out, t.name);
    put_u32(out, (uint32_t)t.dims.size());
    for (auto d : t.dims) put_u64(out, d);
    put_u32(out, t.ggml_type);
    put_u64(out, offsets[i]);
  }
  while (out.size() < data_start) out.push_back(0);
  // data section
  size_t data_total = off;
  std::vector<uint8_t> dsec(data_total, 0);
  for (size_t i = 0; i < tensors.size(); ++i)
    std::memcpy(dsec.data() + offsets[i], tensors[i].data.data(), tensors[i].data.size());
  out.insert(out.end(), dsec.begin(), dsec.end());
  return out;
}

bool deserialize_gguf(const uint8_t* data, size_t n, ModelIR& out_model, RotationSet& out_rs) {
  out_model = ModelIR{};
  out_rs = RotationSet{};
  if (n < 4 + 4 + 8 + 8) return false;
  if (std::memcmp(data, "GGUF", 4) != 0) return false;
  size_t pos = 4;
  /*version*/ pos += 4;
  uint64_t tensor_count = get_u64(data + pos);
  pos += 8;
  uint64_t kv_count = get_u64(data + pos);
  pos += 8;

  auto skip_value = [&](uint32_t type) -> bool {
    switch (type) {
      case 0: case 1: case 7: pos += 1; return true;       // uint8/int8/bool
      case 2: case 3: pos += 2; return true;               // uint16/int16
      case 4: case 5: case 6: pos += 4; return true;       // uint32/int32/float32
      case 10: case 11: case 12: pos += 8; return true;    // uint64/int64/float64
      case 8: {                                            // string
        if (pos + 8 > n) return false;
        uint64_t len = get_u64(data + pos);
        pos += 8 + len;
        return pos <= n;
      }
      case 9: {                                            // array
        if (pos + 4 + 8 > n) return false;
        uint32_t sub = get_u32(data + pos);
        pos += 4;
        uint64_t cnt = get_u64(data + pos);
        pos += 8;
        size_t esz = 0;
        switch (sub) {
          case 0: case 1: case 7: esz = 1; break;
          case 2: case 3: esz = 2; break;
          case 4: case 5: case 6: esz = 4; break;
          case 10: case 11: case 12: esz = 8; break;
          case 8: {
            // variable-size sub-elements: skip each individually
            for (uint64_t i = 0; i < cnt; ++i) {
              if (pos + 8 > n) return false;
              uint64_t len = get_u64(data + pos);
              pos += 8 + len;
              if (pos > n) return false;
            }
            return true;
          }
          default: return false;
        }
        pos += (size_t)cnt * esz;
        return pos <= n;
      }
      default: return false;
    }
  };

  for (uint64_t i = 0; i < kv_count; ++i) {
    if (pos + 8 > n) return false;
    uint64_t klen = get_u64(data + pos);
    pos += 8;
    if (pos + klen + 4 > n) return false;
    pos += klen;
    uint32_t type = get_u32(data + pos);
    pos += 4;
    if (!skip_value(type)) return false;
  }

  struct TInfo {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t ggml_type;
    uint64_t offset;
  };
  std::vector<TInfo> tinfo;
  for (uint64_t i = 0; i < tensor_count; ++i) {
    if (pos + 8 > n) return false;
    uint64_t nlen = get_u64(data + pos);
    pos += 8;
    if (pos + nlen + 4 > n) return false;
    std::string name((const char*)data + pos, nlen);
    pos += nlen;
    uint32_t ndims = get_u32(data + pos);
    pos += 4;
    if (pos + 8 * ndims + 4 + 8 > n) return false;
    TInfo ti;
    ti.name = name;
    for (uint32_t d = 0; d < ndims; ++d) ti.dims.push_back(get_u64(data + pos + 8 * d));
    pos += 8 * ndims;
    ti.ggml_type = get_u32(data + pos);
    pos += 4;
    ti.offset = get_u64(data + pos);
    pos += 8;
    tinfo.push_back(std::move(ti));
  }
  size_t header_size = pos;
  size_t data_start = (header_size + 31) & ~(size_t)31;

  // index aux tensors by base name
  std::map<std::string, const TInfo*> scale_of, global_of;
  std::vector<const TInfo*> mains, rotations;
  for (const auto& ti : tinfo) {
    if (ti.name.rfind("intfer.rotation.", 0) == 0) {
      rotations.push_back(&ti);
    } else if (ti.name.size() > 6 && ti.name.compare(ti.name.size() - 6, 6, ".scale") == 0) {
      scale_of[ti.name.substr(0, ti.name.size() - 6)] = &ti;
    } else if (ti.name.size() > 7 && ti.name.compare(ti.name.size() - 7, 7, ".global") == 0) {
      global_of[ti.name.substr(0, ti.name.size() - 7)] = &ti;
    } else {
      mains.push_back(&ti);
    }
  }

  // build ModelIR
  uint64_t code_off = 0, scale_off = 0, global_off = 0;
  for (const auto* ti : mains) {
    TensorIR t;
    t.name = ti->name;
    t.N = ti->dims.size() > 0 ? (uint32_t)ti->dims[0] : 0;
    t.K = ti->dims.size() > 1 ? (uint32_t)ti->dims[1] : 0;
    if (ti->ggml_type == 1) {  // F16
      t.qcfg.scheme = QuantScheme::FP16;
      t.qcfg.scale_dtype = ScaleDtype::FP16;
      t.qcfg.act = ActScheme::FP16;
    } else {
      QuantScheme s;
      uint16_t g;
      ScaleDtype sd;
      uint8_t rid;
      ActScheme act;
      decode_intfer_ggml_type(ti->ggml_type, s, g, sd, rid, act);
      t.qcfg.scheme = s;
      t.qcfg.group_size = g;
      t.qcfg.scale_dtype = sd;
      t.qcfg.rotation_id = rid;
      t.qcfg.act = act;
    }
    const TInfo* sc = nullptr;
    auto it = scale_of.find(ti->name);
    if (it != scale_of.end()) {
      sc = it->second;
      t.qcfg.has_global_divisor = true;  // scale present => low-bit; global checked below
    }
    const TInfo* gl = nullptr;
    auto itg = global_of.find(ti->name);
    if (itg != global_of.end()) {
      gl = itg->second;
      t.qcfg.has_global_divisor = true;
    }

    // code slice
    uint64_t cb = gguf_tensor_bytes(ti->ggml_type, ti->dims);
    if (data_start + ti->offset + cb > n) return false;
    t.code_offset = code_off;
    t.code_bytes = cb;
    out_model.code_buf.resize(code_off + cb);
    std::memcpy(out_model.code_buf.data() + code_off, data + data_start + ti->offset, cb);
    code_off += cb;

    if (sc) {
      uint64_t sb = gguf_tensor_bytes(sc->ggml_type, sc->dims);
      if (data_start + sc->offset + sb > n) return false;
      t.scale_offset = scale_off;
      t.scale_bytes = sb;
      out_model.scale_buf.resize(scale_off + sb);
      std::memcpy(out_model.scale_buf.data() + scale_off, data + data_start + sc->offset, sb);
      scale_off += sb;
    }
    if (gl) {
      uint64_t gb = gguf_tensor_bytes(gl->ggml_type, gl->dims);
      if (data_start + gl->offset + gb > n) return false;
      t.global_offset = global_off;
      t.global_bytes = gb;
      out_model.global_buf.resize(global_off + gb);
      std::memcpy(out_model.global_buf.data() + global_off, data + data_start + gl->offset, gb);
      global_off += gb;
    }
    out_model.tensors.push_back(std::move(t));
  }

  // rotations
  for (const auto* ti : rotations) {
    Rotation r;
    // name: intfer.rotation.<id>.<SITE>
    std::string body = ti->name.substr(std::string("intfer.rotation.").size());
    size_t dot = body.find('.');
    if (dot == std::string::npos) return false;
    r.id = (uint8_t)std::strtoul(body.substr(0, dot).c_str(), nullptr, 10);
    r.site = site_from_name(body.substr(dot + 1));
    r.dim = (uint32_t)(ti->dims.size() >= 1 ? ti->dims[0] : 0);
    uint64_t elems = r.dim * r.dim;
    if (data_start + ti->offset + elems * 4 > n) return false;
    r.mat.resize(elems);
    for (uint64_t k = 0; k < elems; ++k) r.mat[k] = get_f32(data + data_start + ti->offset + k * 4);
    out_rs.rotations.push_back(std::move(r));
  }
  return true;
}

// ===========================================================================
// File helpers
// ===========================================================================
bool write_file(const std::string& path, const std::vector<uint8_t>& data) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  size_t written = std::fwrite(data.data(), 1, data.size(), f);
  std::fclose(f);
  return written == data.size();
}
bool read_file(const std::string& path, std::vector<uint8_t>& out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (sz < 0) {
    std::fclose(f);
    return false;
  }
  out.resize((size_t)sz);
  size_t rd = std::fread(out.data(), 1, (size_t)sz, f);
  std::fclose(f);
  return rd == (size_t)sz;
}

}  // namespace intfer
