// examples/intfer_cli/openai.hpp
// OpenAI-compatible API layer for intfer (header-only, no third-party deps).
//
// Surface:
//   GET  /health            {"status":"ok","model":"..."} (+ /v1/models alias form)
//   GET  /v1/models         {"object":"list","data":[{"id","object":"model",
//                            "created","owned_by"}]}
//   POST /v1/chat/completions   OpenAI Chat Completions (stream + non-stream)
//   POST /v1/completions        legacy Completions (same engine, prompt in/out)
//
// Chat request fields honored: model, messages[{role,content}], max_tokens,
// max_completion_tokens, temperature, top_p, top_k (extension), seed, stream,
// stop (string | string[]), n (=1 only), user. Unknown fields are ignored so
// generic OpenAI clients keep working. `logprobs`/`audio`/`modalities` get
// a clean 400 (unsupported, stated explicitly — no silent wrong answers).
// `tools` is accepted-but-ignored (text-only model, no tool calls emitted);
// `tool_choice` is accepted only for "none"/"auto".
//
// Roles: system/user/assistant/tool/developer. Content may be a string or a
// ChatML-style content-part array ([{type:text,text:...}]); image/audio parts
// are rejected with 400 (text-only engine). The Jinja chat_template is NOT
// executed (no Jinja runtime in C++); messages render with the model's
// documented ChatML layout:
//   [<|im_start|>system\n...\n]<|im_start|>user\n...\n...<|im_start|>assistant\n[<think>\n\n</think>\n\n]
// which matches chat_template.jinja's non-tool branch exactly (verified
// against Qwen3.5-0.8B's template; tool/model content is out of scope).
#pragma once

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace intfer_openai {

// ---------------------------------------------------------------------------
// Minimal JSON value model (objects preserve insertion order).
// ---------------------------------------------------------------------------
// NOTE: object members are held by shared_ptr because JVal is recursive —
// std::pair<std::string, JVal> would instantiate pair with an incomplete
// type (hard error in libstdc++ type traits). shared_ptr is fine with an
// incomplete type, and values are never mutated through find(), so the
// shared (non-deep) copy semantics are safe.
struct JVal {
  enum class T { Null, Bool, Num, Str, Arr, Obj } t = T::Null;
  bool b = false;
  double num = 0;
  std::string s;
  std::vector<JVal> a;
  struct Member {
    std::string key;
    std::shared_ptr<JVal> val;
  };
  std::vector<Member> o;

  static JVal null() { return JVal(); }
  static JVal boolean(bool v) { JVal j; j.t = T::Bool; j.b = v; return j; }
  static JVal number(double v) { JVal j; j.t = T::Num; j.num = v; return j; }
  static JVal str(const std::string& v) { JVal j; j.t = T::Str; j.s = v; return j; }
  static JVal arr() { JVal j; j.t = T::Arr; return j; }
  static JVal obj() { JVal j; j.t = T::Obj; return j; }
  void push(const JVal& v) { a.push_back(v); }
  void set(const std::string& k, const JVal& v) {
    o.push_back(Member{k, std::make_shared<JVal>(v)});
  }
  const JVal* find(const std::string& k) const {
    if (t != T::Obj) return nullptr;
    for (auto& m : o)
      if (m.key == k) return m.val.get();
    return nullptr;
  }
  bool is_num() const { return t == T::Num; }
  bool is_str() const { return t == T::Str; }
  bool is_arr() const { return t == T::Arr; }
  bool is_obj() const { return t == T::Obj; }
  double num_or(double d) const { return t == T::Num ? num : d; }
  const std::string& str_or(const std::string& d) const { return t == T::Str ? s : d; }
};

struct JParser {
  const char* p = nullptr;
  const char* end = nullptr;
  std::string err;

  explicit JParser(const std::string& s) : p(s.data()), end(s.data() + s.size()) {}

  void ws() {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
  }
  bool lit(const char* w) {
    size_t n = strlen(w);
    if ((size_t)(end - p) < n || memcmp(p, w, n) != 0) return false;
    p += n;
    return true;
  }
  bool parse_string(std::string& out) {
    if (p >= end || *p != '"') return false;
    p++;
    out.clear();
    while (p < end) {
      char c = *p++;
      if (c == '"') return true;
      if (c == '\\') {
        if (p >= end) return false;
        char e = *p++;
        if (e == '"') out += '"';
        else if (e == '\\') out += '\\';
        else if (e == '/') out += '/';
        else if (e == 'n') out += '\n';
        else if (e == 'r') out += '\r';
        else if (e == 't') out += '\t';
        else if (e == 'b') out += '\b';
        else if (e == 'f') out += '\f';
        else if (e == 'u') {
          unsigned v = 0;
          if (end - p < 4) return false;
          for (int i = 0; i < 4; i++) {
            char h = *p++;
            v <<= 4;
            if (h >= '0' && h <= '9') v |= (unsigned)(h - '0');
            else if (h >= 'a' && h <= 'f') v |= (unsigned)(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') v |= (unsigned)(h - 'A' + 10);
            else return false;
          }
          // BMP -> UTF-8 (surrogates passed through as replacement).
          if (v < 0x80) out += (char)v;
          else if (v < 0x800) {
            out += (char)(0xC0 | (v >> 6));
            out += (char)(0x80 | (v & 0x3F));
          } else {
            out += (char)(0xE0 | (v >> 12));
            out += (char)(0x80 | ((v >> 6) & 0x3F));
            out += (char)(0x80 | (v & 0x3F));
          }
        } else {
          return false;
        }
      } else {
        out += c;
      }
    }
    return false;
  }
  bool parse_num(double& out) {
    const char* s = p;
    if (p < end && (*p == '-')) p++;
    bool any = false;
    while (p < end && isdigit((unsigned char)*p)) { p++; any = true; }
    if (p < end && *p == '.') {
      p++;
      while (p < end && isdigit((unsigned char)*p)) { p++; any = true; }
    }
    if (!any) return false;
    if (p < end && (*p == 'e' || *p == 'E')) {
      p++;
      if (p < end && (*p == '+' || *p == '-')) p++;
      bool eany = false;
      while (p < end && isdigit((unsigned char)*p)) { p++; eany = true; }
      if (!eany) return false;
    }
    try {
      out = std::stod(std::string(s, p));
    } catch (...) {
      return false;
    }
    return true;
  }
  bool parse_val(JVal& out) {
    ws();
    if (p >= end) return false;
    if (*p == '{') {
      p++;
      out = JVal::obj();
      ws();
      if (p < end && *p == '}') { p++; return true; }
      while (p < end) {
        ws();
        std::string k;
        if (!parse_string(k)) return false;
        ws();
        if (p >= end || *p != ':') return false;
        p++;
        JVal v;
        if (!parse_val(v)) return false;
        out.set(k, v);
        ws();
        if (p >= end) return false;
        if (*p == ',') { p++; continue; }
        if (*p == '}') { p++; return true; }
        return false;
      }
      return false;
    }
    if (*p == '[') {
      p++;
      out = JVal::arr();
      ws();
      if (p < end && *p == ']') { p++; return true; }
      while (p < end) {
        JVal v;
        if (!parse_val(v)) return false;
        out.push(v);
        ws();
        if (p >= end) return false;
        if (*p == ',') { p++; continue; }
        if (*p == ']') { p++; return true; }
        return false;
      }
      return false;
    }
    if (*p == '"') {
      std::string s;
      if (!parse_string(s)) return false;
      out = JVal::str(s);
      return true;
    }
    if (*p == 't') {
      if (!lit("true")) return false;
      out = JVal::boolean(true);
      return true;
    }
    if (*p == 'f') {
      if (!lit("false")) return false;
      out = JVal::boolean(false);
      return true;
    }
    if (*p == 'n') {
      if (!lit("null")) return false;
      out = JVal::null();
      return true;
    }
    double n = 0;
    if (!parse_num(n)) return false;
    out = JVal::number(n);
    return true;
  }
  bool parse(JVal& out) {
    if (!parse_val(out)) return false;
    ws();
    return p == end;
  }
};

static inline std::string json_escape_str(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (c < 0x20) {
          char b[8];
          snprintf(b, sizeof(b), "\\u%04x", c);
          o += b;
        } else {
          o += (char)c;
        }
    }
  }
  return o;
}

static inline void json_render(const JVal& j, std::string& o) {
  switch (j.t) {
    case JVal::T::Null: o += "null"; break;
    case JVal::T::Bool: o += (j.b ? "true" : "false"); break;
    case JVal::T::Num: {
      char b[32];
      double v = j.num;
      long long iv = (long long)v;
      if (v == (double)iv)
        snprintf(b, sizeof(b), "%lld", iv);
      else
        snprintf(b, sizeof(b), "%.6g", v);
      o += b;
      break;
    }
    case JVal::T::Str: o += '"'; o += json_escape_str(j.s); o += '"'; break;
    case JVal::T::Arr:
      o += '[';
      for (size_t i = 0; i < j.a.size(); i++) {
        if (i) o += ',';
        json_render(j.a[i], o);
      }
      o += ']';
      break;
    case JVal::T::Obj:
      o += '{';
      for (size_t i = 0; i < j.o.size(); i++) {
        if (i) o += ',';
        o += '"';
        o += json_escape_str(j.o[i].key);
        o += "\":";
        json_render(*j.o[i].val, o);
      }
      o += '}';
      break;
  }
}
static inline std::string json_dump(const JVal& j) {
  std::string o;
  json_render(j, o);
  return o;
}

static inline JVal make_error(const std::string& msg, const std::string& type,
                              const std::string& code = "") {
  JVal e = JVal::obj();
  JVal inner = JVal::obj();
  inner.set("message", JVal::str(msg));
  inner.set("type", JVal::str(type));
  if (!code.empty()) inner.set("code", JVal::str(code));
  e.set("error", inner);
  return e;
}

// ---------------------------------------------------------------------------
// Chat request model + validation.
// ---------------------------------------------------------------------------
struct ChatMessage {
  std::string role;
  std::string content;
};

struct ChatRequest {
  std::string model;
  std::vector<ChatMessage> messages;
  int max_tokens = 0;  // 0 = server default
  double temperature = 0.0;
  bool has_temperature = false;
  double top_p = 0.0;
  bool has_top_p = false;
  int top_k = 0;
  int seed = 0;
  bool has_seed = false;
  bool stream = false;
  std::vector<std::string> stop;
  std::string validation_error;  // set when parse fails
};

// Flatten message content: string passthrough, or [{type:text,...}] parts.
// Returns false + reason for non-textual parts.
static bool flatten_content(const JVal& c, std::string& out, std::string& reason) {
  if (c.is_str()) {
    out = c.s;
    return true;
  }
  if (c.is_arr()) {
    out.clear();
    for (auto& part : c.a) {
      if (!part.is_obj()) {
        reason = "message content parts must be objects";
        return false;
      }
      const JVal* t = part.find("type");
      std::string ty = t ? t->str_or("") : "text";
      if (ty == "text") {
        const JVal* tx = part.find("text");
        if (tx && tx->is_str()) out += tx->s;
      } else if (ty == "image_url" || ty == "image" || ty == "input_audio" ||
                 ty == "file") {
        reason = "multimodal content parts are not supported by this text-only engine";
        return false;
      } else {
        reason = std::string("unsupported content part type: ") + ty;
        return false;
      }
    }
    return true;
  }
  if (c.t == JVal::T::Null) {
    out.clear();
    return true;
  }
  reason = "message content must be a string or content-part array";
  return false;
}

static inline bool parse_chat_request(const JVal& j, ChatRequest& out) {
  const JVal* m = j.find("model");
  out.model = (m && m->is_str()) ? m->s : "";
  const JVal* msgs = j.find("messages");
  if (!msgs || !msgs->is_arr() || msgs->a.empty()) {
    out.validation_error = "messages must be a non-empty array";
    return false;
  }
  for (auto& mj : msgs->a) {
    if (!mj.is_obj()) {
      out.validation_error = "each message must be an object";
      return false;
    }
    const JVal* r = mj.find("role");
    const JVal* c = mj.find("content");
    if (!r || !r->is_str() || r->s.empty()) {
      out.validation_error = "each message needs a role string";
      return false;
    }
    std::string role = r->s;
    if (role != "system" && role != "user" && role != "assistant" &&
        role != "tool" && role != "developer") {
      out.validation_error = std::string("unsupported role: ") + role;
      return false;
    }
    std::string content;
    if (c) {
      std::string reason;
      if (!flatten_content(*c, content, reason)) {
        out.validation_error = reason;
        return false;
      }
    }
    out.messages.push_back({role, content});
  }
  const JVal* mt = j.find("max_tokens");
  if (!mt) mt = j.find("max_completion_tokens");
  if (mt && mt->is_num()) out.max_tokens = (int)mt->num;
  const JVal* tp = j.find("temperature");
  if (tp && tp->is_num()) {
    out.temperature = tp->num;
    out.has_temperature = true;
  }
  const JVal* pp = j.find("top_p");
  if (pp && pp->is_num()) {
    out.top_p = pp->num;
    out.has_top_p = true;
  }
  const JVal* tk = j.find("top_k");
  if (tk && tk->is_num()) out.top_k = (int)tk->num;
  const JVal* sd = j.find("seed");
  if (sd && sd->is_num()) {
    out.seed = (int)sd->num;
    out.has_seed = true;
  }
  const JVal* st = j.find("stream");
  if (st && st->t == JVal::T::Bool) out.stream = st->b;
  const JVal* sp = j.find("stop");
  if (sp) {
    if (sp->is_str())
      out.stop.push_back(sp->s);
    else if (sp->is_arr()) {
      for (auto& e : sp->a) {
        if (!e.is_str()) {
          out.validation_error = "stop must be a string or array of strings";
          return false;
        }
        out.stop.push_back(e.s);
      }
    } else if (sp->t != JVal::T::Null) {
      out.validation_error = "stop must be a string or array of strings";
      return false;
    }
  }
  const JVal* n = j.find("n");
  if (n && n->is_num() && (int)n->num != 1) {
    out.validation_error = "n > 1 is not supported (single completion only)";
    return false;
  }
  // Explicit rejections: silent wrong answers are worse than a 400.
  // Tool calling is not implemented (text-only model): `tools` is accepted
  // and ignored (chat UIs like open-webui attach tool lists to every
  // request; answering text beats a 400), but a `tool_choice` that REQUIRES
  // a tool call cannot be honored. `response_format` text/null is accepted
  // (output is plain text anyway); structured formats are rejected.
  // `reasoning_effort` is accepted and ignored (single fixed pipeline).
  for (const char* k : {"logprobs", "top_logprobs",
                        "audio", "modalities"}) {
    const JVal* v = j.find(k);
    if (v && v->t != JVal::T::Null) {
      if (v->t == JVal::T::Bool && !v->b) continue;
      if (v->t == JVal::T::Arr && v->a.empty()) continue;
      out.validation_error = std::string(k) + " is not supported by this server";
      return false;
    }
  }
  {
    const JVal* tc = j.find("tool_choice");
    if (tc && tc->t != JVal::T::Null) {
      bool ok = false;
      if (tc->is_str())
        ok = (tc->s == "none" || tc->s == "auto");
      else if (tc->t == JVal::T::Bool && !tc->b)
        ok = true;
      if (!ok) {
        out.validation_error = "tool_choice requiring a tool call is not supported by this server";
        return false;
      }
    }
  }
  {
    const JVal* rf = j.find("response_format");
    if (rf && rf->t != JVal::T::Null && rf->is_obj()) {
      const JVal* t = rf->find("type");
      if (!t || !t->is_str() || (t->s != "text" && t->s != "json_object")) {
        out.validation_error = "response_format is not supported by this server (text output only)";
        return false;
      }
    }
  }
  if (out.max_tokens < 0) {
    out.validation_error = "max_tokens must be >= 0";
    return false;
  }
  return true;
}

// Render messages with the model's ChatML layout (non-tool branch of the
// Qwen3.5 chat_template.jinja):
//   [<|im_start|>system\n{sys}<|im_end|>\n]
//   <|im_start|>user\n{content}<|im_end|>\n ...
//   <|im_start|>assistant\n[<think>\n\n</think>\n\n]
static inline std::string render_chat_prompt(const std::vector<ChatMessage>& msgs,
                                             bool thinking_suffix = true) {
  std::string o;
  for (auto& m : msgs) {
    if (m.role == "system") {
      o += "<|im_start|>system\n" + m.content + "<|im_end|>\n";
    } else if (m.role == "user" || m.role == "developer") {
      o += "<|im_start|>user\n" + m.content + "<|im_end|>\n";
    } else if (m.role == "assistant") {
      o += "<|im_start|>assistant\n" + m.content + "<|im_end|>\n";
    } else if (m.role == "tool") {
      o += "<|im_start|>user\n<tool_response>\n" + m.content +
           "\n</tool_response><|im_end|>\n";
    }
  }
  o += "<|im_start|>assistant\n";
  if (thinking_suffix) o += "<think>\n\n</think>\n\n";
  return o;
}

}  // namespace intfer_openai
