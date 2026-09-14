// examples/intfer_cli/http.hpp
// Minimal dependency-free HTTP/1.1 server for intfer (POSIX sockets only).
//
// A proper little server, not an MVP shim:
//   * request-line + header parsing with sane limits (64KB headers, 8MB body)
//   * Content-Length bodies, `Expect: 100-continue`, keep-alive
//   * full buffered responses AND chunked SSE streaming (`begin_stream`)
//   * thread-per-connection with a connection cap (429 beyond it)
//   * recv/send timeouts, access-log hook, clean shutdown via stop()
//
// No third-party deps; C++20 + POSIX. Header-only so the intfer_cli target
// picks it up without CMake changes.
#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cctype>
#include <cstring>
#include <ctime>
#include <functional>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace intfer_http {

struct HttpRequest {
  std::string method;   // "GET", "POST", "OPTIONS", ...
  std::string target;   // raw request target
  std::string path;     // target minus query string
  std::string query;    // raw query string (no leading '?')
  std::map<std::string, std::string> headers;  // lowercased names
  std::string body;

  std::string header(const std::string& name) const {
    auto it = headers.find(name);
    return it == headers.end() ? std::string() : it->second;
  }
};

static inline std::string to_lower(std::string s) {
  for (char& c : s) c = (char)std::tolower((unsigned char)c);
  return s;
}
static inline std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && (s[a] == ' ' || s[a] == '\t')) a++;
  while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t')) b--;
  return s.substr(a, b - a);
}

static const char* status_text(int code) {
  switch (code) {
    case 100: return "Continue";
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 413: return "Content Too Large";
    case 429: return "Too Many Requests";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    default: return "Unknown";
  }
}

// Write helper with a poll-bounded send (avoids hanging on dead peers).
static bool send_all(int fd, const char* data, size_t len, int timeout_ms = 30000) {
  size_t sent = 0;
  while (sent < len) {
    struct pollfd pfd{fd, POLLOUT, 0};
    int r = poll(&pfd, 1, timeout_ms);
    if (r <= 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))) return false;
    ssize_t n = ::send(fd, data + sent, len - sent, MSG_NOSIGNAL);
    if (n <= 0) return false;
    sent += (size_t)n;
  }
  return true;
}
static bool send_all(int fd, const std::string& s, int timeout_ms = 30000) {
  return send_all(fd, s.data(), s.size(), timeout_ms);
}

// Per-connection responder: either one buffered response or a chunked
// event stream (SSE). Chunk writes return false when the peer is gone.
class Responder {
 public:
  explicit Responder(int fd) : fd_(fd) {}

  bool send_response(int status, const std::string& content_type,
                     const std::string& body,
                     const std::vector<std::pair<std::string, std::string>>& extra = {}) {
    if (headers_sent_) return false;
    headers_sent_ = true;
    std::string h = "HTTP/1.1 " + std::to_string(status) + " " + status_text(status) + "\r\n";
    h += "Content-Type: " + content_type + "\r\n";
    h += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    h += "Access-Control-Allow-Origin: *\r\n";
    for (auto& kv : extra) h += kv.first + ": " + kv.second + "\r\n";
    h += "Connection: close\r\n\r\n";
    if (!send_all(fd_, h)) return false;
    if (!body.empty() && !send_all(fd_, body)) return false;
    return true;
  }

  bool send_json(int status, const std::string& body,
                 const std::vector<std::pair<std::string, std::string>>& extra = {}) {
    return send_response(status, "application/json", body, extra);
  }

  // Start a `Transfer-Encoding: chunked` stream (SSE uses text/event-stream).
  bool begin_stream(const std::string& content_type,
                    const std::vector<std::pair<std::string, std::string>>& extra = {}) {
    if (headers_sent_) return false;
    headers_sent_ = true;
    chunked_ = true;
    std::string h = "HTTP/1.1 200 OK\r\n";
    h += "Content-Type: " + content_type + "\r\n";
    h += "Transfer-Encoding: chunked\r\n";
    h += "Cache-Control: no-cache\r\n";
    h += "Access-Control-Allow-Origin: *\r\n";
    for (auto& kv : extra) h += kv.first + ": " + kv.second + "\r\n";
    h += "Connection: close\r\n\r\n";
    return send_all(fd_, h);
  }

  bool chunk(const std::string& data) {
    if (!chunked_ || closed_) return false;
    if (data.empty()) return true;
    char hb[32];
    int n = snprintf(hb, sizeof(hb), "%zx\r\n", data.size());
    if (!send_all(fd_, hb, (size_t)n)) { closed_ = true; return false; }
    if (!send_all(fd_, data)) { closed_ = true; return false; }
    if (!send_all(fd_, std::string("\r\n"))) { closed_ = true; return false; }
    return true;
  }

  bool finish() {
    if (!chunked_) return true;
    chunked_ = false;
    return send_all(fd_, "0\r\n\r\n");
  }

  bool peer_gone() const { return closed_; }

 private:
  int fd_ = -1;
  bool headers_sent_ = false;
  bool chunked_ = false;
  bool closed_ = false;
};

using Handler = std::function<void(const HttpRequest&, Responder&)>;

class HttpServer {
 public:
  HttpServer() = default;
  ~HttpServer() { stop(); }

  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  void on_request(Handler h) { handler_ = std::move(h); }
  void on_log(std::function<void(const std::string&, const std::string&, int, long)> f) {
    log_ = std::move(f);
  }
  void set_max_connections(int n) { max_conn_ = n; }

  // Bind 127.0.0.1:port when host is empty. Returns false on bind/listen error.
  bool listen(const std::string& host, int port) {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) return false;
    int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (host.empty() || host == "localhost") {
      addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    } else if (host == "0.0.0.0") {
      addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else {
      if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        ::close(listen_fd_);
        listen_fd_ = -1;
        return false;
      }
    }
    if (::bind(listen_fd_, (struct sockaddr*)&addr, sizeof(addr)) != 0 ||
       ::listen(listen_fd_, 128) != 0) {
      ::close(listen_fd_);
      listen_fd_ = -1;
      return false;
    }
    // Report the effective endpoint (lets port 0 pick an ephemeral port).
    struct sockaddr_in bound{};
    socklen_t blen = sizeof(bound);
    if (::getsockname(listen_fd_, (struct sockaddr*)&bound, &blen) == 0)
      bound_port_ = ntohs(bound.sin_port);
    else
      bound_port_ = port;
    return true;
  }

  int bound_port() const { return bound_port_; }

  // Accept loop. Returns when stop() is called (or on fatal accept error).
  void run() {
    running_ = true;
    while (running_) {
      struct pollfd pfd{listen_fd_, POLLIN, 0};
      int r = poll(&pfd, 1, 200);
      if (!running_) break;
      if (r <= 0) continue;
      int fd = ::accept(listen_fd_, nullptr, nullptr);
      if (fd < 0) {
        if (running_) continue;
        break;
      }
      struct timeval tv{60, 0};
      ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
      ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
      std::thread(&HttpServer::serve_one, this, fd).detach();
    }
  }

  void stop() {
    running_ = false;
    if (listen_fd_ >= 0) {
      ::shutdown(listen_fd_, SHUT_RDWR);
      ::close(listen_fd_);
      listen_fd_ = -1;
    }
  }

  int active_connections() const { return active_.load(); }

 private:
  static bool read_line(int fd, std::string& out, size_t cap) {
    out.clear();
    char c = 0;
    while (out.size() < cap) {
      ssize_t n = ::recv(fd, &c, 1, 0);
      if (n <= 0) return false;
      if (c == '\n') break;
      if (c != '\r') out.push_back(c);
    }
    return true;
  }

  // Read exactly len bytes (bounded); false on timeout/close/oversize.
  static bool read_body(int fd, std::string& out, size_t len) {
    out.clear();
    if (len > 8 * 1024 * 1024) return false;
    out.resize(len);
    size_t got = 0;
    while (got < len) {
      ssize_t n = ::recv(fd, out.data() + got, len - got, 0);
      if (n <= 0) return false;
      got += (size_t)n;
    }
    return true;
  }

  void serve_one(int fd) {
    active_++;
    auto t0 = std::chrono::steady_clock::now();
    int status = 500;
    std::string method, path;
    bool keep = false;
    do {
      keep = false;
      HttpRequest req;
      std::string line;
      if (!read_line(fd, line, 8192)) break;
      if (line.empty()) {
        // Tolerate one stray CRLF between keep-alive requests.
        if (!read_line(fd, line, 8192) || line.empty()) break;
      }
      // Request line: METHOD SP TARGET SP HTTP/x.
      {
        size_t s1 = line.find(' ');
        size_t s2 = s1 == std::string::npos ? std::string::npos : line.find(' ', s1 + 1);
        if (s1 == std::string::npos || s2 == std::string::npos) {
          Responder(fd).send_response(400, "text/plain", "malformed request line");
          status = 400;
          break;
        }
        req.method = line.substr(0, s1);
        req.target = line.substr(s1 + 1, s2 - s1 - 1);
      }
      method = req.method;
      {
        size_t q = req.target.find('?');
        if (q == std::string::npos) req.path = req.target;
        else {
          req.path = req.target.substr(0, q);
          req.query = req.target.substr(q + 1);
        }
      }
      path = req.path;
      // Headers (64KB cap total).
      size_t htotal = 0;
      bool hok = true;
      for (;;) {
        if (!read_line(fd, line, 8192)) { hok = false; break; }
        if (line.empty()) break;
        htotal += line.size();
        if (htotal > 64 * 1024) { hok = false; break; }
        size_t c = line.find(':');
        if (c == std::string::npos) { hok = false; break; }
        req.headers[to_lower(trim(line.substr(0, c)))] = trim(line.substr(c + 1));
      }
      if (!hok) {
        Responder(fd).send_response(400, "text/plain", "malformed headers");
        status = 400;
        break;
      }
      std::string conn = req.header("connection");
      keep = (to_lower(conn) == "keep-alive");
      // Body.
      size_t clen = 0;
      std::string cl = req.header("content-length");
      if (!cl.empty()) {
        try {
          clen = (size_t)std::stoul(trim(cl));
        } catch (...) {
          Responder(fd).send_response(400, "text/plain", "bad Content-Length");
          status = 400;
          break;
        }
      }
      if (req.header("transfer-encoding").find("chunked") != std::string::npos) {
        Responder(fd).send_response(501, "text/plain", "chunked request bodies unsupported");
        status = 501;
        break;
      }
      if (clen > 8 * 1024 * 1024) {
        Responder(fd).send_response(413, "text/plain", "request body too large");
        status = 413;
        // Drain nothing; close (can't reliably resync).
        break;
      }
      if (to_lower(req.header("expect")).find("100-continue") != std::string::npos) {
        if (!send_all(fd, "HTTP/1.1 100 Continue\r\n\r\n")) break;
      }
      if (clen > 0 && !read_body(fd, req.body, clen)) break;

      if (active_.load() > max_conn_ && req.path != "/health") {
        Responder(fd).send_json(429, "{\"error\":{\"message\":\"server busy\",\"type\":\"server_error\"}}",
                                {{"Retry-After", "5"}});
        status = 429;
        break;
      }
      Responder rsp(fd);
      status = 200;
      try {
        if (handler_) handler_(req, rsp);
      } catch (...) {
        if (!rsp.peer_gone()) rsp.send_json(500, "{\"error\":{\"message\":\"internal error\",\"type\":\"server_error\"}}");
        status = 500;
      }
      // Handler always closes (Connection: close); no second request.
      keep = false;
    } while (keep);
    long ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - t0)
                  .count();
    if (log_) log_(method, path, status, ms);
    ::close(fd);
    active_--;
  }

  Handler handler_;
  std::function<void(const std::string&, const std::string&, int, long)> log_;
  int listen_fd_ = -1;
  int bound_port_ = 0;
  int max_conn_ = 64;
  std::atomic<int> active_{0};
  std::atomic<bool> running_{false};
};

}  // namespace intfer_http
