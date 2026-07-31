#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tatara::service {

// HTTP/1.1 message parsing and framing, with no socket anywhere. Parsing is
// the part that faces untrusted input, so it is proven in isolation before a
// listener exists.

inline constexpr std::size_t kMaximumHeaderBytes = 32 * 1024;
inline constexpr std::size_t kMaximumHeaderFields = 128;
inline constexpr std::uint64_t kMaximumBodyBytes = 64ull * 1024 * 1024;

enum class ParseState : std::uint8_t {
    Incomplete,
    Complete,
    MalformedRequestLine,
    InvalidHeaderSyntax,
    TooManyHeaders,
    HeadersTooLarge,
    UnsupportedVersion,
    MissingHost,
    DuplicateHost,
    UnsupportedTransferEncoding,
    DuplicateContentLength,
    InvalidContentLength,
    BodyTooLarge,
};

struct Header {
    std::string name;
    std::string value;
};

struct HttpRequest {
    std::string method;
    std::string target;
    std::vector<Header> headers;
    std::string body;
    bool keep_alive{true};

    std::optional<std::string_view> header(std::string_view name) const;
};

struct ParseResult {
    ParseState state{ParseState::Incomplete};
    HttpRequest request;
    std::size_t consumed_bytes{0};
};

// Parses one request from a buffer. Returns Incomplete when more bytes are
// needed, so a caller can accumulate without guessing at framing.
ParseResult parse_request(std::string_view buffer);

// Pure public error mapping. Incomplete produces status 0 because no response
// is ready; Complete produces 200; every parse failure closes the connection.
int parse_state_http_status(ParseState state) noexcept;
std::string_view parse_state_code(ParseState state) noexcept;

std::string_view reason_phrase(int status);

std::string render_response(int status, std::string_view content_type, std::string_view body,
                            bool keep_alive);

// Server-sent events. A stream is opened once, then each token is one event,
// and the stream is closed with the sentinel OpenAI clients expect.
std::string render_sse_headers(bool keep_alive);
std::string render_sse_event(std::string_view data);
std::string render_sse_done();

} // namespace tatara::service
