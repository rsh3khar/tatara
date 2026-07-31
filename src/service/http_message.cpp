#include "tatara/service/http_message.h"

#include <algorithm>
#include <charconv>
#include <sstream>
#include <utility>

namespace tatara::service {
namespace {

constexpr std::string_view kHeaderTerminator = "\r\n\r\n";

std::string lowercase(std::string_view value) {
    std::string out(value);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char character) {
        if (character >= static_cast<unsigned char>('A') &&
            character <= static_cast<unsigned char>('Z')) {
            return static_cast<char>(character + ('a' - 'A'));
        }
        return static_cast<char>(character);
    });
    return out;
}

std::string_view trim(std::string_view value) {
    const auto first = value.find_first_not_of(" \t");
    if (first == std::string_view::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t");
    return value.substr(first, last - first + 1);
}

bool is_token_byte(unsigned char byte) {
    if ((byte >= static_cast<unsigned char>('0') && byte <= static_cast<unsigned char>('9')) ||
        (byte >= static_cast<unsigned char>('A') && byte <= static_cast<unsigned char>('Z')) ||
        (byte >= static_cast<unsigned char>('a') && byte <= static_cast<unsigned char>('z'))) {
        return true;
    }
    constexpr std::string_view punctuation = "!#$%&'*+-.^_`|~";
    return punctuation.find(static_cast<char>(byte)) != std::string_view::npos;
}

bool is_valid_token(std::string_view value) {
    return !value.empty() && std::all_of(value.begin(), value.end(),
                                         [](unsigned char byte) { return is_token_byte(byte); });
}

bool is_valid_request_target(std::string_view value) {
    return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char byte) {
        return byte >= 0x21u && byte <= 0x7eu;
    });
}

bool is_valid_field_value(std::string_view value) {
    return std::all_of(value.begin(), value.end(), [](unsigned char byte) {
        return byte == '\t' || (byte >= 0x20u && byte != 0x7fu);
    });
}

} // namespace

std::optional<std::string_view> HttpRequest::header(std::string_view name) const {
    const std::string wanted = lowercase(name);
    for (const auto& entry : headers) {
        if (entry.name == wanted) {
            return std::string_view(entry.value);
        }
    }
    return std::nullopt;
}

ParseResult parse_request(std::string_view buffer) {
    ParseResult result;
    const std::size_t maximum_header_wire_bytes =
        kMaximumHeaderBytes + kHeaderTerminator.size();
    const std::string_view header_window =
        buffer.substr(0, std::min(buffer.size(), maximum_header_wire_bytes));
    const auto terminator = header_window.find(kHeaderTerminator);
    if (terminator == std::string_view::npos) {
        // Bound both the scan and retained bytes before the header is
        // complete. Bytes past the header limit remain viable only when an
        // end-of-buffer suffix is a proper prefix of CRLF-CRLF whose candidate
        // start is within the limit.
        bool has_viable_terminator_prefix = buffer.size() <= kMaximumHeaderBytes;
        if (!has_viable_terminator_prefix &&
            buffer.size() < maximum_header_wire_bytes) {
            for (std::size_t prefix_size = 1; prefix_size < kHeaderTerminator.size();
                 ++prefix_size) {
                if (prefix_size > buffer.size()) {
                    break;
                }
                const std::size_t candidate_start = buffer.size() - prefix_size;
                if (candidate_start <= kMaximumHeaderBytes &&
                    buffer.substr(candidate_start) ==
                        kHeaderTerminator.substr(0, prefix_size)) {
                    has_viable_terminator_prefix = true;
                    break;
                }
            }
        }
        if (!has_viable_terminator_prefix) {
            result.state = ParseState::HeadersTooLarge;
        }
        return result;
    }
    if (terminator > kMaximumHeaderBytes) {
        result.state = ParseState::HeadersTooLarge;
        return result;
    }

    HttpRequest request;
    std::string_view head = buffer.substr(0, terminator);
    // A request with no headers is legal, so an absent CRLF means the head is
    // the request line alone rather than a malformed message.
    const auto first_line_end = head.find("\r\n");
    const std::string_view request_line =
        first_line_end == std::string_view::npos ? head : head.substr(0, first_line_end);
    const auto method_end = request_line.find(' ');
    if (method_end == std::string_view::npos ||
        !is_valid_token(request_line.substr(0, method_end))) {
        result.state = ParseState::MalformedRequestLine;
        return result;
    }
    const auto target_end = request_line.find(' ', method_end + 1);
    if (target_end == std::string_view::npos || !is_valid_request_target(request_line.substr(
                                                    method_end + 1, target_end - method_end - 1))) {
        result.state = ParseState::MalformedRequestLine;
        return result;
    }
    request.method.assign(request_line.substr(0, method_end));
    request.target.assign(request_line.substr(method_end + 1, target_end - method_end - 1));
    const std::string_view version = request_line.substr(target_end + 1);

    std::string_view rest = first_line_end == std::string_view::npos
                                ? std::string_view{}
                                : head.substr(first_line_end + 2);
    while (!rest.empty()) {
        if (request.headers.size() == kMaximumHeaderFields) {
            result.state = ParseState::TooManyHeaders;
            return result;
        }
        const auto line_end = rest.find("\r\n");
        const std::string_view line =
            line_end == std::string_view::npos ? rest : rest.substr(0, line_end);
        const auto colon = line.find(':');
        if (colon == std::string_view::npos || !is_valid_token(line.substr(0, colon)) ||
            !is_valid_field_value(line.substr(colon + 1))) {
            result.state = ParseState::InvalidHeaderSyntax;
            return result;
        }
        request.headers.push_back(
            {lowercase(line.substr(0, colon)), std::string(trim(line.substr(colon + 1)))});
        if (line_end == std::string_view::npos) {
            break;
        }
        rest = rest.substr(line_end + 2);
    }

    if (version != "HTTP/1.1" && version != "HTTP/1.0") {
        result.state = ParseState::UnsupportedVersion;
        return result;
    }
    request.keep_alive = version == "HTTP/1.1";

    std::size_t host_count = 0;
    bool host_value_empty = false;
    std::size_t transfer_encoding_count = 0;
    std::size_t content_length_count = 0;
    std::string_view content_length_value;
    for (const auto& header : request.headers) {
        if (header.name == "host") {
            ++host_count;
            host_value_empty = host_value_empty || header.value.empty();
        } else if (header.name == "transfer-encoding") {
            ++transfer_encoding_count;
        } else if (header.name == "content-length") {
            ++content_length_count;
            content_length_value = header.value;
        } else if (header.name == "connection") {
            request.keep_alive = lowercase(header.value) != "close";
        }
    }

    if (host_count > 1) {
        result.state = ParseState::DuplicateHost;
        return result;
    }
    if (host_value_empty || (version == "HTTP/1.1" && host_count == 0)) {
        result.state = ParseState::MissingHost;
        return result;
    }
    if (transfer_encoding_count != 0) {
        result.state = ParseState::UnsupportedTransferEncoding;
        return result;
    }
    if (content_length_count > 1) {
        result.state = ParseState::DuplicateContentLength;
        return result;
    }

    std::uint64_t content_length = 0;
    if (content_length_count == 1) {
        const auto* begin = content_length_value.data();
        const auto* end = begin + content_length_value.size();
        const auto parsed = std::from_chars(begin, end, content_length);
        if (begin == end || parsed.ec != std::errc{} || parsed.ptr != end) {
            result.state = ParseState::InvalidContentLength;
            return result;
        }
        if (content_length > kMaximumBodyBytes) {
            result.state = ParseState::BodyTooLarge;
            return result;
        }
    }

    const std::size_t body_offset = terminator + kHeaderTerminator.size();
    if (buffer.size() - body_offset < content_length) {
        result.state = ParseState::Incomplete;
        return result;
    }
    request.body.assign(buffer.substr(body_offset, content_length));
    result.request = std::move(request);
    result.consumed_bytes = body_offset + content_length;
    result.state = ParseState::Complete;
    return result;
}

int parse_state_http_status(ParseState state) noexcept {
    switch (state) {
    case ParseState::Incomplete:
        return 0;
    case ParseState::Complete:
        return 200;
    case ParseState::BodyTooLarge:
        return 413;
    case ParseState::MalformedRequestLine:
    case ParseState::InvalidHeaderSyntax:
    case ParseState::TooManyHeaders:
    case ParseState::HeadersTooLarge:
    case ParseState::UnsupportedVersion:
    case ParseState::MissingHost:
    case ParseState::DuplicateHost:
    case ParseState::UnsupportedTransferEncoding:
    case ParseState::DuplicateContentLength:
    case ParseState::InvalidContentLength:
        return 400;
    }
    return 500;
}

std::string_view parse_state_code(ParseState state) noexcept {
    switch (state) {
    case ParseState::Incomplete:
    case ParseState::Complete:
        return {};
    case ParseState::BodyTooLarge:
        return "tatara.payload_too_large";
    case ParseState::MalformedRequestLine:
    case ParseState::InvalidHeaderSyntax:
    case ParseState::TooManyHeaders:
    case ParseState::HeadersTooLarge:
    case ParseState::UnsupportedVersion:
    case ParseState::MissingHost:
    case ParseState::DuplicateHost:
    case ParseState::UnsupportedTransferEncoding:
    case ParseState::DuplicateContentLength:
    case ParseState::InvalidContentLength:
        return "tatara.invalid_request";
    }
    return "tatara.internal";
}

std::string_view reason_phrase(int status) {
    switch (status) {
    case 200:
        return "OK";
    case 400:
        return "Bad Request";
    case 404:
        return "Not Found";
    case 405:
        return "Method Not Allowed";
    case 413:
        return "Payload Too Large";
    case 429:
        return "Too Many Requests";
    case 499:
        return "Client Closed Request";
    case 500:
        return "Internal Server Error";
    case 503:
        return "Service Unavailable";
    case 504:
        return "Gateway Timeout";
    default:
        return "Unknown";
    }
}

std::string render_response(int status, std::string_view content_type, std::string_view body,
                            bool keep_alive) {
    std::ostringstream out;
    out << "HTTP/1.1 " << status << ' ' << reason_phrase(status) << "\r\n";
    out << "content-type: " << content_type << "\r\n";
    out << "content-length: " << body.size() << "\r\n";
    out << "connection: " << (keep_alive ? "keep-alive" : "close") << "\r\n\r\n";
    out << body;
    return out.str();
}

std::string render_sse_headers(bool keep_alive) {
    std::ostringstream out;
    out << "HTTP/1.1 200 OK\r\n";
    out << "content-type: text/event-stream\r\n";
    out << "cache-control: no-cache\r\n";
    // Chunked framing is not used: each event is flushed as written and the
    // stream ends when the connection closes or [DONE] is sent.
    out << "connection: " << (keep_alive ? "keep-alive" : "close") << "\r\n\r\n";
    return out.str();
}

std::string render_sse_event(std::string_view data) {
    std::string out("data: ");
    out.append(data);
    out.append("\n\n");
    return out;
}

std::string render_sse_done() {
    return "data: [DONE]\n\n";
}

} // namespace tatara::service
