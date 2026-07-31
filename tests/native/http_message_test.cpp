#include "tatara/service/http_message.h"

#include <array>
#include <cstdio>
#include <string>

namespace {

using namespace tatara::service;

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

void check_request_is_empty(const ParseResult& result) {
    check(result.request.method.empty(), "a non-complete parse exposes no method");
    check(result.request.target.empty(), "a non-complete parse exposes no target");
    check(result.request.headers.empty(), "a non-complete parse exposes no headers");
    check(result.request.body.empty(), "a non-complete parse exposes no body");
}

void check_failure(const ParseResult& result, ParseState expected, const char* what) {
    check(result.state == expected, what);
    check(result.consumed_bytes == 0, "a non-complete parse consumes no bytes");
    check_request_is_empty(result);
}

std::string post(std::string_view body, std::string_view extra = {}) {
    std::string out = "POST /v1/chat/completions HTTP/1.1\r\nhost: localhost\r\n";
    out.append("content-length: ").append(std::to_string(body.size())).append("\r\n");
    out.append(extra);
    out.append("\r\n");
    out.append(body);
    return out;
}

void parses_a_complete_request() {
    const auto result = parse_request(post("{\"model\":\"qwen\"}"));

    check(result.state == ParseState::Complete, "complete request parses");
    check(result.request.method == "POST", "method");
    check(result.request.target == "/v1/chat/completions", "target");
    check(result.request.body == "{\"model\":\"qwen\"}", "body");
    check(result.consumed_bytes > 0, "consumed byte count is reported for pipelining");
}

// Framing must be decided by the parser, not guessed at by the caller.
void an_incomplete_body_is_not_a_failure() {
    std::string wire = post("0123456789");
    wire.resize(wire.size() - 4);

    const auto result = parse_request(wire);

    check(result.state == ParseState::Incomplete, "a short body asks for more bytes");
    check(result.consumed_bytes == 0, "an incomplete body consumes no bytes");
    check_request_is_empty(result);
}

void missing_header_terminator_is_incomplete() {
    const auto result = parse_request("GET /health/live HTTP/1.1\r\nhost: x\r\n");
    check(result.state == ParseState::Incomplete, "headers not yet terminated");
    check(result.consumed_bytes == 0, "incomplete headers consume no bytes");
    check_request_is_empty(result);
}

void headers_are_case_insensitive() {
    const auto result = parse_request(post("", "X-Tatara-Priority: interactive\r\n"));

    check(result.state == ParseState::Complete, "parses");
    const auto value = result.request.header("X-TaTaRa-PrIoRiTy");
    check(value.has_value() && *value == "interactive", "lookup is case-insensitive");
}

void keep_alive_follows_the_connection_header() {
    check(parse_request(post("", "connection: close\r\n")).request.keep_alive == false,
          "close is honoured");
    check(parse_request(post("")).request.keep_alive == true, "HTTP/1.1 defaults to keep-alive");
    const auto old = parse_request("GET / HTTP/1.0\r\nhost: x\r\n\r\n");
    check(old.state == ParseState::Complete && !old.request.keep_alive,
          "HTTP/1.0 defaults to close");
}

// Untrusted input must be bounded before it is understood, not after.
void an_oversized_header_section_is_refused() {
    std::string wire = "GET / HTTP/1.1\r\n";
    wire.append("x-pad: ").append(std::string(kMaximumHeaderBytes + 16, 'a')).append("\r\n\r\n");

    check_failure(parse_request(wire), ParseState::HeadersTooLarge, "oversized headers refused");
}

void the_header_byte_bound_is_exact() {
    const std::string prefix = "GET / HTTP/1.1\r\nhost: x\r\nx-pad: ";
    std::string exact = prefix;
    exact.append(kMaximumHeaderBytes - prefix.size(), 'a').append("\r\n\r\n");
    check(parse_request(exact).state == ParseState::Complete,
          "the exact maximum bytes before the header terminator are accepted");
    for (std::size_t suffix_bytes = 1; suffix_bytes < 4; ++suffix_bytes) {
        const auto fragmented = parse_request(
            std::string_view(exact).substr(0, kMaximumHeaderBytes + suffix_bytes));
        check(fragmented.state == ParseState::Incomplete,
              "an exact-bound fragmented terminator remains incomplete");
        check(fragmented.consumed_bytes == 0,
              "a fragmented exact-bound terminator consumes no bytes");
        check_request_is_empty(fragmented);
    }

    std::string over = prefix;
    over.append(kMaximumHeaderBytes - prefix.size() + 1, 'a').append("\r\n\r\n");
    check_failure(parse_request(over), ParseState::HeadersTooLarge,
                  "one byte past the header-section bound is refused");
}

void an_unterminated_header_section_is_still_bounded() {
    for (std::size_t prefix_size = 1; prefix_size < 4; ++prefix_size) {
        std::string viable(kMaximumHeaderBytes, 'a');
        viable.append(std::string_view("\r\n\r\n").substr(0, prefix_size));
        check_failure(parse_request(viable), ParseState::Incomplete,
                      "a viable terminator prefix remains incomplete");
    }

    std::string earlier_candidate(kMaximumHeaderBytes - 1, 'a');
    earlier_candidate.append("\r\n");
    check_failure(parse_request(earlier_candidate), ParseState::Incomplete,
                  "a terminator prefix may begin before the exact limit");

    check_failure(parse_request(std::string(kMaximumHeaderBytes + 1, 'a')),
                  ParseState::HeadersTooLarge,
                  "arbitrary content one byte past the limit is refused immediately");
    check_failure(parse_request(std::string(kMaximumHeaderBytes + 4, 'a')),
                  ParseState::HeadersTooLarge,
                  "a client that never terminates cannot grow the buffer without limit");
}

void an_oversized_declared_body_is_refused_before_it_arrives() {
    std::string wire =
        "POST /v1/completions HTTP/1.1\r\nhost: x\r\ncontent-length: 999999999999\r\n\r\n";

    check_failure(parse_request(wire), ParseState::BodyTooLarge,
                  "refused on the declaration, not after buffering it");
    std::string one_over =
        "POST /v1/completions HTTP/1.1\r\nhost: x\r\ncontent-length: ";
    one_over.append(std::to_string(kMaximumBodyBytes + 1)).append("\r\n\r\n");
    check_failure(parse_request(one_over), ParseState::BodyTooLarge,
                  "one body byte past the bound is refused");
}

void a_bad_request_line_is_malformed() {
    check_failure(parse_request("GARBAGE\r\n\r\n"), ParseState::MalformedRequestLine,
                  "no method/target");
    check_failure(parse_request("GET /\r\n\r\n"), ParseState::MalformedRequestLine, "no version");
    check_failure(parse_request("G@T / HTTP/1.1\r\nhost: x\r\n\r\n"),
                  ParseState::MalformedRequestLine, "method is an HTTP token");
    check_failure(parse_request("GET /\x01 HTTP/1.1\r\nhost: x\r\n\r\n"),
                  ParseState::MalformedRequestLine, "target contains no control bytes");
}

void a_non_numeric_content_length_is_malformed() {
    check_failure(parse_request("POST / HTTP/1.1\r\nhost: x\r\ncontent-length: abc\r\n\r\n"),
                  ParseState::InvalidContentLength, "content-length must be a number");
    check_failure(parse_request("POST / HTTP/1.1\r\nhost: x\r\ncontent-length: 1junk\r\n\r\n"),
                  ParseState::InvalidContentLength, "content-length must have no suffix");
    check_failure(parse_request("POST / HTTP/1.1\r\nhost: x\r\ncontent-length:\r\n\r\n"),
                  ParseState::InvalidContentLength, "content-length cannot be empty");
    check_failure(parse_request("POST / HTTP/1.1\r\nhost: x\r\ncontent-length: +1\r\n\r\n"),
                  ParseState::InvalidContentLength, "content-length cannot have a sign");
    check_failure(parse_request("POST / HTTP/1.1\r\nhost: x\r\ncontent-length: 1, 1\r\n\r\n"),
                  ParseState::InvalidContentLength,
                  "a comma-normalized content-length is outside Tatara's subset");
    check_failure(parse_request("POST / HTTP/1.1\r\nhost: x\r\n"
                                "content-length: 18446744073709551616\r\n\r\n"),
                  ParseState::InvalidContentLength, "content-length overflow is typed");
}

void the_body_byte_bound_is_exact() {
    std::string wire = "POST / HTTP/1.1\r\nhost: x\r\ncontent-length: ";
    wire.append(std::to_string(kMaximumBodyBytes)).append("\r\n\r\n");
    const std::size_t body_offset = wire.size();
    wire.resize(body_offset + static_cast<std::size_t>(kMaximumBodyBytes), 'x');

    const auto exact = parse_request(wire);
    check(exact.state == ParseState::Complete, "the exact maximum body is accepted");
    check(exact.request.body.size() == kMaximumBodyBytes, "the exact maximum body is owned");
    check(exact.consumed_bytes == wire.size(), "the exact maximum body range is consumed");
}

void an_unsupported_version_is_typed() {
    check_failure(parse_request("GET / HTTP/2.0\r\nhost: x\r\n\r\n"),
                  ParseState::UnsupportedVersion,
                  "unsupported version is its own state, not malformed");
    check_failure(parse_request("GET / HTTP/2.0\r\nbad header\r\n\r\n"),
                  ParseState::InvalidHeaderSyntax,
                  "header syntax precedes unsupported-version classification");
}

void framing_is_single_and_unambiguous() {
    check_failure(
        parse_request(
            "POST / HTTP/1.1\r\nhost: x\r\ncontent-length: 0\r\ncontent-length: 0\r\n\r\n"),
        ParseState::DuplicateContentLength, "identical duplicate lengths are refused");
    check_failure(
        parse_request(
            "POST / HTTP/1.1\r\nhost: x\r\ncontent-length: 0\r\ncontent-length: 1\r\n\r\n"),
        ParseState::DuplicateContentLength, "conflicting duplicate lengths are refused");
    check_failure(parse_request("POST / HTTP/1.1\r\nhost: x\r\ntransfer-encoding: chunked\r\n\r\n"),
                  ParseState::UnsupportedTransferEncoding, "unsupported transfer coding is typed");
    check_failure(parse_request("POST / HTTP/1.1\r\nhost: x\r\ntransfer-encoding: chunked\r\n"
                                "content-length: 0\r\n\r\n"),
                  ParseState::UnsupportedTransferEncoding,
                  "transfer coding plus a length is refused");

    const std::string first = post("one");
    const std::string second = "GET /health/live HTTP/1.1\r\nhost: localhost\r\n\r\n";
    const auto pipelined = parse_request(first + second);
    check(pipelined.state == ParseState::Complete, "the first pipelined request parses");
    check(pipelined.consumed_bytes == first.size(), "exactly the first byte range is consumed");

    const auto zero_body = parse_request("GET / HTTP/1.1\r\nhost: x\r\n\r\nnext-request-bytes");
    check(zero_body.state == ParseState::Complete, "absence of framing means a zero-length body");
    check(zero_body.request.body.empty(), "no body is inferred from following bytes");
    check(zero_body.consumed_bytes == std::string_view("GET / HTTP/1.1\r\nhost: x\r\n\r\n").size(),
          "following bytes remain outside the request");
}

void host_is_exact_for_http_1_1() {
    check_failure(parse_request("GET / HTTP/1.1\r\n\r\n"), ParseState::MissingHost,
                  "HTTP/1.1 requires Host");
    check_failure(parse_request("GET / HTTP/1.1\r\nhost:\r\n\r\n"), ParseState::MissingHost,
                  "Host cannot be empty");
    check_failure(parse_request("GET / HTTP/1.1\r\nhost: a\r\nhost: b\r\n\r\n"),
                  ParseState::DuplicateHost, "Host cannot be duplicated");
    check(parse_request("GET / HTTP/1.0\r\n\r\n").state == ParseState::Complete,
          "HTTP/1.0 may omit Host");
}

void header_syntax_is_closed() {
    check_failure(parse_request("GET / HTTP/1.1\r\nhost : x\r\n\r\n"),
                  ParseState::InvalidHeaderSyntax, "whitespace before colon is refused");
    check_failure(parse_request("GET / HTTP/1.1\r\n: x\r\n\r\n"), ParseState::InvalidHeaderSyntax,
                  "empty header name is refused");
    check_failure(parse_request("GET / HTTP/1.1\r\nho(st: x\r\n\r\n"),
                  ParseState::InvalidHeaderSyntax, "invalid header-name token is refused");
    check_failure(parse_request("GET / HTTP/1.1\r\nhost: x\r\n folded: value\r\n\r\n"),
                  ParseState::InvalidHeaderSyntax, "obsolete folding is refused");

    std::string nul = "GET / HTTP/1.1\r\nhost: x";
    nul.push_back('\0');
    nul.append("\r\n\r\n");
    check_failure(parse_request(nul), ParseState::InvalidHeaderSyntax,
                  "NUL in a field value is refused");

    std::string del = "GET / HTTP/1.1\r\nhost: x";
    del.push_back(static_cast<char>(0x7f));
    del.append("\r\n\r\n");
    check_failure(parse_request(del), ParseState::InvalidHeaderSyntax,
                  "DEL in a field value is refused");

    std::string control = "GET / HTTP/1.1\r\nhost: x";
    control.push_back(static_cast<char>(0x1f));
    control.append("\r\n\r\n");
    check_failure(parse_request(control), ParseState::InvalidHeaderSyntax,
                  "non-tab control bytes in a field value are refused");

    std::string opaque = "GET / HTTP/1.1\r\nhost: x\r\nx-opaque: ";
    opaque.push_back(static_cast<char>(0x80));
    opaque.append("\r\n\r\n");
    check(parse_request(opaque).state == ParseState::Complete,
          "opaque field bytes are retained after syntax validation");
}

void header_count_is_bounded_independently() {
    std::string exact = "GET / HTTP/1.1\r\nhost: x\r\n";
    for (std::size_t index = 1; index < kMaximumHeaderFields; ++index) {
        exact.append("x").append(std::to_string(index)).append(": a\r\n");
    }
    exact.append("\r\n");
    check(parse_request(exact).state == ParseState::Complete,
          "the exact maximum header-field count is accepted");

    std::string over = exact;
    over.insert(over.size() - 2, "overflow: a\r\n");
    check_failure(parse_request(over), ParseState::TooManyHeaders,
                  "one header field past the bound is refused");
}

void parser_errors_map_to_frozen_public_codes() {
    check(parse_state_http_status(ParseState::Incomplete) == 0,
          "incomplete input has no response yet");
    check(parse_state_http_status(ParseState::Complete) == 200, "complete input is successful");
    check(parse_state_http_status(ParseState::BodyTooLarge) == 413 &&
              parse_state_code(ParseState::BodyTooLarge) == "tatara.payload_too_large",
          "body byte overflow maps to the frozen 413 code");

    constexpr std::array framing_failures{
        ParseState::MalformedRequestLine,   ParseState::InvalidHeaderSyntax,
        ParseState::TooManyHeaders,         ParseState::HeadersTooLarge,
        ParseState::UnsupportedVersion,     ParseState::MissingHost,
        ParseState::DuplicateHost,          ParseState::UnsupportedTransferEncoding,
        ParseState::DuplicateContentLength, ParseState::InvalidContentLength,
    };
    for (const auto state : framing_failures) {
        check(parse_state_http_status(state) == 400, "framing and syntax failures map to HTTP 400");
        check(parse_state_code(state) == "tatara.invalid_request",
              "framing and syntax failures map to the stable Tatara code");
    }
}

void responses_are_well_formed() {
    const auto text = render_response(429, "application/json", "{\"error\":1}", false);

    check(text.find("HTTP/1.1 429 Too Many Requests\r\n") == 0, "status line");
    check(text.find("content-length: 11\r\n") != std::string::npos, "length matches the body");
    check(text.find("connection: close\r\n") != std::string::npos, "connection reported");
    check(text.find("\r\n\r\n{\"error\":1}") != std::string::npos, "body follows the blank line");
}

void sse_framing_matches_what_clients_expect() {
    const auto headers = render_sse_headers(true);
    check(headers.find("content-type: text/event-stream\r\n") != std::string::npos,
          "event-stream content type");
    check(headers.find("cache-control: no-cache\r\n") != std::string::npos, "no caching");

    check(render_sse_event("{\"delta\":\"hi\"}") == "data: {\"delta\":\"hi\"}\n\n",
          "one event per token");
    check(render_sse_done() == "data: [DONE]\n\n", "the terminator OpenAI clients expect");
}

} // namespace

int main() {
    parses_a_complete_request();
    an_incomplete_body_is_not_a_failure();
    missing_header_terminator_is_incomplete();
    headers_are_case_insensitive();
    keep_alive_follows_the_connection_header();
    an_oversized_header_section_is_refused();
    the_header_byte_bound_is_exact();
    an_unterminated_header_section_is_still_bounded();
    an_oversized_declared_body_is_refused_before_it_arrives();
    a_bad_request_line_is_malformed();
    a_non_numeric_content_length_is_malformed();
    the_body_byte_bound_is_exact();
    an_unsupported_version_is_typed();
    framing_is_single_and_unambiguous();
    host_is_exact_for_http_1_1();
    header_syntax_is_closed();
    header_count_is_bounded_independently();
    parser_errors_map_to_frozen_public_codes();
    responses_are_well_formed();
    sse_framing_matches_what_clients_expect();
    if (failures == 0) {
        std::printf("http message: PASS\n");
    }
    return failures == 0 ? 0 : 1;
}
