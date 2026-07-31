#include "tatara/text/tokenizer.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <string>
#include <utility>

namespace tatara::text {
namespace {

struct Utf8Prefix {
    std::size_t complete_bytes;
    bool valid;
};

Utf8Prefix complete_utf8_prefix(std::string_view bytes) {
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const std::size_t beginning = offset;
        const auto first = static_cast<std::uint8_t>(bytes[offset]);
        if (first < 0x80U) {
            ++offset;
            continue;
        }
        std::size_t count = 0;
        std::uint32_t codepoint = 0;
        std::uint32_t minimum = 0;
        if ((first & 0xE0U) == 0xC0U) {
            count = 2;
            codepoint = first & 0x1FU;
            minimum = 0x80U;
        } else if ((first & 0xF0U) == 0xE0U) {
            count = 3;
            codepoint = first & 0x0FU;
            minimum = 0x800U;
        } else if ((first & 0xF8U) == 0xF0U) {
            count = 4;
            codepoint = first & 0x07U;
            minimum = 0x10000U;
        } else {
            return {beginning, false};
        }
        if (count > bytes.size() - beginning) {
            return {beginning, true};
        }
        for (std::size_t index = 1; index < count; ++index) {
            const auto tail =
                static_cast<std::uint8_t>(bytes[beginning + index]);
            if ((tail & 0xC0U) != 0x80U) {
                return {beginning, false};
            }
            codepoint = (codepoint << 6U) | (tail & 0x3FU);
        }
        if (codepoint < minimum || codepoint > 0x10FFFFU ||
            (codepoint >= 0xD800U && codepoint <= 0xDFFFU)) {
            return {beginning, false};
        }
        offset += count;
    }
    return {bytes.size(), true};
}

StreamChunkResult stream_error(TokenizerError error, std::string detail) {
    return {
        .bytes = {},
        .stopped = false,
        .error = error,
        .detail = std::move(detail),
    };
}

StreamingDecoderCreateResult create_error(TokenizerError error,
                                          std::string detail) {
    return {
        .decoder = nullptr,
        .error = error,
        .detail = std::move(detail),
    };
}

} // namespace

struct StreamingDecoder::Impl {
    const Tokenizer* tokenizer;
    StreamingDecoderConfig configuration;
    std::string incomplete_utf8;
    std::size_t accepted_output_bytes = 0;
    bool inside_thinking = false;
    bool closed = false;

    StreamChunkResult push(std::uint32_t token) {
        if (closed) {
            return stream_error(TokenizerError::StreamClosed,
                                "streaming decoder is closed");
        }
        if (std::find(configuration.stop_token_ids.begin(),
                      configuration.stop_token_ids.end(),
                      token) != configuration.stop_token_ids.end()) {
            if (!incomplete_utf8.empty()) {
                closed = true;
                return stream_error(
                    TokenizerError::IncompleteUtf8,
                    "stop token followed an incomplete UTF-8 sequence");
            }
            closed = true;
            return {
                .bytes = {},
                .stopped = true,
                .error = TokenizerError::None,
                .detail = {},
            };
        }
        if (configuration.suppress_thinking) {
            if (token == configuration.thinking_start_id) {
                if (!incomplete_utf8.empty()) {
                    closed = true;
                    return stream_error(
                        TokenizerError::IncompleteUtf8,
                        "thinking boundary followed an incomplete UTF-8 sequence");
                }
                inside_thinking = true;
                return {};
            }
            if (token == configuration.thinking_end_id) {
                if (!incomplete_utf8.empty()) {
                    closed = true;
                    return stream_error(
                        TokenizerError::IncompleteUtf8,
                        "thinking boundary followed an incomplete UTF-8 sequence");
                }
                inside_thinking = false;
                return {};
            }
            if (inside_thinking) {
                return {};
            }
        }

        const std::array<std::uint32_t, 1> identifier{token};
        const std::size_t remaining =
            configuration.maximum_output_bytes - accepted_output_bytes;
        DecodeResult decoded = tokenizer->decode(
            identifier, configuration.skip_special,
            TokenizerLimits{
                .maximum_input_bytes = 0,
                .maximum_tokens = 0,
                .maximum_output_bytes = remaining,
            });
        if (!decoded) {
            closed = true;
            return stream_error(decoded.error, std::move(decoded.detail));
        }
        incomplete_utf8 += decoded.bytes;
        accepted_output_bytes += decoded.bytes.size();
        const Utf8Prefix prefix = complete_utf8_prefix(incomplete_utf8);
        if (!prefix.valid) {
            closed = true;
            return stream_error(TokenizerError::InvalidDecodedUtf8,
                                "generated token bytes are not valid UTF-8");
        }
        std::string complete = incomplete_utf8.substr(0, prefix.complete_bytes);
        incomplete_utf8.erase(0, prefix.complete_bytes);
        return {
            .bytes = std::move(complete),
            .stopped = false,
            .error = TokenizerError::None,
            .detail = {},
        };
    }

    StreamChunkResult finish() {
        if (closed) {
            return stream_error(TokenizerError::StreamClosed,
                                "streaming decoder is closed");
        }
        closed = true;
        if (!incomplete_utf8.empty()) {
            return stream_error(TokenizerError::IncompleteUtf8,
                                "stream ended with incomplete UTF-8");
        }
        return {};
    }
};

StreamingDecoderCreateResult StreamingDecoder::create(
    const Tokenizer& tokenizer, StreamingDecoderConfig configuration) {
    try {
        if (configuration.maximum_output_bytes == 0) {
            return create_error(TokenizerError::OutputLimitExceeded,
                                "stream output limit must be non-zero");
        }
        if (configuration.stop_token_ids.empty()) {
            return create_error(TokenizerError::InvalidTokenId,
                                "stream stop-token set must be non-empty");
        }
        for (std::size_t index = 0;
             index < configuration.stop_token_ids.size(); ++index) {
            const std::uint32_t token = configuration.stop_token_ids[index];
            if (token >= tokenizer.populated_vocabulary()) {
                return create_error(TokenizerError::InvalidTokenId,
                                    "stream stop token is not populated");
            }
            if (std::find(configuration.stop_token_ids.begin(),
                          configuration.stop_token_ids.begin() +
                              static_cast<std::ptrdiff_t>(index),
                          token) !=
                configuration.stop_token_ids.begin() +
                    static_cast<std::ptrdiff_t>(index)) {
                return create_error(TokenizerError::InvalidTokenId,
                                    "stream stop-token set repeats an ID");
            }
        }
        if (configuration.suppress_thinking &&
            (configuration.thinking_start_id >= tokenizer.populated_vocabulary() ||
             configuration.thinking_end_id >= tokenizer.populated_vocabulary() ||
             configuration.thinking_start_id ==
                 configuration.thinking_end_id)) {
            return create_error(TokenizerError::InvalidTokenId,
                                "thinking token IDs are invalid");
        }
        auto implementation = std::make_unique<Impl>(Impl{
            .tokenizer = &tokenizer,
            .configuration = std::move(configuration),
            .incomplete_utf8 = {},
            .accepted_output_bytes = 0,
            .inside_thinking = false,
            .closed = false,
        });
        return {
            .decoder = std::unique_ptr<StreamingDecoder>(
                new StreamingDecoder(std::move(implementation))),
            .error = TokenizerError::None,
            .detail = {},
        };
    } catch (const std::bad_alloc&) {
        return create_error(TokenizerError::AllocationFailure,
                            "streaming decoder allocation failed");
    }
}

StreamingDecoder::StreamingDecoder(
    std::unique_ptr<Impl> implementation) noexcept
    : implementation_(std::move(implementation)) {}

StreamingDecoder::StreamingDecoder(StreamingDecoder&&) noexcept = default;
StreamingDecoder& StreamingDecoder::operator=(StreamingDecoder&&) noexcept =
    default;
StreamingDecoder::~StreamingDecoder() = default;

StreamChunkResult StreamingDecoder::push(std::uint32_t token) {
    try {
        return implementation_->push(token);
    } catch (const std::bad_alloc&) {
        implementation_->closed = true;
        return stream_error(TokenizerError::AllocationFailure,
                            "streaming decoder allocation failed");
    }
}

StreamChunkResult StreamingDecoder::finish() {
    return implementation_->finish();
}

} // namespace tatara::text
