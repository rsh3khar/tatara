#pragma once

#include "tatara/model/qwen36_plan.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace tatara::text {

enum class TokenizerLoadError : std::uint8_t {
    None,
    UnsafePath,
    FileUnavailable,
    WrongFileType,
    EmptyFile,
    FileTooLarge,
    FileSizeChanged,
    DigestMismatch,
    InvalidJson,
    MissingField,
    WrongFieldType,
    UnsupportedNormalizer,
    UnsupportedPretokenizer,
    UnsupportedDecoder,
    UnsupportedModel,
    InvalidVocabulary,
    InvalidMerge,
    InvalidAddedToken,
    VocabularyMismatch,
    RegexCompilation,
    ArithmeticOverflow,
    AllocationFailure,
};

enum class TokenizerError : std::uint8_t {
    None,
    InvalidUtf8,
    InputLimitExceeded,
    NormalizationLimitExceeded,
    TokenLimitExceeded,
    UnknownByteSymbol,
    InvalidTokenId,
    InvalidDecodedUtf8,
    IncompleteUtf8,
    OutputLimitExceeded,
    StreamClosed,
    ArithmeticOverflow,
    AllocationFailure,
};

struct TokenizerLimits {
    std::size_t maximum_input_bytes;
    std::size_t maximum_tokens;
    std::size_t maximum_output_bytes;
};

struct EncodeResult {
    std::vector<std::uint32_t> tokens;
    TokenizerError error = TokenizerError::None;
    std::string detail;

    explicit operator bool() const noexcept { return error == TokenizerError::None; }
};

struct DecodeResult {
    // Byte-level decoding is intentionally a byte string. Public UTF-8
    // streaming validation is a separate state-machine boundary.
    std::string bytes;
    TokenizerError error = TokenizerError::None;
    std::string detail;

    explicit operator bool() const noexcept { return error == TokenizerError::None; }
};

class Tokenizer;
struct TokenizerLoadResult;
class StreamingDecoder;
struct StreamingDecoderCreateResult;

struct StreamingDecoderConfig {
    bool skip_special;
    bool suppress_thinking;
    std::size_t maximum_output_bytes;
    std::vector<std::uint32_t> stop_token_ids;
    std::uint32_t thinking_start_id;
    std::uint32_t thinking_end_id;
};

struct StreamChunkResult {
    std::string bytes;
    bool stopped = false;
    TokenizerError error = TokenizerError::None;
    std::string detail;

    explicit operator bool() const noexcept { return error == TokenizerError::None; }
};

class Tokenizer final {
  public:
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;
    ~Tokenizer();

    static TokenizerLoadResult load(
        const model::qwen36::TokenizerSpec& specification,
        const std::filesystem::path& artifact_root);

    EncodeResult encode(std::string_view utf8, const TokenizerLimits& limits) const;
    DecodeResult decode(std::span<const std::uint32_t> tokens, bool skip_special,
                        const TokenizerLimits& limits) const;

    std::size_t populated_vocabulary() const noexcept;
    std::size_t merge_count() const noexcept;
    std::size_t added_token_count() const noexcept;
    std::size_t table_bytes() const noexcept;

  private:
    struct Impl;

    explicit Tokenizer(std::unique_ptr<Impl> implementation) noexcept;
    std::unique_ptr<Impl> implementation_;
};

struct TokenizerLoadResult {
    std::unique_ptr<Tokenizer> tokenizer;
    TokenizerLoadError error = TokenizerLoadError::None;
    std::string detail;

    explicit operator bool() const noexcept {
        return error == TokenizerLoadError::None && tokenizer != nullptr;
    }
};

class StreamingDecoder final {
  public:
    StreamingDecoder(const StreamingDecoder&) = delete;
    StreamingDecoder& operator=(const StreamingDecoder&) = delete;
    StreamingDecoder(StreamingDecoder&&) noexcept;
    StreamingDecoder& operator=(StreamingDecoder&&) noexcept;
    ~StreamingDecoder();

    static StreamingDecoderCreateResult create(
        const Tokenizer& tokenizer, StreamingDecoderConfig configuration);

    StreamChunkResult push(std::uint32_t token);
    StreamChunkResult finish();

  private:
    struct Impl;

    explicit StreamingDecoder(std::unique_ptr<Impl> implementation) noexcept;
    std::unique_ptr<Impl> implementation_;
};

struct StreamingDecoderCreateResult {
    std::unique_ptr<StreamingDecoder> decoder;
    TokenizerError error = TokenizerError::None;
    std::string detail;

    explicit operator bool() const noexcept {
        return error == TokenizerError::None && decoder != nullptr;
    }
};

} // namespace tatara::text
