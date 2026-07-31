#include "tatara/model/sha256.h"
#include "tatara/text/tokenizer.h"

#include "tatara/generated/model_plan.h"

#import <Foundation/Foundation.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace {

using tatara::model::Sha256;
using tatara::model::qwen36::ChatTemplateKind;
using tatara::model::qwen36::TokenizerDecoder;
using tatara::model::qwen36::TokenizerKind;
using tatara::model::qwen36::TokenizerNormalization;
using tatara::model::qwen36::TokenizerPretokenizer;
using tatara::model::qwen36::TokenizerSpec;
using tatara::model::qwen36::generated::kModelPlan;
using tatara::model::sha256_hex;
using tatara::text::Tokenizer;
using tatara::text::TokenizerError;
using tatara::text::TokenizerLimits;
using tatara::text::TokenizerLoadError;
using tatara::text::StreamingDecoder;
using tatara::text::StreamingDecoderConfig;

void check(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void append_utf8(std::string& output, std::uint32_t codepoint) {
    if (codepoint < 0x80U) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint < 0x800U) {
        output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
        output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
}

std::string json_string(std::string_view value) {
    std::string output = "\"";
    for (const char character : value) {
        const auto byte = static_cast<unsigned char>(character);
        switch (byte) {
        case '"':
            output += "\\\"";
            break;
        case '\\':
            output += "\\\\";
            break;
        case '\b':
            output += "\\b";
            break;
        case '\f':
            output += "\\f";
            break;
        case '\n':
            output += "\\n";
            break;
        case '\r':
            output += "\\r";
            break;
        case '\t':
            output += "\\t";
            break;
        default:
            output.push_back(static_cast<char>(byte));
            break;
        }
    }
    output.push_back('"');
    return output;
}

std::array<std::string, 256> byte_alphabet() {
    std::array<std::string, 256> result{};
    std::uint32_t extension = 0;
    for (std::uint32_t byte = 0; byte < result.size(); ++byte) {
        const bool direct = (byte >= 33 && byte <= 126) ||
                            (byte >= 161 && byte <= 172) ||
                            (byte >= 174 && byte <= 255);
        append_utf8(result[byte], direct ? byte : 256 + extension++);
    }
    return result;
}

std::string synthetic_json() {
    const auto alphabet = byte_alphabet();
    std::string vocabulary = "{";
    for (std::size_t index = 0; index < alphabet.size(); ++index) {
        if (index != 0) {
            vocabulary.push_back(',');
        }
        vocabulary += json_string(alphabet[index]) + ":" + std::to_string(index);
    }
    vocabulary += "," + json_string(alphabet['a'] + alphabet['a']) + ":256";
    vocabulary += "," + json_string(alphabet['a'] + alphabet['b']) + ":257}";

    const std::string pattern = ".+";
    return
        "{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,"
        "\"added_tokens\":[{\"id\":258,\"content\":\"<x>\","
        "\"single_word\":false,\"lstrip\":false,\"rstrip\":false,"
        "\"normalized\":false,\"special\":true}],"
        "\"normalizer\":{\"type\":\"NFC\"},"
        "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":["
        "{\"type\":\"Split\",\"pattern\":{\"Regex\":" +
        json_string(pattern) +
        "},\"behavior\":\"Isolated\",\"invert\":false},"
        "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,"
        "\"trim_offsets\":false,\"use_regex\":false}]},"
        "\"decoder\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,"
        "\"trim_offsets\":false,\"use_regex\":false},"
        "\"post_processor\":{\"type\":\"ByteLevel\","
        "\"add_prefix_space\":false,\"trim_offsets\":false,"
        "\"use_regex\":false},"
        "\"model\":{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":null,"
        "\"continuing_subword_prefix\":\"\",\"end_of_word_suffix\":\"\","
        "\"fuse_unk\":false,\"byte_fallback\":false,\"ignore_merges\":false,"
        "\"vocab\":" +
        vocabulary + ",\"merges\":[[" + json_string(alphabet['a']) + "," +
        json_string(alphabet['a']) + "],[" + json_string(alphabet['a']) + "," +
        json_string(alphabet['b']) + "]]}}";
}

std::string sha256(std::string_view value) {
    Sha256 hasher;
    hasher.update(std::as_bytes(std::span(value.data(), value.size())));
    return sha256_hex(hasher.finish());
}

TokenizerSpec synthetic_spec(std::string_view digest, std::size_t size) {
    return TokenizerSpec{
        .kind = TokenizerKind::ByteLevelBpe,
        .normalization = TokenizerNormalization::Nfc,
        .pretokenizer = TokenizerPretokenizer::QwenRegexByteLevelV1,
        .decoder = TokenizerDecoder::ByteLevel,
        .template_kind = ChatTemplateKind::Qwen36TextV1,
        .data_path = "tokenizer.json",
        .data_sha256 = digest,
        .data_size_bytes = size,
        .config_path = "tokenizer_config.json",
        .config_sha256 = std::string_view{"0", 1},
        .config_size_bytes = 1,
        .template_path = "chat_template.jinja",
        .template_sha256 = std::string_view{"0", 1},
        .template_size_bytes = 1,
        .split_pattern = ".+",
        .vocabulary = 261,
        .populated_vocabulary = 259,
        .maximum_context = 100,
        .end_of_text_id = 258,
        .message_start_id = 258,
        .message_end_id = 258,
        .thinking_start_id = 258,
        .thinking_end_id = 258,
        .padding_id = 258,
        .stop_token_ids = std::array<std::uint32_t, 8>{258},
        .stop_token_count = 1,
        .default_thinking = true,
    };
}

class TemporaryDirectory {
  public:
    TemporaryDirectory()
        : path_(std::filesystem::temp_directory_path() /
                ("tatara-tokenizer-" + std::to_string(::getpid()))) {
        std::filesystem::remove_all(path_);
        std::filesystem::create_directory(path_);
    }

    ~TemporaryDirectory() { std::filesystem::remove_all(path_); }

    const std::filesystem::path& path() const noexcept { return path_; }

  private:
    std::filesystem::path path_;
};

void synthetic_contract() {
    TemporaryDirectory directory;
    const std::string json = synthetic_json();
    {
        std::ofstream stream(directory.path() / "tokenizer.json", std::ios::binary);
        stream.write(json.data(), static_cast<std::streamsize>(json.size()));
    }
    const std::string digest = sha256(json);
    TokenizerSpec specification = synthetic_spec(digest, json.size());
    auto loaded = Tokenizer::load(specification, directory.path());
    check(static_cast<bool>(loaded), loaded.detail);
    check(loaded.tokenizer->populated_vocabulary() == 259,
          "synthetic populated vocabulary");
    check(loaded.tokenizer->merge_count() == 2, "synthetic merge count");
    check(loaded.tokenizer->added_token_count() == 1, "synthetic added count");
    check(loaded.tokenizer->table_bytes() > 0, "synthetic table accounting");

    const TokenizerLimits limits{
        .maximum_input_bytes = 128,
        .maximum_tokens = 128,
        .maximum_output_bytes = 128,
    };
    auto encoded = loaded.tokenizer->encode("ab<x>ab", limits);
    check(static_cast<bool>(encoded), encoded.detail);
    check(encoded.tokens == std::vector<std::uint32_t>({257, 258, 257}),
          "heap BPE and added-token boundary");

    auto decoded = loaded.tokenizer->decode(encoded.tokens, false, limits);
    check(static_cast<bool>(decoded) && decoded.bytes == "ab<x>ab",
          "byte-level complete decode");
    auto skipped = loaded.tokenizer->decode(encoded.tokens, true, limits);
    check(static_cast<bool>(skipped) && skipped.bytes == "abab",
          "special-token skip");

    auto overlapping = loaded.tokenizer->encode("aaa", limits);
    check(static_cast<bool>(overlapping) &&
              overlapping.tokens == std::vector<std::uint32_t>({256, 'a'}),
          "overlapping merge chooses the lower left position");
    auto repeated = loaded.tokenizer->encode("aaaa", limits);
    check(static_cast<bool>(repeated) &&
              repeated.tokens == std::vector<std::uint32_t>({256, 256}),
          "stale candidates do not suppress a later repeated pair");

    const std::array<std::uint32_t, 1> reserved{259};
    check(loaded.tokenizer->decode(reserved, false, limits).error ==
              TokenizerError::InvalidTokenId,
          "reserved model-vocabulary tail is rejected");
    check(loaded.tokenizer
              ->encode("ab", TokenizerLimits{128, 0, 128})
              .error == TokenizerError::TokenLimitExceeded,
          "token limit is typed");
    const std::string invalid_utf8{"\xC0\xAF", 2};
    check(loaded.tokenizer->encode(invalid_utf8, limits).error ==
              TokenizerError::InvalidUtf8,
          "invalid UTF-8 is typed");

    TokenizerSpec wrong_digest = specification;
    wrong_digest.data_sha256 = std::string_view{
        "0000000000000000000000000000000000000000000000000000000000000000"};
    check(Tokenizer::load(wrong_digest, directory.path()).error ==
              TokenizerLoadError::DigestMismatch,
          "digest disagreement is typed");

    auto stream_configuration = [] {
        return StreamingDecoderConfig{
            .skip_special = false,
            .suppress_thinking = false,
            .maximum_output_bytes = 128,
            .stop_token_ids = {258},
            .thinking_start_id = 'a',
            .thinking_end_id = 'b',
        };
    };
    auto emoji_tokens = loaded.tokenizer->encode("🙂", limits);
    check(static_cast<bool>(emoji_tokens), emoji_tokens.detail);
    auto stream =
        StreamingDecoder::create(*loaded.tokenizer, stream_configuration());
    check(static_cast<bool>(stream), stream.detail);
    std::string streamed_emoji;
    for (const std::uint32_t token : emoji_tokens.tokens) {
        auto chunk = stream.decoder->push(token);
        check(static_cast<bool>(chunk), chunk.detail);
        streamed_emoji += chunk.bytes;
    }
    check(stream.decoder->finish().error == TokenizerError::None,
          "complete UTF-8 stream finishes");
    check(streamed_emoji == "🙂",
          "streaming decoder buffers an incomplete codepoint");

    auto stopped =
        StreamingDecoder::create(*loaded.tokenizer, stream_configuration());
    auto stop = stopped.decoder->push(258);
    check(static_cast<bool>(stop) && stop.stopped && stop.bytes.empty(),
          "stop token is hidden before decode");

    auto incomplete =
        StreamingDecoder::create(*loaded.tokenizer, stream_configuration());
    check(incomplete.decoder->push(0xF0).error == TokenizerError::None,
          "incomplete UTF-8 prefix is buffered");
    check(incomplete.decoder->finish().error == TokenizerError::IncompleteUtf8,
          "incomplete UTF-8 fails at stream end");

    auto invalid =
        StreamingDecoder::create(*loaded.tokenizer, stream_configuration());
    check(invalid.decoder->push(0xC0).error == TokenizerError::None,
          "invalid lead remains incomplete until discriminated");
    check(invalid.decoder->push('x').error ==
              TokenizerError::InvalidDecodedUtf8,
          "invalid streamed UTF-8 is typed");

    StreamingDecoderConfig suppress_configuration = stream_configuration();
    suppress_configuration.suppress_thinking = true;
    auto suppressed =
        StreamingDecoder::create(*loaded.tokenizer,
                                 std::move(suppress_configuration));
    check(suppressed.decoder->push('a').bytes.empty(),
          "thinking start is hidden");
    check(suppressed.decoder->push('x').bytes.empty(),
          "thinking content is hidden");
    check(suppressed.decoder->push('b').bytes.empty(),
          "thinking end is hidden");
    check(suppressed.decoder->push('y').bytes == "y",
          "content after thinking is emitted");

    StreamingDecoderConfig tiny_configuration = stream_configuration();
    tiny_configuration.maximum_output_bytes = 1;
    auto tiny =
        StreamingDecoder::create(*loaded.tokenizer, std::move(tiny_configuration));
    check(tiny.decoder->push(257).error ==
              TokenizerError::OutputLimitExceeded,
          "stream output bound is typed");
}

std::string std_string(NSString* value) {
    NSData* data = [value dataUsingEncoding:NSUTF8StringEncoding allowLossyConversion:NO];
    return std::string(static_cast<const char*>(data.bytes), data.length);
}

void exact_artifact_parity(const std::filesystem::path& model_root,
                           const std::filesystem::path& fixture_path) {
    auto loaded = Tokenizer::load(kModelPlan.tokenizer, model_root);
    check(static_cast<bool>(loaded), loaded.detail);
    check(loaded.tokenizer->populated_vocabulary() == 248'077,
          "real populated vocabulary");
    check(loaded.tokenizer->merge_count() == 247'587, "real merge count");
    check(loaded.tokenizer->added_token_count() == 33, "real added-token count");

    std::ifstream fixture(fixture_path);
    check(static_cast<bool>(fixture), "parity fixture opens");
    std::string line;
    std::size_t rows = 0;
    while (std::getline(fixture, line)) {
        if (line.empty()) {
            continue;
        }
        @autoreleasepool {
            NSData* data = [NSData dataWithBytes:line.data() length:line.size()];
            NSError* error = nil;
            id parsed = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
            check(parsed != nil && [parsed isKindOfClass:NSDictionary.class],
                  "fixture row is valid JSON");
            NSDictionary* row = static_cast<NSDictionary*>(parsed);
            NSString* text = row[@"text"];
            NSString* expected_decoded = row[@"decoded"];
            NSArray* expected_ids = row[@"ids"];
            check([text isKindOfClass:NSString.class] &&
                      [expected_decoded isKindOfClass:NSString.class] &&
                      [expected_ids isKindOfClass:NSArray.class],
                  "fixture row schema");
            std::vector<std::uint32_t> ids;
            ids.reserve(expected_ids.count);
            for (NSNumber* value in expected_ids) {
                ids.push_back(value.unsignedIntValue);
            }
            const TokenizerLimits limits{
                .maximum_input_bytes = 1U << 20U,
                .maximum_tokens = 1U << 18U,
                .maximum_output_bytes = 1U << 20U,
            };
            auto encoded = loaded.tokenizer->encode(std_string(text), limits);
            check(static_cast<bool>(encoded), encoded.detail);
            check(encoded.tokens == ids, "real encode parity row " + std::to_string(rows));
            auto decoded = loaded.tokenizer->decode(ids, false, limits);
            check(static_cast<bool>(decoded), decoded.detail);
            check(decoded.bytes == std_string(expected_decoded),
                  "real decode parity row " + std::to_string(rows));
            ++rows;
        }
    }
    check(rows > 0, "parity fixture is non-empty");
    std::cout << "real tokenizer parity " << rows << '/' << rows
              << " encode+decode exact, table_bytes="
              << loaded.tokenizer->table_bytes() << '\n';
}

} // namespace

int main(int argument_count, char** arguments) {
    @autoreleasepool {
        synthetic_contract();
        if (argument_count == 3) {
            exact_artifact_parity(arguments[1], arguments[2]);
        } else {
            check(argument_count == 1,
                  "usage: tokenizer_test [model-root parity-fixture]");
        }
    }
    return 0;
}
