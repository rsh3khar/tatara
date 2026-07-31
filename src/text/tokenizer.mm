#include "tatara/text/tokenizer.h"

#include "tatara/model/sha256.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <queue>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tatara::text {
namespace {

using model::qwen36::TokenizerSpec;

constexpr std::uint64_t kMaximumTokenizerBytes = 64ULL * 1024ULL * 1024ULL;
constexpr std::uint32_t kNoSymbol = std::numeric_limits<std::uint32_t>::max();

struct LoadFailure {
    TokenizerLoadError error;
    std::string detail;
};

template <typename Value>
bool checked_add(Value left, Value right, Value& result) noexcept {
    static_assert(std::is_unsigned_v<Value>);
    if (right > std::numeric_limits<Value>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

bool safe_relative_path(std::string_view value) {
    if (value.empty()) {
        return false;
    }
    const std::filesystem::path path(value);
    if (path.is_absolute() || path.has_root_name() || path.has_root_directory()) {
        return false;
    }
    for (const auto& component : path) {
        if (component == ".." || component == ".") {
            return false;
        }
    }
    return true;
}

std::string ns_string(NSString* value) {
    if (value == nil) {
        return {};
    }
    NSData* data = [value dataUsingEncoding:NSUTF8StringEncoding allowLossyConversion:NO];
    if (data == nil) {
        return {};
    }
    return std::string(static_cast<const char*>(data.bytes), data.length);
}

NSString* foundation_string(std::string_view value) {
    return [[NSString alloc] initWithBytes:value.data()
                                   length:value.size()
                                 encoding:NSUTF8StringEncoding];
}

NSDictionary* dictionary_field(NSDictionary* dictionary, NSString* key,
                               LoadFailure& failure) {
    id value = dictionary[key];
    if (value == nil) {
        failure = {TokenizerLoadError::MissingField,
                   "missing object field: " + ns_string(key)};
        return nil;
    }
    if (![value isKindOfClass:NSDictionary.class]) {
        failure = {TokenizerLoadError::WrongFieldType,
                   "object field has wrong type: " + ns_string(key)};
        return nil;
    }
    return static_cast<NSDictionary*>(value);
}

NSArray* array_field(NSDictionary* dictionary, NSString* key, LoadFailure& failure) {
    id value = dictionary[key];
    if (value == nil) {
        failure = {TokenizerLoadError::MissingField,
                   "missing array field: " + ns_string(key)};
        return nil;
    }
    if (![value isKindOfClass:NSArray.class]) {
        failure = {TokenizerLoadError::WrongFieldType,
                   "array field has wrong type: " + ns_string(key)};
        return nil;
    }
    return static_cast<NSArray*>(value);
}

NSString* string_field(NSDictionary* dictionary, NSString* key, LoadFailure& failure) {
    id value = dictionary[key];
    if (value == nil) {
        failure = {TokenizerLoadError::MissingField,
                   "missing string field: " + ns_string(key)};
        return nil;
    }
    if (![value isKindOfClass:NSString.class]) {
        failure = {TokenizerLoadError::WrongFieldType,
                   "string field has wrong type: " + ns_string(key)};
        return nil;
    }
    return static_cast<NSString*>(value);
}

bool boolean_value(NSDictionary* dictionary, NSString* key, bool& result,
                   LoadFailure& failure) {
    id value = dictionary[key];
    if (value == nil) {
        failure = {TokenizerLoadError::MissingField,
                   "missing boolean field: " + ns_string(key)};
        return false;
    }
    if (![value isKindOfClass:NSNumber.class] ||
        CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID()) {
        failure = {TokenizerLoadError::WrongFieldType,
                   "boolean field has wrong type: " + ns_string(key)};
        return false;
    }
    result = [static_cast<NSNumber*>(value) boolValue];
    return true;
}

bool unsigned_value(id value, std::uint32_t& result) {
    if (![value isKindOfClass:NSNumber.class] ||
        CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID()) {
        return false;
    }
    NSNumber* number = static_cast<NSNumber*>(value);
    const long long signed_value = number.longLongValue;
    const unsigned long long unsigned_value = number.unsignedLongLongValue;
    if (signed_value < 0 || unsigned_value > std::numeric_limits<std::uint32_t>::max() ||
        number.doubleValue != static_cast<double>(unsigned_value)) {
        return false;
    }
    result = static_cast<std::uint32_t>(unsigned_value);
    return true;
}

bool equals(NSString* value, std::string_view expected) {
    return value != nil && ns_string(value) == expected;
}

std::uint64_t pair_key(std::uint32_t left, std::uint32_t right) noexcept {
    return (static_cast<std::uint64_t>(left) << 32U) | right;
}

void append_utf8(std::string& output, std::uint32_t codepoint) {
    if (codepoint < 0x80U) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint < 0x800U) {
        output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint < 0x10000U) {
        output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
        output.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
}

bool decode_utf8_codepoint(std::string_view value, std::size_t& offset,
                           std::uint32_t& codepoint) {
    if (offset >= value.size()) {
        return false;
    }
    const auto first = static_cast<std::uint8_t>(value[offset]);
    std::size_t count = 0;
    std::uint32_t minimum = 0;
    if (first < 0x80U) {
        codepoint = first;
        ++offset;
        return true;
    }
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
        return false;
    }
    if (count > value.size() - offset) {
        return false;
    }
    for (std::size_t index = 1; index < count; ++index) {
        const auto tail = static_cast<std::uint8_t>(value[offset + index]);
        if ((tail & 0xC0U) != 0x80U) {
            return false;
        }
        codepoint = (codepoint << 6U) | (tail & 0x3FU);
    }
    if (codepoint < minimum || codepoint > 0x10FFFFU ||
        (codepoint >= 0xD800U && codepoint <= 0xDFFFU)) {
        return false;
    }
    offset += count;
    return true;
}

TokenizerLoadResult load_error(TokenizerLoadError error, std::string detail) {
    return {
        .tokenizer = nullptr,
        .error = error,
        .detail = std::move(detail),
    };
}

EncodeResult encode_error(TokenizerError error, std::string detail) {
    return {
        .tokens = {},
        .error = error,
        .detail = std::move(detail),
    };
}

DecodeResult decode_error(TokenizerError error, std::string detail) {
    return {
        .bytes = {},
        .error = error,
        .detail = std::move(detail),
    };
}

} // namespace

struct Tokenizer::Impl {
    struct MergeSlot {
        std::uint64_t key = 0;
        std::uint32_t rank;
        std::uint32_t result;
    };

    struct AddedToken {
        std::string content;
        std::uint32_t identifier;
        bool special;
    };

    struct Symbol {
        std::uint32_t identifier;
        std::uint32_t previous;
        std::uint32_t next;
        bool alive;
    };

    struct Candidate {
        std::uint32_t rank;
        std::uint32_t left_position;
        std::uint32_t right_position;
        std::uint32_t left_identifier;
        std::uint32_t right_identifier;
        std::uint32_t result;
    };

    struct CandidateLater {
        bool operator()(const Candidate& left, const Candidate& right) const noexcept {
            if (left.rank != right.rank) {
                return left.rank > right.rank;
            }
            return left.left_position > right.left_position;
        }
    };

    std::vector<std::string> identifier_to_token;
    std::vector<std::uint8_t> is_added;
    std::vector<std::uint8_t> is_special;
    std::vector<AddedToken> added_tokens;
    std::vector<MergeSlot> merge_table;
    std::size_t merge_rule_count = 0;
    std::array<std::uint32_t, 256> byte_identifiers{};
    std::array<std::uint16_t, 512> character_bytes{};
    __strong NSRegularExpression* split_expression = nil;
    std::size_t immutable_table_bytes = 0;

    static TokenizerLoadResult create(const TokenizerSpec& specification,
                                      const std::filesystem::path& artifact_root);

    EncodeResult encode(std::string_view utf8, const TokenizerLimits& limits) const;
    DecodeResult decode(std::span<const std::uint32_t> tokens, bool skip_special,
                        const TokenizerLimits& limits) const;

    TokenizerError encode_text(std::string_view utf8, const TokenizerLimits& limits,
                               std::vector<std::uint32_t>& output,
                               std::string& detail) const;
    TokenizerError bpe(std::string_view bytes, const TokenizerLimits& limits,
                       std::vector<std::uint32_t>& output,
                       std::string& detail) const;

    static std::size_t merge_slot(std::uint64_t key, std::size_t mask) noexcept {
        key ^= key >> 30U;
        key *= 0xBF58476D1CE4E5B9ULL;
        key ^= key >> 27U;
        key *= 0x94D049BB133111EBULL;
        key ^= key >> 31U;
        return static_cast<std::size_t>(key) & mask;
    }

    const MergeSlot* find_merge(std::uint32_t left,
                                std::uint32_t right) const noexcept {
        if (merge_table.empty()) {
            return nullptr;
        }
        const std::uint64_t key = pair_key(left, right);
        const std::size_t mask = merge_table.size() - 1;
        std::size_t index = merge_slot(key, mask);
        for (std::size_t probe = 0; probe < merge_table.size(); ++probe) {
            const MergeSlot& slot = merge_table[index];
            if (slot.rank == kNoSymbol) {
                return nullptr;
            }
            if (slot.key == key) {
                return &slot;
            }
            index = (index + 1) & mask;
        }
        return nullptr;
    }

    bool insert_merge(std::uint64_t key, std::uint32_t rank,
                      std::uint32_t result) noexcept {
        const std::size_t mask = merge_table.size() - 1;
        std::size_t index = merge_slot(key, mask);
        for (std::size_t probe = 0; probe < merge_table.size(); ++probe) {
            MergeSlot& slot = merge_table[index];
            if (slot.rank == kNoSymbol) {
                slot = MergeSlot{key, rank, result};
                ++merge_rule_count;
                return true;
            }
            if (slot.key == key) {
                return false;
            }
            index = (index + 1) & mask;
        }
        return false;
    }

    void offer_candidate(
        std::uint32_t left, const std::vector<Symbol>& symbols,
        std::priority_queue<Candidate, std::vector<Candidate>, CandidateLater>& heap) const {
        if (left == kNoSymbol || left >= symbols.size() || !symbols[left].alive) {
            return;
        }
        const std::uint32_t right = symbols[left].next;
        if (right == kNoSymbol || right >= symbols.size() || !symbols[right].alive) {
            return;
        }
        const MergeSlot* found =
            find_merge(symbols[left].identifier, symbols[right].identifier);
        if (found == nullptr) {
            return;
        }
        heap.push(Candidate{
            .rank = found->rank,
            .left_position = left,
            .right_position = right,
            .left_identifier = symbols[left].identifier,
            .right_identifier = symbols[right].identifier,
            .result = found->result,
        });
    }
};

TokenizerLoadResult Tokenizer::Impl::create(
    const TokenizerSpec& specification, const std::filesystem::path& artifact_root) {
    if (specification.kind != model::qwen36::TokenizerKind::ByteLevelBpe) {
        return load_error(TokenizerLoadError::UnsupportedModel,
                          "generated tokenizer kind is unsupported");
    }
    if (specification.normalization != model::qwen36::TokenizerNormalization::Nfc) {
        return load_error(TokenizerLoadError::UnsupportedNormalizer,
                          "generated tokenizer normalizer is unsupported");
    }
    if (specification.pretokenizer !=
        model::qwen36::TokenizerPretokenizer::QwenRegexByteLevelV1) {
        return load_error(TokenizerLoadError::UnsupportedPretokenizer,
                          "generated pre-tokenizer is unsupported");
    }
    if (specification.decoder != model::qwen36::TokenizerDecoder::ByteLevel) {
        return load_error(TokenizerLoadError::UnsupportedDecoder,
                          "generated tokenizer decoder is unsupported");
    }
    if (!safe_relative_path(specification.data_path)) {
        return load_error(TokenizerLoadError::UnsafePath,
                          "tokenizer data path is not a safe relative path");
    }
    if (specification.data_size_bytes == 0) {
        return load_error(TokenizerLoadError::EmptyFile,
                          "tokenizer contract declares an empty file");
    }
    if (specification.data_size_bytes > kMaximumTokenizerBytes) {
        return load_error(TokenizerLoadError::FileTooLarge,
                          "tokenizer contract exceeds 64 MiB");
    }
    if (specification.populated_vocabulary == 0 ||
        specification.populated_vocabulary > specification.vocabulary) {
        return load_error(TokenizerLoadError::VocabularyMismatch,
                          "tokenizer populated/model vocabulary bounds are invalid");
    }

    const std::filesystem::path path =
        artifact_root / std::filesystem::path(specification.data_path);
    std::error_code filesystem_error;
    const auto status = std::filesystem::symlink_status(path, filesystem_error);
    if (filesystem_error || !std::filesystem::exists(status)) {
        return load_error(TokenizerLoadError::FileUnavailable,
                          "tokenizer data file is unavailable");
    }
    if (std::filesystem::is_symlink(status) ||
        !std::filesystem::is_regular_file(status)) {
        return load_error(TokenizerLoadError::WrongFileType,
                          "tokenizer data must be a regular non-symlink file");
    }
    const std::uintmax_t file_size = std::filesystem::file_size(path, filesystem_error);
    if (filesystem_error) {
        return load_error(TokenizerLoadError::FileUnavailable,
                          "tokenizer data size is unavailable");
    }
    if (file_size != specification.data_size_bytes) {
        return load_error(TokenizerLoadError::FileSizeChanged,
                          "tokenizer data size differs from the generated contract");
    }

    std::vector<std::byte> bytes(static_cast<std::size_t>(file_size));
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return load_error(TokenizerLoadError::FileUnavailable,
                          "tokenizer data could not be opened");
    }
    stream.read(reinterpret_cast<char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
    if (!stream || stream.gcount() != static_cast<std::streamsize>(bytes.size())) {
        return load_error(TokenizerLoadError::FileSizeChanged,
                          "tokenizer data changed while it was read");
    }
    model::Sha256 hasher;
    hasher.update(bytes);
    if (model::sha256_hex(hasher.finish()) != specification.data_sha256) {
        return load_error(TokenizerLoadError::DigestMismatch,
                          "tokenizer data digest differs from the generated contract");
    }

    @autoreleasepool {
        NSData* data = [NSData dataWithBytes:bytes.data() length:bytes.size()];
        NSError* json_error = nil;
        id parsed = [NSJSONSerialization JSONObjectWithData:data options:0 error:&json_error];
        if (parsed == nil || ![parsed isKindOfClass:NSDictionary.class]) {
            return load_error(TokenizerLoadError::InvalidJson,
                              json_error == nil ? "tokenizer root is not an object"
                                                : ns_string(json_error.localizedDescription));
        }
        NSDictionary* root = static_cast<NSDictionary*>(parsed);
        LoadFailure failure{TokenizerLoadError::None, {}};
        NSString* version = string_field(root, @"version", failure);
        if (version == nil) {
            return load_error(failure.error, std::move(failure.detail));
        }
        if (!equals(version, "1.0") || root[@"truncation"] != NSNull.null ||
            root[@"padding"] != NSNull.null) {
            return load_error(TokenizerLoadError::UnsupportedModel,
                              "tokenizer version, truncation, or padding is unsupported");
        }

        NSDictionary* normalizer = dictionary_field(root, @"normalizer", failure);
        if (normalizer == nil) {
            return load_error(failure.error, std::move(failure.detail));
        }
        NSString* normalizer_type = string_field(normalizer, @"type", failure);
        if (normalizer_type == nil) {
            return load_error(failure.error, std::move(failure.detail));
        }
        if (!equals(normalizer_type, "NFC")) {
            return load_error(TokenizerLoadError::UnsupportedNormalizer,
                              "only NFC normalization is admitted");
        }

        NSDictionary* pretokenizer =
            dictionary_field(root, @"pre_tokenizer", failure);
        if (pretokenizer == nil) {
            return load_error(failure.error, std::move(failure.detail));
        }
        NSString* pretokenizer_type =
            string_field(pretokenizer, @"type", failure);
        NSArray* stages = array_field(pretokenizer, @"pretokenizers", failure);
        if (pretokenizer_type == nil || stages == nil) {
            return load_error(failure.error, std::move(failure.detail));
        }
        if (!equals(pretokenizer_type, "Sequence") || stages.count != 2 ||
            ![stages[0] isKindOfClass:NSDictionary.class] ||
            ![stages[1] isKindOfClass:NSDictionary.class]) {
            return load_error(TokenizerLoadError::UnsupportedPretokenizer,
                              "pre-tokenizer must be Split then ByteLevel");
        }
        NSDictionary* split = static_cast<NSDictionary*>(stages[0]);
        NSDictionary* byte_level = static_cast<NSDictionary*>(stages[1]);
        NSDictionary* pattern = dictionary_field(split, @"pattern", failure);
        NSString* split_type = string_field(split, @"type", failure);
        NSString* behavior = string_field(split, @"behavior", failure);
        NSString* byte_level_type = string_field(byte_level, @"type", failure);
        if (pattern == nil || split_type == nil || behavior == nil ||
            byte_level_type == nil) {
            return load_error(failure.error, std::move(failure.detail));
        }
        NSString* regex = string_field(pattern, @"Regex", failure);
        bool invert = false;
        bool add_prefix_space = false;
        bool trim_offsets = false;
        bool use_regex = false;
        if (regex == nil || !boolean_value(split, @"invert", invert, failure) ||
            !boolean_value(byte_level, @"add_prefix_space", add_prefix_space, failure) ||
            !boolean_value(byte_level, @"trim_offsets", trim_offsets, failure) ||
            !boolean_value(byte_level, @"use_regex", use_regex, failure)) {
            return load_error(failure.error, std::move(failure.detail));
        }
        if (!equals(split_type, "Split") || !equals(behavior, "Isolated") || invert ||
            !equals(byte_level_type, "ByteLevel") || add_prefix_space || trim_offsets ||
            use_regex || ns_string(regex) != specification.split_pattern) {
            return load_error(TokenizerLoadError::UnsupportedPretokenizer,
                              "pre-tokenizer semantics differ from the package");
        }

        NSDictionary* decoder = dictionary_field(root, @"decoder", failure);
        if (decoder == nil) {
            return load_error(failure.error, std::move(failure.detail));
        }
        NSString* decoder_type = string_field(decoder, @"type", failure);
        bool decoder_prefix = false;
        bool decoder_trim = false;
        bool decoder_regex = false;
        if (decoder_type == nil ||
            !boolean_value(decoder, @"add_prefix_space", decoder_prefix, failure) ||
            !boolean_value(decoder, @"trim_offsets", decoder_trim, failure) ||
            !boolean_value(decoder, @"use_regex", decoder_regex, failure)) {
            return load_error(failure.error, std::move(failure.detail));
        }
        if (!equals(decoder_type, "ByteLevel") || decoder_prefix || decoder_trim ||
            decoder_regex) {
            return load_error(TokenizerLoadError::UnsupportedDecoder,
                              "decoder semantics differ from the package");
        }
        NSDictionary* postprocessor =
            dictionary_field(root, @"post_processor", failure);
        if (postprocessor == nil) {
            return load_error(failure.error, std::move(failure.detail));
        }
        NSString* postprocessor_type =
            string_field(postprocessor, @"type", failure);
        bool postprocessor_prefix = false;
        bool postprocessor_trim = false;
        bool postprocessor_regex = false;
        if (postprocessor_type == nil ||
            !boolean_value(postprocessor, @"add_prefix_space",
                           postprocessor_prefix, failure) ||
            !boolean_value(postprocessor, @"trim_offsets",
                           postprocessor_trim, failure) ||
            !boolean_value(postprocessor, @"use_regex",
                           postprocessor_regex, failure)) {
            return load_error(failure.error, std::move(failure.detail));
        }
        if (!equals(postprocessor_type, "ByteLevel") || postprocessor_prefix ||
            postprocessor_trim || postprocessor_regex) {
            return load_error(TokenizerLoadError::UnsupportedDecoder,
                              "post-processor semantics differ from the package");
        }

        NSDictionary* model = dictionary_field(root, @"model", failure);
        if (model == nil) {
            return load_error(failure.error, std::move(failure.detail));
        }
        NSString* model_type = string_field(model, @"type", failure);
        if (model_type == nil) {
            return load_error(failure.error, std::move(failure.detail));
        }
        bool fuse_unknown = false;
        bool byte_fallback = false;
        bool ignore_merges = false;
        if (!boolean_value(model, @"fuse_unk", fuse_unknown, failure) ||
            !boolean_value(model, @"byte_fallback", byte_fallback, failure) ||
            !boolean_value(model, @"ignore_merges", ignore_merges, failure)) {
            return load_error(failure.error, std::move(failure.detail));
        }
        id dropout = model[@"dropout"];
        id unknown = model[@"unk_token"];
        NSString* continuing =
            string_field(model, @"continuing_subword_prefix", failure);
        NSString* suffix = string_field(model, @"end_of_word_suffix", failure);
        if (continuing == nil || suffix == nil) {
            return load_error(failure.error, std::move(failure.detail));
        }
        if (!equals(model_type, "BPE") || dropout != NSNull.null ||
            unknown != NSNull.null || !equals(continuing, "") || !equals(suffix, "") ||
            fuse_unknown || byte_fallback || ignore_merges) {
            return load_error(TokenizerLoadError::UnsupportedModel,
                              "BPE semantics differ from the admitted family");
        }

        NSDictionary* vocabulary = dictionary_field(model, @"vocab", failure);
        NSArray* merge_values = array_field(model, @"merges", failure);
        NSArray* added_values = array_field(root, @"added_tokens", failure);
        if (vocabulary == nil || merge_values == nil || added_values == nil) {
            return load_error(failure.error, std::move(failure.detail));
        }

        auto implementation = std::make_unique<Impl>();
        implementation->identifier_to_token.resize(
            specification.populated_vocabulary);
        implementation->is_added.assign(specification.populated_vocabulary, 0);
        implementation->is_special.assign(specification.populated_vocabulary, 0);
        std::vector<std::uint8_t> occupied(specification.populated_vocabulary, 0);
        std::unordered_map<std::string, std::uint32_t> token_identifiers;
        token_identifiers.reserve(vocabulary.count * 2U);

        for (id key in vocabulary) {
            if (![key isKindOfClass:NSString.class]) {
                return load_error(TokenizerLoadError::InvalidVocabulary,
                                  "vocabulary key is not a string");
            }
            std::uint32_t identifier = 0;
            if (!unsigned_value(vocabulary[key], identifier) ||
                identifier >= specification.populated_vocabulary ||
                occupied[identifier] != 0) {
                return load_error(TokenizerLoadError::InvalidVocabulary,
                                  "vocabulary ID is invalid or repeated");
            }
            std::string token = ns_string(static_cast<NSString*>(key));
            if (token.empty() || !token_identifiers.emplace(token, identifier).second) {
                return load_error(TokenizerLoadError::InvalidVocabulary,
                                  "vocabulary token is empty or repeated");
            }
            implementation->identifier_to_token[identifier] = std::move(token);
            occupied[identifier] = 1;
        }

        implementation->added_tokens.reserve(added_values.count);
        for (id value in added_values) {
            if (![value isKindOfClass:NSDictionary.class]) {
                return load_error(TokenizerLoadError::InvalidAddedToken,
                                  "added token is not an object");
            }
            NSDictionary* added = static_cast<NSDictionary*>(value);
            std::uint32_t identifier = 0;
            if (!unsigned_value(added[@"id"], identifier) ||
                identifier >= specification.populated_vocabulary ||
                occupied[identifier] != 0) {
                return load_error(TokenizerLoadError::InvalidAddedToken,
                                  "added-token ID is invalid or repeated");
            }
            NSString* content_value = string_field(added, @"content", failure);
            bool special = false;
            bool single_word = false;
            bool left_strip = false;
            bool right_strip = false;
            bool normalized = false;
            if (content_value == nil ||
                !boolean_value(added, @"special", special, failure) ||
                !boolean_value(added, @"single_word", single_word, failure) ||
                !boolean_value(added, @"lstrip", left_strip, failure) ||
                !boolean_value(added, @"rstrip", right_strip, failure) ||
                !boolean_value(added, @"normalized", normalized, failure)) {
                return load_error(failure.error, std::move(failure.detail));
            }
            std::string content = ns_string(content_value);
            if (content.empty() || single_word || left_strip || right_strip || normalized) {
                return load_error(TokenizerLoadError::InvalidAddedToken,
                                  "added-token semantics are unsupported");
            }
            for (const auto& prior : implementation->added_tokens) {
                if (prior.content == content) {
                    return load_error(TokenizerLoadError::InvalidAddedToken,
                                      "added-token content is repeated");
                }
            }
            implementation->identifier_to_token[identifier] = content;
            implementation->is_added[identifier] = 1;
            implementation->is_special[identifier] = special ? 1 : 0;
            implementation->added_tokens.push_back(
                AddedToken{std::move(content), identifier, special});
            occupied[identifier] = 1;
        }
        if (std::find(occupied.begin(), occupied.end(), 0) != occupied.end()) {
            return load_error(TokenizerLoadError::VocabularyMismatch,
                              "tokenizer IDs do not form the generated populated prefix");
        }

        if (merge_values.count >= std::numeric_limits<std::uint32_t>::max()) {
            return load_error(TokenizerLoadError::ArithmeticOverflow,
                              "merge count exceeds the admitted uint32 bound");
        }
        std::size_t doubled_merge_count = 0;
        if (!checked_add(static_cast<std::size_t>(merge_values.count),
                         static_cast<std::size_t>(merge_values.count),
                         doubled_merge_count)) {
            return load_error(TokenizerLoadError::ArithmeticOverflow,
                              "merge-table capacity overflowed");
        }
        const std::size_t maximum_power =
            std::size_t{1} << (std::numeric_limits<std::size_t>::digits - 1);
        if (doubled_merge_count > maximum_power) {
            return load_error(TokenizerLoadError::ArithmeticOverflow,
                              "merge-table power-of-two capacity overflowed");
        }
        const std::size_t merge_capacity =
            std::bit_ceil(std::max<std::size_t>(2, doubled_merge_count));
        implementation->merge_table.assign(
            merge_capacity, MergeSlot{.key = 0, .rank = kNoSymbol, .result = 0});
        std::uint32_t rank = 0;
        for (id value in merge_values) {
            if (![value isKindOfClass:NSArray.class]) {
                return load_error(TokenizerLoadError::InvalidMerge,
                                  "merge entry is not a pair");
            }
            NSArray* pair = static_cast<NSArray*>(value);
            if (pair.count != 2 || ![pair[0] isKindOfClass:NSString.class] ||
                ![pair[1] isKindOfClass:NSString.class]) {
                return load_error(TokenizerLoadError::InvalidMerge,
                                  "merge entry is not two strings");
            }
            const std::string left = ns_string(static_cast<NSString*>(pair[0]));
            const std::string right = ns_string(static_cast<NSString*>(pair[1]));
            const auto left_found = token_identifiers.find(left);
            const auto right_found = token_identifiers.find(right);
            const auto result_found = token_identifiers.find(left + right);
            if (left_found == token_identifiers.end() ||
                right_found == token_identifiers.end() ||
                result_found == token_identifiers.end()) {
                return load_error(TokenizerLoadError::InvalidMerge,
                                  "merge input or output is absent from the vocabulary");
            }
            if (!implementation->insert_merge(
                    pair_key(left_found->second, right_found->second), rank,
                    result_found->second)) {
                return load_error(TokenizerLoadError::InvalidMerge,
                                  "merge pair is repeated");
            }
            ++rank;
        }

        implementation->character_bytes.fill(
            std::numeric_limits<std::uint16_t>::max());
        std::uint32_t extension = 0;
        for (std::uint32_t byte = 0; byte < 256; ++byte) {
            const bool direct = (byte >= 33 && byte <= 126) ||
                                (byte >= 161 && byte <= 172) ||
                                (byte >= 174 && byte <= 255);
            const std::uint32_t codepoint = direct ? byte : 256 + extension++;
            std::string mapped;
            append_utf8(mapped, codepoint);
            const auto found = token_identifiers.find(mapped);
            if (found == token_identifiers.end()) {
                return load_error(TokenizerLoadError::InvalidVocabulary,
                                  "byte alphabet token is absent from the vocabulary");
            }
            implementation->byte_identifiers[byte] = found->second;
            implementation->character_bytes[codepoint] =
                static_cast<std::uint16_t>(byte);
        }

        NSError* regex_error = nil;
        implementation->split_expression =
            [NSRegularExpression regularExpressionWithPattern:regex
                                                      options:0
                                                        error:&regex_error];
        if (implementation->split_expression == nil) {
            return load_error(
                TokenizerLoadError::RegexCompilation,
                regex_error == nil ? "regular expression compilation failed"
                                   : ns_string(regex_error.localizedDescription));
        }

        std::size_t table_bytes = 0;
        for (const auto& token : implementation->identifier_to_token) {
            if (!checked_add(table_bytes, token.capacity() + 1, table_bytes)) {
                return load_error(TokenizerLoadError::ArithmeticOverflow,
                                  "token table byte accounting overflowed");
            }
        }
        for (const auto& token : implementation->added_tokens) {
            if (!checked_add(table_bytes, token.content.capacity() + 1, table_bytes)) {
                return load_error(TokenizerLoadError::ArithmeticOverflow,
                                  "added-token byte accounting overflowed");
            }
        }
        const std::array<std::size_t, 5> fixed_counts{
            implementation->identifier_to_token.capacity() * sizeof(std::string),
            implementation->is_added.capacity(),
            implementation->is_special.capacity(),
            implementation->added_tokens.capacity() * sizeof(AddedToken),
            implementation->merge_table.capacity() * sizeof(MergeSlot),
        };
        for (const std::size_t count : fixed_counts) {
            if (!checked_add(table_bytes, count, table_bytes)) {
                return load_error(TokenizerLoadError::ArithmeticOverflow,
                                  "token table byte accounting overflowed");
            }
        }
        constexpr std::size_t kFoundationRegexReserve = 1U << 20U;
        if (!checked_add(table_bytes, sizeof(implementation->byte_identifiers),
                         table_bytes) ||
            !checked_add(table_bytes, sizeof(implementation->character_bytes),
                         table_bytes) ||
            !checked_add(table_bytes, kFoundationRegexReserve, table_bytes)) {
            return load_error(TokenizerLoadError::ArithmeticOverflow,
                              "token table byte accounting overflowed");
        }
        implementation->immutable_table_bytes = table_bytes;
        return {
            .tokenizer =
                std::unique_ptr<Tokenizer>(new Tokenizer(std::move(implementation))),
            .error = TokenizerLoadError::None,
            .detail = {},
        };
    }
}

TokenizerError Tokenizer::Impl::bpe(std::string_view bytes,
                                    const TokenizerLimits& limits,
                                    std::vector<std::uint32_t>& output,
                                    std::string& detail) const {
    if (bytes.empty()) {
        return TokenizerError::None;
    }
    if (bytes.size() > std::numeric_limits<std::uint32_t>::max()) {
        detail = "pre-token symbol count exceeds uint32";
        return TokenizerError::ArithmeticOverflow;
    }
    std::size_t heap_bound = 0;
    if (!checked_add(bytes.size(), bytes.size(), heap_bound) ||
        !checked_add(heap_bound, bytes.size(), heap_bound)) {
        detail = "BPE heap bound overflowed";
        return TokenizerError::ArithmeticOverflow;
    }

    std::vector<Symbol> symbols;
    symbols.reserve(bytes.size());
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        const auto byte = static_cast<std::uint8_t>(bytes[index]);
        symbols.push_back(Symbol{
            .identifier = byte_identifiers[byte],
            .previous = index == 0 ? kNoSymbol : static_cast<std::uint32_t>(index - 1),
            .next = index + 1 == bytes.size() ? kNoSymbol
                                              : static_cast<std::uint32_t>(index + 1),
            .alive = true,
        });
    }
    std::vector<Candidate> heap_storage;
    heap_storage.reserve(heap_bound);
    std::priority_queue<Candidate, std::vector<Candidate>, CandidateLater> heap(
        CandidateLater{}, std::move(heap_storage));
    for (std::uint32_t index = 0; index + 1 < symbols.size(); ++index) {
        offer_candidate(index, symbols, heap);
    }

    while (!heap.empty()) {
        const Candidate candidate = heap.top();
        heap.pop();
        if (candidate.left_position >= symbols.size() ||
            candidate.right_position >= symbols.size()) {
            continue;
        }
        Symbol& left = symbols[candidate.left_position];
        Symbol& right = symbols[candidate.right_position];
        if (!left.alive || !right.alive || left.next != candidate.right_position ||
            right.previous != candidate.left_position ||
            left.identifier != candidate.left_identifier ||
            right.identifier != candidate.right_identifier) {
            continue;
        }
        const MergeSlot* rule = find_merge(left.identifier, right.identifier);
        if (rule == nullptr || rule->rank != candidate.rank ||
            rule->result != candidate.result) {
            continue;
        }
        left.identifier = candidate.result;
        left.next = right.next;
        right.alive = false;
        if (right.next != kNoSymbol) {
            symbols[right.next].previous = candidate.left_position;
        }
        offer_candidate(left.previous, symbols, heap);
        offer_candidate(candidate.left_position, symbols, heap);
    }

    for (std::uint32_t index = 0; index != kNoSymbol; index = symbols[index].next) {
        if (output.size() >= limits.maximum_tokens) {
            detail = "encoded token count exceeds the request limit";
            return TokenizerError::TokenLimitExceeded;
        }
        output.push_back(symbols[index].identifier);
    }
    return TokenizerError::None;
}

TokenizerError Tokenizer::Impl::encode_text(
    std::string_view utf8, const TokenizerLimits& limits,
    std::vector<std::uint32_t>& output, std::string& detail) const {
    if (utf8.empty()) {
        return TokenizerError::None;
    }
    @autoreleasepool {
        NSString* raw = foundation_string(utf8);
        if (raw == nil) {
            detail = "input is not valid UTF-8";
            return TokenizerError::InvalidUtf8;
        }
        NSString* normalized = raw.precomposedStringWithCanonicalMapping;
        NSData* normalized_data =
            [normalized dataUsingEncoding:NSUTF8StringEncoding allowLossyConversion:NO];
        if (normalized_data == nil) {
            detail = "NFC normalization did not produce valid UTF-8";
            return TokenizerError::InvalidUtf8;
        }
        if (normalized_data.length > limits.maximum_input_bytes) {
            detail = "normalized input exceeds the request byte limit";
            return TokenizerError::NormalizationLimitExceeded;
        }
        NSArray<NSTextCheckingResult*>* matches =
            [split_expression matchesInString:normalized
                                      options:0
                                        range:NSMakeRange(0, normalized.length)];
        TokenizerError result = TokenizerError::None;
        std::string block_detail;
        auto emit = [&](NSRange range) {
            if (range.length == 0 || result != TokenizerError::None) {
                return;
            }
            NSString* piece = [normalized substringWithRange:range];
            NSData* data =
                [piece dataUsingEncoding:NSUTF8StringEncoding allowLossyConversion:NO];
            if (data == nil) {
                result = TokenizerError::InvalidUtf8;
                block_detail = "pre-token is not valid UTF-8";
                return;
            }
            result = bpe(
                std::string_view(static_cast<const char*>(data.bytes), data.length),
                limits, output, block_detail);
        };
        NSUInteger previous = 0;
        for (NSTextCheckingResult* match in matches) {
            if (match.range.location > previous) {
                emit(NSMakeRange(previous, match.range.location - previous));
            }
            emit(match.range);
            previous = match.range.location + match.range.length;
        }
        if (previous < normalized.length) {
            emit(NSMakeRange(previous, normalized.length - previous));
        }
        if (result != TokenizerError::None) {
            detail = std::move(block_detail);
        }
        return result;
    }
}

EncodeResult Tokenizer::Impl::encode(std::string_view utf8,
                                     const TokenizerLimits& limits) const {
    if (utf8.size() > limits.maximum_input_bytes) {
        return encode_error(TokenizerError::InputLimitExceeded,
                            "input exceeds the request byte limit");
    }
    std::vector<std::uint32_t> output;
    output.reserve(std::min(utf8.size(), limits.maximum_tokens));
    std::size_t position = 0;
    while (position < utf8.size()) {
        std::size_t best_position = std::string_view::npos;
        std::size_t best_length = 0;
        std::uint32_t best_identifier = 0;
        for (const AddedToken& added : added_tokens) {
            const std::size_t found = utf8.find(added.content, position);
            if (found == std::string_view::npos) {
                continue;
            }
            if (found < best_position ||
                (found == best_position &&
                 (added.content.size() > best_length ||
                  (added.content.size() == best_length &&
                   added.identifier < best_identifier)))) {
                best_position = found;
                best_length = added.content.size();
                best_identifier = added.identifier;
            }
        }
        std::string detail;
        if (best_position == std::string_view::npos) {
            const TokenizerError error =
                encode_text(utf8.substr(position), limits, output, detail);
            if (error != TokenizerError::None) {
                return encode_error(error, std::move(detail));
            }
            break;
        }
        if (best_position > position) {
            const TokenizerError error = encode_text(
                utf8.substr(position, best_position - position), limits, output, detail);
            if (error != TokenizerError::None) {
                return encode_error(error, std::move(detail));
            }
        }
        if (output.size() >= limits.maximum_tokens) {
            return encode_error(TokenizerError::TokenLimitExceeded,
                                "encoded token count exceeds the request limit");
        }
        output.push_back(best_identifier);
        position = best_position + best_length;
    }
    return {
        .tokens = std::move(output),
        .error = TokenizerError::None,
        .detail = {},
    };
}

DecodeResult Tokenizer::Impl::decode(std::span<const std::uint32_t> tokens,
                                     bool skip_special,
                                     const TokenizerLimits& limits) const {
    std::string output;
    for (const std::uint32_t identifier : tokens) {
        if (identifier >= identifier_to_token.size() ||
            identifier_to_token[identifier].empty()) {
            return decode_error(TokenizerError::InvalidTokenId,
                                "token ID is not in the populated vocabulary");
        }
        const std::string& token = identifier_to_token[identifier];
        if (is_added[identifier] != 0) {
            if (skip_special && is_special[identifier] != 0) {
                continue;
            }
            if (token.size() > limits.maximum_output_bytes - output.size()) {
                return decode_error(TokenizerError::OutputLimitExceeded,
                                    "decoded bytes exceed the request limit");
            }
            output += token;
            continue;
        }
        std::size_t offset = 0;
        while (offset < token.size()) {
            std::uint32_t codepoint = 0;
            if (!decode_utf8_codepoint(token, offset, codepoint)) {
                return decode_error(TokenizerError::UnknownByteSymbol,
                                    "vocabulary token is not valid UTF-8");
            }
            if (codepoint >= character_bytes.size() ||
                character_bytes[codepoint] ==
                    std::numeric_limits<std::uint16_t>::max()) {
                return decode_error(TokenizerError::UnknownByteSymbol,
                                    "vocabulary token is outside the byte alphabet");
            }
            if (output.size() >= limits.maximum_output_bytes) {
                return decode_error(TokenizerError::OutputLimitExceeded,
                                    "decoded bytes exceed the request limit");
            }
            output.push_back(static_cast<char>(character_bytes[codepoint]));
        }
    }
    return {
        .bytes = std::move(output),
        .error = TokenizerError::None,
        .detail = {},
    };
}

Tokenizer::Tokenizer(std::unique_ptr<Impl> implementation) noexcept
    : implementation_(std::move(implementation)) {}

Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;
Tokenizer::~Tokenizer() = default;

TokenizerLoadResult Tokenizer::load(
    const model::qwen36::TokenizerSpec& specification,
    const std::filesystem::path& artifact_root) {
    try {
        return Impl::create(specification, artifact_root);
    } catch (const std::bad_alloc&) {
        return load_error(TokenizerLoadError::AllocationFailure,
                          "tokenizer table allocation failed");
    }
}

EncodeResult Tokenizer::encode(std::string_view utf8,
                               const TokenizerLimits& limits) const {
    try {
        return implementation_->encode(utf8, limits);
    } catch (const std::bad_alloc&) {
        return encode_error(TokenizerError::AllocationFailure,
                            "tokenizer request allocation failed");
    }
}

DecodeResult Tokenizer::decode(std::span<const std::uint32_t> tokens,
                               bool skip_special,
                               const TokenizerLimits& limits) const {
    try {
        return implementation_->decode(tokens, skip_special, limits);
    } catch (const std::bad_alloc&) {
        return decode_error(TokenizerError::AllocationFailure,
                            "detokenizer request allocation failed");
    }
}

std::size_t Tokenizer::populated_vocabulary() const noexcept {
    return implementation_->identifier_to_token.size();
}

std::size_t Tokenizer::merge_count() const noexcept {
    return implementation_->merge_rule_count;
}

std::size_t Tokenizer::added_token_count() const noexcept {
    return implementation_->added_tokens.size();
}

std::size_t Tokenizer::table_bytes() const noexcept {
    return implementation_->immutable_table_bytes;
}

} // namespace tatara::text
