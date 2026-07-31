#include "tatara/evidence/codecs.h"

#include "tatara/evidence/wire.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <limits>

namespace tatara::evidence {
namespace {

// --- Shared byte helpers ---------------------------------------------------

constexpr std::array<std::byte, 8> kRawLogMagic = {
    std::byte{0x54}, std::byte{0x41}, std::byte{0x54}, std::byte{0x52},
    std::byte{0x4c}, std::byte{0x4f}, std::byte{0x47}, std::byte{0x00},
};
constexpr std::uint16_t kRawLogTerminalTruncatedFlag = 0x0001;

void append_u16_be(std::vector<std::byte>& out, std::uint16_t value) {
    out.push_back(static_cast<std::byte>(value >> 8));
    out.push_back(static_cast<std::byte>(value & 0xffU));
}

void append_u32_be(std::vector<std::byte>& out, std::uint32_t value) {
    for (std::size_t index = 0; index < 4; ++index) {
        const std::size_t shift = (3 - index) * 8;
        out.push_back(static_cast<std::byte>(value >> shift));
    }
}

void append_u64_be(std::vector<std::byte>& out, std::uint64_t value) {
    for (std::size_t index = 0; index < 8; ++index) {
        const std::size_t shift = (7 - index) * 8;
        out.push_back(static_cast<std::byte>(value >> shift));
    }
}

std::uint16_t read_u16_be(std::span<const std::byte> bytes, std::size_t offset) noexcept {
    std::uint16_t value = 0;
    for (std::size_t index = 0; index < 2; ++index) {
        value = static_cast<std::uint16_t>(
            static_cast<std::uint16_t>(value << 8) |
            std::to_integer<std::uint16_t>(bytes[offset + index]));
    }
    return value;
}

std::uint32_t read_u32_be(std::span<const std::byte> bytes, std::size_t offset) noexcept {
    std::uint32_t value = 0;
    for (std::size_t index = 0; index < 4; ++index) {
        value = (value << 8) | std::to_integer<std::uint32_t>(bytes[offset + index]);
    }
    return value;
}

std::uint64_t read_u64_be(std::span<const std::byte> bytes, std::size_t offset) noexcept {
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < 8; ++index) {
        value = (value << 8) | std::to_integer<std::uint64_t>(bytes[offset + index]);
    }
    return value;
}

std::uint64_t effective_budget(std::uint64_t maximum_artifact_bytes,
                               std::uint64_t format_ceiling) noexcept {
    return std::min(maximum_artifact_bytes, format_ceiling);
}

// --- Raw log validation ----------------------------------------------------

bool valid_raw_log_stream(RawLogStreamId stream_id) noexcept {
    const auto value = static_cast<std::uint8_t>(stream_id);
    return value >= static_cast<std::uint8_t>(RawLogStreamId::Stdout) &&
           value <= static_cast<std::uint8_t>(RawLogStreamId::Harness);
}

RawLogError validate_raw_log_data_frame(const RawLogDataFrame& frame) noexcept {
    if (!valid_raw_log_stream(frame.stream_id)) {
        return RawLogError::InvalidEnum;
    }
    if (frame.payload.size() < kMinimumRawLogDataPayloadBytes ||
        frame.payload.size() > kMaximumRawLogDataPayloadBytes) {
        return RawLogError::NumericRange;
    }
    return RawLogError::None;
}

RawLogError validate_raw_log_terminal_frame(const RawLogTerminalFrame& terminal) noexcept {
    // TRUNCATED = 1 exactly when bytes were dropped and the count is known.
    if (terminal.truncated != (terminal.discarded_payload_bytes > 0)) {
        return RawLogError::ConservationMismatch;
    }
    return RawLogError::None;
}

void append_raw_log_data_frame(std::vector<std::byte>& out, const RawLogDataFrame& frame) {
    out.push_back(static_cast<std::byte>(static_cast<std::uint8_t>(RawLogRecordType::Data)));
    out.push_back(static_cast<std::byte>(static_cast<std::uint8_t>(frame.stream_id)));
    append_u16_be(out, 0);
    append_u64_be(out, frame.monotonic_offset_ns);
    append_u32_be(out, static_cast<std::uint32_t>(frame.payload.size()));
    out.insert(out.end(), frame.payload.begin(), frame.payload.end());
}

void append_raw_log_terminal_frame(std::vector<std::byte>& out,
                                   const RawLogTerminalFrame& terminal) {
    out.push_back(static_cast<std::byte>(static_cast<std::uint8_t>(RawLogRecordType::Terminal)));
    out.push_back(std::byte{0});
    append_u16_be(out, static_cast<std::uint16_t>(
                           terminal.truncated ? kRawLogTerminalTruncatedFlag : 0));
    append_u64_be(out, terminal.monotonic_offset_ns);
    append_u32_be(out, static_cast<std::uint32_t>(kRawLogTerminalPayloadBytes));
    append_u64_be(out, terminal.data_frame_count);
    append_u64_be(out, terminal.captured_payload_bytes);
    append_u64_be(out, terminal.discarded_payload_bytes);
}

// Classifies the leading record-type byte for a decoder expecting `expected`.
RawLogError check_raw_log_record_type(std::byte record_type_byte,
                                      RawLogRecordType expected) noexcept {
    const auto value = std::to_integer<std::uint8_t>(record_type_byte);
    if (value != static_cast<std::uint8_t>(RawLogRecordType::Data) &&
        value != static_cast<std::uint8_t>(RawLogRecordType::Terminal)) {
        return RawLogError::InvalidEnum;
    }
    if (value != static_cast<std::uint8_t>(expected)) {
        return RawLogError::InvalidFrame;
    }
    return RawLogError::None;
}

// --- Sample stream vocabulary ----------------------------------------------

constexpr std::string_view kSampleStreamSchemaName = "tatara.sample_stream";
constexpr std::uint64_t kSampleStreamSchemaVersion = 1;

constexpr std::array<std::string_view, 5> kSampleArmSpellings = {
    "COMPETITOR", "CONTROL_AFTER", "CONTROL_BEFORE", "DESCRIPTIVE", "TREATMENT",
};
constexpr std::array<std::string_view, 3> kSampleDirectionSpellings = {
    "LOWER_IS_BETTER", "HIGHER_IS_BETTER", "DESCRIPTIVE_ONLY",
};
constexpr std::array<std::string_view, 3> kSampleDispositionSpellings = {
    "WARMUP", "RETAINED", "REJECTED",
};
// NotApplicable first; the seven concrete reasons follow in ASCII order so the
// enum order is also the canonical sorted order for the header set.
constexpr std::array<std::string_view, 8> kSampleRejectionReasonSpellings = {
    "NOT_APPLICABLE",  "CANCELLED",     "MEASUREMENT_UNAVAILABLE",
    "MEMORY_PRESSURE_POLICY", "ORDER_PROTOCOL_ABORT", "REQUEST_ERROR",
    "THERMAL_POLICY",  "TIMEOUT",
};

enum class SampleRecordKind : std::uint8_t {
    Header,
    Sample,
    Terminal,
};
constexpr std::array<std::string_view, 3> kSampleRecordKindSpellings = {
    "HEADER", "SAMPLE", "TERMINAL",
};

constexpr std::array<std::string_view, 11> kSampleHeaderKeys = {
    "allowed_rejection_reasons",
    "arm",
    "decimal_scale",
    "direction",
    "maximum_samples",
    "metric_name",
    "planned_observation_count",
    "record_kind",
    "schema_name",
    "schema_version",
    "unit",
};
constexpr std::array<std::string_view, 6> kSampleRecordKeys = {
    "arm",       "disposition",  "record_kind",
    "rejection_reason", "sample_index", "value_scaled",
};
constexpr std::array<std::string_view, 7> kSampleTerminalKeys = {
    "complete",       "observed_count", "record_kind", "rejected_count",
    "retained_count", "truncated",      "warmup_count",
};

template <typename Enum, std::size_t Count>
bool parse_enum_spelling(const std::array<std::string_view, Count>& spellings,
                         std::string_view text, Enum& value) noexcept {
    for (std::size_t index = 0; index < Count; ++index) {
        if (spellings[index] == text) {
            value = static_cast<Enum>(index);
            return true;
        }
    }
    return false;
}

template <typename Enum, std::size_t Count>
bool valid_enum_value(const std::array<std::string_view, Count>& spellings,
                      Enum value) noexcept {
    (void)spellings;
    return static_cast<std::size_t>(value) < Count;
}

// --- Strict flat-record JSON tokenizer -------------------------------------

// Capacity exceeds the widest schema record (11 keys, 7 set members) so a
// duplicate or unknown key is always classifiable by pigeonhole.
constexpr std::size_t kMaximumRecordPairs = 12;
constexpr std::size_t kMaximumArrayElements = 8;

enum class JsonValueKind : std::uint8_t {
    String,
    Integer,
    Boolean,
    StringArray,
};

struct JsonValue {
    JsonValueKind kind{JsonValueKind::String};
    std::string_view text{};
    bool boolean{false};
    std::array<std::string_view, kMaximumArrayElements> elements{};
    std::size_t element_count{0};
};

struct JsonPair {
    std::string_view key{};
    JsonValue value{};
};

struct RecordTokens {
    std::array<JsonPair, kMaximumRecordPairs> pairs{};
    std::size_t pair_count{0};
    bool overflowed{false};
};

bool json_whitespace(char character) noexcept {
    return character == ' ' || character == '\t' || character == '\r' ||
           character == '\n';
}

SampleStreamError expect_character(std::string_view line, std::size_t& position,
                                   char expected) noexcept {
    if (position >= line.size()) {
        return SampleStreamError::InvalidFrame;
    }
    if (line[position] == expected) {
        ++position;
        return SampleStreamError::None;
    }
    if (json_whitespace(line[position])) {
        return SampleStreamError::NonCanonicalBytes;
    }
    return SampleStreamError::InvalidFrame;
}

// Reads the raw bytes of a string token. Canonical schema strings never need
// an escape, so any reverse solidus or raw control byte is noncanonical.
SampleStreamError scan_string_body(std::string_view line, std::size_t& position,
                                   std::string_view& text) noexcept {
    const std::size_t begin = position;
    while (position < line.size()) {
        const char character = line[position];
        if (character == '"') {
            text = line.substr(begin, position - begin);
            ++position;
            return SampleStreamError::None;
        }
        if (character == '\\' || static_cast<unsigned char>(character) < 0x20) {
            return SampleStreamError::NonCanonicalBytes;
        }
        ++position;
    }
    return SampleStreamError::InvalidFrame;
}

SampleStreamError scan_integer_token(std::string_view line, std::size_t& position,
                                     std::string_view& text) noexcept {
    const std::size_t begin = position;
    if (position < line.size() && line[position] == '-') {
        ++position;
    }
    const std::size_t digits_begin = position;
    while (position < line.size() && line[position] >= '0' && line[position] <= '9') {
        ++position;
    }
    if (position == digits_begin) {
        return SampleStreamError::InvalidInteger;
    }
    if (position < line.size() &&
        (line[position] == '.' || line[position] == 'e' || line[position] == 'E' ||
         line[position] == '+')) {
        return SampleStreamError::InvalidInteger;
    }
    text = line.substr(begin, position - begin);
    return SampleStreamError::None;
}

SampleStreamError expect_literal(std::string_view line, std::size_t& position,
                                 std::string_view literal) noexcept {
    if (line.size() - position < literal.size() ||
        line.substr(position, literal.size()) != literal) {
        return SampleStreamError::InvalidFrame;
    }
    position += literal.size();
    return SampleStreamError::None;
}

SampleStreamError scan_value(std::string_view line, std::size_t& position,
                             JsonValue& value) noexcept {
    if (position >= line.size()) {
        return SampleStreamError::InvalidFrame;
    }
    const char character = line[position];
    if (character == '"') {
        ++position;
        value.kind = JsonValueKind::String;
        return scan_string_body(line, position, value.text);
    }
    if (character == '-' || (character >= '0' && character <= '9')) {
        value.kind = JsonValueKind::Integer;
        return scan_integer_token(line, position, value.text);
    }
    if (character == 't' || character == 'f') {
        value.kind = JsonValueKind::Boolean;
        value.boolean = character == 't';
        return expect_literal(line, position, value.boolean ? "true" : "false");
    }
    if (character == 'n') {
        const SampleStreamError error = expect_literal(line, position, "null");
        return error == SampleStreamError::None ? SampleStreamError::WrongType : error;
    }
    if (character == '[') {
        ++position;
        value.kind = JsonValueKind::StringArray;
        value.element_count = 0;
        if (position < line.size() && line[position] == ']') {
            ++position;
            return SampleStreamError::None;
        }
        while (true) {
            SampleStreamError error = expect_character(line, position, '"');
            if (error != SampleStreamError::None) {
                return error;
            }
            std::string_view element;
            error = scan_string_body(line, position, element);
            if (error != SampleStreamError::None) {
                return error;
            }
            if (value.element_count == kMaximumArrayElements) {
                return SampleStreamError::InvalidCount;
            }
            value.elements[value.element_count] = element;
            ++value.element_count;
            if (position >= line.size()) {
                return SampleStreamError::InvalidFrame;
            }
            if (line[position] == ',') {
                ++position;
                continue;
            }
            if (line[position] == ']') {
                ++position;
                return SampleStreamError::None;
            }
            return json_whitespace(line[position]) ? SampleStreamError::NonCanonicalBytes
                                                   : SampleStreamError::InvalidFrame;
        }
    }
    if (json_whitespace(character)) {
        return SampleStreamError::NonCanonicalBytes;
    }
    return SampleStreamError::InvalidFrame;
}

SampleStreamError tokenize_record(std::string_view line, RecordTokens& tokens) noexcept {
    std::size_t position = 0;
    SampleStreamError error = expect_character(line, position, '{');
    if (error != SampleStreamError::None) {
        return error;
    }
    if (position < line.size() && line[position] == '}') {
        ++position;
    } else {
        while (true) {
            error = expect_character(line, position, '"');
            if (error != SampleStreamError::None) {
                return error;
            }
            std::string_view key;
            error = scan_string_body(line, position, key);
            if (error != SampleStreamError::None) {
                return error;
            }
            error = expect_character(line, position, ':');
            if (error != SampleStreamError::None) {
                return error;
            }
            JsonValue value;
            error = scan_value(line, position, value);
            if (error != SampleStreamError::None) {
                return error;
            }
            if (tokens.pair_count == kMaximumRecordPairs) {
                tokens.overflowed = true;
                return SampleStreamError::None;
            }
            tokens.pairs[tokens.pair_count] = {key, value};
            ++tokens.pair_count;
            if (position >= line.size()) {
                return SampleStreamError::InvalidFrame;
            }
            if (line[position] == ',') {
                ++position;
                continue;
            }
            if (line[position] == '}') {
                ++position;
                break;
            }
            return json_whitespace(line[position]) ? SampleStreamError::NonCanonicalBytes
                                                   : SampleStreamError::InvalidFrame;
        }
    }
    if (position != line.size()) {
        return SampleStreamError::TrailingData;
    }
    return SampleStreamError::None;
}

bool has_duplicate_key(const RecordTokens& tokens) noexcept {
    for (std::size_t left = 0; left < tokens.pair_count; ++left) {
        for (std::size_t right = left + 1; right < tokens.pair_count; ++right) {
            if (tokens.pairs[left].key == tokens.pairs[right].key) {
                return true;
            }
        }
    }
    return false;
}

SampleStreamError analyze_record_keys(const RecordTokens& tokens,
                                      std::span<const std::string_view> expected) noexcept {
    if (has_duplicate_key(tokens)) {
        return SampleStreamError::DuplicateKey;
    }
    for (std::size_t index = 0; index < tokens.pair_count; ++index) {
        if (std::find(expected.begin(), expected.end(), tokens.pairs[index].key) ==
            expected.end()) {
            return SampleStreamError::UnknownKey;
        }
    }
    for (const std::string_view key : expected) {
        bool found = false;
        for (std::size_t index = 0; index < tokens.pair_count; ++index) {
            if (tokens.pairs[index].key == key) {
                found = true;
                break;
            }
        }
        if (!found) {
            return SampleStreamError::MissingField;
        }
    }
    if (tokens.overflowed) {
        // Unreachable: capacity exceeds every schema key set, so an overflow
        // always carries a duplicate or unknown key.
        return SampleStreamError::InvalidFrame;
    }
    for (std::size_t index = 0; index < tokens.pair_count; ++index) {
        if (tokens.pairs[index].key != expected[index]) {
            return SampleStreamError::NonCanonicalOrder;
        }
    }
    return SampleStreamError::None;
}

// --- Canonical JSON integer parsing ----------------------------------------

SampleStreamError parse_integer_text(std::string_view text, bool& negative,
                                     std::uint64_t& magnitude) noexcept {
    negative = false;
    std::string_view digits = text;
    if (!digits.empty() && digits.front() == '-') {
        negative = true;
        digits.remove_prefix(1);
    }
    if (digits.empty()) {
        return SampleStreamError::InvalidInteger;
    }
    if (digits.size() > 1 && digits.front() == '0') {
        return SampleStreamError::InvalidInteger;
    }
    std::uint64_t parsed = 0;
    for (const char character : digits) {
        if (character < '0' || character > '9') {
            return SampleStreamError::InvalidInteger;
        }
        const auto digit = static_cast<std::uint64_t>(character - '0');
        if (parsed > (std::numeric_limits<std::uint64_t>::max() - digit) / 10U) {
            return SampleStreamError::NumericRange;
        }
        parsed = parsed * 10U + digit;
    }
    if (negative && parsed == 0) {
        return SampleStreamError::InvalidInteger;
    }
    magnitude = parsed;
    return SampleStreamError::None;
}

SampleStreamError take_string_value(const JsonValue& value, std::string_view& text) noexcept {
    if (value.kind != JsonValueKind::String) {
        return SampleStreamError::WrongType;
    }
    text = value.text;
    return SampleStreamError::None;
}

SampleStreamError take_boolean_value(const JsonValue& value, bool& out) noexcept {
    if (value.kind != JsonValueKind::Boolean) {
        return SampleStreamError::WrongType;
    }
    out = value.boolean;
    return SampleStreamError::None;
}

SampleStreamError parse_unsigned_value(const JsonValue& value, std::uint64_t& out) noexcept {
    if (value.kind != JsonValueKind::Integer) {
        return SampleStreamError::WrongType;
    }
    bool negative = false;
    std::uint64_t magnitude = 0;
    const SampleStreamError error = parse_integer_text(value.text, negative, magnitude);
    if (error != SampleStreamError::None) {
        return error;
    }
    if (negative) {
        return SampleStreamError::NumericRange;
    }
    out = magnitude;
    return SampleStreamError::None;
}

SampleStreamError parse_signed_value(const JsonValue& value, std::int64_t& out) noexcept {
    if (value.kind != JsonValueKind::Integer) {
        return SampleStreamError::WrongType;
    }
    bool negative = false;
    std::uint64_t magnitude = 0;
    const SampleStreamError error = parse_integer_text(value.text, negative, magnitude);
    if (error != SampleStreamError::None) {
        return error;
    }
    constexpr std::uint64_t kPositiveLimit = 9223372036854775807ULL;
    if (negative) {
        if (magnitude > kPositiveLimit + 1) {
            return SampleStreamError::NumericRange;
        }
        out = magnitude == kPositiveLimit + 1
                  ? std::numeric_limits<std::int64_t>::min()
                  : -static_cast<std::int64_t>(magnitude);
        return SampleStreamError::None;
    }
    if (magnitude > kPositiveLimit) {
        return SampleStreamError::NumericRange;
    }
    out = static_cast<std::int64_t>(magnitude);
    return SampleStreamError::None;
}

SampleStreamError parse_bounded_count_value(const JsonValue& value, std::uint64_t minimum,
                                            std::uint64_t maximum,
                                            SampleStreamError range_error,
                                            std::uint32_t& out) noexcept {
    std::uint64_t parsed = 0;
    const SampleStreamError error = parse_unsigned_value(value, parsed);
    if (error != SampleStreamError::None) {
        return error;
    }
    if (parsed < minimum || parsed > maximum) {
        return range_error;
    }
    out = static_cast<std::uint32_t>(parsed);
    return SampleStreamError::None;
}

SampleStreamError check_record_kind_value(const JsonValue& value,
                                          SampleRecordKind expected) noexcept {
    std::string_view text;
    const SampleStreamError error = take_string_value(value, text);
    if (error != SampleStreamError::None) {
        return error;
    }
    SampleRecordKind kind{};
    if (!parse_enum_spelling(kSampleRecordKindSpellings, text, kind)) {
        return SampleStreamError::InvalidEnum;
    }
    // A valid kind in the wrong decoder is a structural fault, not enum drift.
    return kind == expected ? SampleStreamError::None : SampleStreamError::InvalidFrame;
}

SampleStreamError check_identifier_value(const JsonValue& value,
                                         std::string_view& text) noexcept {
    const SampleStreamError error = take_string_value(value, text);
    if (error != SampleStreamError::None) {
        return error;
    }
    if (validate_identifier(text) != IdentifierError::None) {
        return SampleStreamError::InvalidIdentifier;
    }
    return SampleStreamError::None;
}

// --- Sample record decoders over tokens ------------------------------------

SampleStreamError header_from_tokens(const RecordTokens& tokens,
                                     SampleStreamHeader& header) {
    SampleStreamError error = analyze_record_keys(tokens, kSampleHeaderKeys);
    if (error != SampleStreamError::None) {
        return error;
    }

    const JsonValue& reasons_value = tokens.pairs[0].value;
    if (reasons_value.kind != JsonValueKind::StringArray) {
        return SampleStreamError::WrongType;
    }
    header.allowed_rejection_reasons.clear();
    header.allowed_rejection_reasons.reserve(reasons_value.element_count);
    for (std::size_t index = 0; index < reasons_value.element_count; ++index) {
        SampleRejectionReason reason{};
        if (!parse_enum_spelling(kSampleRejectionReasonSpellings,
                                 reasons_value.elements[index], reason) ||
            reason == SampleRejectionReason::NotApplicable) {
            return SampleStreamError::InvalidEnum;
        }
        if (!header.allowed_rejection_reasons.empty() &&
            reason <= header.allowed_rejection_reasons.back()) {
            return SampleStreamError::NonCanonicalOrder;
        }
        header.allowed_rejection_reasons.push_back(reason);
    }

    std::string_view arm_text;
    error = take_string_value(tokens.pairs[1].value, arm_text);
    if (error != SampleStreamError::None) {
        return error;
    }
    if (!parse_enum_spelling(kSampleArmSpellings, arm_text, header.arm)) {
        return SampleStreamError::InvalidEnum;
    }

    error = parse_bounded_count_value(tokens.pairs[2].value, 0, kMaximumSampleDecimalScale,
                                      SampleStreamError::NumericRange,
                                      header.decimal_scale);
    if (error != SampleStreamError::None) {
        return error;
    }

    std::string_view direction_text;
    error = take_string_value(tokens.pairs[3].value, direction_text);
    if (error != SampleStreamError::None) {
        return error;
    }
    if (!parse_enum_spelling(kSampleDirectionSpellings, direction_text, header.direction)) {
        return SampleStreamError::InvalidEnum;
    }

    error = parse_bounded_count_value(tokens.pairs[4].value, 1, kMaximumSampleRecords,
                                      SampleStreamError::InvalidCount,
                                      header.maximum_samples);
    if (error != SampleStreamError::None) {
        return error;
    }

    std::string_view metric_name;
    error = check_identifier_value(tokens.pairs[5].value, metric_name);
    if (error != SampleStreamError::None) {
        return error;
    }
    header.metric_name.assign(metric_name);

    error = parse_bounded_count_value(tokens.pairs[6].value, 1, header.maximum_samples,
                                      SampleStreamError::InvalidCount,
                                      header.planned_observation_count);
    if (error != SampleStreamError::None) {
        return error;
    }

    error = check_record_kind_value(tokens.pairs[7].value, SampleRecordKind::Header);
    if (error != SampleStreamError::None) {
        return error;
    }

    std::string_view schema_name;
    error = take_string_value(tokens.pairs[8].value, schema_name);
    if (error != SampleStreamError::None) {
        return error;
    }
    if (schema_name != kSampleStreamSchemaName) {
        return SampleStreamError::SchemaUnsupported;
    }

    std::uint64_t schema_version = 0;
    error = parse_unsigned_value(tokens.pairs[9].value, schema_version);
    if (error != SampleStreamError::None) {
        return error;
    }
    if (schema_version != kSampleStreamSchemaVersion) {
        return SampleStreamError::SchemaUnsupported;
    }

    std::string_view unit;
    error = check_identifier_value(tokens.pairs[10].value, unit);
    if (error != SampleStreamError::None) {
        return error;
    }
    header.unit.assign(unit);
    return SampleStreamError::None;
}

SampleStreamError sample_from_tokens(const RecordTokens& tokens,
                                     SampleRecord& record) noexcept {
    SampleStreamError error = analyze_record_keys(tokens, kSampleRecordKeys);
    if (error != SampleStreamError::None) {
        return error;
    }

    std::string_view arm_text;
    error = take_string_value(tokens.pairs[0].value, arm_text);
    if (error != SampleStreamError::None) {
        return error;
    }
    if (!parse_enum_spelling(kSampleArmSpellings, arm_text, record.arm)) {
        return SampleStreamError::InvalidEnum;
    }

    std::string_view disposition_text;
    error = take_string_value(tokens.pairs[1].value, disposition_text);
    if (error != SampleStreamError::None) {
        return error;
    }
    if (!parse_enum_spelling(kSampleDispositionSpellings, disposition_text,
                             record.disposition)) {
        return SampleStreamError::InvalidEnum;
    }

    error = check_record_kind_value(tokens.pairs[2].value, SampleRecordKind::Sample);
    if (error != SampleStreamError::None) {
        return error;
    }

    std::string_view reason_text;
    error = take_string_value(tokens.pairs[3].value, reason_text);
    if (error != SampleStreamError::None) {
        return error;
    }
    if (!parse_enum_spelling(kSampleRejectionReasonSpellings, reason_text,
                             record.rejection_reason)) {
        return SampleStreamError::InvalidEnum;
    }

    error = parse_bounded_count_value(tokens.pairs[4].value, 0, kMaximumSampleRecords - 1,
                                      SampleStreamError::InvalidCount,
                                      record.sample_index);
    if (error != SampleStreamError::None) {
        return error;
    }

    error = parse_signed_value(tokens.pairs[5].value, record.value_scaled);
    if (error != SampleStreamError::None) {
        return error;
    }

    // A rejected sample carries a concrete reason; every other disposition
    // carries the NOT_APPLICABLE sentinel.
    const bool rejected = record.disposition == SampleDisposition::Rejected;
    const bool has_reason = record.rejection_reason != SampleRejectionReason::NotApplicable;
    if (rejected != has_reason) {
        return SampleStreamError::ConservationMismatch;
    }
    return SampleStreamError::None;
}

SampleStreamError terminal_from_tokens(const RecordTokens& tokens,
                                       SampleStreamTerminal& terminal) noexcept {
    SampleStreamError error = analyze_record_keys(tokens, kSampleTerminalKeys);
    if (error != SampleStreamError::None) {
        return error;
    }

    error = take_boolean_value(tokens.pairs[0].value, terminal.complete);
    if (error != SampleStreamError::None) {
        return error;
    }
    error = parse_bounded_count_value(tokens.pairs[1].value, 0, kMaximumSampleRecords,
                                      SampleStreamError::InvalidCount,
                                      terminal.observed_count);
    if (error != SampleStreamError::None) {
        return error;
    }
    error = check_record_kind_value(tokens.pairs[2].value, SampleRecordKind::Terminal);
    if (error != SampleStreamError::None) {
        return error;
    }
    error = parse_bounded_count_value(tokens.pairs[3].value, 0, kMaximumSampleRecords,
                                      SampleStreamError::InvalidCount,
                                      terminal.rejected_count);
    if (error != SampleStreamError::None) {
        return error;
    }
    error = parse_bounded_count_value(tokens.pairs[4].value, 0, kMaximumSampleRecords,
                                      SampleStreamError::InvalidCount,
                                      terminal.retained_count);
    if (error != SampleStreamError::None) {
        return error;
    }
    error = take_boolean_value(tokens.pairs[5].value, terminal.truncated);
    if (error != SampleStreamError::None) {
        return error;
    }
    error = parse_bounded_count_value(tokens.pairs[6].value, 0, kMaximumSampleRecords,
                                      SampleStreamError::InvalidCount,
                                      terminal.warmup_count);
    if (error != SampleStreamError::None) {
        return error;
    }

    const std::uint64_t disposition_total =
        static_cast<std::uint64_t>(terminal.warmup_count) +
        static_cast<std::uint64_t>(terminal.retained_count) +
        static_cast<std::uint64_t>(terminal.rejected_count);
    if (disposition_total != terminal.observed_count) {
        return SampleStreamError::ConservationMismatch;
    }
    if (terminal.complete && terminal.truncated) {
        return SampleStreamError::ConservationMismatch;
    }
    return SampleStreamError::None;
}

SampleStreamError record_kind_of(const RecordTokens& tokens,
                                 SampleRecordKind& kind) noexcept {
    for (std::size_t index = 0; index < tokens.pair_count; ++index) {
        if (tokens.pairs[index].key != "record_kind") {
            continue;
        }
        std::string_view text;
        const SampleStreamError error = take_string_value(tokens.pairs[index].value, text);
        if (error != SampleStreamError::None) {
            return error;
        }
        if (!parse_enum_spelling(kSampleRecordKindSpellings, text, kind)) {
            return SampleStreamError::InvalidEnum;
        }
        return SampleStreamError::None;
    }
    if (has_duplicate_key(tokens)) {
        return SampleStreamError::DuplicateKey;
    }
    return tokens.overflowed ? SampleStreamError::UnknownKey
                             : SampleStreamError::MissingField;
}

// --- Sample record emitters -------------------------------------------------

// Counts when `out` is null so artifact bounds are known before any write;
// writes the identical bytes when `out` is set.
class LineSink {
  public:
    explicit LineSink(std::string* out) noexcept : out_(out) {}

    void append(std::string_view text) {
        length_ += text.size();
        if (out_ != nullptr) {
            out_->append(text);
        }
    }

    void append_unsigned(std::uint64_t value) {
        std::array<char, 20> digits{};
        const auto converted =
            std::to_chars(digits.data(), digits.data() + digits.size(), value);
        append({digits.data(), static_cast<std::size_t>(converted.ptr - digits.data())});
    }

    void append_signed(std::int64_t value) {
        std::array<char, 20> digits{};
        const auto converted =
            std::to_chars(digits.data(), digits.data() + digits.size(), value);
        append({digits.data(), static_cast<std::size_t>(converted.ptr - digits.data())});
    }

    void append_boolean(bool value) { append(value ? "true" : "false"); }

    std::size_t length() const noexcept { return length_; }

  private:
    std::string* out_;
    std::size_t length_{0};
};

void append_sample_stream_header(LineSink& sink, const SampleStreamHeader& header) {
    sink.append("{\"allowed_rejection_reasons\":[");
    for (std::size_t index = 0; index < header.allowed_rejection_reasons.size(); ++index) {
        if (index > 0) {
            sink.append(",");
        }
        sink.append("\"");
        sink.append(kSampleRejectionReasonSpellings[static_cast<std::size_t>(
            header.allowed_rejection_reasons[index])]);
        sink.append("\"");
    }
    sink.append("],\"arm\":\"");
    sink.append(kSampleArmSpellings[static_cast<std::size_t>(header.arm)]);
    sink.append("\",\"decimal_scale\":");
    sink.append_unsigned(header.decimal_scale);
    sink.append(",\"direction\":\"");
    sink.append(kSampleDirectionSpellings[static_cast<std::size_t>(header.direction)]);
    sink.append("\",\"maximum_samples\":");
    sink.append_unsigned(header.maximum_samples);
    sink.append(",\"metric_name\":\"");
    sink.append(header.metric_name);
    sink.append("\",\"planned_observation_count\":");
    sink.append_unsigned(header.planned_observation_count);
    sink.append(",\"record_kind\":\"HEADER\",\"schema_name\":\"tatara.sample_stream\","
                "\"schema_version\":1,\"unit\":\"");
    sink.append(header.unit);
    sink.append("\"}");
}

void append_sample_record(LineSink& sink, const SampleRecord& record) {
    sink.append("{\"arm\":\"");
    sink.append(kSampleArmSpellings[static_cast<std::size_t>(record.arm)]);
    sink.append("\",\"disposition\":\"");
    sink.append(kSampleDispositionSpellings[static_cast<std::size_t>(record.disposition)]);
    sink.append("\",\"record_kind\":\"SAMPLE\",\"rejection_reason\":\"");
    sink.append(kSampleRejectionReasonSpellings[static_cast<std::size_t>(
        record.rejection_reason)]);
    sink.append("\",\"sample_index\":");
    sink.append_unsigned(record.sample_index);
    sink.append(",\"value_scaled\":");
    sink.append_signed(record.value_scaled);
    sink.append("}");
}

void append_sample_stream_terminal(LineSink& sink, const SampleStreamTerminal& terminal) {
    sink.append("{\"complete\":");
    sink.append_boolean(terminal.complete);
    sink.append(",\"observed_count\":");
    sink.append_unsigned(terminal.observed_count);
    sink.append(",\"record_kind\":\"TERMINAL\",\"rejected_count\":");
    sink.append_unsigned(terminal.rejected_count);
    sink.append(",\"retained_count\":");
    sink.append_unsigned(terminal.retained_count);
    sink.append(",\"truncated\":");
    sink.append_boolean(terminal.truncated);
    sink.append(",\"warmup_count\":");
    sink.append_unsigned(terminal.warmup_count);
    sink.append("}");
}

SampleStreamError validate_sample_stream_header(const SampleStreamHeader& header) noexcept {
    for (std::size_t index = 0; index < header.allowed_rejection_reasons.size(); ++index) {
        const SampleRejectionReason reason = header.allowed_rejection_reasons[index];
        if (!valid_enum_value(kSampleRejectionReasonSpellings, reason) ||
            reason == SampleRejectionReason::NotApplicable) {
            return SampleStreamError::InvalidEnum;
        }
        if (index > 0 && reason <= header.allowed_rejection_reasons[index - 1]) {
            return SampleStreamError::NonCanonicalOrder;
        }
    }
    if (!valid_enum_value(kSampleArmSpellings, header.arm) ||
        !valid_enum_value(kSampleDirectionSpellings, header.direction)) {
        return SampleStreamError::InvalidEnum;
    }
    if (header.decimal_scale > kMaximumSampleDecimalScale) {
        return SampleStreamError::NumericRange;
    }
    if (header.maximum_samples < 1 || header.maximum_samples > kMaximumSampleRecords) {
        return SampleStreamError::InvalidCount;
    }
    if (header.planned_observation_count < 1 ||
        header.planned_observation_count > header.maximum_samples) {
        return SampleStreamError::InvalidCount;
    }
    if (validate_identifier(header.metric_name) != IdentifierError::None ||
        validate_identifier(header.unit) != IdentifierError::None) {
        return SampleStreamError::InvalidIdentifier;
    }
    return SampleStreamError::None;
}

SampleStreamError validate_sample_record(const SampleRecord& record) noexcept {
    if (!valid_enum_value(kSampleArmSpellings, record.arm) ||
        !valid_enum_value(kSampleDispositionSpellings, record.disposition) ||
        !valid_enum_value(kSampleRejectionReasonSpellings, record.rejection_reason)) {
        return SampleStreamError::InvalidEnum;
    }
    if (record.sample_index >= kMaximumSampleRecords) {
        return SampleStreamError::InvalidCount;
    }
    const bool rejected = record.disposition == SampleDisposition::Rejected;
    const bool has_reason = record.rejection_reason != SampleRejectionReason::NotApplicable;
    if (rejected != has_reason) {
        return SampleStreamError::ConservationMismatch;
    }
    return SampleStreamError::None;
}

SampleStreamError
validate_sample_stream_terminal(const SampleStreamTerminal& terminal) noexcept {
    if (terminal.observed_count > kMaximumSampleRecords ||
        terminal.warmup_count > kMaximumSampleRecords ||
        terminal.retained_count > kMaximumSampleRecords ||
        terminal.rejected_count > kMaximumSampleRecords) {
        return SampleStreamError::InvalidCount;
    }
    const std::uint64_t disposition_total =
        static_cast<std::uint64_t>(terminal.warmup_count) +
        static_cast<std::uint64_t>(terminal.retained_count) +
        static_cast<std::uint64_t>(terminal.rejected_count);
    if (disposition_total != terminal.observed_count) {
        return SampleStreamError::ConservationMismatch;
    }
    if (terminal.complete && terminal.truncated) {
        return SampleStreamError::ConservationMismatch;
    }
    return SampleStreamError::None;
}

bool rejection_reason_allowed(const SampleStreamHeader& header,
                              SampleRejectionReason reason) noexcept {
    return std::find(header.allowed_rejection_reasons.begin(),
                     header.allowed_rejection_reasons.end(),
                     reason) != header.allowed_rejection_reasons.end();
}

} // namespace

// --- Raw log public codecs -------------------------------------------------

std::array<std::byte, kRawLogFileHeaderBytes> encode_raw_log_file_header() noexcept {
    std::array<std::byte, kRawLogFileHeaderBytes> header{};
    std::copy(kRawLogMagic.begin(), kRawLogMagic.end(), header.begin());
    header[8] = static_cast<std::byte>(kRawLogFormatVersion >> 8);
    header[9] = static_cast<std::byte>(kRawLogFormatVersion & 0xffU);
    header[10] = static_cast<std::byte>(kRawLogFileHeaderBytes >> 8);
    header[11] = static_cast<std::byte>(kRawLogFileHeaderBytes & 0xffU);
    // Bytes 12 through 15 are the required all-zero flags.
    return header;
}

RawLogError decode_raw_log_file_header(std::span<const std::byte> bytes) noexcept {
    if (bytes.size() < kRawLogFileHeaderBytes) {
        return RawLogError::InvalidFrame;
    }
    if (bytes.size() > kRawLogFileHeaderBytes) {
        return RawLogError::TrailingData;
    }
    if (!std::equal(kRawLogMagic.begin(), kRawLogMagic.end(), bytes.begin())) {
        return RawLogError::InvalidFrame;
    }
    if (read_u16_be(bytes, 8) != kRawLogFormatVersion) {
        return RawLogError::SchemaUnsupported;
    }
    if (read_u16_be(bytes, 10) != kRawLogFileHeaderBytes) {
        return RawLogError::InvalidFrame;
    }
    if (read_u32_be(bytes, 12) != 0) {
        return RawLogError::InvalidFrame;
    }
    return RawLogError::None;
}

RawLogEncodeResult encode_raw_log_data_frame(const RawLogDataFrame& frame) {
    RawLogEncodeResult result;
    result.error = validate_raw_log_data_frame(frame);
    if (result.error != RawLogError::None) {
        return result;
    }
    result.bytes.reserve(kRawLogFrameHeaderBytes + frame.payload.size());
    append_raw_log_data_frame(result.bytes, frame);
    return result;
}

RawLogDataFrameDecodeResult
decode_raw_log_data_frame(std::span<const std::byte> bytes) noexcept {
    RawLogDataFrameDecodeResult result;
    if (bytes.size() < kRawLogFrameHeaderBytes) {
        result.error = RawLogError::InvalidFrame;
        return result;
    }
    result.error = check_raw_log_record_type(bytes[0], RawLogRecordType::Data);
    if (result.error != RawLogError::None) {
        return result;
    }
    const auto stream_value = std::to_integer<std::uint8_t>(bytes[1]);
    if (stream_value < static_cast<std::uint8_t>(RawLogStreamId::Stdout) ||
        stream_value > static_cast<std::uint8_t>(RawLogStreamId::Harness)) {
        result.error = RawLogError::InvalidEnum;
        return result;
    }
    if (read_u16_be(bytes, 2) != 0) {
        result.error = RawLogError::InvalidFrame;
        return result;
    }
    const std::uint32_t payload_length = read_u32_be(bytes, 12);
    if (payload_length < kMinimumRawLogDataPayloadBytes ||
        payload_length > kMaximumRawLogDataPayloadBytes) {
        result.error = RawLogError::NumericRange;
        return result;
    }
    const std::size_t frame_bytes = kRawLogFrameHeaderBytes + payload_length;
    if (bytes.size() < frame_bytes) {
        result.error = RawLogError::InvalidFrame;
        return result;
    }
    if (bytes.size() > frame_bytes) {
        result.error = RawLogError::TrailingData;
        return result;
    }
    result.frame.stream_id = static_cast<RawLogStreamId>(stream_value);
    result.frame.monotonic_offset_ns = read_u64_be(bytes, 4);
    result.frame.payload = bytes.subspan(kRawLogFrameHeaderBytes, payload_length);
    return result;
}

RawLogEncodeResult encode_raw_log_terminal_frame(const RawLogTerminalFrame& terminal) {
    RawLogEncodeResult result;
    result.error = validate_raw_log_terminal_frame(terminal);
    if (result.error != RawLogError::None) {
        return result;
    }
    result.bytes.reserve(kRawLogTerminalFrameBytes);
    append_raw_log_terminal_frame(result.bytes, terminal);
    return result;
}

RawLogTerminalFrameDecodeResult
decode_raw_log_terminal_frame(std::span<const std::byte> bytes) noexcept {
    RawLogTerminalFrameDecodeResult result;
    if (bytes.size() < kRawLogTerminalFrameBytes) {
        result.error = RawLogError::InvalidFrame;
        return result;
    }
    result.error = check_raw_log_record_type(bytes[0], RawLogRecordType::Terminal);
    if (result.error != RawLogError::None) {
        return result;
    }
    if (bytes.size() > kRawLogTerminalFrameBytes) {
        result.error = RawLogError::TrailingData;
        return result;
    }
    if (std::to_integer<std::uint8_t>(bytes[1]) != 0) {
        result.error = RawLogError::InvalidFrame;
        return result;
    }
    const std::uint16_t flags = read_u16_be(bytes, 2);
    // Only bit 0 (TRUNCATED) is defined for a terminal frame.
    if ((flags >> 1) != 0) {
        result.error = RawLogError::InvalidFrame;
        return result;
    }
    if (read_u32_be(bytes, 12) != kRawLogTerminalPayloadBytes) {
        result.error = RawLogError::InvalidFrame;
        return result;
    }
    result.terminal.monotonic_offset_ns = read_u64_be(bytes, 4);
    result.terminal.truncated = (flags & kRawLogTerminalTruncatedFlag) != 0;
    result.terminal.data_frame_count = read_u64_be(bytes, 16);
    result.terminal.captured_payload_bytes = read_u64_be(bytes, 24);
    result.terminal.discarded_payload_bytes = read_u64_be(bytes, 32);
    result.error = validate_raw_log_terminal_frame(result.terminal);
    if (result.error != RawLogError::None) {
        result.terminal = {};
    }
    return result;
}

RawLogEncodeResult encode_raw_log(std::span<const RawLogDataFrame> frames,
                                  const RawLogTerminalFrame& terminal,
                                  std::uint64_t maximum_artifact_bytes) {
    RawLogEncodeResult result;
    if (frames.size() > kMaximumRawLogDataFrames) {
        result.error = RawLogError::InvalidCount;
        return result;
    }
    std::uint64_t captured_payload_bytes = 0;
    std::uint64_t last_offset_ns = 0;
    for (const RawLogDataFrame& frame : frames) {
        result.error = validate_raw_log_data_frame(frame);
        if (result.error != RawLogError::None) {
            return result;
        }
        if (frame.monotonic_offset_ns < last_offset_ns) {
            result.error = RawLogError::InvalidFrame;
            return result;
        }
        last_offset_ns = frame.monotonic_offset_ns;
        captured_payload_bytes += frame.payload.size();
    }
    result.error = validate_raw_log_terminal_frame(terminal);
    if (result.error != RawLogError::None) {
        return result;
    }
    if (terminal.monotonic_offset_ns < last_offset_ns) {
        result.error = RawLogError::InvalidFrame;
        return result;
    }
    if (terminal.data_frame_count != frames.size() ||
        terminal.captured_payload_bytes != captured_payload_bytes) {
        result.error = RawLogError::ConservationMismatch;
        return result;
    }
    // Bounded above by 16 + 2^20 * 16 + 2^36 + 40 < 2^37, so plain 64-bit
    // arithmetic cannot overflow after the count and payload checks.
    const std::uint64_t total_bytes =
        static_cast<std::uint64_t>(kRawLogFileHeaderBytes) +
        static_cast<std::uint64_t>(frames.size()) * kRawLogFrameHeaderBytes +
        captured_payload_bytes + static_cast<std::uint64_t>(kRawLogTerminalFrameBytes);
    if (total_bytes > effective_budget(maximum_artifact_bytes, kMaximumRawLogArtifactBytes)) {
        result.error = RawLogError::ArtifactTooLarge;
        return result;
    }
    result.bytes.reserve(static_cast<std::size_t>(total_bytes));
    const auto file_header = encode_raw_log_file_header();
    result.bytes.insert(result.bytes.end(), file_header.begin(), file_header.end());
    for (const RawLogDataFrame& frame : frames) {
        append_raw_log_data_frame(result.bytes, frame);
    }
    append_raw_log_terminal_frame(result.bytes, terminal);
    return result;
}

RawLogDecodeResult decode_raw_log(std::span<const std::byte> bytes,
                                  std::uint64_t maximum_artifact_bytes) {
    RawLogDecodeResult result;
    const auto fail = [&result](RawLogError error) {
        RawLogDecodeResult failure;
        failure.error = error;
        result = std::move(failure);
    };
    if (bytes.size() > effective_budget(maximum_artifact_bytes, kMaximumRawLogArtifactBytes)) {
        fail(RawLogError::ArtifactTooLarge);
        return result;
    }
    if (bytes.size() < kMinimumRawLogArtifactBytes) {
        fail(RawLogError::InvalidFrame);
        return result;
    }
    result.error = decode_raw_log_file_header(bytes.first(kRawLogFileHeaderBytes));
    if (result.error != RawLogError::None) {
        fail(result.error);
        return result;
    }

    std::size_t cursor = kRawLogFileHeaderBytes;
    std::uint64_t last_offset_ns = 0;
    std::uint64_t captured_payload_bytes = 0;
    while (true) {
        const std::size_t remaining = bytes.size() - cursor;
        if (remaining < kRawLogFrameHeaderBytes) {
            fail(RawLogError::InvalidFrame);
            return result;
        }
        const auto record_type = std::to_integer<std::uint8_t>(bytes[cursor]);
        if (record_type == static_cast<std::uint8_t>(RawLogRecordType::Data)) {
            const std::uint32_t payload_length = read_u32_be(bytes, cursor + 12);
            if (payload_length < kMinimumRawLogDataPayloadBytes ||
                payload_length > kMaximumRawLogDataPayloadBytes) {
                fail(RawLogError::NumericRange);
                return result;
            }
            const std::size_t frame_bytes = kRawLogFrameHeaderBytes + payload_length;
            if (remaining < frame_bytes) {
                fail(RawLogError::InvalidFrame);
                return result;
            }
            const auto decoded = decode_raw_log_data_frame(bytes.subspan(cursor, frame_bytes));
            if (!decoded) {
                fail(decoded.error);
                return result;
            }
            if (decoded.frame.monotonic_offset_ns < last_offset_ns) {
                fail(RawLogError::InvalidFrame);
                return result;
            }
            if (result.frames.size() == kMaximumRawLogDataFrames) {
                fail(RawLogError::InvalidCount);
                return result;
            }
            last_offset_ns = decoded.frame.monotonic_offset_ns;
            captured_payload_bytes += payload_length;
            result.frames.push_back(decoded.frame);
            cursor += frame_bytes;
            continue;
        }
        if (record_type != static_cast<std::uint8_t>(RawLogRecordType::Terminal)) {
            fail(RawLogError::InvalidEnum);
            return result;
        }
        if (remaining < kRawLogTerminalFrameBytes) {
            fail(RawLogError::InvalidFrame);
            return result;
        }
        const auto terminal =
            decode_raw_log_terminal_frame(bytes.subspan(cursor, kRawLogTerminalFrameBytes));
        if (!terminal) {
            fail(terminal.error);
            return result;
        }
        if (terminal.terminal.monotonic_offset_ns < last_offset_ns) {
            fail(RawLogError::InvalidFrame);
            return result;
        }
        if (terminal.terminal.data_frame_count != result.frames.size() ||
            terminal.terminal.captured_payload_bytes != captured_payload_bytes) {
            fail(RawLogError::ConservationMismatch);
            return result;
        }
        cursor += kRawLogTerminalFrameBytes;
        if (cursor != bytes.size()) {
            fail(RawLogError::TrailingData);
            return result;
        }
        result.terminal = terminal.terminal;
        return result;
    }
}

// --- Sample stream public codecs -------------------------------------------

SampleStreamEncodeResult encode_sample_stream_header(const SampleStreamHeader& header) {
    SampleStreamEncodeResult result;
    result.error = validate_sample_stream_header(header);
    if (result.error != SampleStreamError::None) {
        return result;
    }
    LineSink counter(nullptr);
    append_sample_stream_header(counter, header);
    result.bytes.reserve(counter.length());
    LineSink writer(&result.bytes);
    append_sample_stream_header(writer, header);
    return result;
}

SampleStreamHeaderDecodeResult decode_sample_stream_header(std::string_view line) {
    SampleStreamHeaderDecodeResult result;
    RecordTokens tokens;
    result.error = tokenize_record(line, tokens);
    if (result.error != SampleStreamError::None) {
        return result;
    }
    result.error = header_from_tokens(tokens, result.header);
    if (result.error != SampleStreamError::None) {
        result.header = {};
    }
    return result;
}

SampleStreamEncodeResult encode_sample_record(const SampleRecord& record) {
    SampleStreamEncodeResult result;
    result.error = validate_sample_record(record);
    if (result.error != SampleStreamError::None) {
        return result;
    }
    LineSink counter(nullptr);
    append_sample_record(counter, record);
    result.bytes.reserve(counter.length());
    LineSink writer(&result.bytes);
    append_sample_record(writer, record);
    return result;
}

SampleRecordDecodeResult decode_sample_record(std::string_view line) noexcept {
    SampleRecordDecodeResult result;
    RecordTokens tokens;
    result.error = tokenize_record(line, tokens);
    if (result.error != SampleStreamError::None) {
        return result;
    }
    result.error = sample_from_tokens(tokens, result.record);
    if (result.error != SampleStreamError::None) {
        result.record = {};
    }
    return result;
}

SampleStreamEncodeResult
encode_sample_stream_terminal(const SampleStreamTerminal& terminal) {
    SampleStreamEncodeResult result;
    result.error = validate_sample_stream_terminal(terminal);
    if (result.error != SampleStreamError::None) {
        return result;
    }
    LineSink counter(nullptr);
    append_sample_stream_terminal(counter, terminal);
    result.bytes.reserve(counter.length());
    LineSink writer(&result.bytes);
    append_sample_stream_terminal(writer, terminal);
    return result;
}

SampleStreamTerminalDecodeResult
decode_sample_stream_terminal(std::string_view line) noexcept {
    SampleStreamTerminalDecodeResult result;
    RecordTokens tokens;
    result.error = tokenize_record(line, tokens);
    if (result.error != SampleStreamError::None) {
        return result;
    }
    result.error = terminal_from_tokens(tokens, result.terminal);
    if (result.error != SampleStreamError::None) {
        result.terminal = {};
    }
    return result;
}

SampleStreamEncodeResult encode_sample_stream(const SampleStreamHeader& header,
                                              std::span<const SampleRecord> samples,
                                              const SampleStreamTerminal& terminal,
                                              std::uint64_t maximum_artifact_bytes) {
    SampleStreamEncodeResult result;
    result.error = validate_sample_stream_header(header);
    if (result.error != SampleStreamError::None) {
        return result;
    }
    if (samples.size() > kMaximumSampleRecords ||
        samples.size() > header.maximum_samples) {
        result.error = SampleStreamError::InvalidCount;
        return result;
    }
    std::uint32_t warmup_count = 0;
    std::uint32_t retained_count = 0;
    std::uint32_t rejected_count = 0;
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const SampleRecord& record = samples[index];
        result.error = validate_sample_record(record);
        if (result.error != SampleStreamError::None) {
            return result;
        }
        if (record.sample_index != index) {
            result.error = SampleStreamError::InvalidCount;
            return result;
        }
        if (record.arm != header.arm) {
            result.error = SampleStreamError::ConservationMismatch;
            return result;
        }
        if (record.disposition == SampleDisposition::Rejected &&
            !rejection_reason_allowed(header, record.rejection_reason)) {
            result.error = SampleStreamError::InvalidEnum;
            return result;
        }
        switch (record.disposition) {
        case SampleDisposition::Warmup:
            ++warmup_count;
            break;
        case SampleDisposition::Retained:
            ++retained_count;
            break;
        case SampleDisposition::Rejected:
            ++rejected_count;
            break;
        }
    }
    result.error = validate_sample_stream_terminal(terminal);
    if (result.error != SampleStreamError::None) {
        return result;
    }
    if (terminal.observed_count != samples.size() ||
        terminal.warmup_count != warmup_count ||
        terminal.retained_count != retained_count ||
        terminal.rejected_count != rejected_count) {
        result.error = SampleStreamError::ConservationMismatch;
        return result;
    }
    if (terminal.complete &&
        terminal.observed_count != header.planned_observation_count) {
        result.error = SampleStreamError::ConservationMismatch;
        return result;
    }

    // Count pass; every line stays far below the 4,096-byte bound because all
    // fields are identifier- or enum-bounded, so only the artifact budget can
    // be exceeded.
    LineSink counter(nullptr);
    append_sample_stream_header(counter, header);
    counter.append("\n");
    for (const SampleRecord& record : samples) {
        append_sample_record(counter, record);
        counter.append("\n");
    }
    append_sample_stream_terminal(counter, terminal);
    counter.append("\n");
    if (counter.length() >
        effective_budget(maximum_artifact_bytes, kMaximumSampleStreamArtifactBytes)) {
        result.error = SampleStreamError::ArtifactTooLarge;
        return result;
    }

    result.bytes.reserve(counter.length());
    LineSink writer(&result.bytes);
    append_sample_stream_header(writer, header);
    writer.append("\n");
    for (const SampleRecord& record : samples) {
        append_sample_record(writer, record);
        writer.append("\n");
    }
    append_sample_stream_terminal(writer, terminal);
    writer.append("\n");
    return result;
}

SampleStreamDecodeResult decode_sample_stream(std::string_view bytes,
                                              std::uint64_t maximum_artifact_bytes) {
    SampleStreamDecodeResult result;
    const auto fail = [&result](SampleStreamError error) {
        SampleStreamDecodeResult failure;
        failure.error = error;
        result = std::move(failure);
    };
    if (bytes.size() >
        effective_budget(maximum_artifact_bytes, kMaximumSampleStreamArtifactBytes)) {
        fail(SampleStreamError::ArtifactTooLarge);
        return result;
    }
    if (bytes.empty()) {
        fail(SampleStreamError::InvalidFrame);
        return result;
    }
    if (bytes.back() != '\n') {
        fail(SampleStreamError::NonCanonicalBytes);
        return result;
    }

    bool have_header = false;
    bool have_terminal = false;
    std::uint32_t warmup_count = 0;
    std::uint32_t retained_count = 0;
    std::uint32_t rejected_count = 0;
    std::size_t position = 0;
    while (position < bytes.size()) {
        const std::size_t line_feed = bytes.find('\n', position);
        const std::string_view line = bytes.substr(position, line_feed - position);
        position = line_feed + 1;
        if (have_terminal) {
            fail(SampleStreamError::TrailingData);
            return result;
        }
        if (line.size() > kMaximumSampleStreamLineBytes) {
            fail(SampleStreamError::ArtifactTooLarge);
            return result;
        }
        RecordTokens tokens;
        result.error = tokenize_record(line, tokens);
        if (result.error != SampleStreamError::None) {
            fail(result.error);
            return result;
        }
        SampleRecordKind kind{};
        result.error = record_kind_of(tokens, kind);
        if (result.error != SampleStreamError::None) {
            fail(result.error);
            return result;
        }
        if (!have_header) {
            if (kind != SampleRecordKind::Header) {
                fail(SampleStreamError::InvalidFrame);
                return result;
            }
            result.error = header_from_tokens(tokens, result.header);
            if (result.error != SampleStreamError::None) {
                fail(result.error);
                return result;
            }
            have_header = true;
            continue;
        }
        if (kind == SampleRecordKind::Header) {
            fail(SampleStreamError::InvalidFrame);
            return result;
        }
        if (kind == SampleRecordKind::Sample) {
            SampleRecord record;
            result.error = sample_from_tokens(tokens, record);
            if (result.error != SampleStreamError::None) {
                fail(result.error);
                return result;
            }
            if (record.sample_index != result.samples.size()) {
                fail(SampleStreamError::InvalidCount);
                return result;
            }
            if (result.samples.size() == result.header.maximum_samples) {
                fail(SampleStreamError::InvalidCount);
                return result;
            }
            if (record.arm != result.header.arm) {
                fail(SampleStreamError::ConservationMismatch);
                return result;
            }
            if (record.disposition == SampleDisposition::Rejected &&
                !rejection_reason_allowed(result.header, record.rejection_reason)) {
                fail(SampleStreamError::InvalidEnum);
                return result;
            }
            switch (record.disposition) {
            case SampleDisposition::Warmup:
                ++warmup_count;
                break;
            case SampleDisposition::Retained:
                ++retained_count;
                break;
            case SampleDisposition::Rejected:
                ++rejected_count;
                break;
            }
            result.samples.push_back(record);
            continue;
        }
        result.error = terminal_from_tokens(tokens, result.terminal);
        if (result.error != SampleStreamError::None) {
            fail(result.error);
            return result;
        }
        if (result.terminal.observed_count != result.samples.size() ||
            result.terminal.warmup_count != warmup_count ||
            result.terminal.retained_count != retained_count ||
            result.terminal.rejected_count != rejected_count) {
            fail(SampleStreamError::ConservationMismatch);
            return result;
        }
        if (result.terminal.complete &&
            result.terminal.observed_count != result.header.planned_observation_count) {
            fail(SampleStreamError::ConservationMismatch);
            return result;
        }
        have_terminal = true;
    }
    if (!have_terminal) {
        fail(SampleStreamError::InvalidFrame);
        return result;
    }
    return result;
}

} // namespace tatara::evidence
