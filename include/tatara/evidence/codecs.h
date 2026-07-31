#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// Schema-v1 evidence record codecs over the EVI-1 wire primitives.
//
// Two record-stream artifact formats are covered, exactly as frozen by
// V01_EVIDENCE_WIRE_SCHEMA.md:
//   - the framed binary raw log `tatara.raw-log` version 1 (section 7) with
//     record types file header, data frame, and terminal frame;
//   - the canonical JSON Lines sample stream `tatara.sample_stream` version 1
//     (section 8) with record kinds HEADER, SAMPLE, and TERMINAL.
//
// Every decode rejects truncated, overlong, invalid-enum, and non-canonical
// input with one typed error; every encode checks its bounds before writing a
// byte. Encode(decode(bytes)) reproduces the input bytes exactly.

namespace tatara::evidence {

// --- Framed raw log `tatara.raw-log` version 1 -----------------------------

inline constexpr std::size_t kRawLogFileHeaderBytes = 16;
inline constexpr std::size_t kRawLogFrameHeaderBytes = 16;
inline constexpr std::size_t kRawLogTerminalPayloadBytes = 24;
inline constexpr std::size_t kRawLogTerminalFrameBytes =
    kRawLogFrameHeaderBytes + kRawLogTerminalPayloadBytes;
inline constexpr std::size_t kMinimumRawLogArtifactBytes =
    kRawLogFileHeaderBytes + kRawLogTerminalFrameBytes;
inline constexpr std::uint16_t kRawLogFormatVersion = 1;
inline constexpr std::size_t kMinimumRawLogDataPayloadBytes = 1;
inline constexpr std::size_t kMaximumRawLogDataPayloadBytes = 65536;
inline constexpr std::size_t kMaximumRawLogDataFrames = 1048576;
inline constexpr std::uint64_t kMaximumRawLogArtifactBytes = 4294967296ULL;

enum class RawLogRecordType : std::uint8_t {
    Data = 1,
    Terminal = 2,
};

enum class RawLogStreamId : std::uint8_t {
    Stdout = 1,
    Stderr = 2,
    Harness = 3,
};

enum class RawLogError : std::uint8_t {
    None,
    ArtifactTooLarge,
    ConservationMismatch,
    InvalidCount,
    InvalidEnum,
    InvalidFrame,
    NumericRange,
    SchemaUnsupported,
    TrailingData,
};

struct RawLogDataFrame {
    RawLogStreamId stream_id{RawLogStreamId::Stdout};
    std::uint64_t monotonic_offset_ns{0};
    // Caller-owned bytes; decoded frames alias the input artifact bytes and
    // remain valid only while those bytes live.
    std::span<const std::byte> payload{};
};

struct RawLogTerminalFrame {
    std::uint64_t monotonic_offset_ns{0};
    bool truncated{false};
    std::uint64_t data_frame_count{0};
    std::uint64_t captured_payload_bytes{0};
    std::uint64_t discarded_payload_bytes{0};
};

struct RawLogEncodeResult {
    RawLogError error{RawLogError::None};
    std::vector<std::byte> bytes;

    explicit operator bool() const noexcept { return error == RawLogError::None; }
};

struct RawLogDataFrameDecodeResult {
    RawLogError error{RawLogError::None};
    RawLogDataFrame frame{};

    explicit operator bool() const noexcept { return error == RawLogError::None; }
};

struct RawLogTerminalFrameDecodeResult {
    RawLogError error{RawLogError::None};
    RawLogTerminalFrame terminal{};

    explicit operator bool() const noexcept { return error == RawLogError::None; }
};

struct RawLogDecodeResult {
    RawLogError error{RawLogError::None};
    std::vector<RawLogDataFrame> frames;
    RawLogTerminalFrame terminal{};

    explicit operator bool() const noexcept { return error == RawLogError::None; }
};

std::array<std::byte, kRawLogFileHeaderBytes> encode_raw_log_file_header() noexcept;
// Validates exactly one 16-byte file header record.
RawLogError decode_raw_log_file_header(std::span<const std::byte> bytes) noexcept;

RawLogEncodeResult encode_raw_log_data_frame(const RawLogDataFrame& frame);
// Validates exactly one data frame record; the decoded payload aliases `bytes`.
RawLogDataFrameDecodeResult
decode_raw_log_data_frame(std::span<const std::byte> bytes) noexcept;

RawLogEncodeResult encode_raw_log_terminal_frame(const RawLogTerminalFrame& terminal);
// Validates exactly one 40-byte terminal frame record.
RawLogTerminalFrameDecodeResult
decode_raw_log_terminal_frame(std::span<const std::byte> bytes) noexcept;

// Whole-artifact codec. The effective byte budget is
// min(maximum_artifact_bytes, kMaximumRawLogArtifactBytes) and is enforced
// before any write or parse. The terminal counters are caller evidence and are
// validated against the frames, never recomputed silently.
RawLogEncodeResult encode_raw_log(std::span<const RawLogDataFrame> frames,
                                  const RawLogTerminalFrame& terminal,
                                  std::uint64_t maximum_artifact_bytes);
RawLogDecodeResult decode_raw_log(std::span<const std::byte> bytes,
                                  std::uint64_t maximum_artifact_bytes);

// --- Sample stream `tatara.sample_stream` version 1 ------------------------

inline constexpr std::size_t kMaximumSampleStreamLineBytes = 4096;
inline constexpr std::size_t kMaximumSampleRecords = 16777216;
inline constexpr std::uint64_t kMaximumSampleStreamArtifactBytes = 4294967296ULL;
inline constexpr std::uint32_t kMaximumSampleDecimalScale = 18;

enum class SampleArm : std::uint8_t {
    Competitor,
    ControlAfter,
    ControlBefore,
    Descriptive,
    Treatment,
};

enum class SampleDirection : std::uint8_t {
    LowerIsBetter,
    HigherIsBetter,
    DescriptiveOnly,
};

enum class SampleDisposition : std::uint8_t {
    Warmup,
    Retained,
    Rejected,
};

// NotApplicable is the sample sentinel for warmup/retained records; it is not
// a member of the header's allowed rejection-reason set.
enum class SampleRejectionReason : std::uint8_t {
    NotApplicable,
    Cancelled,
    MeasurementUnavailable,
    MemoryPressurePolicy,
    OrderProtocolAbort,
    RequestError,
    ThermalPolicy,
    Timeout,
};

enum class SampleStreamError : std::uint8_t {
    None,
    ArtifactTooLarge,
    ConservationMismatch,
    DuplicateKey,
    InvalidCount,
    InvalidEnum,
    InvalidFrame,
    InvalidIdentifier,
    InvalidInteger,
    MissingField,
    NonCanonicalBytes,
    NonCanonicalOrder,
    NumericRange,
    SchemaUnsupported,
    TrailingData,
    UnknownKey,
    WrongType,
};

struct SampleStreamHeader {
    // Sorted unique, drawn from the seven concrete rejection reasons.
    std::vector<SampleRejectionReason> allowed_rejection_reasons;
    SampleArm arm{SampleArm::Descriptive};
    std::uint32_t decimal_scale{0};
    SampleDirection direction{SampleDirection::DescriptiveOnly};
    std::uint32_t maximum_samples{1};
    std::string metric_name;
    std::uint32_t planned_observation_count{1};
    std::string unit;
};

struct SampleRecord {
    SampleArm arm{SampleArm::Descriptive};
    SampleDisposition disposition{SampleDisposition::Retained};
    SampleRejectionReason rejection_reason{SampleRejectionReason::NotApplicable};
    std::uint32_t sample_index{0};
    std::int64_t value_scaled{0};
};

struct SampleStreamTerminal {
    bool complete{false};
    std::uint32_t observed_count{0};
    std::uint32_t rejected_count{0};
    std::uint32_t retained_count{0};
    bool truncated{false};
    std::uint32_t warmup_count{0};
};

struct SampleStreamEncodeResult {
    SampleStreamError error{SampleStreamError::None};
    std::string bytes;

    explicit operator bool() const noexcept {
        return error == SampleStreamError::None;
    }
};

struct SampleStreamHeaderDecodeResult {
    SampleStreamError error{SampleStreamError::None};
    SampleStreamHeader header{};

    explicit operator bool() const noexcept {
        return error == SampleStreamError::None;
    }
};

struct SampleRecordDecodeResult {
    SampleStreamError error{SampleStreamError::None};
    SampleRecord record{};

    explicit operator bool() const noexcept {
        return error == SampleStreamError::None;
    }
};

struct SampleStreamTerminalDecodeResult {
    SampleStreamError error{SampleStreamError::None};
    SampleStreamTerminal terminal{};

    explicit operator bool() const noexcept {
        return error == SampleStreamError::None;
    }
};

struct SampleStreamDecodeResult {
    SampleStreamError error{SampleStreamError::None};
    SampleStreamHeader header{};
    std::vector<SampleRecord> samples;
    SampleStreamTerminal terminal{};

    explicit operator bool() const noexcept {
        return error == SampleStreamError::None;
    }
};

// Record codecs operate on one canonical JSON object without the line feed;
// the artifact codec owns line framing. Record-local rules are enforced here;
// cross-record rules (arm homogeneity, index contiguity, declared rejection
// reasons, terminal conservation against observed records) belong to the
// artifact codec.
SampleStreamEncodeResult encode_sample_stream_header(const SampleStreamHeader& header);
SampleStreamHeaderDecodeResult decode_sample_stream_header(std::string_view line);

SampleStreamEncodeResult encode_sample_record(const SampleRecord& record);
SampleRecordDecodeResult decode_sample_record(std::string_view line) noexcept;

SampleStreamEncodeResult
encode_sample_stream_terminal(const SampleStreamTerminal& terminal);
SampleStreamTerminalDecodeResult
decode_sample_stream_terminal(std::string_view line) noexcept;

// Whole-artifact codec over canonical JSON Lines. The effective byte budget is
// min(maximum_artifact_bytes, kMaximumSampleStreamArtifactBytes); the encoder
// sizes the complete artifact before writing any byte.
SampleStreamEncodeResult encode_sample_stream(const SampleStreamHeader& header,
                                              std::span<const SampleRecord> samples,
                                              const SampleStreamTerminal& terminal,
                                              std::uint64_t maximum_artifact_bytes);
SampleStreamDecodeResult decode_sample_stream(std::string_view bytes,
                                              std::uint64_t maximum_artifact_bytes);

} // namespace tatara::evidence
