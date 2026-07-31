#include "tatara/evidence/codecs.h"

#include "tatara/model/sha256.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using tatara::evidence::RawLogDataFrame;
using tatara::evidence::RawLogError;
using tatara::evidence::RawLogStreamId;
using tatara::evidence::RawLogTerminalFrame;
using tatara::evidence::SampleArm;
using tatara::evidence::SampleDirection;
using tatara::evidence::SampleDisposition;
using tatara::evidence::SampleRecord;
using tatara::evidence::SampleRejectionReason;
using tatara::evidence::SampleStreamError;
using tatara::evidence::SampleStreamHeader;
using tatara::evidence::SampleStreamTerminal;
using tatara::evidence::decode_raw_log;
using tatara::evidence::decode_raw_log_data_frame;
using tatara::evidence::decode_raw_log_file_header;
using tatara::evidence::decode_raw_log_terminal_frame;
using tatara::evidence::decode_sample_record;
using tatara::evidence::decode_sample_stream;
using tatara::evidence::decode_sample_stream_header;
using tatara::evidence::decode_sample_stream_terminal;
using tatara::evidence::encode_raw_log;
using tatara::evidence::encode_raw_log_data_frame;
using tatara::evidence::encode_raw_log_file_header;
using tatara::evidence::encode_raw_log_terminal_frame;
using tatara::evidence::encode_sample_record;
using tatara::evidence::encode_sample_stream;
using tatara::evidence::encode_sample_stream_header;
using tatara::evidence::encode_sample_stream_terminal;
using tatara::evidence::kMaximumRawLogArtifactBytes;
using tatara::evidence::kMaximumRawLogDataFrames;
using tatara::evidence::kMaximumRawLogDataPayloadBytes;
using tatara::evidence::kMaximumSampleStreamArtifactBytes;
using tatara::evidence::kRawLogFileHeaderBytes;
using tatara::evidence::kRawLogTerminalFrameBytes;
using tatara::model::Sha256;
using tatara::model::sha256_hex;

// RL-001 golden 75-byte raw log from V01_EVIDENCE_WIRE_SCHEMA.md section 18.4.
constexpr std::string_view kRawLogGoldenHex =
    "544154524c4f47000001001000000000010100000000000000000000000000036162630200"
    "0000000000000000000500000018000000000000000100000000000000030000000000000000";
static_assert(kRawLogGoldenHex.size() == 150);
constexpr std::string_view kRawLogGoldenSha256 =
    "ddceef7bbe62c60e32543927f7d578875b7d765a0ed6590cd015c3ea3f8cdadb";

// SS-001 golden 570-byte sample stream from section 18.5.
constexpr std::string_view kSampleStreamGolden =
    "{\"allowed_rejection_reasons\":[],\"arm\":\"DESCRIPTIVE\",\"decimal_scale\":3,"
    "\"direction\":\"HIGHER_IS_BETTER\",\"maximum_samples\":3,"
    "\"metric_name\":\"tokens_per_second\",\"planned_observation_count\":1,"
    "\"record_kind\":\"HEADER\",\"schema_name\":\"tatara.sample_stream\","
    "\"schema_version\":1,\"unit\":\"tokens_per_second\"}\n"
    "{\"arm\":\"DESCRIPTIVE\",\"disposition\":\"RETAINED\",\"record_kind\":\"SAMPLE\","
    "\"rejection_reason\":\"NOT_APPLICABLE\",\"sample_index\":0,\"value_scaled\":1000}\n"
    "{\"complete\":true,\"observed_count\":1,\"record_kind\":\"TERMINAL\","
    "\"rejected_count\":0,\"retained_count\":1,\"truncated\":false,\"warmup_count\":0}\n";
static_assert(kSampleStreamGolden.size() == 570);
constexpr std::string_view kSampleStreamGoldenSha256 =
    "006d31d96e3a4ce4ce91af4597f490eb7c2f04b36419da2d9eb2f1bec276d241";

void check(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

std::span<const std::byte> bytes_of(std::string_view value) {
    return {reinterpret_cast<const std::byte*>(value.data()), value.size()};
}

std::string_view text_of(std::span<const std::byte> bytes) {
    return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

std::string sha256_hex_of(std::span<const std::byte> bytes) {
    Sha256 hasher;
    hasher.update(bytes);
    return sha256_hex(hasher.finish());
}

std::vector<std::byte> bytes_from_hex(std::string_view hex) {
    check(hex.size() % 2 == 0, "hex fixture length");
    const auto nibble = [](char character) -> unsigned {
        if (character >= '0' && character <= '9') {
            return static_cast<unsigned>(character - '0');
        }
        check(character >= 'a' && character <= 'f', "hex fixture character");
        return static_cast<unsigned>(character - 'a') + 10U;
    };
    std::vector<std::byte> bytes;
    bytes.reserve(hex.size() / 2);
    for (std::size_t index = 0; index < hex.size(); index += 2) {
        bytes.push_back(
            static_cast<std::byte>((nibble(hex[index]) << 4) | nibble(hex[index + 1])));
    }
    return bytes;
}

void append_bytes(std::vector<std::byte>& out, std::span<const std::byte> bytes) {
    out.insert(out.end(), bytes.begin(), bytes.end());
}

std::string replaced(std::string_view artifact, std::string_view from, std::string_view to) {
    std::string result(artifact);
    const std::size_t position = result.find(from);
    check(position != std::string::npos, "fixture substring present");
    check(result.find(from, position + 1) == std::string::npos, "fixture substring unique");
    result.replace(position, from.size(), to);
    return result;
}

// --- Raw log record codecs -------------------------------------------------

void check_raw_log_file_header_record() {
    const auto header = encode_raw_log_file_header();
    check(decode_raw_log_file_header(header) == RawLogError::None, "file header round trip");

    const auto golden = bytes_from_hex(kRawLogGoldenHex);
    check(std::equal(header.begin(), header.end(), golden.begin()),
          "file header equals golden prefix");

    std::vector<std::byte> mutated(header.begin(), header.end());
    mutated[0] = std::byte{0x55};
    check(decode_raw_log_file_header(mutated) == RawLogError::InvalidFrame,
          "file header magic");

    mutated.assign(header.begin(), header.end());
    mutated[9] = std::byte{2};
    check(decode_raw_log_file_header(mutated) == RawLogError::SchemaUnsupported,
          "file header version 2");

    mutated.assign(header.begin(), header.end());
    mutated[11] = std::byte{0x11};
    check(decode_raw_log_file_header(mutated) == RawLogError::InvalidFrame,
          "file header size 17");

    mutated.assign(header.begin(), header.end());
    mutated[12] = std::byte{1};
    check(decode_raw_log_file_header(mutated) == RawLogError::InvalidFrame,
          "file header flags");

    check(decode_raw_log_file_header(std::span<const std::byte>(header).first(15)) ==
              RawLogError::InvalidFrame,
          "file header truncated");
    mutated.assign(header.begin(), header.end());
    mutated.push_back(std::byte{0});
    check(decode_raw_log_file_header(mutated) == RawLogError::TrailingData,
          "file header overlong");
}

void check_raw_log_data_frame_record() {
    const std::array<RawLogStreamId, 3> streams = {
        RawLogStreamId::Stdout, RawLogStreamId::Stderr, RawLogStreamId::Harness};
    for (const RawLogStreamId stream : streams) {
        const RawLogDataFrame frame{stream, 42, bytes_of("payload")};
        const auto encoded = encode_raw_log_data_frame(frame);
        check(static_cast<bool>(encoded), "data frame encode");
        const auto decoded = decode_raw_log_data_frame(encoded.bytes);
        check(static_cast<bool>(decoded), "data frame decode");
        check(decoded.frame.stream_id == stream &&
                  decoded.frame.monotonic_offset_ns == 42 &&
                  text_of(decoded.frame.payload) == "payload",
              "data frame fields");
        const auto reencoded = encode_raw_log_data_frame(decoded.frame);
        check(reencoded.bytes == encoded.bytes, "data frame byte stability");
    }

    const auto encoded = encode_raw_log_data_frame({RawLogStreamId::Stdout, 0, bytes_of("abc")});
    std::vector<std::byte> mutated = encoded.bytes;
    mutated[1] = std::byte{4};
    check(decode_raw_log_data_frame(mutated).error == RawLogError::InvalidEnum,
          "data frame unknown stream");
    mutated = encoded.bytes;
    mutated[0] = std::byte{3};
    check(decode_raw_log_data_frame(mutated).error == RawLogError::InvalidEnum,
          "data frame unknown record type");
    mutated = encoded.bytes;
    mutated[0] = std::byte{2};
    check(decode_raw_log_data_frame(mutated).error == RawLogError::InvalidFrame,
          "data frame terminal type");
    mutated = encoded.bytes;
    mutated[3] = std::byte{1};
    check(decode_raw_log_data_frame(mutated).error == RawLogError::InvalidFrame,
          "data frame nonzero flags");
    mutated = encoded.bytes;
    mutated[15] = std::byte{0};
    check(decode_raw_log_data_frame(mutated).error == RawLogError::NumericRange,
          "data frame zero payload length");
    check(decode_raw_log_data_frame(std::span<const std::byte>(encoded.bytes).first(18))
                  .error == RawLogError::InvalidFrame,
          "data frame truncated payload");
    mutated = encoded.bytes;
    mutated.push_back(std::byte{0});
    check(decode_raw_log_data_frame(mutated).error == RawLogError::TrailingData,
          "data frame overlong");

    check(encode_raw_log_data_frame({RawLogStreamId::Stdout, 0, {}}).error ==
              RawLogError::NumericRange,
          "data frame empty payload encode");
    check(encode_raw_log_data_frame({static_cast<RawLogStreamId>(4), 0, bytes_of("x")})
                  .error == RawLogError::InvalidEnum,
          "data frame invalid stream encode");

    const std::vector<std::byte> maximum_payload(kMaximumRawLogDataPayloadBytes,
                                                 std::byte{0x5a});
    const auto maximum_frame =
        encode_raw_log_data_frame({RawLogStreamId::Harness, 7, maximum_payload});
    check(static_cast<bool>(maximum_frame), "data frame maximum payload encode");
    const auto maximum_decoded = decode_raw_log_data_frame(maximum_frame.bytes);
    check(static_cast<bool>(maximum_decoded) &&
              maximum_decoded.frame.payload.size() == kMaximumRawLogDataPayloadBytes,
          "data frame maximum payload round trip");
    std::vector<std::byte> oversize = maximum_frame.bytes;
    oversize[15] = std::byte{0x01}; // payload length 65,537
    oversize.push_back(std::byte{0});
    check(decode_raw_log_data_frame(oversize).error == RawLogError::NumericRange,
          "data frame payload maximum plus one decode");
    const std::vector<std::byte> oversize_payload(kMaximumRawLogDataPayloadBytes + 1,
                                                  std::byte{0});
    check(encode_raw_log_data_frame({RawLogStreamId::Stdout, 0, oversize_payload}).error ==
              RawLogError::NumericRange,
          "data frame payload maximum plus one encode");
}

void check_raw_log_terminal_frame_record() {
    const RawLogTerminalFrame complete{9, false, 2, 6, 0};
    const auto encoded = encode_raw_log_terminal_frame(complete);
    check(static_cast<bool>(encoded) && encoded.bytes.size() == kRawLogTerminalFrameBytes,
          "terminal encode");
    const auto decoded = decode_raw_log_terminal_frame(encoded.bytes);
    check(static_cast<bool>(decoded) && decoded.terminal.monotonic_offset_ns == 9 &&
              !decoded.terminal.truncated && decoded.terminal.data_frame_count == 2 &&
              decoded.terminal.captured_payload_bytes == 6 &&
              decoded.terminal.discarded_payload_bytes == 0,
          "terminal fields");
    check(encode_raw_log_terminal_frame(decoded.terminal).bytes == encoded.bytes,
          "terminal byte stability");

    const RawLogTerminalFrame truncated{9, true, 2, 6, 17};
    const auto truncated_encoded = encode_raw_log_terminal_frame(truncated);
    const auto truncated_decoded = decode_raw_log_terminal_frame(truncated_encoded.bytes);
    check(static_cast<bool>(truncated_decoded) && truncated_decoded.terminal.truncated &&
              truncated_decoded.terminal.discarded_payload_bytes == 17,
          "terminal truncated round trip");

    check(encode_raw_log_terminal_frame({0, true, 0, 0, 0}).error ==
              RawLogError::ConservationMismatch,
          "terminal truncated without discarded encode");
    check(encode_raw_log_terminal_frame({0, false, 0, 0, 5}).error ==
              RawLogError::ConservationMismatch,
          "terminal discarded without truncated encode");

    std::vector<std::byte> mutated = encoded.bytes;
    mutated[1] = std::byte{1};
    check(decode_raw_log_terminal_frame(mutated).error == RawLogError::InvalidFrame,
          "terminal nonzero stream");
    mutated = encoded.bytes;
    mutated[3] = std::byte{2};
    check(decode_raw_log_terminal_frame(mutated).error == RawLogError::InvalidFrame,
          "terminal reserved flag bit");
    mutated = encoded.bytes;
    mutated[15] = std::byte{23};
    check(decode_raw_log_terminal_frame(mutated).error == RawLogError::InvalidFrame,
          "terminal payload length 23");
    mutated = encoded.bytes;
    mutated[0] = std::byte{1};
    check(decode_raw_log_terminal_frame(mutated).error == RawLogError::InvalidFrame,
          "terminal data record type");
    mutated = encoded.bytes;
    mutated[0] = std::byte{5};
    check(decode_raw_log_terminal_frame(mutated).error == RawLogError::InvalidEnum,
          "terminal unknown record type");
    mutated = encoded.bytes;
    mutated[3] = std::byte{1}; // TRUNCATED flag with discarded still zero
    check(decode_raw_log_terminal_frame(mutated).error == RawLogError::ConservationMismatch,
          "terminal truncated without discarded decode");
    mutated = encoded.bytes;
    mutated[39] = std::byte{3}; // discarded 3 without TRUNCATED flag
    check(decode_raw_log_terminal_frame(mutated).error == RawLogError::ConservationMismatch,
          "terminal discarded without truncated decode");
    check(decode_raw_log_terminal_frame(std::span<const std::byte>(encoded.bytes).first(39))
                  .error == RawLogError::InvalidFrame,
          "terminal truncated record");
    mutated = encoded.bytes;
    mutated.push_back(std::byte{0});
    check(decode_raw_log_terminal_frame(mutated).error == RawLogError::TrailingData,
          "terminal overlong record");
}

// --- Raw log whole-artifact codec ------------------------------------------

void check_raw_log_golden_artifact() {
    const auto golden = bytes_from_hex(kRawLogGoldenHex);
    check(sha256_hex_of(golden) == kRawLogGoldenSha256, "RL-001 fixture digest");

    const auto decoded = decode_raw_log(golden, kMaximumRawLogArtifactBytes);
    check(static_cast<bool>(decoded), "RL-001 decode");
    check(decoded.frames.size() == 1 &&
              decoded.frames[0].stream_id == RawLogStreamId::Stdout &&
              decoded.frames[0].monotonic_offset_ns == 0 &&
              text_of(decoded.frames[0].payload) == "abc",
          "RL-001 data frame");
    check(decoded.terminal.monotonic_offset_ns == 5 && !decoded.terminal.truncated &&
              decoded.terminal.data_frame_count == 1 &&
              decoded.terminal.captured_payload_bytes == 3 &&
              decoded.terminal.discarded_payload_bytes == 0,
          "RL-001 terminal");

    const auto reencoded =
        encode_raw_log(decoded.frames, decoded.terminal, kMaximumRawLogArtifactBytes);
    check(static_cast<bool>(reencoded) && reencoded.bytes == golden,
          "RL-001 encode-decode-encode byte stability");

    check(decode_raw_log(golden, 75).error == RawLogError::None, "RL-001 exact budget");
    check(decode_raw_log(golden, 74).error == RawLogError::ArtifactTooLarge, "RL-012");
}

void check_raw_log_artifact_negatives() {
    const auto golden = bytes_from_hex(kRawLogGoldenHex);
    const std::span<const std::byte> golden_span(golden);

    check(decode_raw_log({}, kMaximumRawLogArtifactBytes).error == RawLogError::InvalidFrame,
          "raw log empty input");
    check(decode_raw_log(golden_span.first(35), kMaximumRawLogArtifactBytes).error ==
              RawLogError::InvalidFrame,
          "RL-002 missing terminal");
    check(decode_raw_log(golden_span.first(33), kMaximumRawLogArtifactBytes).error ==
              RawLogError::InvalidFrame,
          "RL-005 payload-length mismatch");
    check(decode_raw_log(golden_span.first(74), kMaximumRawLogArtifactBytes).error ==
              RawLogError::InvalidFrame,
          "raw log truncated terminal");

    auto mutated = golden;
    mutated[17] = std::byte{4};
    check(decode_raw_log(mutated, kMaximumRawLogArtifactBytes).error ==
              RawLogError::InvalidEnum,
          "RL-003 unknown stream");
    mutated = golden;
    mutated[16] = std::byte{3};
    check(decode_raw_log(mutated, kMaximumRawLogArtifactBytes).error ==
              RawLogError::InvalidEnum,
          "raw log unknown record type");
    mutated = golden;
    mutated[27] = std::byte{9}; // data offset 9 exceeds the terminal offset 5
    check(decode_raw_log(mutated, kMaximumRawLogArtifactBytes).error ==
              RawLogError::InvalidFrame,
          "RL-004 decreasing offset");
    mutated = golden;
    mutated[58] = std::byte{2};
    check(decode_raw_log(mutated, kMaximumRawLogArtifactBytes).error ==
              RawLogError::ConservationMismatch,
          "RL-006 terminal frame-count mismatch");
    mutated = golden;
    mutated[66] = std::byte{5};
    check(decode_raw_log(mutated, kMaximumRawLogArtifactBytes).error ==
              RawLogError::ConservationMismatch,
          "RL-007 captured-byte mismatch");
    mutated = golden;
    mutated[38] = std::byte{1};
    check(decode_raw_log(mutated, kMaximumRawLogArtifactBytes).error ==
              RawLogError::ConservationMismatch,
          "RL-008 truncated flag with zero discarded");
    mutated = golden;
    mutated[74] = std::byte{3};
    check(decode_raw_log(mutated, kMaximumRawLogArtifactBytes).error ==
              RawLogError::ConservationMismatch,
          "RL-009 discarded without truncated flag");
    mutated = golden;
    mutated.push_back(std::byte{0});
    check(decode_raw_log(mutated, kMaximumRawLogArtifactBytes).error ==
              RawLogError::TrailingData,
          "RL-010 bytes after terminal");
    mutated = golden;
    mutated[9] = std::byte{2};
    check(decode_raw_log(mutated, kMaximumRawLogArtifactBytes).error ==
              RawLogError::SchemaUnsupported,
          "raw log format version 2");
    mutated = golden;
    mutated[0] = std::byte{0x55};
    check(decode_raw_log(mutated, kMaximumRawLogArtifactBytes).error ==
              RawLogError::InvalidFrame,
          "raw log corrupt magic");
    mutated = golden;
    mutated[31] = std::byte{0};
    check(decode_raw_log(mutated, kMaximumRawLogArtifactBytes).error ==
              RawLogError::NumericRange,
          "raw log zero payload length");

    // Two individually valid frames with a decreasing offset pair.
    std::vector<std::byte> crafted;
    const auto file_header = encode_raw_log_file_header();
    append_bytes(crafted, file_header);
    append_bytes(crafted, encode_raw_log_data_frame({RawLogStreamId::Stdout, 5, bytes_of("aa")}).bytes);
    append_bytes(crafted, encode_raw_log_data_frame({RawLogStreamId::Stdout, 4, bytes_of("bb")}).bytes);
    append_bytes(crafted, encode_raw_log_terminal_frame({5, false, 2, 4, 0}).bytes);
    check(decode_raw_log(crafted, kMaximumRawLogArtifactBytes).error ==
              RawLogError::InvalidFrame,
          "raw log decreasing data offsets");
}

void check_raw_log_round_trips() {
    // Zero-output child execution: exactly one terminal frame.
    const RawLogTerminalFrame empty_terminal{0, false, 0, 0, 0};
    const auto empty_log =
        encode_raw_log({}, empty_terminal, kMaximumRawLogArtifactBytes);
    check(static_cast<bool>(empty_log) &&
              empty_log.bytes.size() == kRawLogFileHeaderBytes + kRawLogTerminalFrameBytes,
          "zero-frame raw log encode");
    const auto empty_decoded = decode_raw_log(empty_log.bytes, kMaximumRawLogArtifactBytes);
    check(static_cast<bool>(empty_decoded) && empty_decoded.frames.empty(),
          "zero-frame raw log decode");
    check(encode_raw_log(empty_decoded.frames, empty_decoded.terminal,
                         kMaximumRawLogArtifactBytes)
                  .bytes == empty_log.bytes,
          "zero-frame raw log byte stability");

    // Every stream enum plus an equal-offset pair and a truncated terminal.
    const std::array<RawLogDataFrame, 3> frames = {
        RawLogDataFrame{RawLogStreamId::Stdout, 1, bytes_of("alpha")},
        RawLogDataFrame{RawLogStreamId::Stderr, 1, bytes_of("beta")},
        RawLogDataFrame{RawLogStreamId::Harness, 2, bytes_of("gamma!")},
    };
    const RawLogTerminalFrame terminal{2, true, 3, 15, 9};
    const auto encoded = encode_raw_log(frames, terminal, kMaximumRawLogArtifactBytes);
    check(static_cast<bool>(encoded), "multi-stream raw log encode");
    const auto decoded = decode_raw_log(encoded.bytes, kMaximumRawLogArtifactBytes);
    check(static_cast<bool>(decoded) && decoded.frames.size() == 3 &&
              decoded.terminal.truncated &&
              decoded.terminal.discarded_payload_bytes == 9,
          "multi-stream raw log decode");
    check(encode_raw_log(decoded.frames, decoded.terminal, kMaximumRawLogArtifactBytes)
                  .bytes == encoded.bytes,
          "multi-stream raw log byte stability");
}

void check_raw_log_encoder_negatives() {
    const RawLogDataFrame frame{RawLogStreamId::Stdout, 3, bytes_of("abc")};
    const std::array<RawLogDataFrame, 1> frames = {frame};

    check(encode_raw_log(frames, {3, false, 2, 3, 0}, kMaximumRawLogArtifactBytes).error ==
              RawLogError::ConservationMismatch,
          "encode frame-count mismatch");
    check(encode_raw_log(frames, {3, false, 1, 4, 0}, kMaximumRawLogArtifactBytes).error ==
              RawLogError::ConservationMismatch,
          "encode captured-byte mismatch");
    check(encode_raw_log(frames, {2, false, 1, 3, 0}, kMaximumRawLogArtifactBytes).error ==
              RawLogError::InvalidFrame,
          "encode terminal offset before data offset");
    check(encode_raw_log({}, {0, false, 0, 0, 0}, 55).error == RawLogError::ArtifactTooLarge,
          "encode budget below minimum artifact");

    const std::array<RawLogDataFrame, 2> decreasing = {
        RawLogDataFrame{RawLogStreamId::Stdout, 5, bytes_of("aa")},
        RawLogDataFrame{RawLogStreamId::Stdout, 4, bytes_of("bb")},
    };
    check(encode_raw_log(decreasing, {5, false, 2, 4, 0}, kMaximumRawLogArtifactBytes)
                  .error == RawLogError::InvalidFrame,
          "encode decreasing offsets");

    const std::array<RawLogDataFrame, 1> invalid_stream = {
        RawLogDataFrame{static_cast<RawLogStreamId>(4), 0, bytes_of("x")}};
    check(encode_raw_log(invalid_stream, {0, false, 1, 1, 0}, kMaximumRawLogArtifactBytes)
                  .error == RawLogError::InvalidEnum,
          "encode invalid stream enum");
}

void check_raw_log_frame_count_bounds() {
    const RawLogDataFrame frame{RawLogStreamId::Stdout, 0, bytes_of("x")};
    const auto frame_bytes = encode_raw_log_data_frame(frame);
    check(static_cast<bool>(frame_bytes), "bound fixture frame");
    const auto file_header = encode_raw_log_file_header();

    const auto build = [&](std::size_t frame_count) {
        std::vector<std::byte> artifact;
        artifact.reserve(file_header.size() + frame_count * frame_bytes.bytes.size() +
                         kRawLogTerminalFrameBytes);
        append_bytes(artifact, file_header);
        for (std::size_t index = 0; index < frame_count; ++index) {
            append_bytes(artifact, frame_bytes.bytes);
        }
        const RawLogTerminalFrame terminal{
            0, false, static_cast<std::uint64_t>(frame_count),
            static_cast<std::uint64_t>(frame_count), 0};
        append_bytes(artifact, encode_raw_log_terminal_frame(terminal).bytes);
        return artifact;
    };

    const auto exact = build(kMaximumRawLogDataFrames);
    const auto decoded = decode_raw_log(exact, kMaximumRawLogArtifactBytes);
    check(static_cast<bool>(decoded) && decoded.frames.size() == kMaximumRawLogDataFrames,
          "raw log exact frame maximum");
    check(encode_raw_log(decoded.frames, decoded.terminal, kMaximumRawLogArtifactBytes)
                  .bytes == exact,
          "raw log frame maximum byte stability");

    const auto over = build(kMaximumRawLogDataFrames + 1);
    check(decode_raw_log(over, kMaximumRawLogArtifactBytes).error == RawLogError::InvalidCount,
          "RL-011 frame maximum plus one decode");

    std::vector<RawLogDataFrame> frames(kMaximumRawLogDataFrames + 1, frame);
    const RawLogTerminalFrame terminal{
        0, false, static_cast<std::uint64_t>(frames.size()),
        static_cast<std::uint64_t>(frames.size()), 0};
    check(encode_raw_log(frames, terminal, kMaximumRawLogArtifactBytes).error ==
              RawLogError::InvalidCount,
          "RL-011 frame maximum plus one encode");
}

// --- Sample stream record codecs -------------------------------------------

void check_sample_stream_golden() {
    check(sha256_hex_of(bytes_of(kSampleStreamGolden)) == kSampleStreamGoldenSha256,
          "SS-001 fixture digest");

    const auto decoded =
        decode_sample_stream(kSampleStreamGolden, kMaximumSampleStreamArtifactBytes);
    check(static_cast<bool>(decoded), "SS-001 decode");
    check(decoded.header.allowed_rejection_reasons.empty() &&
              decoded.header.arm == SampleArm::Descriptive &&
              decoded.header.decimal_scale == 3 &&
              decoded.header.direction == SampleDirection::HigherIsBetter &&
              decoded.header.maximum_samples == 3 &&
              decoded.header.metric_name == "tokens_per_second" &&
              decoded.header.planned_observation_count == 1 &&
              decoded.header.unit == "tokens_per_second",
          "SS-001 header");
    check(decoded.samples.size() == 1 &&
              decoded.samples[0].disposition == SampleDisposition::Retained &&
              decoded.samples[0].rejection_reason == SampleRejectionReason::NotApplicable &&
              decoded.samples[0].sample_index == 0 &&
              decoded.samples[0].value_scaled == 1000,
          "SS-001 sample");
    check(decoded.terminal.complete && decoded.terminal.observed_count == 1 &&
              decoded.terminal.retained_count == 1 && !decoded.terminal.truncated,
          "SS-001 terminal");

    const auto reencoded =
        encode_sample_stream(decoded.header, decoded.samples, decoded.terminal,
                             kMaximumSampleStreamArtifactBytes);
    check(static_cast<bool>(reencoded) && reencoded.bytes == kSampleStreamGolden,
          "SS-001 encode-decode-encode byte stability");

    check(decode_sample_stream(kSampleStreamGolden, 570).error == SampleStreamError::None,
          "SS-001 exact budget");
    check(decode_sample_stream(kSampleStreamGolden, 569).error ==
              SampleStreamError::ArtifactTooLarge,
          "sample stream budget maximum plus one");
}

void check_sample_record_codecs() {
    const std::size_t first_feed = kSampleStreamGolden.find('\n');
    const std::size_t second_feed = kSampleStreamGolden.find('\n', first_feed + 1);
    const std::string_view header_line = kSampleStreamGolden.substr(0, first_feed);
    const std::string_view sample_line =
        kSampleStreamGolden.substr(first_feed + 1, second_feed - first_feed - 1);
    const std::string_view terminal_line = kSampleStreamGolden.substr(
        second_feed + 1, kSampleStreamGolden.size() - second_feed - 2);

    const auto header = decode_sample_stream_header(header_line);
    check(static_cast<bool>(header), "header record decode");
    check(encode_sample_stream_header(header.header).bytes == header_line,
          "header record byte stability");

    const auto sample = decode_sample_record(sample_line);
    check(static_cast<bool>(sample), "sample record decode");
    check(encode_sample_record(sample.record).bytes == sample_line,
          "sample record byte stability");

    const auto terminal = decode_sample_stream_terminal(terminal_line);
    check(static_cast<bool>(terminal), "terminal record decode");
    check(encode_sample_stream_terminal(terminal.terminal).bytes == terminal_line,
          "terminal record byte stability");

    // Populated rejection-reason set and every disposition round trip.
    SampleStreamHeader synthetic;
    synthetic.allowed_rejection_reasons = {SampleRejectionReason::MeasurementUnavailable,
                                           SampleRejectionReason::Timeout};
    synthetic.arm = SampleArm::Treatment;
    synthetic.decimal_scale = 0;
    synthetic.direction = SampleDirection::LowerIsBetter;
    synthetic.maximum_samples = 4;
    synthetic.metric_name = "latency_ns";
    synthetic.planned_observation_count = 3;
    synthetic.unit = "nanoseconds";
    const auto synthetic_encoded = encode_sample_stream_header(synthetic);
    check(static_cast<bool>(synthetic_encoded), "synthetic header encode");
    const auto synthetic_decoded = decode_sample_stream_header(synthetic_encoded.bytes);
    check(static_cast<bool>(synthetic_decoded) &&
              synthetic_decoded.header.allowed_rejection_reasons ==
                  synthetic.allowed_rejection_reasons &&
              synthetic_decoded.header.arm == SampleArm::Treatment,
          "synthetic header round trip");

    const SampleRecord rejected{SampleArm::Treatment, SampleDisposition::Rejected,
                                SampleRejectionReason::Timeout, 2, -12};
    const auto rejected_encoded = encode_sample_record(rejected);
    const auto rejected_decoded = decode_sample_record(rejected_encoded.bytes);
    check(static_cast<bool>(rejected_decoded) &&
              rejected_decoded.record.disposition == SampleDisposition::Rejected &&
              rejected_decoded.record.value_scaled == -12,
          "rejected sample round trip");

    const SampleRecord extreme{SampleArm::Competitor, SampleDisposition::Warmup,
                               SampleRejectionReason::NotApplicable, 0,
                               std::numeric_limits<std::int64_t>::min()};
    const auto extreme_encoded = encode_sample_record(extreme);
    const auto extreme_decoded = decode_sample_record(extreme_encoded.bytes);
    check(static_cast<bool>(extreme_decoded) &&
              extreme_decoded.record.value_scaled ==
                  std::numeric_limits<std::int64_t>::min(),
          "I64 minimum round trip");
    check(encode_sample_record(extreme_decoded.record).bytes == extreme_encoded.bytes,
          "I64 minimum byte stability");

    check(encode_sample_record({SampleArm::Descriptive, SampleDisposition::Rejected,
                                SampleRejectionReason::NotApplicable, 0, 0})
                  .error == SampleStreamError::ConservationMismatch,
          "rejected sample without reason encode");
    check(encode_sample_record({static_cast<SampleArm>(9), SampleDisposition::Retained,
                                SampleRejectionReason::NotApplicable, 0, 0})
                  .error == SampleStreamError::InvalidEnum,
          "invalid arm enum encode");
    check(decode_sample_record(replaced(sample_line, "\"sample_index\":0",
                                        "\"sample_index\":16777216"))
                  .error == SampleStreamError::InvalidCount,
          "sample index maximum plus one");
}

void check_sample_stream_negatives() {
    const std::string_view golden = kSampleStreamGolden;
    const auto decode_error = [](std::string_view artifact) {
        return decode_sample_stream(artifact, kMaximumSampleStreamArtifactBytes).error;
    };

    check(decode_error("") == SampleStreamError::InvalidFrame, "empty artifact");
    std::string long_line(4097, 'x');
    long_line.push_back('\n');
    check(decode_error(long_line) == SampleStreamError::ArtifactTooLarge,
          "SS-009 line maximum plus one");

    check(decode_error(replaced(golden, "\"rejection_reason\":\"NOT_APPLICABLE\"",
                                "\"rejection_reason\":\"TIMEOUT\"")) ==
              SampleStreamError::ConservationMismatch,
          "SS-004 retained sample with rejection reason");
    check(decode_error(replaced(golden, "\"warmup_count\":0", "\"warmup_count\":1")) ==
              SampleStreamError::ConservationMismatch,
          "SS-005 terminal conservation mismatch");
    check(decode_error(replaced(golden, "{\"arm\":\"DESCRIPTIVE\",\"disposition\"",
                                "{\"arm\":\"TREATMENT\",\"disposition\"")) ==
              SampleStreamError::ConservationMismatch,
          "SS-011 sample arm differs from header");
    check(decode_error(replaced(golden, "\"truncated\":false", "\"truncated\":true")) ==
              SampleStreamError::ConservationMismatch,
          "terminal complete with truncated");

    const std::size_t first_feed = golden.find('\n');
    const std::size_t second_feed = golden.find('\n', first_feed + 1);
    const std::string header_line_feed(golden.substr(0, first_feed + 1));
    const std::string sample_line_feed(
        golden.substr(first_feed + 1, second_feed - first_feed));
    const std::string terminal_line_feed(golden.substr(second_feed + 1));
    std::string incomplete = header_line_feed;
    std::string empty_observation_terminal = replaced(
        replaced(terminal_line_feed, "\"observed_count\":1", "\"observed_count\":0"),
        "\"retained_count\":1", "\"retained_count\":0");
    incomplete.append(empty_observation_terminal);
    check(decode_error(incomplete) == SampleStreamError::ConservationMismatch,
          "SS-012 observed below planned with complete true");

    check(decode_error(replaced(golden, "\"observed_count\":1", "\"complete\":true")) ==
              SampleStreamError::DuplicateKey,
          "duplicate key");
    check(decode_error(replaced(golden, "\"sample_index\":0", "\"sample_index\":1")) ==
              SampleStreamError::InvalidCount,
          "SS-002 noncontiguous sample index");
    check(decode_error(replaced(golden, "\"maximum_samples\":3",
                                "\"maximum_samples\":16777217")) ==
              SampleStreamError::InvalidCount,
          "SS-010 sample maximum plus one");
    check(decode_error(replaced(golden, "\"planned_observation_count\":1",
                                "\"planned_observation_count\":4")) ==
              SampleStreamError::InvalidCount,
          "planned above maximum");

    check(decode_error(replaced(golden, "\"arm\":\"DESCRIPTIVE\",\"decimal_scale\"",
                                "\"arm\":\"DESCRIPTIVEX\",\"decimal_scale\"")) ==
              SampleStreamError::InvalidEnum,
          "unknown arm enum");
    check(decode_error(replaced(
              replaced(golden, "\"disposition\":\"RETAINED\"", "\"disposition\":\"REJECTED\""),
              "\"rejection_reason\":\"NOT_APPLICABLE\"", "\"rejection_reason\":\"TIMEOUT\"")) ==
              SampleStreamError::InvalidEnum,
          "SS-003 undeclared rejection reason");
    check(decode_error(replaced(golden, "\"allowed_rejection_reasons\":[]",
                                "\"allowed_rejection_reasons\":[\"NOT_APPLICABLE\"]")) ==
              SampleStreamError::InvalidEnum,
          "NOT_APPLICABLE inside allowed set");
    check(decode_error(replaced(golden, "\"record_kind\":\"SAMPLE\"",
                                "\"record_kind\":\"BOGUS\"")) ==
              SampleStreamError::InvalidEnum,
          "unknown record kind");

    std::string missing_terminal = header_line_feed;
    missing_terminal.append(sample_line_feed);
    check(decode_error(missing_terminal) == SampleStreamError::InvalidFrame,
          "SS-006 missing terminal");
    std::string headerless = sample_line_feed;
    headerless.append(terminal_line_feed);
    check(decode_error(headerless) == SampleStreamError::InvalidFrame,
          "first record not a header");
    std::string double_header = header_line_feed;
    double_header.append(header_line_feed);
    double_header.append(terminal_line_feed);
    check(decode_error(double_header) == SampleStreamError::InvalidFrame, "second header");
    check(decode_error("{\"x\n") == SampleStreamError::InvalidFrame,
          "unterminated record line");

    check(decode_error(replaced(golden, "\"metric_name\":\"tokens_per_second\"",
                                "\"metric_name\":\"_tokens\"")) ==
              SampleStreamError::InvalidIdentifier,
          "metric name identifier");
    check(decode_error(replaced(golden, "\"decimal_scale\":3", "\"decimal_scale\":03")) ==
              SampleStreamError::InvalidInteger,
          "leading-zero integer");
    check(decode_error(replaced(golden, "\"value_scaled\":1000", "\"value_scaled\":-0")) ==
              SampleStreamError::InvalidInteger,
          "negative zero integer");
    check(decode_error(replaced(golden, ",\"unit\":\"tokens_per_second\"}", "}")) ==
              SampleStreamError::MissingField,
          "missing header field");
    std::string dispatch_missing = header_line_feed;
    dispatch_missing.append("{\"complete\":true}\n");
    dispatch_missing.append(terminal_line_feed);
    check(decode_error(dispatch_missing) == SampleStreamError::MissingField,
          "record kind missing");

    check(decode_error(golden.substr(0, golden.size() - 1)) ==
              SampleStreamError::NonCanonicalBytes,
          "missing final line feed");
    check(decode_error(replaced(golden, "\"value_scaled\":1000", "\"value_scaled\": 1000")) ==
              SampleStreamError::NonCanonicalBytes,
          "insignificant whitespace");
    check(decode_error(replaced(golden, "\"arm\":\"DESCRIPTIVE\",\"decimal_scale\"",
                                "\"arm\":\"\\u0044ESCRIPTIVE\",\"decimal_scale\"")) ==
              SampleStreamError::NonCanonicalBytes,
          "noncanonical escape");

    check(decode_error(replaced(golden,
                                "{\"arm\":\"DESCRIPTIVE\",\"disposition\":\"RETAINED\",",
                                "{\"disposition\":\"RETAINED\",\"arm\":\"DESCRIPTIVE\",")) ==
              SampleStreamError::NonCanonicalOrder,
          "keys out of canonical order");
    check(decode_error(replaced(golden, "\"allowed_rejection_reasons\":[]",
                                "\"allowed_rejection_reasons\":[\"TIMEOUT\",\"CANCELLED\"]")) ==
              SampleStreamError::NonCanonicalOrder,
          "unsorted rejection reasons");
    check(decode_error(replaced(golden, "\"allowed_rejection_reasons\":[]",
                                "\"allowed_rejection_reasons\":[\"TIMEOUT\",\"TIMEOUT\"]")) ==
              SampleStreamError::NonCanonicalOrder,
          "duplicate rejection reason");

    check(decode_error(replaced(golden, "\"value_scaled\":1000",
                                "\"value_scaled\":9223372036854775808")) ==
              SampleStreamError::NumericRange,
          "SS-008 value_scaled overflow");
    check(decode_error(replaced(golden, "\"decimal_scale\":3", "\"decimal_scale\":19")) ==
              SampleStreamError::NumericRange,
          "decimal scale maximum plus one");

    check(decode_error(replaced(golden, "\"schema_version\":1", "\"schema_version\":2")) ==
              SampleStreamError::SchemaUnsupported,
          "schema version 2");
    check(decode_error(replaced(golden, "\"schema_name\":\"tatara.sample_stream\"",
                                "\"schema_name\":\"tatara.other_stream\"")) ==
              SampleStreamError::SchemaUnsupported,
          "unknown schema name");

    std::string trailing(golden);
    trailing.append(terminal_line_feed);
    check(decode_error(trailing) == SampleStreamError::TrailingData,
          "SS-007 record after terminal");
    check(decode_error(replaced(golden, "}\n{\"arm\"", "}\r\n{\"arm\"")) ==
              SampleStreamError::TrailingData,
          "carriage return line ending");

    check(decode_error(replaced(golden, ",\"warmup_count\":0}",
                                ",\"warmup_count\":0,\"zz_extra\":1}")) ==
              SampleStreamError::UnknownKey,
          "unknown key");
    check(decode_error(replaced(golden, "\"complete\":true", "\"complete\":1")) ==
              SampleStreamError::WrongType,
          "boolean field with integer value");
    check(decode_error(replaced(golden, "\"decimal_scale\":3", "\"decimal_scale\":\"3\"")) ==
              SampleStreamError::WrongType,
          "integer field with string value");
    check(decode_error(replaced(golden, "\"value_scaled\":1000", "\"value_scaled\":null")) ==
              SampleStreamError::WrongType,
          "null value");
}

void check_sample_stream_round_trips() {
    SampleStreamHeader header;
    header.allowed_rejection_reasons = {SampleRejectionReason::MeasurementUnavailable,
                                        SampleRejectionReason::Timeout};
    header.arm = SampleArm::Treatment;
    header.decimal_scale = 18;
    header.direction = SampleDirection::LowerIsBetter;
    header.maximum_samples = 4;
    header.metric_name = "latency_ns";
    header.planned_observation_count = 3;
    header.unit = "nanoseconds";
    const std::array<SampleRecord, 3> samples = {
        SampleRecord{SampleArm::Treatment, SampleDisposition::Warmup,
                     SampleRejectionReason::NotApplicable, 0, -5},
        SampleRecord{SampleArm::Treatment, SampleDisposition::Retained,
                     SampleRejectionReason::NotApplicable, 1,
                     std::numeric_limits<std::int64_t>::max()},
        SampleRecord{SampleArm::Treatment, SampleDisposition::Rejected,
                     SampleRejectionReason::Timeout, 2, 0},
    };
    const SampleStreamTerminal terminal{true, 3, 1, 1, false, 1};
    const auto encoded =
        encode_sample_stream(header, samples, terminal, kMaximumSampleStreamArtifactBytes);
    check(static_cast<bool>(encoded), "synthetic stream encode");
    const auto decoded =
        decode_sample_stream(encoded.bytes, kMaximumSampleStreamArtifactBytes);
    check(static_cast<bool>(decoded) && decoded.samples.size() == 3 &&
              decoded.samples[1].value_scaled ==
                  std::numeric_limits<std::int64_t>::max() &&
              decoded.terminal.complete,
          "synthetic stream decode");
    check(encode_sample_stream(decoded.header, decoded.samples, decoded.terminal,
                               kMaximumSampleStreamArtifactBytes)
                  .bytes == encoded.bytes,
          "synthetic stream byte stability");

    // Zero samples with an incomplete terminal is a legal closed stream.
    SampleStreamHeader empty_header = header;
    empty_header.allowed_rejection_reasons.clear();
    const SampleStreamTerminal empty_terminal{false, 0, 0, 0, true, 0};
    const auto empty_encoded = encode_sample_stream(empty_header, {}, empty_terminal,
                                                    kMaximumSampleStreamArtifactBytes);
    check(static_cast<bool>(empty_encoded), "empty stream encode");
    const auto empty_decoded =
        decode_sample_stream(empty_encoded.bytes, kMaximumSampleStreamArtifactBytes);
    check(static_cast<bool>(empty_decoded) && empty_decoded.samples.empty() &&
              empty_decoded.terminal.truncated,
          "empty stream decode");
    check(encode_sample_stream(empty_decoded.header, empty_decoded.samples,
                               empty_decoded.terminal, kMaximumSampleStreamArtifactBytes)
                  .bytes == empty_encoded.bytes,
          "empty stream byte stability");
}

void check_sample_stream_encoder_negatives() {
    SampleStreamHeader header;
    header.arm = SampleArm::Descriptive;
    header.direction = SampleDirection::DescriptiveOnly;
    header.maximum_samples = 2;
    header.metric_name = "tokens_per_second";
    header.planned_observation_count = 1;
    header.unit = "tokens_per_second";
    const SampleRecord sample{SampleArm::Descriptive, SampleDisposition::Retained,
                              SampleRejectionReason::NotApplicable, 0, 7};
    const std::array<SampleRecord, 1> samples = {sample};
    const SampleStreamTerminal terminal{true, 1, 0, 1, false, 0};

    SampleStreamHeader bad_identifier = header;
    bad_identifier.metric_name = "bad name";
    check(encode_sample_stream_header(bad_identifier).error ==
              SampleStreamError::InvalidIdentifier,
          "encode invalid metric identifier");
    SampleStreamHeader long_identifier = header;
    long_identifier.unit.assign(129, 'a');
    check(encode_sample_stream_header(long_identifier).error ==
              SampleStreamError::InvalidIdentifier,
          "encode identifier maximum plus one");
    SampleStreamHeader unsorted = header;
    unsorted.allowed_rejection_reasons = {SampleRejectionReason::Timeout,
                                          SampleRejectionReason::Cancelled};
    check(encode_sample_stream_header(unsorted).error ==
              SampleStreamError::NonCanonicalOrder,
          "encode unsorted rejection reasons");
    SampleStreamHeader sentinel_reason = header;
    sentinel_reason.allowed_rejection_reasons = {SampleRejectionReason::NotApplicable};
    check(encode_sample_stream_header(sentinel_reason).error ==
              SampleStreamError::InvalidEnum,
          "encode NOT_APPLICABLE in allowed set");
    SampleStreamHeader bad_scale = header;
    bad_scale.decimal_scale = 19;
    check(encode_sample_stream_header(bad_scale).error == SampleStreamError::NumericRange,
          "encode decimal scale maximum plus one");
    SampleStreamHeader zero_maximum = header;
    zero_maximum.maximum_samples = 0;
    check(encode_sample_stream_header(zero_maximum).error == SampleStreamError::InvalidCount,
          "encode zero maximum samples");
    SampleStreamHeader planned_overflow = header;
    planned_overflow.planned_observation_count = 3;
    check(encode_sample_stream_header(planned_overflow).error ==
              SampleStreamError::InvalidCount,
          "encode planned above maximum");
    SampleStreamHeader invalid_arm = header;
    invalid_arm.arm = static_cast<SampleArm>(9);
    check(encode_sample_stream_header(invalid_arm).error == SampleStreamError::InvalidEnum,
          "encode invalid arm enum");

    const std::array<SampleRecord, 1> wrong_arm = {
        SampleRecord{SampleArm::Treatment, SampleDisposition::Retained,
                     SampleRejectionReason::NotApplicable, 0, 7}};
    check(encode_sample_stream(header, wrong_arm, terminal,
                               kMaximumSampleStreamArtifactBytes)
                  .error == SampleStreamError::ConservationMismatch,
          "encode arm mismatch");
    const std::array<SampleRecord, 1> gap = {
        SampleRecord{SampleArm::Descriptive, SampleDisposition::Retained,
                     SampleRejectionReason::NotApplicable, 1, 7}};
    check(encode_sample_stream(header, gap, terminal, kMaximumSampleStreamArtifactBytes)
                  .error == SampleStreamError::InvalidCount,
          "encode noncontiguous index");
    const std::array<SampleRecord, 1> undeclared = {
        SampleRecord{SampleArm::Descriptive, SampleDisposition::Rejected,
                     SampleRejectionReason::Timeout, 0, 7}};
    const SampleStreamTerminal rejected_terminal{true, 1, 1, 0, false, 0};
    check(encode_sample_stream(header, undeclared, rejected_terminal,
                               kMaximumSampleStreamArtifactBytes)
                  .error == SampleStreamError::InvalidEnum,
          "encode undeclared rejection reason");
    const SampleStreamTerminal wrong_counts{true, 1, 0, 0, false, 1};
    check(encode_sample_stream(header, samples, wrong_counts,
                               kMaximumSampleStreamArtifactBytes)
                  .error == SampleStreamError::ConservationMismatch,
          "encode terminal count mismatch");
    const SampleStreamTerminal incomplete_complete{true, 0, 0, 0, false, 0};
    check(encode_sample_stream(header, {}, incomplete_complete,
                               kMaximumSampleStreamArtifactBytes)
                  .error == SampleStreamError::ConservationMismatch,
          "encode complete below planned");
    check(encode_sample_stream(header, samples, terminal, 100).error ==
              SampleStreamError::ArtifactTooLarge,
          "encode budget too small");

    SampleStreamHeader tight = header;
    tight.maximum_samples = 1;
    const std::array<SampleRecord, 2> two_samples = {
        SampleRecord{SampleArm::Descriptive, SampleDisposition::Retained,
                     SampleRejectionReason::NotApplicable, 0, 1},
        SampleRecord{SampleArm::Descriptive, SampleDisposition::Retained,
                     SampleRejectionReason::NotApplicable, 1, 2}};
    const SampleStreamTerminal two_terminal{false, 2, 0, 2, false, 0};
    check(encode_sample_stream(tight, two_samples, two_terminal,
                               kMaximumSampleStreamArtifactBytes)
                  .error == SampleStreamError::InvalidCount,
          "encode samples above header maximum");
}

} // namespace

int main() {
    check_raw_log_file_header_record();
    check_raw_log_data_frame_record();
    check_raw_log_terminal_frame_record();
    check_raw_log_golden_artifact();
    check_raw_log_artifact_negatives();
    check_raw_log_round_trips();
    check_raw_log_encoder_negatives();
    check_raw_log_frame_count_bounds();
    check_sample_stream_golden();
    check_sample_record_codecs();
    check_sample_stream_negatives();
    check_sample_stream_round_trips();
    check_sample_stream_encoder_negatives();
    return 0;
}
