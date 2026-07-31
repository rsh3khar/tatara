#pragma once

#include "tatara/model/artifact_identity.h"
#include "tatara/model/checkpoint_layout.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace tatara::model {

inline constexpr std::array<std::byte, 8> kPreparedCheckpointMagic = {
    std::byte{0x54}, std::byte{0x41}, std::byte{0x54}, std::byte{0x43},
    std::byte{0x4b}, std::byte{0x50}, std::byte{0x54}, std::byte{0},
};
inline constexpr std::uint32_t kPreparedCheckpointSchemaVersion = 1;

struct PreparedCheckpointParseResult;

struct PreparedCheckpointIdentity {
    std::string package_id;
    std::string package_sha256;
    std::string artifact_id;
    std::string artifact_manifest_sha256;
    std::string model_type;
    std::string format;
    std::string source_repository;
    std::string source_revision;
    std::uint32_t artifact_file_count;
};

struct PreparedShard {
    std::string path;
    std::string sha256;
    std::uint64_t file_size_bytes;
    std::uint64_t data_offset_bytes;
    std::uint64_t data_size_bytes;
};

class PreparedCheckpoint {
  public:
    const PreparedCheckpointIdentity& identity() const noexcept;
    std::span<const PreparedShard> shards() const noexcept;
    std::span<const TensorRecord> tensors() const noexcept;
    std::uint64_t tensor_payload_bytes() const noexcept;

  private:
    friend PreparedCheckpointParseResult parse_prepared_checkpoint(std::span<const std::byte>);

    PreparedCheckpoint(PreparedCheckpointIdentity identity, std::vector<PreparedShard> shards,
                       std::vector<TensorRecord> tensors, std::uint64_t tensor_payload_bytes);

    PreparedCheckpointIdentity identity_;
    std::vector<PreparedShard> shards_;
    std::vector<TensorRecord> tensors_;
    std::uint64_t tensor_payload_bytes_;
};

enum class PreparedCheckpointError : std::uint8_t {
    None,
    RecordTooSmall,
    RecordTooLarge,
    InvalidMagic,
    UnsupportedSchema,
    InvalidFlags,
    SizeMismatch,
    InvalidCount,
    Truncated,
    InvalidString,
    InvalidDigest,
    InvalidShardPath,
    DuplicateShardPath,
    NonCanonicalOrder,
    InvalidShardRegion,
    InvalidDataType,
    InvalidTensorRank,
    InvalidReservedField,
    TrailingData,
    InvalidLayout,
};

struct PreparedCheckpointParseResult {
    PreparedCheckpointError error;
    CheckpointLayoutIssue layout_issue;
    std::optional<PreparedCheckpoint> checkpoint;

    explicit operator bool() const noexcept {
        return error == PreparedCheckpointError::None && checkpoint.has_value();
    }
};

struct PreparedCheckpointExpectation {
    std::string_view package_id;
    std::string_view package_sha256;
    ArtifactIdentity artifact;
};

enum class PreparedCheckpointIdentityError : std::uint8_t {
    None,
    PackageIdMismatch,
    PackageDigestMismatch,
    ArtifactIdMismatch,
    ManifestDigestMismatch,
    ModelTypeMismatch,
    FormatMismatch,
    RepositoryMismatch,
    RevisionMismatch,
    ArtifactFileCountMismatch,
    WeightFileCountMismatch,
    TensorCountMismatch,
    TensorBytesMismatch,
};

PreparedCheckpointParseResult parse_prepared_checkpoint(std::span<const std::byte> bytes);

PreparedCheckpointIdentityError
validate_prepared_checkpoint_identity(const PreparedCheckpoint& checkpoint,
                                      const PreparedCheckpointExpectation& expectation) noexcept;

} // namespace tatara::model
