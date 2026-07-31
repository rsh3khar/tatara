#include "tatara/model/prepared_checkpoint.h"

#include <algorithm>
#include <limits>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace tatara::model {
namespace {

constexpr std::size_t kPrefixBytes = 48;
constexpr std::size_t kMaximumRecordBytes = 512ULL * 1024 * 1024;
constexpr std::uint32_t kMaximumStringBytes = 1024 * 1024;
constexpr std::uint32_t kMaximumArtifactFileCount = 1'000'000;
constexpr std::uint32_t kMaximumShardCount = 65'536;
constexpr std::uint32_t kMaximumTensorCount = 1'000'000;
constexpr std::uint32_t kMaximumTensorRank = 64;
constexpr std::size_t kMinimumShardRecordBytes = 97;
constexpr std::size_t kMinimumTensorRecordBytes = 37;

class Reader {
  public:
    explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

    template <typename Integer> bool read(Integer& value) noexcept {
        static_assert(std::is_unsigned_v<Integer>);
        if (remaining() < sizeof(Integer)) {
            return false;
        }
        value = 0;
        for (std::size_t index = 0; index < sizeof(Integer); ++index) {
            value |= static_cast<Integer>(std::to_integer<unsigned int>(bytes_[offset_ + index]))
                     << (index * 8);
        }
        offset_ += sizeof(Integer);
        return true;
    }

    bool read_bytes(std::span<const std::byte>& value, std::size_t size) noexcept {
        if (remaining() < size) {
            return false;
        }
        value = bytes_.subspan(offset_, size);
        offset_ += size;
        return true;
    }

    std::size_t remaining() const noexcept {
        return bytes_.size() - offset_;
    }

  private:
    std::span<const std::byte> bytes_;
    std::size_t offset_ = 0;
};

PreparedCheckpointParseResult failure(PreparedCheckpointError error,
                                      CheckpointLayoutIssue layout_issue = {}) {
    return {
        .error = error,
        .layout_issue = std::move(layout_issue),
        .checkpoint = std::nullopt,
    };
}

bool valid_utf8(std::span<const std::byte> bytes) noexcept {
    std::size_t index = 0;
    while (index < bytes.size()) {
        const auto first = std::to_integer<unsigned int>(bytes[index]);
        if (first == 0) {
            return false;
        }
        if (first <= 0x7f) {
            ++index;
            continue;
        }
        std::size_t continuation = 0;
        std::uint32_t code_point = 0;
        if (first >= 0xc2 && first <= 0xdf) {
            continuation = 1;
            code_point = first & 0x1f;
        } else if (first >= 0xe0 && first <= 0xef) {
            continuation = 2;
            code_point = first & 0x0f;
        } else if (first >= 0xf0 && first <= 0xf4) {
            continuation = 3;
            code_point = first & 0x07;
        } else {
            return false;
        }
        if (index + continuation >= bytes.size()) {
            return false;
        }
        for (std::size_t offset = 1; offset <= continuation; ++offset) {
            const auto next = std::to_integer<unsigned int>(bytes[index + offset]);
            if ((next & 0xc0) != 0x80) {
                return false;
            }
            code_point = (code_point << 6) | (next & 0x3f);
        }
        if ((continuation == 2 && code_point < 0x800) ||
            (continuation == 3 && code_point < 0x10000) ||
            (code_point >= 0xd800 && code_point <= 0xdfff) || code_point > 0x10ffff) {
            return false;
        }
        index += continuation + 1;
    }
    return true;
}

bool read_string(Reader& reader, std::string& value) {
    std::uint32_t size = 0;
    if (!reader.read(size) || size == 0 || size > kMaximumStringBytes) {
        return false;
    }
    std::span<const std::byte> bytes;
    if (!reader.read_bytes(bytes, size) || !valid_utf8(bytes)) {
        return false;
    }
    value.assign(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    return true;
}

bool valid_sha256(std::string_view value) noexcept {
    return value.size() == 64 && std::ranges::all_of(value, [](char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
           });
}

bool valid_shard_path(std::string_view path) noexcept {
    if (path.empty() || path.front() == '/' || path.back() == '/' ||
        path.find('\\') != std::string_view::npos) {
        return false;
    }
    std::size_t begin = 0;
    while (begin < path.size()) {
        const std::size_t end = path.find('/', begin);
        const std::string_view part = path.substr(begin, end - begin);
        if (part.empty() || part == "." || part == "..") {
            return false;
        }
        if (end == std::string_view::npos) {
            break;
        }
        begin = end + 1;
    }
    return true;
}

std::optional<TensorDataType> tensor_data_type(std::uint32_t value) noexcept {
    if (value < static_cast<std::uint32_t>(TensorDataType::Bool) ||
        value > static_cast<std::uint32_t>(TensorDataType::F8E5M2)) {
        return std::nullopt;
    }
    return static_cast<TensorDataType>(value);
}

bool add_overflows(std::uint64_t left, std::uint64_t right) noexcept {
    return left > std::numeric_limits<std::uint64_t>::max() - right;
}

} // namespace

PreparedCheckpoint::PreparedCheckpoint(PreparedCheckpointIdentity identity,
                                       std::vector<PreparedShard> shards,
                                       std::vector<TensorRecord> tensors,
                                       std::uint64_t tensor_payload_bytes)
    : identity_(std::move(identity)), shards_(std::move(shards)), tensors_(std::move(tensors)),
      tensor_payload_bytes_(tensor_payload_bytes) {}

const PreparedCheckpointIdentity& PreparedCheckpoint::identity() const noexcept {
    return identity_;
}

std::span<const PreparedShard> PreparedCheckpoint::shards() const noexcept {
    return shards_;
}

std::span<const TensorRecord> PreparedCheckpoint::tensors() const noexcept {
    return tensors_;
}

std::uint64_t PreparedCheckpoint::tensor_payload_bytes() const noexcept {
    return tensor_payload_bytes_;
}

PreparedCheckpointParseResult parse_prepared_checkpoint(std::span<const std::byte> bytes) {
    if (bytes.size() < kPrefixBytes) {
        return failure(PreparedCheckpointError::RecordTooSmall);
    }
    if (bytes.size() > kMaximumRecordBytes) {
        return failure(PreparedCheckpointError::RecordTooLarge);
    }

    Reader reader(bytes);
    std::span<const std::byte> magic;
    if (!reader.read_bytes(magic, kPreparedCheckpointMagic.size()) ||
        !std::ranges::equal(magic, kPreparedCheckpointMagic)) {
        return failure(PreparedCheckpointError::InvalidMagic);
    }
    std::uint32_t schema = 0;
    std::uint32_t flags = 0;
    std::uint64_t record_bytes = 0;
    std::uint32_t artifact_file_count = 0;
    std::uint32_t shard_count = 0;
    std::uint32_t tensor_count = 0;
    std::uint32_t reserved = 0;
    std::uint64_t tensor_payload_bytes = 0;
    if (!reader.read(schema) || !reader.read(flags) || !reader.read(record_bytes) ||
        !reader.read(artifact_file_count) || !reader.read(shard_count) ||
        !reader.read(tensor_count) || !reader.read(reserved) ||
        !reader.read(tensor_payload_bytes)) {
        return failure(PreparedCheckpointError::Truncated);
    }
    if (schema != kPreparedCheckpointSchemaVersion) {
        return failure(PreparedCheckpointError::UnsupportedSchema);
    }
    if (flags != 0 || reserved != 0) {
        return failure(PreparedCheckpointError::InvalidFlags);
    }
    if (record_bytes != bytes.size()) {
        return failure(PreparedCheckpointError::SizeMismatch);
    }
    if (artifact_file_count > kMaximumArtifactFileCount || shard_count == 0 ||
        shard_count > kMaximumShardCount || tensor_count == 0 ||
        tensor_count > kMaximumTensorCount || artifact_file_count < shard_count) {
        return failure(PreparedCheckpointError::InvalidCount);
    }

    PreparedCheckpointIdentity identity;
    identity.artifact_file_count = artifact_file_count;
    if (!read_string(reader, identity.package_id) ||
        !read_string(reader, identity.package_sha256) ||
        !read_string(reader, identity.artifact_id) ||
        !read_string(reader, identity.artifact_manifest_sha256) ||
        !read_string(reader, identity.model_type) || !read_string(reader, identity.format) ||
        !read_string(reader, identity.source_repository) ||
        !read_string(reader, identity.source_revision)) {
        return failure(PreparedCheckpointError::InvalidString);
    }
    if (!valid_sha256(identity.package_sha256) ||
        !valid_sha256(identity.artifact_manifest_sha256)) {
        return failure(PreparedCheckpointError::InvalidDigest);
    }
    if (shard_count > reader.remaining() / kMinimumShardRecordBytes ||
        tensor_count > reader.remaining() / kMinimumTensorRecordBytes) {
        return failure(PreparedCheckpointError::InvalidCount);
    }

    std::vector<PreparedShard> shards;
    shards.reserve(shard_count);
    std::unordered_set<std::string> shard_paths;
    shard_paths.reserve(shard_count);
    std::string previous_shard_path;
    for (std::uint32_t index = 0; index < shard_count; ++index) {
        PreparedShard shard;
        if (!read_string(reader, shard.path) || !read_string(reader, shard.sha256) ||
            !reader.read(shard.file_size_bytes) || !reader.read(shard.data_offset_bytes) ||
            !reader.read(shard.data_size_bytes)) {
            return failure(PreparedCheckpointError::Truncated);
        }
        if (!valid_shard_path(shard.path)) {
            return failure(PreparedCheckpointError::InvalidShardPath);
        }
        if (!shard_paths.emplace(shard.path).second) {
            return failure(PreparedCheckpointError::DuplicateShardPath);
        }
        if (!previous_shard_path.empty() && shard.path <= previous_shard_path) {
            return failure(PreparedCheckpointError::NonCanonicalOrder);
        }
        previous_shard_path = shard.path;
        if (!valid_sha256(shard.sha256)) {
            return failure(PreparedCheckpointError::InvalidDigest);
        }
        if (shard.data_offset_bytes < 10 ||
            add_overflows(shard.data_offset_bytes, shard.data_size_bytes) ||
            shard.data_offset_bytes + shard.data_size_bytes != shard.file_size_bytes) {
            return failure(PreparedCheckpointError::InvalidShardRegion);
        }
        shards.push_back(std::move(shard));
    }

    std::vector<TensorRecord> tensors;
    tensors.reserve(tensor_count);
    std::string previous_tensor_name;
    for (std::uint32_t index = 0; index < tensor_count; ++index) {
        TensorRecord tensor;
        std::uint32_t raw_data_type = 0;
        std::uint32_t shard = 0;
        std::uint32_t rank = 0;
        std::uint32_t tensor_reserved = 0;
        if (!read_string(reader, tensor.name) || !reader.read(raw_data_type) ||
            !reader.read(shard) || !reader.read(rank) || !reader.read(tensor_reserved) ||
            !reader.read(tensor.shard_offset_bytes) || !reader.read(tensor.size_bytes)) {
            return failure(PreparedCheckpointError::Truncated);
        }
        const auto data_type = tensor_data_type(raw_data_type);
        if (!data_type) {
            return failure(PreparedCheckpointError::InvalidDataType);
        }
        if (rank > kMaximumTensorRank || rank > reader.remaining() / sizeof(std::uint64_t)) {
            return failure(PreparedCheckpointError::InvalidTensorRank);
        }
        if (tensor_reserved != 0) {
            return failure(PreparedCheckpointError::InvalidReservedField);
        }
        if (!previous_tensor_name.empty() && tensor.name <= previous_tensor_name) {
            return failure(PreparedCheckpointError::NonCanonicalOrder);
        }
        previous_tensor_name = tensor.name;
        tensor.data_type = *data_type;
        tensor.shard = shard;
        tensor.shape.reserve(rank);
        for (std::uint32_t dimension = 0; dimension < rank; ++dimension) {
            std::uint64_t value = 0;
            if (!reader.read(value)) {
                return failure(PreparedCheckpointError::Truncated);
            }
            tensor.shape.push_back(value);
        }
        tensors.push_back(std::move(tensor));
    }
    if (reader.remaining() != 0) {
        return failure(PreparedCheckpointError::TrailingData);
    }

    std::vector<std::uint64_t> shard_sizes;
    shard_sizes.reserve(shards.size());
    for (const PreparedShard& shard : shards) {
        shard_sizes.push_back(shard.data_size_bytes);
    }
    const CheckpointExpectation expectation = {
        .shard_count = shard_count,
        .tensor_count = tensor_count,
        .tensor_payload_bytes = tensor_payload_bytes,
    };
    CheckpointLayoutIssue layout_issue =
        validate_checkpoint_layout(tensors, shard_sizes, expectation);
    if (layout_issue) {
        return failure(PreparedCheckpointError::InvalidLayout, std::move(layout_issue));
    }
    return {
        .error = PreparedCheckpointError::None,
        .layout_issue = {},
        .checkpoint = PreparedCheckpoint(std::move(identity), std::move(shards), std::move(tensors),
                                         tensor_payload_bytes),
    };
}

PreparedCheckpointIdentityError
validate_prepared_checkpoint_identity(const PreparedCheckpoint& checkpoint,
                                      const PreparedCheckpointExpectation& expectation) noexcept {
    const PreparedCheckpointIdentity& identity = checkpoint.identity();
    if (identity.package_id != expectation.package_id) {
        return PreparedCheckpointIdentityError::PackageIdMismatch;
    }
    if (identity.package_sha256 != expectation.package_sha256) {
        return PreparedCheckpointIdentityError::PackageDigestMismatch;
    }
    if (identity.artifact_id != expectation.artifact.id) {
        return PreparedCheckpointIdentityError::ArtifactIdMismatch;
    }
    if (identity.artifact_manifest_sha256 != expectation.artifact.manifest_sha256) {
        return PreparedCheckpointIdentityError::ManifestDigestMismatch;
    }
    if (identity.model_type != expectation.artifact.model_type) {
        return PreparedCheckpointIdentityError::ModelTypeMismatch;
    }
    if (identity.format != expectation.artifact.format) {
        return PreparedCheckpointIdentityError::FormatMismatch;
    }
    if (identity.source_repository != expectation.artifact.source_repository) {
        return PreparedCheckpointIdentityError::RepositoryMismatch;
    }
    if (identity.source_revision != expectation.artifact.source_revision) {
        return PreparedCheckpointIdentityError::RevisionMismatch;
    }
    if (identity.artifact_file_count != expectation.artifact.file_count) {
        return PreparedCheckpointIdentityError::ArtifactFileCountMismatch;
    }
    if (checkpoint.shards().size() != expectation.artifact.weight_file_count) {
        return PreparedCheckpointIdentityError::WeightFileCountMismatch;
    }
    if (checkpoint.tensors().size() != expectation.artifact.tensor_count) {
        return PreparedCheckpointIdentityError::TensorCountMismatch;
    }
    if (checkpoint.tensor_payload_bytes() != expectation.artifact.tensor_bytes) {
        return PreparedCheckpointIdentityError::TensorBytesMismatch;
    }
    return PreparedCheckpointIdentityError::None;
}

} // namespace tatara::model
