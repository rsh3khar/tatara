#include "tatara/model/image_population.h"

#include "tatara/generated/model_plan.h"
#include "tatara/model/sha256.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace tatara::model {
namespace {

constexpr std::size_t kPopulationChunkBytes = std::size_t{4} * 1024 * 1024;

struct PlacedExtent {
    std::size_t tensor;
    std::uint64_t file_begin;
    std::uint64_t file_end;
};

bool power_of_two(std::uint64_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

bool align_up(std::uint64_t value, std::uint64_t alignment, std::uint64_t& aligned) noexcept {
    const std::uint64_t mask = alignment - 1;
    if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
        return false;
    }
    aligned = (value + mask) & ~mask;
    return true;
}

bool checked_add(std::uint64_t left, std::uint64_t right, std::uint64_t& sum) noexcept {
    if (left > std::numeric_limits<std::uint64_t>::max() - right) {
        return false;
    }
    sum = left + right;
    return true;
}

bool excluded(std::string_view name, std::span<const std::string_view> excluded_prefixes) noexcept {
    return std::ranges::any_of(
        excluded_prefixes, [name](std::string_view prefix) { return name.starts_with(prefix); });
}

PopulateResult populate_failure(PopulateError error, std::size_t shard_index = 0,
                                int system_error = 0, std::string computed_digest_hex = {}) {
    return {
        .error = error,
        .shard_index = shard_index,
        .system_error = system_error,
        .computed_digest_hex = std::move(computed_digest_hex),
        .shard_digests_hex = {},
    };
}

} // namespace

ImageLayoutResult plan_image_layout(std::span<const TensorRecord> tensors,
                                    std::uint64_t alignment_bytes,
                                    std::span<const std::string_view> excluded_prefixes) {
    if (tensors.empty()) {
        return {.error = ImageLayoutError::NoTensors, .layout = std::nullopt};
    }
    if (!power_of_two(alignment_bytes)) {
        return {.error = ImageLayoutError::InvalidAlignment, .layout = std::nullopt};
    }
    ImageLayout layout{
        .tensor_offsets = {},
        .total_bytes = 0,
        .alignment_bytes = alignment_bytes,
    };
    layout.tensor_offsets.reserve(tensors.size());
    std::uint64_t cursor = 0;
    std::size_t resident_tensors = 0;
    for (const TensorRecord& tensor : tensors) {
        if (excluded(tensor.name, excluded_prefixes)) {
            layout.tensor_offsets.push_back(kExcludedTensorOffset);
            continue;
        }
        std::uint64_t offset = 0;
        if (!align_up(cursor, alignment_bytes, offset) ||
            !checked_add(offset, tensor.size_bytes, cursor)) {
            return {.error = ImageLayoutError::OffsetOverflow, .layout = std::nullopt};
        }
        layout.tensor_offsets.push_back(offset);
        ++resident_tensors;
    }
    if (resident_tensors == 0) {
        return {.error = ImageLayoutError::NoResidentTensors, .layout = std::nullopt};
    }
    layout.total_bytes = cursor;
    return {.error = ImageLayoutError::None, .layout = std::move(layout)};
}

ImageLayoutResult plan_image_layout(std::span<const TensorRecord> tensors,
                                    std::uint64_t alignment_bytes) {
    return plan_image_layout(tensors, alignment_bytes, generated::kExcludedTensorPrefixes);
}

PopulateResult populate_model_image(const PreparedCheckpoint& checkpoint, const ImageLayout& layout,
                                    std::span<const SourceShardView> shards,
                                    std::span<std::byte> destination) {
    const std::span<const TensorRecord> tensors = checkpoint.tensors();
    const std::span<const PreparedShard> prepared_shards = checkpoint.shards();

    if (layout.tensor_offsets.size() != tensors.size()) {
        return populate_failure(PopulateError::LayoutTensorCountMismatch);
    }
    if (!power_of_two(layout.alignment_bytes)) {
        return populate_failure(PopulateError::LayoutMismatch);
    }
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        const std::uint64_t offset = layout.tensor_offsets[index];
        if (offset == kExcludedTensorOffset) {
            continue;
        }
        std::uint64_t extent_end = 0;
        if (offset % layout.alignment_bytes != 0 ||
            !checked_add(offset, tensors[index].size_bytes, extent_end) ||
            extent_end > layout.total_bytes) {
            return populate_failure(PopulateError::LayoutMismatch);
        }
    }
    if (destination.size() < layout.total_bytes) {
        return populate_failure(PopulateError::DestinationTooSmall);
    }
    if (reinterpret_cast<std::uintptr_t>(destination.data()) % layout.alignment_bytes != 0) {
        return populate_failure(PopulateError::MisalignedDestination);
    }
    if (shards.size() != prepared_shards.size()) {
        return populate_failure(PopulateError::ShardViewCountMismatch);
    }
    for (std::size_t index = 0; index < shards.size(); ++index) {
        const SourceShardView& view = shards[index];
        const PreparedShard& prepared = prepared_shards[index];
        if (view.file_size_bytes != prepared.file_size_bytes ||
            view.data_offset_bytes != prepared.data_offset_bytes ||
            view.data_size_bytes != prepared.data_size_bytes) {
            return populate_failure(PopulateError::ShardViewMismatch, index);
        }
    }

    // Excluded tensors are still hashed as part of their shard's full-file
    // digest; they are simply never placed.
    std::vector<std::vector<PlacedExtent>> extents_by_shard(prepared_shards.size());
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        if (layout.tensor_offsets[index] == kExcludedTensorOffset) {
            continue;
        }
        const TensorRecord& tensor = tensors[index];
        const std::uint64_t file_begin =
            prepared_shards[tensor.shard].data_offset_bytes + tensor.shard_offset_bytes;
        extents_by_shard[tensor.shard].push_back({
            .tensor = index,
            .file_begin = file_begin,
            .file_end = file_begin + tensor.size_bytes,
        });
    }
    for (std::vector<PlacedExtent>& extents : extents_by_shard) {
        std::sort(extents.begin(), extents.end(),
                  [](const PlacedExtent& left, const PlacedExtent& right) {
                      return left.file_begin < right.file_begin;
                  });
    }

    std::vector<std::byte> chunk(kPopulationChunkBytes);
    std::vector<std::string> shard_digests;
    shard_digests.reserve(prepared_shards.size());
    Sha256 hasher;

    for (std::size_t shard_index = 0; shard_index < prepared_shards.size(); ++shard_index) {
        const SourceShardView& view = shards[shard_index];
        const std::vector<PlacedExtent>& extents = extents_by_shard[shard_index];
        std::size_t next_extent = 0;
        std::uint64_t position = 0;

        while (position < view.file_size_bytes) {
            const std::uint64_t remaining = view.file_size_bytes - position;
            const std::size_t wanted =
                remaining < chunk.size() ? static_cast<std::size_t>(remaining) : chunk.size();
            std::size_t received = 0;
            while (received < wanted) {
                const ::ssize_t outcome =
                    ::pread(view.borrowed_file_descriptor, chunk.data() + received,
                            wanted - received, static_cast<::off_t>(position + received));
                if (outcome < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    return populate_failure(PopulateError::ReadFailed, shard_index, errno);
                }
                if (outcome == 0) {
                    return populate_failure(PopulateError::ShardTruncated, shard_index);
                }
                received += static_cast<std::size_t>(outcome);
            }
            hasher.update(std::span<const std::byte>(chunk.data(), wanted));

            const std::uint64_t chunk_begin = position;
            const std::uint64_t chunk_end = position + wanted;
            for (std::size_t extent = next_extent; extent < extents.size(); ++extent) {
                const PlacedExtent& placed = extents[extent];
                if (placed.file_begin >= chunk_end) {
                    break;
                }
                if (placed.file_end <= chunk_begin) {
                    next_extent = extent + 1;
                    continue;
                }
                const std::uint64_t copy_begin = std::max(placed.file_begin, chunk_begin);
                const std::uint64_t copy_end = std::min(placed.file_end, chunk_end);
                const std::uint64_t destination_offset =
                    layout.tensor_offsets[placed.tensor] + (copy_begin - placed.file_begin);
                std::memcpy(destination.data() + destination_offset,
                            chunk.data() + (copy_begin - chunk_begin),
                            static_cast<std::size_t>(copy_end - copy_begin));
                if (placed.file_end <= chunk_end) {
                    next_extent = extent + 1;
                }
            }
            position = chunk_end;
        }

        std::string computed = sha256_hex(hasher.finish());
        if (computed != prepared_shards[shard_index].sha256) {
            return populate_failure(PopulateError::DigestMismatch, shard_index, 0,
                                    std::move(computed));
        }
        shard_digests.push_back(std::move(computed));
    }

    return {
        .error = PopulateError::None,
        .shard_index = 0,
        .system_error = 0,
        .computed_digest_hex = {},
        .shard_digests_hex = std::move(shard_digests),
    };
}

} // namespace tatara::model
