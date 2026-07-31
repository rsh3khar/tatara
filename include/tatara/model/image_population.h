#pragma once

#include "tatara/model/checkpoint_layout.h"
#include "tatara/model/prepared_checkpoint.h"
#include "tatara/model/source_shards.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace tatara::model {

// The sealed reference arena packed tensors at 256-byte boundaries, which also
// conservatively satisfies Metal buffer-offset binding for the first package.
inline constexpr std::uint64_t kTensorAlignmentBytes = 256;

// The offset a tensor the package excludes from serving carries instead of a
// place in the image. Population never writes it and no binding may read it.
inline constexpr std::uint64_t kExcludedTensorOffset = std::numeric_limits<std::uint64_t>::max();

// One offset per record tensor, so a resolved record index addresses the image
// directly. Excluded tensors hold kExcludedTensorOffset and occupy no bytes.
struct ImageLayout {
    std::vector<std::uint64_t> tensor_offsets;
    std::uint64_t total_bytes;
    std::uint64_t alignment_bytes;
};

enum class ImageLayoutError : std::uint8_t {
    None,
    NoTensors,
    InvalidAlignment,
    OffsetOverflow,
    NoResidentTensors,
};

struct ImageLayoutResult {
    ImageLayoutError error;
    std::optional<ImageLayout> layout;

    explicit operator bool() const noexcept {
        return error == ImageLayoutError::None && layout.has_value();
    }
};

// Maps every tensor of a parsed record, in record order, to an aligned offset,
// skipping those whose name starts with one of the excluded prefixes. Pure
// arithmetic over the records: no file, shard, or identity is consulted.
ImageLayoutResult plan_image_layout(std::span<const TensorRecord> tensors,
                                    std::uint64_t alignment_bytes,
                                    std::span<const std::string_view> excluded_prefixes);

// The same planning against the compiled package's declared exclusions, which
// is what serving that package uses.
ImageLayoutResult plan_image_layout(std::span<const TensorRecord> tensors,
                                    std::uint64_t alignment_bytes);

enum class PopulateError : std::uint8_t {
    None,
    LayoutTensorCountMismatch,
    LayoutMismatch,
    DestinationTooSmall,
    MisalignedDestination,
    ShardViewCountMismatch,
    ShardViewMismatch,
    ReadFailed,
    ShardTruncated,
    DigestMismatch,
};

struct PopulateResult {
    PopulateError error;
    std::size_t shard_index;
    int system_error;
    std::string computed_digest_hex;
    std::vector<std::string> shard_digests_hex;

    explicit operator bool() const noexcept {
        return error == PopulateError::None;
    }
};

PopulateResult populate_model_image(const PreparedCheckpoint& checkpoint, const ImageLayout& layout,
                                    std::span<const SourceShardView> shards,
                                    std::span<std::byte> destination);

} // namespace tatara::model
