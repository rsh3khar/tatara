#include "tatara/model/image_population.h"

#include "tatara/model/model_image.h"
#include "tatara/model/prepared_checkpoint.h"
#include "tatara/model/source_shards.h"

#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

using tatara::model::ImageLayout;
using tatara::model::ImageLayoutError;
using tatara::model::kExcludedTensorOffset;
using tatara::model::kTensorAlignmentBytes;
using tatara::model::ModelImage;
using tatara::model::ModelImageError;
using tatara::model::open_source_shards;
using tatara::model::parse_prepared_checkpoint;
using tatara::model::plan_image_layout;
using tatara::model::populate_model_image;
using tatara::model::PopulateError;
using tatara::model::PreparedCheckpoint;
using tatara::model::SourceShardView;
using tatara::model::TensorDataType;
using tatara::model::TensorRecord;

constexpr std::byte kSentinel{0xEE};
constexpr char kRecordFileName[] = "population.tatara";

class TemporaryDirectory {
  public:
    TemporaryDirectory() {
        for (int attempt = 0; attempt < 100; ++attempt) {
            std::error_code error;
            const auto candidate = std::filesystem::temp_directory_path() /
                                   ("tatara-image-population-" + std::to_string(::getpid()) + "-" +
                                    std::to_string(attempt));
            if (std::filesystem::create_directory(candidate, error)) {
                path_ = candidate;
                break;
            }
        }
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    const std::filesystem::path& path() const noexcept {
        return path_;
    }

  private:
    std::filesystem::path path_;
};

class AlignedBuffer {
  public:
    AlignedBuffer(std::size_t size_bytes, std::size_t alignment_bytes)
        : data_(static_cast<std::byte*>(
              ::operator new(size_bytes, std::align_val_t{alignment_bytes}))),
          size_bytes_(size_bytes), alignment_bytes_(alignment_bytes) {
        std::fill(data_, data_ + size_bytes_, kSentinel);
    }

    ~AlignedBuffer() {
        ::operator delete(data_, std::align_val_t{alignment_bytes_});
    }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    std::span<std::byte> span() noexcept {
        return {data_, size_bytes_};
    }

  private:
    std::byte* data_;
    std::size_t size_bytes_;
    std::size_t alignment_bytes_;
};

std::vector<std::byte> read_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    std::vector<char> content((std::istreambuf_iterator<char>(stream)),
                              std::istreambuf_iterator<char>());
    const auto* begin = reinterpret_cast<const std::byte*>(content.data());
    return {begin, begin + content.size()};
}

void write_file(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
}

TensorRecord synthetic_tensor(std::string name, std::uint64_t size_bytes) {
    return {
        .name = std::move(name),
        .data_type = TensorDataType::U8,
        .shape = {size_bytes},
        .shard = 0,
        .shard_offset_bytes = 0,
        .size_bytes = size_bytes,
    };
}

int test_layout_planning() {
    const std::vector<TensorRecord> tensors = {
        synthetic_tensor("a", 5),
        synthetic_tensor("b", 16),
        synthetic_tensor("c", 1),
    };
    const auto layout = plan_image_layout(tensors, 8);
    if (!layout || layout.layout->tensor_offsets != std::vector<std::uint64_t>{0, 8, 24} ||
        layout.layout->total_bytes != 25 || layout.layout->alignment_bytes != 8) {
        return 1;
    }
    if (plan_image_layout(tensors, 0).error != ImageLayoutError::InvalidAlignment ||
        plan_image_layout(tensors, 6).error != ImageLayoutError::InvalidAlignment) {
        return 2;
    }
    if (plan_image_layout({}, 8).error != ImageLayoutError::NoTensors) {
        return 3;
    }
    const std::vector<TensorRecord> huge = {
        synthetic_tensor("a", std::numeric_limits<std::uint64_t>::max() - 3),
        synthetic_tensor("b", 16),
    };
    if (plan_image_layout(huge, 8).error != ImageLayoutError::OffsetOverflow) {
        return 4;
    }
    return 0;
}

// An excluded tensor keeps its record index but takes no place in the image,
// and the tensors around it pack exactly as if it were never in the record.
int test_excluded_layout_planning() {
    const std::vector<TensorRecord> tensors = {
        synthetic_tensor("a", 5),  synthetic_tensor("vision_tower.patch_embed", 4096),
        synthetic_tensor("b", 16), synthetic_tensor("vision_tower.blocks.0", 64),
        synthetic_tensor("c", 1),
    };
    constexpr std::string_view kExcluded[] = {"vision_tower."};
    const auto layout = plan_image_layout(tensors, 8, kExcluded);
    if (!layout ||
        layout.layout->tensor_offsets !=
            std::vector<std::uint64_t>{0, kExcludedTensorOffset, 8, kExcludedTensorOffset, 24} ||
        layout.layout->total_bytes != 25) {
        return 5;
    }

    // The compiled package declares its own exclusions, so the two-argument
    // form must drop the same names without being told.
    const auto package_layout = plan_image_layout(tensors, 8);
    if (!package_layout || package_layout.layout->tensor_offsets != layout.layout->tensor_offsets) {
        return 6;
    }

    constexpr std::string_view kEverything[] = {""};
    if (plan_image_layout(tensors, 8, kEverything).error != ImageLayoutError::NoResidentTensors) {
        return 7;
    }
    return 0;
}

struct FixtureState {
    std::filesystem::path root;
    std::vector<std::byte> record_bytes;
    std::optional<PreparedCheckpoint> checkpoint;
};

int load_fixture(const std::filesystem::path& root, FixtureState& state) {
    state.root = root;
    state.record_bytes = read_file(root / kRecordFileName);
    auto parsed = parse_prepared_checkpoint(state.record_bytes);
    if (!parsed) {
        return 1;
    }
    state.checkpoint.emplace(std::move(*parsed.checkpoint));
    return 0;
}

int test_success(const FixtureState& state) {
    const PreparedCheckpoint& checkpoint = *state.checkpoint;
    auto layout = plan_image_layout(checkpoint.tensors(), kTensorAlignmentBytes);
    if (!layout) {
        return 10;
    }
    auto shards = open_source_shards(state.root.string(), checkpoint.shards());
    if (!shards) {
        return 11;
    }
    AlignedBuffer destination(layout.layout->total_bytes, kTensorAlignmentBytes);
    auto result = populate_model_image(checkpoint, *layout.layout, shards.shard_set->shards(),
                                       destination.span());
    if (!result) {
        return 12;
    }
    if (result.shard_digests_hex.size() != checkpoint.shards().size()) {
        return 13;
    }
    for (std::size_t index = 0; index < checkpoint.shards().size(); ++index) {
        if (result.shard_digests_hex[index] != checkpoint.shards()[index].sha256) {
            return 14;
        }
    }

    for (std::size_t index = 0; index < checkpoint.tensors().size(); ++index) {
        const auto& tensor = checkpoint.tensors()[index];
        const auto& shard = checkpoint.shards()[tensor.shard];
        const auto file_bytes = read_file(state.root / shard.path);
        const std::size_t source_begin =
            static_cast<std::size_t>(shard.data_offset_bytes + tensor.shard_offset_bytes);
        const std::byte* placed = destination.span().data() + layout.layout->tensor_offsets[index];
        for (std::size_t offset = 0; offset < tensor.size_bytes; ++offset) {
            if (placed[offset] != file_bytes[source_begin + offset]) {
                return 15;
            }
        }
        const std::uint64_t padding_position =
            layout.layout->tensor_offsets[index] + tensor.size_bytes;
        if (padding_position < layout.layout->total_bytes &&
            destination.span()[static_cast<std::size_t>(padding_position)] != kSentinel) {
            return 16;
        }
    }
    return 0;
}

// Excluding a tensor removes only its payload: every other tensor still lands
// byte-exact, and each shard's full-file digest is unchanged because population
// hashes the whole file whether or not it places what it reads.
int test_excluded_population(const FixtureState& state) {
    const PreparedCheckpoint& checkpoint = *state.checkpoint;
    constexpr std::string_view kExcluded[] = {"tensor-b"};
    auto full = plan_image_layout(checkpoint.tensors(), kTensorAlignmentBytes);
    auto layout = plan_image_layout(checkpoint.tensors(), kTensorAlignmentBytes, kExcluded);
    auto shards = open_source_shards(state.root.string(), checkpoint.shards());
    if (!full || !layout || !shards) {
        return 50;
    }
    if (layout.layout->total_bytes >= full.layout->total_bytes) {
        return 51;
    }

    AlignedBuffer destination(layout.layout->total_bytes, kTensorAlignmentBytes);
    auto result = populate_model_image(checkpoint, *layout.layout, shards.shard_set->shards(),
                                       destination.span());
    if (!result) {
        return 52;
    }
    for (std::size_t index = 0; index < checkpoint.shards().size(); ++index) {
        if (result.shard_digests_hex[index] != checkpoint.shards()[index].sha256) {
            return 53;
        }
    }

    for (std::size_t index = 0; index < checkpoint.tensors().size(); ++index) {
        const auto& tensor = checkpoint.tensors()[index];
        const std::uint64_t offset = layout.layout->tensor_offsets[index];
        if (tensor.name.starts_with(kExcluded[0])) {
            if (offset != kExcludedTensorOffset) {
                return 54;
            }
            continue;
        }
        const auto& shard = checkpoint.shards()[tensor.shard];
        const auto file_bytes = read_file(state.root / shard.path);
        const std::size_t source_begin =
            static_cast<std::size_t>(shard.data_offset_bytes + tensor.shard_offset_bytes);
        const std::byte* placed = destination.span().data() + offset;
        for (std::size_t byte = 0; byte < tensor.size_bytes; ++byte) {
            if (placed[byte] != file_bytes[source_begin + byte]) {
                return 55;
            }
        }
    }
    return 0;
}

int corrupt_and_expect_digest_mismatch(const FixtureState& state, std::size_t shard_index,
                                       std::uint64_t flip_position) {
    TemporaryDirectory copy_root;
    if (copy_root.path().empty()) {
        return 20;
    }
    const PreparedCheckpoint& checkpoint = *state.checkpoint;
    write_file(copy_root.path() / kRecordFileName, state.record_bytes);
    for (const auto& shard : checkpoint.shards()) {
        auto bytes = read_file(state.root / shard.path);
        write_file(copy_root.path() / shard.path, bytes);
    }
    const auto& target = checkpoint.shards()[shard_index];
    auto corrupted = read_file(state.root / target.path);
    corrupted[static_cast<std::size_t>(flip_position)] ^= std::byte{0x01};
    write_file(copy_root.path() / target.path, corrupted);

    auto layout = plan_image_layout(checkpoint.tensors(), kTensorAlignmentBytes);
    auto shards = open_source_shards(copy_root.path().string(), checkpoint.shards());
    if (!layout || !shards) {
        return 21;
    }
    AlignedBuffer destination(layout.layout->total_bytes, kTensorAlignmentBytes);
    auto result = populate_model_image(checkpoint, *layout.layout, shards.shard_set->shards(),
                                       destination.span());
    if (result.error != PopulateError::DigestMismatch || result.shard_index != shard_index) {
        return 22;
    }
    if (result.computed_digest_hex.size() != 64 || result.computed_digest_hex == target.sha256) {
        return 23;
    }
    return 0;
}

int test_rejections(const FixtureState& state) {
    const PreparedCheckpoint& checkpoint = *state.checkpoint;
    auto layout = plan_image_layout(checkpoint.tensors(), kTensorAlignmentBytes);
    auto shards = open_source_shards(state.root.string(), checkpoint.shards());
    if (!layout || !shards) {
        return 30;
    }
    const auto views = shards.shard_set->shards();
    AlignedBuffer destination(layout.layout->total_bytes + kTensorAlignmentBytes,
                              kTensorAlignmentBytes);

    auto small = destination.span().subspan(0, layout.layout->total_bytes - 1);
    if (populate_model_image(checkpoint, *layout.layout, views, small).error !=
        PopulateError::DestinationTooSmall) {
        return 31;
    }
    auto shifted = destination.span().subspan(1, layout.layout->total_bytes);
    if (populate_model_image(checkpoint, *layout.layout, views, shifted).error !=
        PopulateError::MisalignedDestination) {
        return 32;
    }
    if (populate_model_image(checkpoint, *layout.layout, views.subspan(0, 1), destination.span())
            .error != PopulateError::ShardViewCountMismatch) {
        return 33;
    }

    std::vector<SourceShardView> mismatched(views.begin(), views.end());
    mismatched[1].file_size_bytes += 1;
    auto mismatch_result =
        populate_model_image(checkpoint, *layout.layout, mismatched, destination.span());
    if (mismatch_result.error != PopulateError::ShardViewMismatch ||
        mismatch_result.shard_index != 1) {
        return 34;
    }

    ImageLayout short_layout = *layout.layout;
    short_layout.tensor_offsets.pop_back();
    if (populate_model_image(checkpoint, short_layout, views, destination.span()).error !=
        PopulateError::LayoutTensorCountMismatch) {
        return 35;
    }

    ImageLayout misaligned_layout = *layout.layout;
    misaligned_layout.tensor_offsets[0] = 3;
    if (populate_model_image(checkpoint, misaligned_layout, views, destination.span()).error !=
        PopulateError::LayoutMismatch) {
        return 36;
    }

    ImageLayout overflowing_layout = *layout.layout;
    overflowing_layout.total_bytes = 8;
    if (populate_model_image(checkpoint, overflowing_layout, views, destination.span()).error !=
        PopulateError::LayoutMismatch) {
        return 37;
    }
    return 0;
}

int test_adoption_after_population(const FixtureState& state) {
    const PreparedCheckpoint& checkpoint = *state.checkpoint;
    auto layout = plan_image_layout(checkpoint.tensors(), kTensorAlignmentBytes);
    auto shards = open_source_shards(state.root.string(), checkpoint.shards());
    if (!layout || !shards) {
        return 40;
    }
    AlignedBuffer destination(layout.layout->total_bytes, kTensorAlignmentBytes);
    if (!populate_model_image(checkpoint, *layout.layout, shards.shard_set->shards(),
                              destination.span())) {
        return 41;
    }
    int releases = 0;
    {
        auto adoption = ModelImage::adopt(
            destination.span().data(), destination.span().size(), kTensorAlignmentBytes, &releases,
            [](void* context) noexcept { ++*static_cast<int*>(context); });
        if (!adoption || releases != 0) {
            return 42;
        }
    }
    return releases == 1 ? 0 : 43;
}

} // namespace

int main(int argument_count, char** arguments) {
    if (const int result = test_layout_planning(); result != 0) {
        return result;
    }
    if (const int result = test_excluded_layout_planning(); result != 0) {
        return result;
    }
    if (argument_count != 2) {
        return 90;
    }
    FixtureState state;
    if (load_fixture(arguments[1], state) != 0) {
        return 91;
    }
    if (const int result = test_success(state); result != 0) {
        return result;
    }
    if (const int result = test_excluded_population(state); result != 0) {
        return result;
    }
    const auto& first_tensor = state.checkpoint->tensors().front();
    const auto& first_shard = state.checkpoint->shards()[first_tensor.shard];
    const std::uint64_t extent_position =
        first_shard.data_offset_bytes + first_tensor.shard_offset_bytes;
    if (const int result =
            corrupt_and_expect_digest_mismatch(state, first_tensor.shard, extent_position);
        result != 0) {
        return result;
    }
    if (const int result = corrupt_and_expect_digest_mismatch(state, 0, 5); result != 0) {
        return result;
    }
    if (const int result = corrupt_and_expect_digest_mismatch(state, 0, 70); result != 0) {
        return result;
    }
    if (const int result = test_rejections(state); result != 0) {
        return result;
    }
    return test_adoption_after_population(state);
}
