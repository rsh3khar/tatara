#include "tatara/model/checkpoint_layout.h"
#include "tatara/model/model_image.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using tatara::model::CheckpointExpectation;
using tatara::model::CheckpointLayoutError;
using tatara::model::ModelImage;
using tatara::model::ModelImageError;
using tatara::model::TensorDataType;
using tatara::model::TensorRecord;
using tatara::model::validate_checkpoint_layout;

std::vector<TensorRecord> valid_tensors() {
    return {
        {
            .name = "a",
            .data_type = TensorDataType::BF16,
            .shape = {2, 4},
            .shard = 0,
            .shard_offset_bytes = 16,
            .size_bytes = 16,
        },
        {
            .name = "b",
            .data_type = TensorDataType::U32,
            .shape = {2},
            .shard = 0,
            .shard_offset_bytes = 64,
            .size_bytes = 8,
        },
        {
            .name = "c",
            .data_type = TensorDataType::I8,
            .shape = {4},
            .shard = 1,
            .shard_offset_bytes = 8,
            .size_bytes = 4,
        },
    };
}

CheckpointExpectation expectation() {
    return {.shard_count = 2, .tensor_count = 3, .tensor_payload_bytes = 28};
}

bool has_error(std::vector<TensorRecord> tensors, CheckpointLayoutError expected) {
    constexpr std::array<std::uint64_t, 2> shard_sizes = {128, 64};
    return validate_checkpoint_layout(tensors, shard_sizes, expectation()).error == expected;
}

struct ReleaseContext {
    std::byte* data;
    int* releases;
};

void release_image(void* raw_context) noexcept {
    auto& context = *static_cast<ReleaseContext*>(raw_context);
    ::operator delete(context.data, std::align_val_t(64));
    ++*context.releases;
}

void no_op_release(void*) noexcept {}

int test_layout_failures() {
    constexpr std::array<std::uint64_t, 2> shard_sizes = {128, 64};
    if (validate_checkpoint_layout(valid_tensors(), shard_sizes, expectation())) {
        return 1;
    }

    auto duplicate = valid_tensors();
    duplicate[1].name = "a";
    if (!has_error(std::move(duplicate), CheckpointLayoutError::DuplicateTensorName)) {
        return 2;
    }

    auto shape_overflow = valid_tensors();
    shape_overflow[0].shape = {std::numeric_limits<std::uint64_t>::max(), 2};
    if (!has_error(std::move(shape_overflow), CheckpointLayoutError::ShapeOverflow)) {
        return 3;
    }

    auto range_overflow = valid_tensors();
    range_overflow[1].shard_offset_bytes = std::numeric_limits<std::uint64_t>::max() - 3;
    if (!has_error(std::move(range_overflow), CheckpointLayoutError::RangeOverflow)) {
        return 4;
    }

    auto out_of_range = valid_tensors();
    out_of_range[2].shard_offset_bytes = 62;
    if (!has_error(std::move(out_of_range), CheckpointLayoutError::TensorOutOfRange)) {
        return 5;
    }

    auto overlap = valid_tensors();
    overlap[1].shard_offset_bytes = 24;
    if (!has_error(std::move(overlap), CheckpointLayoutError::TensorOverlap)) {
        return 6;
    }

    auto size_mismatch = valid_tensors();
    size_mismatch[0].size_bytes = 8;
    if (!has_error(std::move(size_mismatch), CheckpointLayoutError::TensorSizeMismatch)) {
        return 7;
    }

    const std::vector<TensorRecord> overflowing_total = {
        {
            .name = "first",
            .data_type = TensorDataType::U8,
            .shape = {std::numeric_limits<std::uint64_t>::max()},
            .shard = 0,
            .shard_offset_bytes = 0,
            .size_bytes = std::numeric_limits<std::uint64_t>::max(),
        },
        {
            .name = "second",
            .data_type = TensorDataType::U8,
            .shape = {1},
            .shard = 1,
            .shard_offset_bytes = 0,
            .size_bytes = 1,
        },
    };
    constexpr std::array<std::uint64_t, 2> maximum_shards = {
        std::numeric_limits<std::uint64_t>::max(),
        1,
    };
    const CheckpointExpectation overflowing_expectation = {
        .shard_count = 2,
        .tensor_count = 2,
        .tensor_payload_bytes = 0,
    };
    if (validate_checkpoint_layout(overflowing_total, maximum_shards, overflowing_expectation)
            .error != CheckpointLayoutError::TotalBytesOverflow) {
        return 8;
    }
    return 0;
}

int test_model_image_ownership() {
    static_assert(!std::is_copy_constructible_v<ModelImage>);
    static_assert(!std::is_copy_assignable_v<ModelImage>);
    static_assert(std::is_nothrow_move_constructible_v<ModelImage>);
    static_assert(std::is_nothrow_move_assignable_v<ModelImage>);

    alignas(64) std::array<std::byte, 64> local_bytes{};
    int local_context = 0;
    if (ModelImage::adopt(nullptr, 64, 64, &local_context, no_op_release).error !=
        ModelImageError::NullData) {
        return 9;
    }
    if (ModelImage::adopt(local_bytes.data(), 0, 64, &local_context, no_op_release).error !=
        ModelImageError::ZeroSize) {
        return 10;
    }
    if (ModelImage::adopt(local_bytes.data(), 64, 3, &local_context, no_op_release).error !=
        ModelImageError::InvalidAlignment) {
        return 11;
    }
    if (ModelImage::adopt(local_bytes.data() + 1, 63, 64, &local_context, no_op_release).error !=
        ModelImageError::MisalignedData) {
        return 12;
    }
    if (ModelImage::adopt(local_bytes.data(), 64, 64, nullptr, no_op_release).error !=
        ModelImageError::NullContext) {
        return 13;
    }
    if (ModelImage::adopt(local_bytes.data(), 64, 64, &local_context, nullptr).error !=
        ModelImageError::MissingRelease) {
        return 14;
    }

    int releases = 0;
    auto* bytes = static_cast<std::byte*>(::operator new(64, std::align_val_t(64)));
    ReleaseContext context{.data = bytes, .releases = &releases};
    auto adoption = ModelImage::adopt(bytes, 64, 64, &context, release_image);
    if (!adoption || !adoption.image || adoption.image->data() != bytes) {
        release_image(&context);
        return 15;
    }

    {
        ModelImage first = std::move(*adoption.image);
        ModelImage second = std::move(first);
        if (first || !second || releases != 0) {
            return 16;
        }
    }
    if (releases != 1) {
        return 17;
    }

    auto* unaligned = static_cast<std::byte*>(::operator new(65, std::align_val_t(64)));
    const auto invalid = ModelImage::adopt(unaligned + 1, 64, 64, nullptr, release_image);
    ::operator delete(unaligned, std::align_val_t(64));
    if (invalid.error != ModelImageError::MisalignedData || invalid.image) {
        return 18;
    }
    return 0;
}

} // namespace

int main() {
    if (const int result = test_layout_failures(); result != 0) {
        return result;
    }
    return test_model_image_ownership();
}
