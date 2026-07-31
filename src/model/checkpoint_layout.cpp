#include "tatara/model/checkpoint_layout.h"

#include <algorithm>
#include <limits>
#include <string_view>
#include <unordered_set>

namespace tatara::model {
namespace {

struct TensorExtent {
    std::size_t shard;
    std::uint64_t begin;
    std::uint64_t end;
    std::string_view name;
};

bool multiply_overflows(std::uint64_t left, std::uint64_t right) noexcept {
    return right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right;
}

bool add_overflows(std::uint64_t left, std::uint64_t right) noexcept {
    return left > std::numeric_limits<std::uint64_t>::max() - right;
}

CheckpointLayoutIssue issue(CheckpointLayoutError error, std::string_view tensor = {}) {
    return {.error = error, .tensor = std::string(tensor)};
}

} // namespace

std::uint64_t tensor_data_type_bytes(TensorDataType data_type) noexcept {
    switch (data_type) {
    case TensorDataType::Bool:
    case TensorDataType::U8:
    case TensorDataType::I8:
    case TensorDataType::F8E4M3:
    case TensorDataType::F8E5M2:
        return 1;
    case TensorDataType::F16:
    case TensorDataType::BF16:
    case TensorDataType::U16:
    case TensorDataType::I16:
        return 2;
    case TensorDataType::F32:
    case TensorDataType::U32:
    case TensorDataType::I32:
        return 4;
    case TensorDataType::F64:
    case TensorDataType::U64:
    case TensorDataType::I64:
        return 8;
    }
    return 0;
}

CheckpointLayoutIssue validate_checkpoint_layout(std::span<const TensorRecord> tensors,
                                                 std::span<const std::uint64_t> shard_sizes,
                                                 const CheckpointExpectation& expectation) {
    if (shard_sizes.size() != expectation.shard_count) {
        return issue(CheckpointLayoutError::ShardCountMismatch);
    }
    if (tensors.size() != expectation.tensor_count) {
        return issue(CheckpointLayoutError::TensorCountMismatch);
    }

    std::unordered_set<std::string_view> names;
    names.reserve(tensors.size());
    std::vector<TensorExtent> extents;
    extents.reserve(tensors.size());
    std::uint64_t total_bytes = 0;

    for (const TensorRecord& tensor : tensors) {
        if (tensor.name.empty()) {
            return issue(CheckpointLayoutError::EmptyTensorName);
        }
        if (!names.emplace(tensor.name).second) {
            return issue(CheckpointLayoutError::DuplicateTensorName, tensor.name);
        }
        if (tensor.shard >= shard_sizes.size()) {
            return issue(CheckpointLayoutError::InvalidShard, tensor.name);
        }

        std::uint64_t element_count = 1;
        for (const std::uint64_t dimension : tensor.shape) {
            if (multiply_overflows(element_count, dimension)) {
                return issue(CheckpointLayoutError::ShapeOverflow, tensor.name);
            }
            element_count *= dimension;
        }
        const std::uint64_t element_bytes = tensor_data_type_bytes(tensor.data_type);
        if (element_bytes == 0) {
            return issue(CheckpointLayoutError::InvalidDataType, tensor.name);
        }
        if (multiply_overflows(element_count, element_bytes)) {
            return issue(CheckpointLayoutError::ShapeOverflow, tensor.name);
        }
        if (element_count * element_bytes != tensor.size_bytes) {
            return issue(CheckpointLayoutError::TensorSizeMismatch, tensor.name);
        }
        if (add_overflows(tensor.shard_offset_bytes, tensor.size_bytes)) {
            return issue(CheckpointLayoutError::RangeOverflow, tensor.name);
        }
        const std::uint64_t end = tensor.shard_offset_bytes + tensor.size_bytes;
        if (end > shard_sizes[tensor.shard]) {
            return issue(CheckpointLayoutError::TensorOutOfRange, tensor.name);
        }
        if (add_overflows(total_bytes, tensor.size_bytes)) {
            return issue(CheckpointLayoutError::TotalBytesOverflow, tensor.name);
        }
        total_bytes += tensor.size_bytes;
        if (tensor.size_bytes != 0) {
            extents.push_back({
                .shard = tensor.shard,
                .begin = tensor.shard_offset_bytes,
                .end = end,
                .name = tensor.name,
            });
        }
    }

    std::ranges::sort(extents, [](const TensorExtent& left, const TensorExtent& right) {
        if (left.shard != right.shard) {
            return left.shard < right.shard;
        }
        if (left.begin != right.begin) {
            return left.begin < right.begin;
        }
        return left.end < right.end;
    });
    for (std::size_t index = 1; index < extents.size(); ++index) {
        const TensorExtent& previous = extents[index - 1];
        const TensorExtent& current = extents[index];
        if (previous.shard == current.shard && current.begin < previous.end) {
            return issue(CheckpointLayoutError::TensorOverlap, current.name);
        }
    }

    if (total_bytes != expectation.tensor_payload_bytes) {
        return issue(CheckpointLayoutError::TotalBytesMismatch);
    }
    return issue(CheckpointLayoutError::None);
}

} // namespace tatara::model
