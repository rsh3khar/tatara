#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace tatara::model {

enum class TensorDataType : std::uint8_t {
    Bool = 1,
    U8 = 2,
    I8 = 3,
    F16 = 4,
    BF16 = 5,
    U16 = 6,
    I16 = 7,
    F32 = 8,
    U32 = 9,
    I32 = 10,
    F64 = 11,
    U64 = 12,
    I64 = 13,
    F8E4M3 = 14,
    F8E5M2 = 15,
};

struct TensorRecord {
    std::string name;
    TensorDataType data_type;
    std::vector<std::uint64_t> shape;
    std::size_t shard;
    std::uint64_t shard_offset_bytes;
    std::uint64_t size_bytes;
};

struct CheckpointExpectation {
    std::size_t shard_count;
    std::size_t tensor_count;
    std::uint64_t tensor_payload_bytes;
};

enum class CheckpointLayoutError : std::uint8_t {
    None,
    ShardCountMismatch,
    TensorCountMismatch,
    EmptyTensorName,
    DuplicateTensorName,
    InvalidShard,
    InvalidDataType,
    ShapeOverflow,
    TensorSizeMismatch,
    RangeOverflow,
    TensorOutOfRange,
    TensorOverlap,
    TotalBytesOverflow,
    TotalBytesMismatch,
};

struct CheckpointLayoutIssue {
    CheckpointLayoutError error;
    std::string tensor;

    explicit operator bool() const noexcept {
        return error != CheckpointLayoutError::None;
    }
};

std::uint64_t tensor_data_type_bytes(TensorDataType data_type) noexcept;

CheckpointLayoutIssue validate_checkpoint_layout(std::span<const TensorRecord> tensors,
                                                 std::span<const std::uint64_t> shard_sizes,
                                                 const CheckpointExpectation& expectation);

} // namespace tatara::model
