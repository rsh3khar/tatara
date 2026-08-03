#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace tatara::draft {

// Frozen plan of the DFlash draft companion model for the Qwen3.6-35B-A3B
// target.
// These are draft-model facts, not target facts; every one of them is
// verified against the checkpoint inventory at load time, so a checkpoint
// that disagrees is refused typed rather than silently reinterpreted.
inline constexpr std::uint32_t kDraftLayers = 6;
inline constexpr std::uint32_t kDraftHidden = 2048;
inline constexpr std::uint32_t kDraftIntermediate = 6144;
inline constexpr std::uint32_t kDraftQueryHeads = 32;
inline constexpr std::uint32_t kDraftKeyValueHeads = 8;
inline constexpr std::uint32_t kDraftHeadDimension = 128;
inline constexpr std::uint32_t kDraftCaptureLayers = 8;
inline constexpr std::uint32_t kDraftFeatureWidth =
    kDraftCaptureLayers * kDraftHidden;                    // 16,384
inline constexpr std::uint32_t kDraftSinkPositions = 64;
inline constexpr std::uint32_t kDraftWindowPositions = 4096;
inline constexpr std::uint32_t kDraftNativeBlock = 16;
inline constexpr std::uint32_t kDraftMaskTokenId = 248077;
inline constexpr std::uint64_t kDraftParameterCount = 385'906'176;
inline constexpr std::size_t kDraftTensorCount = 69;
// Target residual-stream capture points: output of target layer index i for
// i in this list (the DFlash conditioning contract; concatenation order is
// exactly this order).
inline constexpr std::array<std::uint32_t, kDraftCaptureLayers>
    kDraftCaptureAfterTargetLayer{1, 6, 11, 16, 22, 27, 32, 37};

enum class DraftCheckpointError : std::uint8_t {
    None,
    FileUnreadable,
    HeaderTruncated,
    HeaderLengthInvalid,
    HeaderParse,
    TensorEntryMalformed,
    TensorCount,
    UnknownTensor,
    MissingTensor,
    BannedTensor,
    DtypeNotBf16,
    ShapeMismatch,
    OffsetsInvalid,
    PayloadMismatch,
    ParameterCount,
};

std::string_view draft_checkpoint_error_name(DraftCheckpointError error) noexcept;

// A resolved bf16 tensor inside the mapped checkpoint. `elements` is the
// shape product; `bytes` is elements * 2. Data points into the checkpoint
// mapping owned by DraftCheckpoint and is valid for its lifetime.
struct DraftTensorView {
    const std::byte* data{nullptr};
    std::uint64_t elements{0};
    std::array<std::uint32_t, 2> shape{0, 0};   // [rows, columns]; vectors are [n, 1]
};

struct DraftCheckpointLoad;

// Maps `model.safetensors` under `root` and verifies the complete frozen
// inventory: exactly 69 tensors, all BF16, exact names and shapes, no
// embedding/head/bias/rotary tensors, contiguous non-overlapping offsets
// covering the payload exactly, and the frozen parameter count. Fail-closed:
// any deviation returns a typed error and an empty checkpoint.
DraftCheckpointLoad load_draft_checkpoint(std::string_view root);

class DraftCheckpoint {
  public:
    DraftCheckpoint() = default;
    DraftCheckpoint(const DraftCheckpoint&) = delete;
    DraftCheckpoint& operator=(const DraftCheckpoint&) = delete;
    DraftCheckpoint(DraftCheckpoint&&) noexcept;
    DraftCheckpoint& operator=(DraftCheckpoint&&) noexcept;
    ~DraftCheckpoint();

    // Named per the checkpoint ("fc.weight", "layers.3.mlp.up_proj.weight", ...).
    // Returns a null view when absent — load() guarantees the full frozen
    // inventory, so absence after a successful load is a programming error.
    DraftTensorView tensor(std::string_view name) const noexcept;

    std::uint64_t payload_bytes() const noexcept {
        return payload_bytes_;
    }

  private:
    friend DraftCheckpointLoad load_draft_checkpoint(std::string_view root);
    void* mapping_{nullptr};
    std::uint64_t mapping_bytes_{0};
    std::uint64_t payload_bytes_{0};
    struct Entry {
        std::string name;
        std::uint64_t offset;
        std::uint64_t elements;
        std::array<std::uint32_t, 2> shape;
    };
    std::vector<Entry> entries_;
};

struct DraftCheckpointLoad {
    DraftCheckpointError error{DraftCheckpointError::None};
    // Name of the tensor implicated in the error, when one is.
    std::string detail;
    DraftCheckpoint checkpoint;

    explicit operator bool() const noexcept {
        return error == DraftCheckpointError::None;
    }
};

} // namespace tatara::draft
