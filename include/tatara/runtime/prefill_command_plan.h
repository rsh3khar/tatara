#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace tatara::runtime {

inline constexpr std::uint32_t kPrefillCommandGraphSchemaVersion = 4;
inline constexpr std::size_t kPrefillCommandIdentityBytes = 32;
inline constexpr std::uint64_t kNoPrefillCommandPredecessor =
    std::numeric_limits<std::uint64_t>::max();
inline constexpr std::uint32_t kNoPrefillLaneEvent =
    std::numeric_limits<std::uint32_t>::max();

using PrefillCommandIdentity =
    std::array<std::uint8_t, kPrefillCommandIdentityBytes>;

struct PrefillCommandPlanKey {
    PrefillCommandIdentity model_package_identity{};
    PrefillCommandIdentity prepared_image_identity{};
    PrefillCommandIdentity pipeline_identity{};
    PrefillCommandIdentity execution_policy_identity{};
    std::uint64_t icb_capability_identity{0};
    std::uint64_t state_slot_identity{0};
    std::uint32_t row_count{0};
    std::uint32_t context_base{0};
    std::uint32_t graph_schema_version{kPrefillCommandGraphSchemaVersion};
    std::span<const std::uint32_t> chunk_rows;
    std::span<const std::uint64_t> persistent_resource_identities;
};

enum class PrefillCommandPlanError : std::uint8_t {
    None,
    InvalidIdentity,
    InvalidSchemaVersion,
    InvalidLayerCount,
    EmptyChunkSchedule,
    ZeroChunkRows,
    RowCountMismatch,
    ContextOutOfRange,
    ScratchLaneMismatch,
    ArithmeticOverflow,
    NodeStorageTooSmall,
    DiagonalStorageTooSmall,
    LaneEventStorageTooSmall,
    InvalidWavefrontNode,
    InvalidEventValue,
    InvalidCommandCount,
    InvalidCommandIndex,
    InvalidBufferIndex,
    InvalidArgumentSize,
    InvalidArgumentAlignment,
    DuplicateArgumentBinding,
    ArgumentStorageTooSmall,
    ArgumentArenaTooSmall,
    InvalidResourceIdentity,
    InvalidResourceUsage,
    ResourceStorageTooSmall,
    InvalidRoutedGrid,
    RoutedGridStorageTooSmall,
};

PrefillCommandPlanError
validate_prefill_command_plan_key(const PrefillCommandPlanKey& key) noexcept;

bool same_prefill_command_plan_key(const PrefillCommandPlanKey& left,
                                   const PrefillCommandPlanKey& right) noexcept;

struct PrefillCommandNode {
    std::uint32_t layer_index{0};
    std::uint32_t chunk_ordinal{0};
    std::uint32_t diagonal_index{0};
    std::uint32_t scratch_lane{0};
    std::uint64_t layer_major_index{0};
    std::uint64_t hidden_predecessor{kNoPrefillCommandPredecessor};
    std::uint64_t state_predecessor{kNoPrefillCommandPredecessor};
};

struct PrefillCommandDiagonal {
    std::uint64_t node_begin{0};
    std::uint32_t node_count{0};
};

struct PrefillWavefrontRequest {
    std::uint32_t layer_count{0};
    std::uint32_t row_count{0};
    std::uint32_t context_base{0};
    std::uint32_t context_capacity{0};
    std::uint32_t scratch_lane_count{0};
    std::span<const std::uint32_t> chunk_rows;
};

struct PrefillWavefrontPlan {
    std::uint64_t node_count{0};
    std::uint32_t diagonal_count{0};
    std::uint32_t scratch_lane_count{0};
};

struct PrefillWavefrontPlanResult {
    PrefillCommandPlanError error{PrefillCommandPlanError::InvalidLayerCount};
    PrefillWavefrontPlan plan{};

    explicit constexpr operator bool() const noexcept {
        return error == PrefillCommandPlanError::None;
    }
};

PrefillWavefrontPlanResult build_prefill_wavefront_plan(
    const PrefillWavefrontRequest& request,
    std::span<PrefillCommandNode> nodes,
    std::span<PrefillCommandDiagonal> diagonals) noexcept;

struct PrefillLaneEventRequest {
    std::uint32_t layer_count{0};
    std::uint32_t scratch_lane_count{0};
    std::uint64_t event_value_base{0};
    std::span<const PrefillCommandNode> nodes;
};

struct PrefillLaneEventNode {
    std::uint64_t node_index{kNoPrefillCommandPredecessor};
    std::uint32_t layer_index{0};
    std::uint32_t scratch_lane{0};
    std::uint32_t wait_event{kNoPrefillLaneEvent};
    std::uint64_t wait_value{0};
    std::uint32_t signal_event{kNoPrefillLaneEvent};
    std::uint64_t signal_value{0};
};

struct PrefillLaneEventPlan {
    std::uint64_t node_count{0};
    std::uint32_t event_count{0};
    std::uint64_t terminal_event_value{0};
};

struct PrefillLaneEventPlanResult {
    PrefillCommandPlanError error{
        PrefillCommandPlanError::InvalidLayerCount};
    PrefillLaneEventPlan plan{};

    explicit constexpr operator bool() const noexcept {
        return error == PrefillCommandPlanError::None;
    }
};

PrefillLaneEventPlanResult build_prefill_lane_event_plan(
    const PrefillLaneEventRequest& request,
    std::span<PrefillLaneEventNode> event_nodes) noexcept;

struct PrefillArgumentBindingRequest {
    std::uint32_t command_index{0};
    std::uint32_t buffer_index{0};
    std::uint64_t size_bytes{0};
    std::uint64_t alignment_bytes{0};
};

struct PrefillArgumentBinding {
    std::uint32_t command_index{0};
    std::uint32_t buffer_index{0};
    std::uint64_t offset_bytes{0};
    std::uint64_t size_bytes{0};
};

struct PrefillArgumentArenaPlan {
    std::uint64_t required_bytes{0};
    std::uint32_t binding_count{0};
};

struct PrefillArgumentArenaPlanResult {
    PrefillCommandPlanError error{
        PrefillCommandPlanError::InvalidCommandCount};
    PrefillArgumentArenaPlan plan{};

    explicit constexpr operator bool() const noexcept {
        return error == PrefillCommandPlanError::None;
    }
};

PrefillArgumentArenaPlanResult build_prefill_argument_arena_plan(
    std::uint32_t command_count, std::uint32_t maximum_buffer_index,
    std::uint64_t arena_capacity_bytes,
    std::span<const PrefillArgumentBindingRequest> requests,
    std::span<PrefillArgumentBinding> bindings) noexcept;

enum class PrefillResourceUsage : std::uint8_t {
    Read = 1,
    Write = 2,
    ReadWrite = 3,
};

struct PrefillResourceRequest {
    std::uint64_t identity{0};
    PrefillResourceUsage usage{PrefillResourceUsage::Read};
};

struct PrefillResourceEntry {
    std::uint64_t identity{0};
    PrefillResourceUsage usage{PrefillResourceUsage::Read};
};

struct PrefillResourceTableResult {
    PrefillCommandPlanError error{
        PrefillCommandPlanError::InvalidResourceIdentity};
    std::uint32_t resource_count{0};

    explicit constexpr operator bool() const noexcept {
        return error == PrefillCommandPlanError::None;
    }
};

PrefillResourceTableResult build_prefill_resource_table(
    std::span<const PrefillResourceRequest> requests,
    std::span<PrefillResourceEntry> resources) noexcept;

struct PrefillRoutedGridRequest {
    std::span<const std::uint32_t> chunk_rows;
    std::uint32_t routes_per_position{0};
    std::uint32_t task_capacity{0};
    std::uint32_t tile_columns{0};
    std::uint32_t upgate_columns{0};
    std::uint32_t down_columns{0};
};

struct PrefillRoutedGrid {
    std::uint32_t maximum_task_count{0};
    std::uint32_t upgate_column_groups{0};
    std::uint32_t down_column_groups{0};
};

struct PrefillRoutedGridPlanResult {
    PrefillCommandPlanError error{
        PrefillCommandPlanError::InvalidRoutedGrid};
    PrefillRoutedGrid grid{};

    explicit constexpr operator bool() const noexcept {
        return error == PrefillCommandPlanError::None;
    }
};

PrefillRoutedGridPlanResult build_prefill_routed_grid_plan(
    const PrefillRoutedGridRequest& request,
    std::span<std::uint32_t> maximum_task_counts) noexcept;

} // namespace tatara::runtime
