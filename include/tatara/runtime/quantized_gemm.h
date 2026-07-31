#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace tatara::runtime {

inline constexpr std::uint32_t kQuantizedGemmProfileVersion = 3;
inline constexpr std::uint32_t
    kQuantizedGemmDeviceTaskDescriptorVersion = 1;
inline constexpr std::size_t kQuantizedGemmRegionCapacity = 16;
inline constexpr std::size_t kQuantizedGemmTaskCapacity = 4096;
inline constexpr std::size_t kQuantizedGemmIdentityBytes = 32;
inline constexpr std::uint64_t kNoQuantizedGemmWorkspaceOffset = UINT64_MAX;

enum class QuantizedGemmPolicy : std::uint8_t {
    ExactRow = 0,
    NativeDenseMma = 1,
    NativeRaggedMma = 2,
};

enum class QuantizedGemmWorkload : std::uint8_t {
    Dense = 0,
    Ragged = 1,
};

enum class QuantizedGemmTaskMode : std::uint8_t {
    HostCompact = 0,
    DevicePaddedSlotsV1 = 1,
};

enum class QuantizedGemmActivationRowMapping : std::uint8_t {
    PositionRows = 0,
    PaddedSlotRows = 1,
};

enum class AffineQuantization : std::uint8_t {
    Q4 = 4,
    Q8 = 8,
};

enum class QuantizedWeightLayout : std::uint8_t {
    OutputMajorAffineGroups = 0,
};

struct QuantizedGemmShapeDescriptor {
    QuantizedGemmWorkload workload{QuantizedGemmWorkload::Dense};
    AffineQuantization quantization{AffineQuantization::Q4};
    QuantizedWeightLayout weight_layout{
        QuantizedWeightLayout::OutputMajorAffineGroups};
    std::uint32_t input_rows{0};
    std::uint32_t output_rows{0};
    std::uint32_t output_columns{0};
    std::uint32_t reduction_columns{0};
    std::uint32_t group_size{0};
    std::uint32_t activation_element_bytes{0};
    std::uint32_t output_element_bytes{0};
    std::uint32_t quantization_parameter_bytes{0};
    std::uint64_t activation_row_stride_elements{0};
    std::uint64_t output_row_stride_elements{0};
    std::uint64_t packed_weight_row_stride_bytes{0};
    std::uint64_t scale_row_stride_bytes{0};
    std::uint64_t bias_row_stride_bytes{0};
    std::uint32_t expert_count{0};
    std::uint32_t route_list_count{0};
    std::uint64_t activation_storage_bytes{0};
    std::uint64_t packed_weight_storage_bytes{0};
    std::uint64_t scale_storage_bytes{0};
    std::uint64_t bias_storage_bytes{0};
    std::uint64_t output_storage_bytes{0};
};

struct QuantizedGemmRegionDescriptor {
    std::uint32_t output_column_begin{0};
    std::uint32_t output_column_count{0};
    std::uint64_t packed_weight_offset_bytes{0};
    std::uint64_t scale_offset_bytes{0};
    std::uint64_t bias_offset_bytes{0};
    std::uint64_t packed_weight_expert_stride_bytes{0};
    std::uint64_t scale_expert_stride_bytes{0};
    std::uint64_t bias_expert_stride_bytes{0};
};

struct QuantizedGemmTaskDescriptor {
    std::uint32_t expert_index{0};
    std::uint32_t route_list_begin{0};
    std::uint32_t row_count{0};
    // DevicePaddedSlotsV1 derives destinations from packed list entries and
    // requires this field to remain zero.
    std::uint32_t output_row_begin{0};
};

struct QuantizedGemmDeviceTaskDescriptor {
    std::uint32_t version{0};
    QuantizedGemmActivationRowMapping activation_row_mapping{
        QuantizedGemmActivationRowMapping::PositionRows};
    std::uint32_t position_capacity{0};
    std::uint32_t routes_per_position{0};
    std::uint32_t padded_slot_stride{0};
    std::uint32_t packed_slot_bits{0};
    std::uint32_t list_capacity_entries{0};
    std::uint64_t list_expert_stride_entries{0};
    std::uint64_t count_storage_bytes{0};
    std::uint64_t list_storage_bytes{0};
};

struct QuantizedGemmImplementationProfile {
    std::uint32_t profile_version{0};
    std::array<std::uint8_t, kQuantizedGemmIdentityBytes>
        implementation_sha256{};
    std::uint32_t specialization_id{0};
    std::uint32_t specialization_version{0};
    std::uint32_t supported_group_size{0};
    std::uint32_t maximum_regions{0};
    std::uint32_t maximum_tasks{0};
    std::uint32_t maximum_experts{0};
    std::uint32_t maximum_route_entries{0};
    std::uint32_t supported_activation_element_bytes{0};
    std::uint32_t supported_output_element_bytes{0};
    std::uint32_t supported_quantization_parameter_bytes{0};
    std::uint32_t tile_rows{0};
    std::uint32_t tile_columns{0};
    std::uint32_t tile_reduction_columns{0};
    std::uint64_t activation_staging_row_stride_elements{0};
    std::uint64_t weight_staging_row_stride_elements{0};
    std::uint32_t threads_per_threadgroup{0};
    std::uint32_t maximum_threads_per_threadgroup{0};
    std::uint64_t maximum_threadgroup_memory_bytes{0};
    std::uint64_t maximum_accumulator_elements{0};
    std::uint64_t required_threadgroup_memory_bytes{0};
    std::uint64_t required_accumulator_elements_per_threadgroup{0};
    std::uint32_t threadgroup_staging_buffer_count{0};
    std::uint32_t partial_partition_count{0};
    std::uint32_t maximum_partial_partition_count{0};
    std::uint32_t staging_element_bytes{0};
    std::uint32_t partial_element_bytes{0};
    std::uint32_t workspace_alignment{0};
    bool q4_available{false};
    bool q8_available{false};
    bool exact_row_available{false};
    bool native_dense_mma_available{false};
    bool native_ragged_mma_available{false};
    bool device_padded_slots_v1_available{false};
    QuantizedGemmActivationRowMapping activation_row_mapping{
        QuantizedGemmActivationRowMapping::PositionRows};
};

struct QuantizedGemmImplementationIdentity {
    std::array<std::uint8_t, kQuantizedGemmIdentityBytes>
        implementation_sha256{};
    std::uint32_t specialization_id{0};
    std::uint32_t specialization_version{0};
    QuantizedGemmPolicy policy{QuantizedGemmPolicy::ExactRow};
    AffineQuantization quantization{AffineQuantization::Q4};
    std::uint32_t group_size{0};
    std::uint32_t tile_rows{0};
    std::uint32_t tile_columns{0};
    std::uint32_t tile_reduction_columns{0};
    std::uint32_t threads_per_threadgroup{0};
    QuantizedGemmTaskMode task_mode{QuantizedGemmTaskMode::HostCompact};
    std::uint32_t device_task_descriptor_version{0};
    std::uint64_t required_threadgroup_memory_bytes{0};
    std::uint64_t required_accumulator_elements_per_threadgroup{0};
    QuantizedGemmActivationRowMapping activation_row_mapping{
        QuantizedGemmActivationRowMapping::PositionRows};
};

struct QuantizedGemmRuntimeCapability {
    QuantizedGemmImplementationIdentity identity;
    bool available{false};
};

enum class QuantizedGemmCreationError : std::uint8_t {
    None = 0,
    InvalidShape = 1,
    UnsupportedQuantization = 2,
    UnsupportedGroup = 3,
    UnsupportedLayout = 4,
    InvalidImplementationProfile = 5,
    DescriptorLimitExceeded = 6,
    InvalidRegion = 7,
    InvalidTask = 8,
    InvalidStorageBounds = 9,
    ArithmeticOverflow = 10,
    ThreadLimitExceeded = 11,
    ThreadgroupMemoryExceeded = 12,
    AccumulatorLimitExceeded = 13,
    PolicyUnavailable = 14,
    WorkspaceInsufficient = 15,
    ExtentValidationUnsupported = 16,
};

struct QuantizedGemmWorkspacePlan {
    std::uint64_t row_grid_groups{0};
    std::uint64_t column_grid_groups{0};
    std::uint64_t reduction_grid_groups{0};
    std::uint64_t threadgroup_count{0};
    std::uint32_t threads_per_threadgroup{0};
    std::uint64_t threadgroup_memory_bytes{0};
    std::uint64_t accumulator_elements_per_threadgroup{0};
    std::uint32_t partial_partition_count{0};
    std::uint64_t region_descriptor_offset{kNoQuantizedGemmWorkspaceOffset};
    std::uint64_t region_descriptor_bytes{0};
    std::uint64_t task_descriptor_offset{kNoQuantizedGemmWorkspaceOffset};
    std::uint64_t task_descriptor_bytes{0};
    std::uint64_t device_task_indirect_argument_offset{
        kNoQuantizedGemmWorkspaceOffset};
    std::uint64_t device_task_indirect_argument_bytes{0};
    std::uint64_t device_task_status_offset{
        kNoQuantizedGemmWorkspaceOffset};
    std::uint64_t device_task_status_bytes{0};
    std::uint64_t route_input_index_offset{kNoQuantizedGemmWorkspaceOffset};
    std::uint64_t route_input_index_bytes{0};
    std::uint64_t route_output_index_offset{kNoQuantizedGemmWorkspaceOffset};
    std::uint64_t route_output_index_bytes{0};
    std::uint64_t partial_offset{kNoQuantizedGemmWorkspaceOffset};
    std::uint64_t partial_bytes{0};
    std::uint64_t required_bytes{0};
    std::uint32_t alignment{0};
};

struct QuantizedGemmCreationRequest {
    QuantizedGemmPolicy policy{QuantizedGemmPolicy::ExactRow};
    QuantizedGemmShapeDescriptor shape;
    std::span<const QuantizedGemmRegionDescriptor> regions;
    std::span<const QuantizedGemmTaskDescriptor> tasks;
    QuantizedGemmTaskMode task_mode{QuantizedGemmTaskMode::HostCompact};
    QuantizedGemmDeviceTaskDescriptor device_tasks;
    QuantizedGemmImplementationProfile implementation;
    std::uint64_t available_workspace_bytes{0};
};

struct QuantizedGemmCreationResult;

class QuantizedGemmRequestPlan {
  public:
    constexpr QuantizedGemmRequestPlan() noexcept = default;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return valid_;
    }
    [[nodiscard]] constexpr QuantizedGemmPolicy selected_policy() const noexcept {
        return selected_policy_;
    }
    [[nodiscard]] static constexpr QuantizedGemmPolicy fallback_policy() noexcept {
        return QuantizedGemmPolicy::ExactRow;
    }
    [[nodiscard]] static constexpr bool automatic_fallback_allowed() noexcept {
        return false;
    }
    [[nodiscard]] constexpr const QuantizedGemmShapeDescriptor& shape() const noexcept {
        return shape_;
    }
    [[nodiscard]] constexpr std::span<const QuantizedGemmRegionDescriptor>
    regions() const noexcept {
        return {regions_.data(), region_count_};
    }
    [[nodiscard]] constexpr std::span<const QuantizedGemmTaskDescriptor>
    tasks() const noexcept {
        return {tasks_.data(), task_count_};
    }
    [[nodiscard]] constexpr QuantizedGemmTaskMode task_mode() const noexcept {
        return task_mode_;
    }
    [[nodiscard]] constexpr const QuantizedGemmDeviceTaskDescriptor&
    device_tasks() const noexcept {
        return device_tasks_;
    }
    [[nodiscard]] constexpr std::uint64_t
    device_route_capacity() const noexcept {
        return device_route_capacity_;
    }
    [[nodiscard]] constexpr std::uint64_t
    device_task_capacity() const noexcept {
        return device_task_capacity_;
    }
    [[nodiscard]] constexpr std::uint64_t
    required_device_task_count_bytes() const noexcept {
        return required_device_task_count_bytes_;
    }
    [[nodiscard]] constexpr std::uint64_t
    required_device_task_list_bytes() const noexcept {
        return required_device_task_list_bytes_;
    }
    [[nodiscard]] constexpr const QuantizedGemmWorkspacePlan&
    workspace() const noexcept {
        return workspace_;
    }
    [[nodiscard]] constexpr std::uint64_t required_output_bytes() const noexcept {
        return required_output_bytes_;
    }
    [[nodiscard]] constexpr const QuantizedGemmImplementationIdentity&
    implementation_identity() const noexcept {
        return implementation_identity_;
    }
    [[nodiscard]] constexpr std::uint64_t required_activation_bytes() const noexcept {
        return required_activation_bytes_;
    }
    [[nodiscard]] constexpr std::uint64_t required_packed_weight_bytes() const noexcept {
        return required_packed_weight_bytes_;
    }
    [[nodiscard]] constexpr std::uint64_t required_scale_bytes() const noexcept {
        return required_scale_bytes_;
    }
    [[nodiscard]] constexpr std::uint64_t required_bias_bytes() const noexcept {
        return required_bias_bytes_;
    }

  private:
    friend QuantizedGemmCreationResult
    create_quantized_gemm_plan(const QuantizedGemmCreationRequest&) noexcept;

    bool valid_{false};
    QuantizedGemmPolicy selected_policy_{QuantizedGemmPolicy::ExactRow};
    QuantizedGemmTaskMode task_mode_{QuantizedGemmTaskMode::HostCompact};
    QuantizedGemmShapeDescriptor shape_;
    QuantizedGemmDeviceTaskDescriptor device_tasks_;
    std::array<QuantizedGemmRegionDescriptor, kQuantizedGemmRegionCapacity>
        regions_{};
    std::array<QuantizedGemmTaskDescriptor, kQuantizedGemmTaskCapacity>
        tasks_{};
    std::uint16_t region_count_{0};
    std::uint16_t task_count_{0};
    QuantizedGemmWorkspacePlan workspace_;
    QuantizedGemmImplementationIdentity implementation_identity_;
    std::uint64_t required_activation_bytes_{0};
    std::uint64_t required_packed_weight_bytes_{0};
    std::uint64_t required_scale_bytes_{0};
    std::uint64_t required_bias_bytes_{0};
    std::uint64_t required_output_bytes_{0};
    std::uint64_t device_route_capacity_{0};
    std::uint64_t device_task_capacity_{0};
    std::uint64_t required_device_task_count_bytes_{0};
    std::uint64_t required_device_task_list_bytes_{0};
};

struct QuantizedGemmCreationResult {
    QuantizedGemmCreationError error{
        QuantizedGemmCreationError::InvalidShape};
    QuantizedGemmWorkspacePlan workspace;
    std::uint64_t available_workspace_bytes{0};
    std::uint64_t required_output_bytes{0};
    QuantizedGemmRequestPlan plan;

    explicit constexpr operator bool() const noexcept {
        return error == QuantizedGemmCreationError::None && plan.valid();
    }
};

enum class QuantizedGemmRuntimeError : std::uint8_t {
    None = 0,
    InvalidPlan = 1,
    PolicyChangeProhibited = 2,
    InvalidRange = 3,
    WorkspaceInsufficient = 4,
    WorkspaceMisaligned = 5,
    OperandInsufficient = 6,
    OutputInsufficient = 7,
    InvalidIndirection = 8,
    CapabilityUnavailable = 9,
    CapabilityIdentityMismatch = 10,
    WorkspaceAliasesOperand = 11,
    DeviceTaskResourceMisaligned = 12,
    DeviceTaskResourceAliasesOperand = 13,
};

struct QuantizedGemmRuntimeRequest {
    QuantizedGemmPolicy policy{QuantizedGemmPolicy::ExactRow};
    std::uint32_t input_rows{0};
    std::uint32_t output_rows{0};
    std::span<std::byte> workspace;
    std::span<const std::byte> activations;
    std::span<const std::byte> packed_weights;
    std::span<const std::byte> scales;
    std::span<const std::byte> biases;
    std::span<std::byte> output;
    std::span<const std::uint32_t> route_input_rows;
    std::span<const std::uint32_t> route_output_rows;
    std::uint32_t device_task_position_count{0};
    std::span<const std::byte> device_task_counts;
    std::span<const std::byte> device_task_lists;
    QuantizedGemmRuntimeCapability capability;
};

enum class QuantizedGemmOperandKind : std::uint8_t {
    None = 0,
    Activations = 1,
    PackedWeights = 2,
    Scales = 3,
    Biases = 4,
    Output = 5,
    DeviceTaskCounts = 6,
    DeviceTaskLists = 7,
};

enum class QuantizedGemmDeviceTaskStatus : std::uint32_t {
    NotProduced = 0,
    Ready = 1,
    CountOutOfRange = 2,
    RouteConservationFailure = 3,
    TaskCapacityExceeded = 4,
    PackedSlotOutOfRange = 5,
};

struct QuantizedGemmRuntimeValidation {
    QuantizedGemmRuntimeError error{QuantizedGemmRuntimeError::InvalidPlan};
    std::uint64_t required_workspace_bytes{0};
    std::uint64_t available_workspace_bytes{0};
    std::uint64_t required_output_bytes{0};
    std::uint64_t available_output_bytes{0};
    QuantizedGemmOperandKind operand{QuantizedGemmOperandKind::None};
    std::uint64_t required_operand_bytes{0};
    std::uint64_t available_operand_bytes{0};

    explicit constexpr operator bool() const noexcept {
        return error == QuantizedGemmRuntimeError::None;
    }
};

enum class QuantizedGemmMaterializationError : std::uint8_t {
    None = 0,
    InvalidPlan = 1,
    WorkspaceInsufficient = 2,
    WorkspaceMisaligned = 3,
    InvalidIndirection = 4,
    SourceAliasesWorkspace = 5,
};

struct QuantizedGemmMaterializationRequest {
    std::span<std::byte> workspace;
    std::span<const std::uint32_t> route_input_rows;
    std::span<const std::uint32_t> route_output_rows;
};

struct QuantizedGemmMaterializationResult {
    QuantizedGemmMaterializationError error{
        QuantizedGemmMaterializationError::InvalidPlan};
    std::uint64_t required_workspace_bytes{0};
    std::uint64_t available_workspace_bytes{0};
    std::uint64_t bytes_written{0};

    explicit constexpr operator bool() const noexcept {
        return error == QuantizedGemmMaterializationError::None;
    }
};

[[nodiscard]] QuantizedGemmCreationResult
create_quantized_gemm_plan(const QuantizedGemmCreationRequest& request) noexcept;

[[nodiscard]] QuantizedGemmRuntimeValidation validate_quantized_gemm_runtime(
    const QuantizedGemmRequestPlan& plan,
    const QuantizedGemmRuntimeRequest& request) noexcept;

[[nodiscard]] QuantizedGemmMaterializationResult
materialize_quantized_gemm_workspace(
    const QuantizedGemmRequestPlan& plan,
    const QuantizedGemmMaterializationRequest& request) noexcept;

} // namespace tatara::runtime
