#include "tatara/runtime/quantized_gemm.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <new>
#include <span>
#include <type_traits>
#include <utility>

namespace {

using tatara::runtime::AffineQuantization;
using tatara::runtime::QuantizedGemmCreationError;
using tatara::runtime::QuantizedGemmCreationRequest;
using tatara::runtime::QuantizedGemmActivationRowMapping;
using tatara::runtime::QuantizedGemmDeviceTaskDescriptor;
using tatara::runtime::QuantizedGemmDeviceTaskStatus;
using tatara::runtime::QuantizedGemmImplementationProfile;
using tatara::runtime::QuantizedGemmPolicy;
using tatara::runtime::QuantizedGemmRegionDescriptor;
using tatara::runtime::QuantizedGemmRuntimeError;
using tatara::runtime::QuantizedGemmRuntimeRequest;
using tatara::runtime::QuantizedGemmShapeDescriptor;
using tatara::runtime::QuantizedGemmTaskDescriptor;
using tatara::runtime::QuantizedGemmTaskMode;
using tatara::runtime::QuantizedGemmWorkload;
using tatara::runtime::QuantizedWeightLayout;

std::uint64_t g_allocation_count = 0;

static_assert(static_cast<std::uint8_t>(QuantizedGemmPolicy::ExactRow) == 0);
static_assert(
    static_cast<std::uint8_t>(QuantizedGemmPolicy::NativeDenseMma) == 1);
static_assert(
    static_cast<std::uint8_t>(QuantizedGemmPolicy::NativeRaggedMma) == 2);
static_assert(
    !std::is_aggregate_v<tatara::runtime::QuantizedGemmRequestPlan>);
static_assert(
    tatara::runtime::QuantizedGemmRequestPlan::fallback_policy() ==
    QuantizedGemmPolicy::ExactRow);
static_assert(
    !tatara::runtime::QuantizedGemmRequestPlan::
        automatic_fallback_allowed());
static_assert(tatara::runtime::kQuantizedGemmProfileVersion == 3);
static_assert(
    tatara::runtime::kQuantizedGemmDeviceTaskDescriptorVersion == 1);
static_assert(
    static_cast<std::uint8_t>(QuantizedGemmTaskMode::HostCompact) == 0);
static_assert(
    static_cast<std::uint8_t>(
        QuantizedGemmTaskMode::DevicePaddedSlotsV1) == 1);
static_assert(sizeof(QuantizedGemmDeviceTaskStatus) ==
              sizeof(std::uint32_t));
static_assert(
    static_cast<std::uint32_t>(
        QuantizedGemmDeviceTaskStatus::NotProduced) == 0);
static_assert(
    static_cast<std::uint32_t>(QuantizedGemmDeviceTaskStatus::Ready) == 1);
static_assert(
    static_cast<std::uint32_t>(
        QuantizedGemmDeviceTaskStatus::CountOutOfRange) == 2);
static_assert(
    static_cast<std::uint32_t>(
        QuantizedGemmDeviceTaskStatus::RouteConservationFailure) == 3);
static_assert(
    static_cast<std::uint32_t>(
        QuantizedGemmDeviceTaskStatus::TaskCapacityExceeded) == 4);
static_assert(
    static_cast<std::uint32_t>(
        QuantizedGemmDeviceTaskStatus::PackedSlotOutOfRange) == 5);
static_assert(std::is_standard_layout_v<QuantizedGemmTaskDescriptor>);
static_assert(std::is_trivially_copyable_v<QuantizedGemmTaskDescriptor>);
static_assert(sizeof(QuantizedGemmTaskDescriptor) == 4U * sizeof(std::uint32_t));
static_assert(alignof(QuantizedGemmTaskDescriptor) == alignof(std::uint32_t));
static_assert(offsetof(QuantizedGemmTaskDescriptor, expert_index) == 0);
static_assert(offsetof(QuantizedGemmTaskDescriptor, route_list_begin) ==
              sizeof(std::uint32_t));
static_assert(offsetof(QuantizedGemmTaskDescriptor, row_count) ==
              2U * sizeof(std::uint32_t));
static_assert(offsetof(QuantizedGemmTaskDescriptor, output_row_begin) ==
              3U * sizeof(std::uint32_t));
static_assert(QuantizedGemmTaskDescriptor{}.output_row_begin == 0);
static_assert(noexcept(tatara::runtime::create_quantized_gemm_plan(
    std::declval<const QuantizedGemmCreationRequest&>())));

QuantizedGemmImplementationProfile make_profile() {
    return {
        .profile_version = tatara::runtime::kQuantizedGemmProfileVersion,
        .implementation_sha256 = {1, 2, 3, 4},
        .specialization_id = 7,
        .specialization_version = 1,
        .supported_group_size = 24,
        .maximum_regions = 4,
        .maximum_tasks = 8,
        .maximum_experts = 16,
        .maximum_route_entries = 1024,
        .supported_activation_element_bytes = 2,
        .supported_output_element_bytes = 2,
        .supported_quantization_parameter_bytes = 2,
        .tile_rows = 8,
        .tile_columns = 16,
        .tile_reduction_columns = 32,
        .activation_staging_row_stride_elements = 32,
        .weight_staging_row_stride_elements = 16,
        .threads_per_threadgroup = 128,
        .maximum_threads_per_threadgroup = 256,
        .maximum_threadgroup_memory_bytes = 65536,
        .maximum_accumulator_elements = 1024,
        .required_threadgroup_memory_bytes = 3072,
        .required_accumulator_elements_per_threadgroup = 128,
        .threadgroup_staging_buffer_count = 2,
        .partial_partition_count = 1,
        .maximum_partial_partition_count = 8,
        .staging_element_bytes = 2,
        .partial_element_bytes = 4,
        .workspace_alignment = 64,
        .q4_available = true,
        .q8_available = true,
        .exact_row_available = true,
        .native_dense_mma_available = true,
        .native_ragged_mma_available = true,
    };
}

QuantizedGemmImplementationProfile make_direct_profile(
    std::uint64_t accumulator_elements,
    QuantizedGemmActivationRowMapping mapping,
    std::uint32_t specialization_id,
    std::uint32_t specialization_version,
    std::uint32_t output_element_bytes = 2) {
    auto profile = make_profile();
    profile.specialization_id = specialization_id;
    profile.specialization_version = specialization_version;
    profile.tile_rows = 16;
    profile.tile_columns = 32;
    profile.tile_reduction_columns = 32;
    profile.activation_staging_row_stride_elements = 0;
    profile.weight_staging_row_stride_elements = 0;
    profile.threads_per_threadgroup = 64;
    profile.required_threadgroup_memory_bytes = 0;
    profile.required_accumulator_elements_per_threadgroup =
        accumulator_elements;
    profile.threadgroup_staging_buffer_count = 0;
    profile.staging_element_bytes = 0;
    profile.supported_output_element_bytes = output_element_bytes;
    profile.device_padded_slots_v1_available = true;
    profile.activation_row_mapping = mapping;
    return profile;
}

QuantizedGemmShapeDescriptor make_dense_shape(AffineQuantization quantization) {
    const bool q4 = quantization == AffineQuantization::Q4;
    return {
        .workload = QuantizedGemmWorkload::Dense,
        .quantization = quantization,
        .weight_layout =
            QuantizedWeightLayout::OutputMajorAffineGroups,
        .input_rows = 9,
        .output_rows = 9,
        .output_columns = 19,
        .reduction_columns = 70,
        .group_size = 24,
        .activation_element_bytes = 2,
        .output_element_bytes = 2,
        .quantization_parameter_bytes = 2,
        .activation_row_stride_elements = 80,
        .output_row_stride_elements = 24,
        .packed_weight_row_stride_bytes = q4 ? 40U : 80U,
        .scale_row_stride_bytes = 8,
        .bias_row_stride_bytes = 8,
        .expert_count = 0,
        .route_list_count = 0,
        .activation_storage_bytes = 9U * 80U * 2U,
        .packed_weight_storage_bytes =
            q4 ? 19U * 40U : 19U * 80U,
        .scale_storage_bytes = 19U * 8U,
        .bias_storage_bytes = 19U * 8U,
        .output_storage_bytes = 9U * 24U * 2U,
    };
}

std::array<QuantizedGemmRegionDescriptor, 2>
make_dense_regions(AffineQuantization quantization) {
    const std::uint64_t weight_stride =
        quantization == AffineQuantization::Q4 ? 40U : 80U;
    return {
        QuantizedGemmRegionDescriptor{
            .output_column_begin = 0,
            .output_column_count = 13,
        },
        QuantizedGemmRegionDescriptor{
            .output_column_begin = 13,
            .output_column_count = 6,
            .packed_weight_offset_bytes = 13U * weight_stride,
            .scale_offset_bytes = 13U * 8U,
            .bias_offset_bytes = 13U * 8U,
        },
    };
}

QuantizedGemmShapeDescriptor make_ragged_shape() {
    return {
        .workload = QuantizedGemmWorkload::Ragged,
        .quantization = AffineQuantization::Q4,
        .weight_layout =
            QuantizedWeightLayout::OutputMajorAffineGroups,
        .input_rows = 7,
        .output_rows = 9,
        .output_columns = 17,
        .reduction_columns = 70,
        .group_size = 24,
        .activation_element_bytes = 2,
        .output_element_bytes = 2,
        .quantization_parameter_bytes = 2,
        .activation_row_stride_elements = 72,
        .output_row_stride_elements = 24,
        .packed_weight_row_stride_bytes = 40,
        .scale_row_stride_bytes = 8,
        .bias_row_stride_bytes = 8,
        .expert_count = 5,
        .route_list_count = 9,
        .activation_storage_bytes = 7U * 72U * 2U,
        .packed_weight_storage_bytes = 3400,
        .scale_storage_bytes = 680,
        .bias_storage_bytes = 680,
        .output_storage_bytes = 9U * 24U * 2U,
    };
}

std::array<QuantizedGemmRegionDescriptor, 1> make_ragged_regions() {
    return {
        QuantizedGemmRegionDescriptor{
            .output_column_begin = 0,
            .output_column_count = 17,
            .packed_weight_expert_stride_bytes = 680,
            .scale_expert_stride_bytes = 136,
            .bias_expert_stride_bytes = 136,
        },
    };
}

std::array<QuantizedGemmTaskDescriptor, 3> make_ragged_tasks() {
    return {
        QuantizedGemmTaskDescriptor{
            .expert_index = 0,
            .route_list_begin = 0,
            .row_count = 2,
            .output_row_begin = 0,
        },
        QuantizedGemmTaskDescriptor{
            .expert_index = 2,
            .route_list_begin = 2,
            .row_count = 3,
            .output_row_begin = 2,
        },
        QuantizedGemmTaskDescriptor{
            .expert_index = 4,
            .route_list_begin = 5,
            .row_count = 4,
            .output_row_begin = 5,
        },
    };
}

QuantizedGemmShapeDescriptor make_device_task_shape(
    QuantizedGemmActivationRowMapping mapping,
    std::uint32_t output_element_bytes = 2) {
    auto shape = make_ragged_shape();
    constexpr std::uint32_t kPositions = 7;
    constexpr std::uint32_t kPaddedSlotStride = 3;
    shape.input_rows =
        mapping == QuantizedGemmActivationRowMapping::PositionRows
            ? kPositions
            : kPositions * kPaddedSlotStride;
    shape.output_rows = kPositions * kPaddedSlotStride;
    shape.output_element_bytes = output_element_bytes;
    shape.route_list_count = kPositions * 2U;
    shape.activation_storage_bytes =
        std::uint64_t{shape.input_rows} *
        shape.activation_row_stride_elements *
        shape.activation_element_bytes;
    shape.output_storage_bytes =
        std::uint64_t{shape.output_rows} *
        shape.output_row_stride_elements *
        shape.output_element_bytes;
    return shape;
}

QuantizedGemmDeviceTaskDescriptor make_device_task_descriptor(
    QuantizedGemmActivationRowMapping mapping) {
    return {
        .version =
            tatara::runtime::kQuantizedGemmDeviceTaskDescriptorVersion,
        .activation_row_mapping = mapping,
        .position_capacity = 7,
        .routes_per_position = 2,
        .padded_slot_stride = 3,
        .packed_slot_bits = 2,
        .list_capacity_entries = 7,
        .list_expert_stride_entries = 8,
        .count_storage_bytes = 5U * sizeof(std::uint32_t),
        .list_storage_bytes = 40U * sizeof(std::uint32_t),
    };
}

int test_task_descriptor_physical_serialization() {
    constexpr QuantizedGemmTaskDescriptor descriptor{
        .expert_index = 0x04030201U,
        .route_list_begin = 0x08070605U,
        .row_count = 0x0c0b0a09U,
        .output_row_begin = 0x100f0e0dU,
    };
    constexpr std::array<std::byte, 16> expected{
        std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04},
        std::byte{0x05}, std::byte{0x06}, std::byte{0x07}, std::byte{0x08},
        std::byte{0x09}, std::byte{0x0a}, std::byte{0x0b}, std::byte{0x0c},
        std::byte{0x0d}, std::byte{0x0e}, std::byte{0x0f}, std::byte{0x10},
    };
    std::array<std::byte, sizeof(descriptor)> serialized{};
    const std::uint64_t allocations_before = g_allocation_count;
    std::memcpy(serialized.data(), &descriptor, sizeof(descriptor));
    return serialized == expected &&
                   g_allocation_count == allocations_before
               ? 0
               : 1;
}

QuantizedGemmShapeDescriptor make_interleaved_shape() {
    auto shape = make_ragged_shape();
    shape.output_columns = 2;
    shape.output_row_stride_elements = 2;
    shape.packed_weight_storage_bytes = 390;
    shape.scale_storage_bytes = 76;
    shape.bias_storage_bytes = 76;
    shape.output_storage_bytes = 9U * 2U * 2U;
    return shape;
}

std::array<QuantizedGemmRegionDescriptor, 2>
make_interleaved_regions() {
    return {
        QuantizedGemmRegionDescriptor{
            .output_column_begin = 0,
            .output_column_count = 1,
            .packed_weight_offset_bytes = 35,
            .scale_offset_bytes = 6,
            .bias_offset_bytes = 6,
            .packed_weight_expert_stride_bytes = 80,
            .scale_expert_stride_bytes = 16,
            .bias_expert_stride_bytes = 16,
        },
        QuantizedGemmRegionDescriptor{
            .output_column_begin = 1,
            .output_column_count = 1,
            .packed_weight_offset_bytes = 0,
            .scale_offset_bytes = 0,
            .bias_offset_bytes = 0,
            .packed_weight_expert_stride_bytes = 80,
            .scale_expert_stride_bytes = 16,
            .bias_expert_stride_bytes = 16,
        },
    };
}

int test_q4_q8_dense_tails_and_exact_fallback() {
    for (const AffineQuantization quantization :
         {AffineQuantization::Q4, AffineQuantization::Q8}) {
        const auto regions = make_dense_regions(quantization);
        const QuantizedGemmCreationRequest request{
            .policy = QuantizedGemmPolicy::NativeDenseMma,
            .shape = make_dense_shape(quantization),
            .regions = regions,
            .tasks = {},
            .implementation = make_profile(),
            .available_workspace_bytes =
                std::numeric_limits<std::uint64_t>::max(),
        };
        const auto result =
            tatara::runtime::create_quantized_gemm_plan(request);
        if (!result || result.workspace.row_grid_groups != 2 ||
            result.workspace.column_grid_groups != 2 ||
            result.workspace.threadgroup_count != 4 ||
            result.workspace.threadgroup_memory_bytes != 3072 ||
            result.workspace.accumulator_elements_per_threadgroup != 128 ||
            result.workspace.region_descriptor_bytes !=
                2U * sizeof(QuantizedGemmRegionDescriptor) ||
            result.workspace.region_descriptor_offset != 0 ||
            result.workspace.task_descriptor_bytes != 0 ||
            result.workspace.device_task_indirect_argument_bytes != 0 ||
            result.workspace.device_task_status_bytes != 0 ||
            result.workspace.partial_bytes != 0 ||
            result.workspace.required_bytes == 0 ||
            result.required_output_bytes != 9U * 24U * 2U ||
            result.plan.selected_policy() !=
                QuantizedGemmPolicy::NativeDenseMma ||
            result.plan.task_mode() != QuantizedGemmTaskMode::HostCompact ||
            result.plan.device_tasks().version != 0 ||
            result.plan.device_task_capacity() != 0 ||
            result.plan.device_route_capacity() != 0 ||
            result.plan.implementation_identity().task_mode !=
                QuantizedGemmTaskMode::HostCompact ||
            result.plan.implementation_identity()
                    .device_task_descriptor_version !=
                0 ||
            result.plan.implementation_identity()
                    .required_threadgroup_memory_bytes !=
                3072 ||
            result.plan.implementation_identity()
                    .required_accumulator_elements_per_threadgroup !=
                128 ||
            result.plan.implementation_identity()
                    .activation_row_mapping !=
                QuantizedGemmActivationRowMapping::PositionRows ||
            result.plan.regions().data() == regions.data() ||
            result.plan.regions()[1].output_column_begin != 13) {
            return quantization == AffineQuantization::Q4 ? 1 : 2;
        }
    }

    auto owned_regions = make_dense_regions(AffineQuantization::Q4);
    const QuantizedGemmCreationRequest owned_request{
        .policy = QuantizedGemmPolicy::NativeDenseMma,
        .shape = make_dense_shape(AffineQuantization::Q4),
        .regions = owned_regions,
        .tasks = {},
        .implementation = make_profile(),
        .available_workspace_bytes =
            std::numeric_limits<std::uint64_t>::max(),
    };
    const auto owned =
        tatara::runtime::create_quantized_gemm_plan(owned_request);
    owned_regions[0].output_column_count = 0;
    if (!owned || owned.plan.regions()[0].output_column_count != 13) {
        return 3;
    }

    const auto regions = make_dense_regions(AffineQuantization::Q4);
    const QuantizedGemmCreationRequest exact_request{
        .policy = QuantizedGemmPolicy::ExactRow,
        .shape = make_dense_shape(AffineQuantization::Q4),
        .regions = regions,
        .tasks = {},
        .implementation = make_profile(),
        .available_workspace_bytes = 0,
    };
    const auto exact =
        tatara::runtime::create_quantized_gemm_plan(exact_request);
    if (!exact || exact.workspace.required_bytes != 0 ||
        exact.workspace.threadgroup_memory_bytes != 0 ||
        exact.plan.selected_policy() != QuantizedGemmPolicy::ExactRow ||
        exact.plan.fallback_policy() != QuantizedGemmPolicy::ExactRow ||
        exact.plan.automatic_fallback_allowed()) {
        return 4;
    }
    const auto exact_materialized =
        tatara::runtime::materialize_quantized_gemm_workspace(
            exact.plan,
            {
                .workspace = {},
                .route_input_rows = {},
                .route_output_rows = {},
            });
    if (!exact_materialized ||
        exact_materialized.bytes_written != 0) {
        return 5;
    }
    return 0;
}

int test_threadgroup_staging_resource_contract() {
    const std::uint64_t allocations_before = g_allocation_count;
    const auto regions = make_dense_regions(AffineQuantization::Q4);
    auto shape = make_dense_shape(AffineQuantization::Q4);
    shape.group_size = 64;

    auto profile = make_profile();
    profile.supported_group_size = 64;
    profile.tile_rows = 32;
    profile.tile_columns = 32;
    profile.tile_reduction_columns = 32;
    profile.activation_staging_row_stride_elements = 40;
    profile.weight_staging_row_stride_elements = 40;
    profile.threadgroup_staging_buffer_count = 1;
    profile.maximum_threadgroup_memory_bytes = 5120;
    profile.required_threadgroup_memory_bytes = 5120;
    profile.required_accumulator_elements_per_threadgroup = 1024;

    QuantizedGemmCreationRequest request{
        .policy = QuantizedGemmPolicy::NativeDenseMma,
        .shape = shape,
        .regions = regions,
        .tasks = {},
        .implementation = profile,
        .available_workspace_bytes =
            std::numeric_limits<std::uint64_t>::max(),
    };
    const auto result =
        tatara::runtime::create_quantized_gemm_plan(request);
    if (!result || result.workspace.threadgroup_memory_bytes != 5120) {
        return 1;
    }

    auto memory_limit = request;
    memory_limit.implementation.maximum_threadgroup_memory_bytes = 5119;
    if (tatara::runtime::create_quantized_gemm_plan(memory_limit).error !=
        QuantizedGemmCreationError::ThreadgroupMemoryExceeded) {
        return 2;
    }

    auto activation_stride = request;
    activation_stride.shape =
        make_dense_shape(AffineQuantization::Q4);
    activation_stride.implementation = make_profile();
    activation_stride.implementation
        .activation_staging_row_stride_elements = 31;
    if (tatara::runtime::create_quantized_gemm_plan(activation_stride).error !=
        QuantizedGemmCreationError::InvalidImplementationProfile) {
        return 3;
    }

    auto weight_stride = request;
    weight_stride.shape = make_dense_shape(AffineQuantization::Q4);
    weight_stride.implementation = make_profile();
    weight_stride.implementation.weight_staging_row_stride_elements = 15;
    if (tatara::runtime::create_quantized_gemm_plan(weight_stride).error !=
        QuantizedGemmCreationError::InvalidImplementationProfile) {
        return 4;
    }

    auto overflow = request;
    overflow.shape = make_dense_shape(AffineQuantization::Q4);
    overflow.implementation = make_profile();
    overflow.implementation.tile_rows =
        std::numeric_limits<std::uint32_t>::max();
    overflow.implementation.activation_staging_row_stride_elements =
        std::numeric_limits<std::uint64_t>::max();
    overflow.implementation.maximum_threadgroup_memory_bytes =
        std::numeric_limits<std::uint64_t>::max();
    overflow.implementation.maximum_accumulator_elements =
        std::numeric_limits<std::uint64_t>::max();
    if (tatara::runtime::create_quantized_gemm_plan(overflow).error !=
        QuantizedGemmCreationError::ArithmeticOverflow) {
        return 5;
    }

    auto exact = request;
    exact.policy = QuantizedGemmPolicy::ExactRow;
    exact.implementation.maximum_threadgroup_memory_bytes = 1;
    exact.available_workspace_bytes = 0;
    const auto exact_result =
        tatara::runtime::create_quantized_gemm_plan(exact);
    if (!exact_result ||
        exact_result.workspace.threadgroup_memory_bytes != 0 ||
        exact_result.workspace.required_bytes != 0) {
        return 6;
    }
    return g_allocation_count == allocations_before ? 0 : 7;
}

int test_ragged_tasks_partials_and_workspace() {
    const auto regions = make_ragged_regions();
    const auto tasks = make_ragged_tasks();
    auto profile = make_profile();
    profile.partial_partition_count = 3;
    const QuantizedGemmCreationRequest request{
        .policy = QuantizedGemmPolicy::NativeRaggedMma,
        .shape = make_ragged_shape(),
        .regions = regions,
        .tasks = tasks,
        .implementation = profile,
        .available_workspace_bytes =
            std::numeric_limits<std::uint64_t>::max(),
    };
    const std::uint64_t allocations_before = g_allocation_count;
    const auto result =
        tatara::runtime::create_quantized_gemm_plan(request);
    if (!result || g_allocation_count != allocations_before ||
        result.workspace.row_grid_groups != 3 ||
        result.workspace.column_grid_groups != 2 ||
        result.workspace.threadgroup_count != 18 ||
        result.workspace.reduction_grid_groups != 6 ||
        result.workspace.task_descriptor_bytes !=
            3U * sizeof(QuantizedGemmTaskDescriptor) ||
        result.workspace.route_input_index_bytes != 9U * sizeof(std::uint32_t) ||
        result.workspace.route_output_index_bytes != 9U * sizeof(std::uint32_t) ||
        result.workspace.partial_partition_count != 3 ||
        result.workspace.partial_bytes != 9U * 17U * 3U * 4U ||
        result.workspace.partial_offset ==
            tatara::runtime::kNoQuantizedGemmWorkspaceOffset ||
        result.workspace.partial_offset % result.workspace.alignment != 0 ||
        result.plan.tasks().data() == tasks.data() ||
        result.plan.tasks()[2].expert_index != 4) {
        return 1;
    }

    alignas(64) std::array<std::byte, 8192> workspace{};
    std::array<std::byte, 7U * 72U * 2U> activations{};
    std::array<std::byte, 3400> packed_weights{};
    std::array<std::byte, 680> scales{};
    std::array<std::byte, 680> biases{};
    std::array<std::byte, 9U * 24U * 2U> output{};
    constexpr std::array<std::uint32_t, 9> kInputRows{0, 1, 1, 2, 3,
                                                      3, 4, 5, 6};
    constexpr std::array<std::uint32_t, 9> kOutputRows{0, 1, 2, 3, 4,
                                                       5, 6, 7, 8};
    QuantizedGemmRuntimeRequest runtime{
        .policy = QuantizedGemmPolicy::NativeRaggedMma,
        .input_rows = result.plan.shape().input_rows,
        .output_rows = result.plan.shape().output_rows,
        .workspace = std::span<std::byte>{workspace}.first(
            result.workspace.required_bytes),
        .activations = activations,
        .packed_weights = packed_weights,
        .scales = scales,
        .biases = biases,
        .output = output,
        .route_input_rows = kInputRows,
        .route_output_rows = kOutputRows,
        .capability =
            {
                .identity = result.plan.implementation_identity(),
                .available = true,
            },
    };
    if (!tatara::runtime::validate_quantized_gemm_runtime(result.plan,
                                                           runtime)) {
        return 2;
    }
    auto invalid_indirection = runtime;
    constexpr std::array<std::uint32_t, 9> kInvalidInputRows{
        0, 1, 1, 2, 3, 3, 4, 5, 7};
    invalid_indirection.route_input_rows = kInvalidInputRows;
    if (tatara::runtime::validate_quantized_gemm_runtime(
            result.plan, invalid_indirection)
            .error != QuantizedGemmRuntimeError::InvalidIndirection) {
        return 3;
    }
    invalid_indirection = runtime;
    constexpr std::array<std::uint32_t, 9> kInvalidOutputRows{
        0, 1, 2, 3, 4, 5, 6, 7, 7};
    invalid_indirection.route_output_rows = kInvalidOutputRows;
    if (tatara::runtime::validate_quantized_gemm_runtime(
            result.plan, invalid_indirection)
            .error != QuantizedGemmRuntimeError::InvalidIndirection) {
        return 4;
    }

    workspace.fill(std::byte{0x5a});
    const tatara::runtime::QuantizedGemmMaterializationRequest
        invalid_materialization{
            .workspace = runtime.workspace,
            .route_input_rows = kInputRows,
            .route_output_rows = kInvalidOutputRows,
        };
    if (tatara::runtime::materialize_quantized_gemm_workspace(
            result.plan, invalid_materialization)
            .error !=
        tatara::runtime::QuantizedGemmMaterializationError::
            InvalidIndirection) {
        return 5;
    }
    for (const std::byte value : workspace) {
        if (value != std::byte{0x5a}) {
            return 6;
        }
    }

    const tatara::runtime::QuantizedGemmMaterializationRequest
        materialization{
            .workspace = runtime.workspace,
            .route_input_rows = kInputRows,
            .route_output_rows = kOutputRows,
        };
    const std::uint64_t materialization_allocations_before =
        g_allocation_count;
    const auto materialized =
        tatara::runtime::materialize_quantized_gemm_workspace(
            result.plan, materialization);
    const auto region_bytes = std::as_bytes(result.plan.regions());
    const auto task_bytes = std::as_bytes(result.plan.tasks());
    const auto input_bytes = std::as_bytes(std::span{kInputRows});
    const auto output_bytes = std::as_bytes(std::span{kOutputRows});
    if (!materialized ||
        g_allocation_count != materialization_allocations_before ||
        materialized.bytes_written !=
            region_bytes.size() + task_bytes.size() + input_bytes.size() +
                output_bytes.size() ||
        std::memcmp(runtime.workspace.data() +
                        result.workspace.region_descriptor_offset,
                    region_bytes.data(), region_bytes.size()) != 0 ||
        std::memcmp(runtime.workspace.data() +
                        result.workspace.task_descriptor_offset,
                    task_bytes.data(), task_bytes.size()) != 0 ||
        std::memcmp(runtime.workspace.data() +
                        result.workspace.route_input_index_offset,
                    input_bytes.data(), input_bytes.size()) != 0 ||
        std::memcmp(runtime.workspace.data() +
                        result.workspace.route_output_index_offset,
                    output_bytes.data(), output_bytes.size()) != 0) {
        return 7;
    }

    auto exact_request = request;
    exact_request.policy = QuantizedGemmPolicy::ExactRow;
    exact_request.available_workspace_bytes = 0;
    const auto exact =
        tatara::runtime::create_quantized_gemm_plan(exact_request);
    if (!exact || exact.workspace.required_bytes != 0 ||
        !tatara::runtime::materialize_quantized_gemm_workspace(
            exact.plan,
            {
                .workspace = {},
                .route_input_rows = kInputRows,
                .route_output_rows = kOutputRows,
            })) {
        return 8;
    }
    if (tatara::runtime::materialize_quantized_gemm_workspace(
            exact.plan,
            {
                .workspace = {},
                .route_input_rows = kInputRows,
                .route_output_rows = kInvalidOutputRows,
            })
            .error !=
        tatara::runtime::QuantizedGemmMaterializationError::
            InvalidIndirection) {
        return 9;
    }

    auto insufficient = request;
    insufficient.available_workspace_bytes =
        result.workspace.required_bytes - 1U;
    const auto insufficient_result =
        tatara::runtime::create_quantized_gemm_plan(insufficient);
    if (insufficient_result ||
        insufficient_result.error !=
            QuantizedGemmCreationError::WorkspaceInsufficient ||
        insufficient_result.workspace.required_bytes !=
            result.workspace.required_bytes ||
        insufficient_result.available_workspace_bytes !=
            result.workspace.required_bytes - 1U) {
        return 10;
    }

    auto invalid_task_values = tasks;
    invalid_task_values[1].route_list_begin = 3;
    auto invalid_task = request;
    invalid_task.tasks = invalid_task_values;
    if (tatara::runtime::create_quantized_gemm_plan(invalid_task).error !=
        QuantizedGemmCreationError::InvalidTask) {
        return 11;
    }

    invalid_task_values = tasks;
    invalid_task_values[2].expert_index = 2;
    invalid_task.tasks = invalid_task_values;
    if (tatara::runtime::create_quantized_gemm_plan(invalid_task).error !=
        QuantizedGemmCreationError::InvalidTask) {
        return 12;
    }
    return 0;
}

int test_creation_rejections_and_overflow() {
    const auto regions = make_dense_regions(AffineQuantization::Q4);
    QuantizedGemmCreationRequest request{
        .policy = QuantizedGemmPolicy::NativeDenseMma,
        .shape = make_dense_shape(AffineQuantization::Q4),
        .regions = regions,
        .tasks = {},
        .implementation = make_profile(),
        .available_workspace_bytes =
            std::numeric_limits<std::uint64_t>::max(),
    };

    auto zero = request;
    zero.shape.input_rows = 0;
    if (tatara::runtime::create_quantized_gemm_plan(zero).error !=
        QuantizedGemmCreationError::InvalidShape) {
        return 1;
    }

    auto bad_group = request;
    bad_group.shape.group_size = 18;
    if (tatara::runtime::create_quantized_gemm_plan(bad_group).error !=
        QuantizedGemmCreationError::UnsupportedGroup) {
        return 2;
    }

    auto partial_q4_group = request;
    partial_q4_group.shape.group_size = 23;
    partial_q4_group.implementation.supported_group_size = 23;
    if (!tatara::runtime::create_quantized_gemm_plan(partial_q4_group)) {
        return 3;
    }

    auto continuous_q4 = request;
    continuous_q4.shape.reduction_columns = 6;
    continuous_q4.shape.group_size = 3;
    continuous_q4.shape.activation_row_stride_elements = 8;
    continuous_q4.shape.activation_storage_bytes = 9U * 8U * 2U;
    continuous_q4.shape.packed_weight_row_stride_bytes = 3;
    continuous_q4.shape.packed_weight_storage_bytes = 19U * 3U;
    continuous_q4.shape.scale_row_stride_bytes = 4;
    continuous_q4.shape.bias_row_stride_bytes = 4;
    continuous_q4.shape.scale_storage_bytes = 19U * 4U;
    continuous_q4.shape.bias_storage_bytes = 19U * 4U;
    continuous_q4.implementation.supported_group_size = 3;
    auto continuous_regions = regions;
    continuous_regions[1].packed_weight_offset_bytes = 13U * 3U;
    continuous_regions[1].scale_offset_bytes = 13U * 4U;
    continuous_regions[1].bias_offset_bytes = 13U * 4U;
    continuous_q4.regions = continuous_regions;
    if (!tatara::runtime::create_quantized_gemm_plan(continuous_q4)) {
        return 4;
    }

    auto q8_odd_group = request;
    q8_odd_group.shape.quantization = AffineQuantization::Q8;
    q8_odd_group.shape.group_size = 9;
    q8_odd_group.shape.reduction_columns = 70;
    q8_odd_group.shape.packed_weight_row_stride_bytes = 72;
    q8_odd_group.shape.packed_weight_storage_bytes = 19U * 72U;
    q8_odd_group.shape.scale_row_stride_bytes = 16;
    q8_odd_group.shape.bias_row_stride_bytes = 16;
    q8_odd_group.shape.scale_storage_bytes = 19U * 16U;
    q8_odd_group.shape.bias_storage_bytes = 19U * 16U;
    q8_odd_group.implementation.supported_group_size = 9;
    auto q8_regions = make_dense_regions(AffineQuantization::Q8);
    q8_regions[1].packed_weight_offset_bytes = 13U * 72U;
    q8_regions[1].scale_offset_bytes = 13U * 16U;
    q8_regions[1].bias_offset_bytes = 13U * 16U;
    q8_odd_group.regions = q8_regions;
    if (!tatara::runtime::create_quantized_gemm_plan(q8_odd_group)) {
        return 5;
    }
    auto unsupported_q8 = q8_odd_group;
    unsupported_q8.implementation.q8_available = false;
    if (tatara::runtime::create_quantized_gemm_plan(unsupported_q8).error !=
        QuantizedGemmCreationError::UnsupportedQuantization) {
        return 6;
    }

    auto invalid_quantization = request;
    invalid_quantization.shape.quantization =
        static_cast<AffineQuantization>(7);
    if (tatara::runtime::create_quantized_gemm_plan(invalid_quantization)
            .error !=
        QuantizedGemmCreationError::UnsupportedQuantization) {
        return 7;
    }

    auto invalid_layout = request;
    invalid_layout.shape.weight_layout =
        static_cast<QuantizedWeightLayout>(99);
    if (tatara::runtime::create_quantized_gemm_plan(invalid_layout).error !=
        QuantizedGemmCreationError::UnsupportedLayout) {
        return 8;
    }

    auto invalid_regions = regions;
    invalid_regions[1].output_column_begin = 12;
    auto invalid_region = request;
    invalid_region.regions = invalid_regions;
    if (tatara::runtime::create_quantized_gemm_plan(invalid_region).error !=
        QuantizedGemmCreationError::InvalidRegion) {
        return 9;
    }

    invalid_regions = regions;
    invalid_regions[1].packed_weight_offset_bytes =
        std::numeric_limits<std::uint64_t>::max();
    invalid_region.regions = invalid_regions;
    if (tatara::runtime::create_quantized_gemm_plan(invalid_region).error !=
        QuantizedGemmCreationError::ArithmeticOverflow) {
        return 10;
    }

    const auto interleaved_shape = make_interleaved_shape();
    const auto interleaved_regions = make_interleaved_regions();
    const auto interleaved_tasks = make_ragged_tasks();
    QuantizedGemmCreationRequest interleaved{
        .policy = QuantizedGemmPolicy::NativeRaggedMma,
        .shape = interleaved_shape,
        .regions = interleaved_regions,
        .tasks = interleaved_tasks,
        .implementation = make_profile(),
        .available_workspace_bytes =
            std::numeric_limits<std::uint64_t>::max(),
    };
    if (!tatara::runtime::create_quantized_gemm_plan(interleaved)) {
        return 11;
    }
    auto overlapping_regions = interleaved_regions;
    overlapping_regions[0].packed_weight_offset_bytes = 34;
    interleaved.regions = overlapping_regions;
    if (tatara::runtime::create_quantized_gemm_plan(interleaved).error !=
        QuantizedGemmCreationError::InvalidRegion) {
        return 12;
    }

    auto unequal_stride_regions = interleaved_regions;
    unequal_stride_regions[0].scale_expert_stride_bytes = 6;
    unequal_stride_regions[1].scale_expert_stride_bytes = 36;
    interleaved.shape = interleaved_shape;
    interleaved.shape.scale_storage_bytes = 150;
    interleaved.regions = unequal_stride_regions;
    if (!tatara::runtime::create_quantized_gemm_plan(interleaved)) {
        return 13;
    }
    interleaved.policy = QuantizedGemmPolicy::ExactRow;
    interleaved.available_workspace_bytes = 0;
    const auto exact_unequal_stride =
        tatara::runtime::create_quantized_gemm_plan(interleaved);
    if (!exact_unequal_stride ||
        exact_unequal_stride.plan.selected_policy() !=
            QuantizedGemmPolicy::ExactRow ||
        exact_unequal_stride.workspace.required_bytes != 0) {
        return 14;
    }

    auto later_overlap_regions = interleaved_regions;
    later_overlap_regions[0].scale_expert_stride_bytes = 18;
    interleaved.policy = QuantizedGemmPolicy::NativeRaggedMma;
    interleaved.shape = interleaved_shape;
    interleaved.shape.scale_storage_bytes = 84;
    interleaved.regions = later_overlap_regions;
    interleaved.available_workspace_bytes =
        std::numeric_limits<std::uint64_t>::max();
    if (tatara::runtime::create_quantized_gemm_plan(interleaved).error !=
        QuantizedGemmCreationError::InvalidRegion) {
        return 15;
    }

    auto storage = request;
    storage.shape.output_storage_bytes -= 1U;
    if (tatara::runtime::create_quantized_gemm_plan(storage).error !=
        QuantizedGemmCreationError::InvalidStorageBounds) {
        return 16;
    }

    auto overflow = request;
    overflow.shape.output_rows = 2;
    overflow.shape.input_rows = 2;
    overflow.shape.output_row_stride_elements =
        std::numeric_limits<std::uint64_t>::max();
    overflow.shape.output_storage_bytes =
        std::numeric_limits<std::uint64_t>::max();
    if (tatara::runtime::create_quantized_gemm_plan(overflow).error !=
        QuantizedGemmCreationError::ArithmeticOverflow) {
        return 17;
    }

    auto thread_limit = request;
    thread_limit.implementation.threads_per_threadgroup = 257;
    if (tatara::runtime::create_quantized_gemm_plan(thread_limit).error !=
        QuantizedGemmCreationError::ThreadLimitExceeded) {
        return 18;
    }

    auto memory_limit = request;
    memory_limit.implementation.maximum_threadgroup_memory_bytes = 3071;
    if (tatara::runtime::create_quantized_gemm_plan(memory_limit).error !=
        QuantizedGemmCreationError::ThreadgroupMemoryExceeded) {
        return 19;
    }

    auto accumulator_limit = request;
    accumulator_limit.implementation.maximum_accumulator_elements = 127;
    if (tatara::runtime::create_quantized_gemm_plan(accumulator_limit).error !=
        QuantizedGemmCreationError::AccumulatorLimitExceeded) {
        return 20;
    }

    auto unavailable = request;
    unavailable.implementation.native_dense_mma_available = false;
    if (tatara::runtime::create_quantized_gemm_plan(unavailable).error !=
        QuantizedGemmCreationError::PolicyUnavailable) {
        return 21;
    }

    auto wrong_policy = request;
    wrong_policy.policy = QuantizedGemmPolicy::NativeRaggedMma;
    if (tatara::runtime::create_quantized_gemm_plan(wrong_policy).error !=
        QuantizedGemmCreationError::PolicyUnavailable) {
        return 22;
    }

    auto too_many_partitions = request;
    too_many_partitions.implementation.partial_partition_count = 4;
    if (tatara::runtime::create_quantized_gemm_plan(too_many_partitions)
            .error !=
        QuantizedGemmCreationError::InvalidImplementationProfile) {
        return 23;
    }

    auto invalid_profile_precedes_policy = request;
    invalid_profile_precedes_policy.implementation.profile_version = 1;
    invalid_profile_precedes_policy.implementation.native_dense_mma_available =
        false;
    if (tatara::runtime::create_quantized_gemm_plan(
            invalid_profile_precedes_policy)
            .error !=
        QuantizedGemmCreationError::InvalidImplementationProfile) {
        return 24;
    }
    return 0;
}

int test_extent_validation_has_fixed_work() {
    auto shape = make_interleaved_shape();
    shape.expert_count = std::numeric_limits<std::uint32_t>::max();
    shape.packed_weight_storage_bytes =
        std::numeric_limits<std::uint64_t>::max();
    shape.scale_storage_bytes =
        std::numeric_limits<std::uint64_t>::max();
    shape.bias_storage_bytes =
        std::numeric_limits<std::uint64_t>::max();
    const auto regions = make_interleaved_regions();
    const auto tasks = make_ragged_tasks();
    auto profile = make_profile();
    profile.maximum_experts =
        std::numeric_limits<std::uint32_t>::max();
    const QuantizedGemmCreationRequest request{
        .policy = QuantizedGemmPolicy::NativeRaggedMma,
        .shape = shape,
        .regions = regions,
        .tasks = tasks,
        .implementation = profile,
        .available_workspace_bytes =
            std::numeric_limits<std::uint64_t>::max(),
    };

    const std::uint64_t allocations_before = g_allocation_count;
    const auto result =
        tatara::runtime::create_quantized_gemm_plan(request);
    if (!result || g_allocation_count != allocations_before ||
        result.plan.shape().expert_count !=
            std::numeric_limits<std::uint32_t>::max() ||
        result.plan.regions().size() != regions.size()) {
        return 1;
    }
    return 0;
}

QuantizedGemmShapeDescriptor make_extent_oracle_shape(
    AffineQuantization quantization, std::uint32_t groups,
    std::uint32_t expert_count) {
    const std::uint64_t parameter_row_bytes =
        std::uint64_t{groups} * 2U;
    return {
        .workload = QuantizedGemmWorkload::Ragged,
        .quantization = quantization,
        .weight_layout =
            QuantizedWeightLayout::OutputMajorAffineGroups,
        .input_rows = 1,
        .output_rows = 1,
        .output_columns = 2,
        .reduction_columns = groups,
        .group_size = 1,
        .activation_element_bytes = 2,
        .output_element_bytes = 2,
        .quantization_parameter_bytes = 2,
        .activation_row_stride_elements = groups,
        .output_row_stride_elements = 2,
        .packed_weight_row_stride_bytes = groups,
        .scale_row_stride_bytes = parameter_row_bytes,
        .bias_row_stride_bytes = parameter_row_bytes,
        .expert_count = expert_count,
        .route_list_count = 1,
        .activation_storage_bytes = std::uint64_t{groups} * 2U,
        .packed_weight_storage_bytes =
            std::numeric_limits<std::uint64_t>::max(),
        .scale_storage_bytes =
            std::numeric_limits<std::uint64_t>::max(),
        .bias_storage_bytes =
            std::numeric_limits<std::uint64_t>::max(),
        .output_storage_bytes = 4,
    };
}

std::array<QuantizedGemmRegionDescriptor, 2> make_extent_oracle_regions(
    std::uint64_t left_begin, std::uint64_t left_stride,
    std::uint64_t right_begin, std::uint64_t right_stride) {
    return {
        QuantizedGemmRegionDescriptor{
            .output_column_begin = 0,
            .output_column_count = 1,
            .packed_weight_offset_bytes = left_begin,
            .scale_offset_bytes = left_begin,
            .bias_offset_bytes = left_begin,
            .packed_weight_expert_stride_bytes = left_stride,
            .scale_expert_stride_bytes = left_stride,
            .bias_expert_stride_bytes = left_stride,
        },
        QuantizedGemmRegionDescriptor{
            .output_column_begin = 1,
            .output_column_count = 1,
            .packed_weight_offset_bytes = right_begin,
            .scale_offset_bytes = right_begin,
            .bias_offset_bytes = right_begin,
            .packed_weight_expert_stride_bytes = right_stride,
            .scale_expert_stride_bytes = right_stride,
            .bias_expert_stride_bytes = right_stride,
        },
    };
}

bool oracle_extents_overlap(std::uint64_t left_begin,
                            std::uint64_t left_stride,
                            std::uint64_t right_begin,
                            std::uint64_t right_stride,
                            std::uint64_t bytes,
                            std::uint32_t count) {
    for (std::uint32_t left_index = 0; left_index < count;
         ++left_index) {
        const std::uint64_t left =
            left_begin + std::uint64_t{left_index} * left_stride;
        for (std::uint32_t right_index = 0; right_index < count;
             ++right_index) {
            const std::uint64_t right =
                right_begin +
                std::uint64_t{right_index} * right_stride;
            if (left < right + bytes && right < left + bytes) {
                return true;
            }
        }
    }
    return false;
}

int test_extent_relation_matches_small_oracle() {
    constexpr std::array<QuantizedGemmTaskDescriptor, 1> kTasks{
        QuantizedGemmTaskDescriptor{
            .expert_index = 0,
            .route_list_begin = 0,
            .row_count = 1,
            .output_row_begin = 0,
        },
    };
    const std::uint64_t allocations_before = g_allocation_count;
    for (const AffineQuantization quantization :
         {AffineQuantization::Q4, AffineQuantization::Q8}) {
        for (std::uint32_t groups = 1; groups <= 2; ++groups) {
            const std::uint64_t bytes = std::uint64_t{groups} * 2U;
            for (std::uint32_t expert_count = 1; expert_count <= 4;
                 ++expert_count) {
                for (std::uint64_t left_stride = bytes;
                     left_stride <= bytes + 4U; ++left_stride) {
                    for (std::uint64_t right_stride = bytes;
                         right_stride <= bytes + 4U; ++right_stride) {
                        for (std::uint64_t left_begin = 0;
                             left_begin <= 8; left_begin += 2U) {
                            for (std::uint64_t right_begin = 0;
                                 right_begin <= 8;
                                 right_begin += 2U) {
                                const auto regions =
                                    make_extent_oracle_regions(
                                        left_begin, left_stride,
                                        right_begin, right_stride);
                                auto profile = make_profile();
                                profile.supported_group_size = 1;
                                const QuantizedGemmCreationRequest request{
                                    .policy =
                                        QuantizedGemmPolicy::ExactRow,
                                    .shape = make_extent_oracle_shape(
                                        quantization, groups,
                                        expert_count),
                                    .regions = regions,
                                    .tasks = kTasks,
                                    .implementation = profile,
                                    .available_workspace_bytes = 0,
                                };
                                const bool overlap =
                                    oracle_extents_overlap(
                                        left_begin, left_stride,
                                        right_begin, right_stride, bytes,
                                        expert_count);
                                const auto result =
                                    tatara::runtime::
                                        create_quantized_gemm_plan(
                                            request);
                                if (overlap) {
                                    if (result.error !=
                                        QuantizedGemmCreationError::
                                            InvalidRegion) {
                                        return 1;
                                    }
                                } else if (!result) {
                                    return 2;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return g_allocation_count == allocations_before ? 0 : 3;
}

int test_extent_relation_near_u64_bounds() {
    constexpr std::array<QuantizedGemmTaskDescriptor, 1> kTasks{
        QuantizedGemmTaskDescriptor{
            .expert_index = 0,
            .route_list_begin = 0,
            .row_count = 1,
            .output_row_begin = 0,
        },
    };
    auto profile = make_profile();
    profile.supported_group_size = 1;
    const std::uint64_t maximum =
        std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t allocations_before = g_allocation_count;

    auto shape = make_extent_oracle_shape(
        AffineQuantization::Q8, 1, 2);
    auto regions = make_extent_oracle_regions(
        0, maximum - 10U, 2, maximum - 6U);
    QuantizedGemmCreationRequest request{
        .policy = QuantizedGemmPolicy::ExactRow,
        .shape = shape,
        .regions = regions,
        .tasks = kTasks,
        .implementation = profile,
        .available_workspace_bytes = 0,
    };
    if (!tatara::runtime::create_quantized_gemm_plan(request)) {
        return 1;
    }

    regions = make_extent_oracle_regions(
        0, maximum - 10U, 2, maximum - 12U);
    request.regions = regions;
    if (tatara::runtime::create_quantized_gemm_plan(request).error !=
        QuantizedGemmCreationError::InvalidRegion) {
        return 2;
    }

    shape.expert_count = 3;
    regions = make_extent_oracle_regions(
        maximum - 31U, 6, maximum - 29U, 8);
    request.shape = shape;
    request.regions = regions;
    if (!tatara::runtime::create_quantized_gemm_plan(request)) {
        return 3;
    }

    regions = make_extent_oracle_regions(
        maximum - 3U, 6, 0, 8);
    request.regions = regions;
    if (tatara::runtime::create_quantized_gemm_plan(request).error !=
        QuantizedGemmCreationError::ArithmeticOverflow) {
        return 4;
    }
    return g_allocation_count == allocations_before ? 0 : 5;
}

int test_runtime_validation_and_policy_immutability() {
    const auto regions = make_dense_regions(AffineQuantization::Q4);
    const QuantizedGemmCreationRequest creation{
        .policy = QuantizedGemmPolicy::NativeDenseMma,
        .shape = make_dense_shape(AffineQuantization::Q4),
        .regions = regions,
        .tasks = {},
        .implementation = make_profile(),
        .available_workspace_bytes =
            std::numeric_limits<std::uint64_t>::max(),
    };
    const auto created =
        tatara::runtime::create_quantized_gemm_plan(creation);
    if (!created || created.workspace.required_bytes > 8192) {
        return 1;
    }

    alignas(64) std::array<std::byte, 8192> workspace{};
    std::array<std::byte, 9U * 80U * 2U> activations{};
    std::array<std::byte, 19U * 40U> packed_weights{};
    std::array<std::byte, 19U * 8U> scales{};
    std::array<std::byte, 19U * 8U> biases{};
    std::array<std::byte, 9U * 24U * 2U> output{};
    QuantizedGemmRuntimeRequest request{
        .policy = QuantizedGemmPolicy::NativeDenseMma,
        .input_rows = created.plan.shape().input_rows,
        .output_rows = created.plan.shape().output_rows,
        .workspace =
            std::span<std::byte>{workspace}.first(
                created.workspace.required_bytes),
        .activations = activations,
        .packed_weights = packed_weights,
        .scales = scales,
        .biases = biases,
        .output = output,
        .capability =
            {
                .identity = created.plan.implementation_identity(),
                .available = true,
            },
    };
    const std::uint64_t allocations_before = g_allocation_count;
    if (!tatara::runtime::validate_quantized_gemm_runtime(created.plan,
                                                          request) ||
        g_allocation_count != allocations_before) {
        return 2;
    }

    auto changed_policy = request;
    changed_policy.policy = QuantizedGemmPolicy::ExactRow;
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, changed_policy)
            .error !=
        QuantizedGemmRuntimeError::PolicyChangeProhibited) {
        return 3;
    }

    auto invalid_range = request;
    --invalid_range.input_rows;
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, invalid_range)
            .error != QuantizedGemmRuntimeError::InvalidRange) {
        return 4;
    }

    invalid_range = request;
    invalid_range.device_task_position_count = 1;
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, invalid_range)
            .error != QuantizedGemmRuntimeError::InvalidRange) {
        return 20;
    }
    invalid_range = request;
    invalid_range.device_task_counts =
        std::span<const std::byte>{activations}.first(1);
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, invalid_range)
            .error != QuantizedGemmRuntimeError::InvalidRange) {
        return 21;
    }
    invalid_range = request;
    invalid_range.device_task_lists =
        std::span<const std::byte>{activations}.first(1);
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, invalid_range)
            .error != QuantizedGemmRuntimeError::InvalidRange) {
        return 22;
    }

    auto insufficient_workspace = request;
    insufficient_workspace.workspace =
        std::span<std::byte>{workspace}.first(
            created.workspace.required_bytes - 1U);
    const auto workspace_error =
        tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, insufficient_workspace);
    if (workspace_error.error !=
            QuantizedGemmRuntimeError::WorkspaceInsufficient ||
        workspace_error.required_workspace_bytes !=
            created.workspace.required_bytes ||
        workspace_error.available_workspace_bytes !=
            created.workspace.required_bytes - 1U) {
        return 5;
    }

    auto misaligned_workspace = request;
    misaligned_workspace.workspace = std::span<std::byte>{
        workspace.data() + 1,
        static_cast<std::size_t>(created.workspace.required_bytes)};
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, misaligned_workspace)
            .error != QuantizedGemmRuntimeError::WorkspaceMisaligned) {
        return 6;
    }

    auto missing_operand = request;
    missing_operand.activations = {};
    auto operand_error =
        tatara::runtime::validate_quantized_gemm_runtime(created.plan,
                                                         missing_operand);
    if (operand_error.error !=
            QuantizedGemmRuntimeError::OperandInsufficient ||
        operand_error.operand !=
            tatara::runtime::QuantizedGemmOperandKind::Activations) {
        return 7;
    }
    missing_operand = request;
    missing_operand.packed_weights = {};
    operand_error = tatara::runtime::validate_quantized_gemm_runtime(
        created.plan, missing_operand);
    if (operand_error.error !=
            QuantizedGemmRuntimeError::OperandInsufficient ||
        operand_error.operand !=
            tatara::runtime::QuantizedGemmOperandKind::PackedWeights) {
        return 8;
    }
    missing_operand = request;
    missing_operand.scales = {};
    operand_error = tatara::runtime::validate_quantized_gemm_runtime(
        created.plan, missing_operand);
    if (operand_error.error !=
            QuantizedGemmRuntimeError::OperandInsufficient ||
        operand_error.operand !=
            tatara::runtime::QuantizedGemmOperandKind::Scales) {
        return 9;
    }
    missing_operand = request;
    missing_operand.biases = {};
    operand_error = tatara::runtime::validate_quantized_gemm_runtime(
        created.plan, missing_operand);
    if (operand_error.error !=
            QuantizedGemmRuntimeError::OperandInsufficient ||
        operand_error.operand !=
            tatara::runtime::QuantizedGemmOperandKind::Biases) {
        return 10;
    }

    auto aliased_operand = request;
    aliased_operand.activations = std::span<const std::byte>{
        workspace.data(),
        static_cast<std::size_t>(
            created.plan.required_activation_bytes())};
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, aliased_operand)
            .error !=
        QuantizedGemmRuntimeError::WorkspaceAliasesOperand) {
        return 11;
    }
    aliased_operand = request;
    aliased_operand.packed_weights = std::span<const std::byte>{
        workspace.data(),
        static_cast<std::size_t>(
            created.plan.required_packed_weight_bytes())};
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, aliased_operand)
            .error !=
        QuantizedGemmRuntimeError::WorkspaceAliasesOperand) {
        return 12;
    }
    aliased_operand = request;
    aliased_operand.scales = std::span<const std::byte>{
        workspace.data(),
        static_cast<std::size_t>(created.plan.required_scale_bytes())};
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, aliased_operand)
            .error !=
        QuantizedGemmRuntimeError::WorkspaceAliasesOperand) {
        return 13;
    }
    aliased_operand = request;
    aliased_operand.biases = std::span<const std::byte>{
        workspace.data(),
        static_cast<std::size_t>(created.plan.required_bias_bytes())};
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, aliased_operand)
            .error !=
        QuantizedGemmRuntimeError::WorkspaceAliasesOperand) {
        return 14;
    }
    aliased_operand = request;
    aliased_operand.output = std::span<std::byte>{
        workspace.data(),
        static_cast<std::size_t>(created.plan.required_output_bytes())};
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, aliased_operand)
            .error !=
        QuantizedGemmRuntimeError::WorkspaceAliasesOperand) {
        return 15;
    }

    auto insufficient_output = request;
    insufficient_output.output =
        std::span<std::byte>{output}.first(
            created.required_output_bytes - 1U);
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, insufficient_output)
            .error != QuantizedGemmRuntimeError::OutputInsufficient) {
        return 16;
    }

    auto unavailable_capability = request;
    unavailable_capability.capability.available = false;
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, unavailable_capability)
            .error !=
        QuantizedGemmRuntimeError::CapabilityUnavailable) {
        return 17;
    }

    auto mismatched_capability = request;
    ++mismatched_capability.capability.identity.specialization_id;
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, mismatched_capability)
            .error !=
        QuantizedGemmRuntimeError::CapabilityIdentityMismatch) {
        return 18;
    }
    mismatched_capability = request;
    ++mismatched_capability.capability.identity.specialization_version;
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, mismatched_capability)
            .error !=
        QuantizedGemmRuntimeError::CapabilityIdentityMismatch) {
        return 21;
    }

    const tatara::runtime::QuantizedGemmRequestPlan invalid_plan;
    if (tatara::runtime::validate_quantized_gemm_runtime(invalid_plan, request)
            .error != QuantizedGemmRuntimeError::InvalidPlan) {
        return 19;
    }
    return 0;
}

int test_device_padded_task_creation_and_materialization() {
    const auto regions = make_ragged_regions();
    auto profile = make_profile();
    profile.device_padded_slots_v1_available = true;
    const QuantizedGemmCreationRequest request{
        .policy = QuantizedGemmPolicy::NativeRaggedMma,
        .shape = make_device_task_shape(
            QuantizedGemmActivationRowMapping::PositionRows),
        .regions = regions,
        .tasks = {},
        .task_mode =
            QuantizedGemmTaskMode::DevicePaddedSlotsV1,
        .device_tasks = make_device_task_descriptor(
            QuantizedGemmActivationRowMapping::PositionRows),
        .implementation = profile,
        .available_workspace_bytes =
            std::numeric_limits<std::uint64_t>::max(),
    };
    const std::uint64_t allocations_before = g_allocation_count;
    const auto created =
        tatara::runtime::create_quantized_gemm_plan(request);
    if (!created || g_allocation_count != allocations_before ||
        created.plan.task_mode() !=
            QuantizedGemmTaskMode::DevicePaddedSlotsV1 ||
        created.plan.device_tasks().version !=
            tatara::runtime::
                kQuantizedGemmDeviceTaskDescriptorVersion ||
        created.plan.device_route_capacity() != 14 ||
        created.plan.device_task_capacity() != 7 ||
        created.plan.required_device_task_count_bytes() != 20 ||
        created.plan.required_device_task_list_bytes() != 156 ||
        !created.plan.tasks().empty() ||
        created.workspace.row_grid_groups != 7 ||
        created.workspace.column_grid_groups != 2 ||
        created.workspace.threadgroup_count != 14 ||
        created.workspace.task_descriptor_bytes !=
            7U * sizeof(QuantizedGemmTaskDescriptor) ||
        created.workspace.device_task_indirect_argument_bytes !=
            3U * sizeof(std::uint32_t) ||
        created.workspace.device_task_status_bytes !=
            sizeof(QuantizedGemmDeviceTaskStatus) ||
        created.workspace.route_input_index_bytes != 0 ||
        created.workspace.route_output_index_bytes != 0 ||
        created.workspace.device_task_indirect_argument_offset ==
            tatara::runtime::kNoQuantizedGemmWorkspaceOffset ||
        created.workspace.device_task_status_offset ==
            tatara::runtime::kNoQuantizedGemmWorkspaceOffset ||
        created.workspace.device_task_indirect_argument_offset %
                created.workspace.alignment !=
            0 ||
        created.workspace.device_task_status_offset %
                created.workspace.alignment !=
            0 ||
        created.plan.implementation_identity().task_mode !=
            QuantizedGemmTaskMode::DevicePaddedSlotsV1 ||
        created.plan.implementation_identity()
                .device_task_descriptor_version !=
            tatara::runtime::
                kQuantizedGemmDeviceTaskDescriptorVersion ||
        created.plan.implementation_identity()
                .required_threadgroup_memory_bytes !=
            profile.required_threadgroup_memory_bytes ||
        created.plan.implementation_identity()
                .required_accumulator_elements_per_threadgroup !=
            profile.required_accumulator_elements_per_threadgroup ||
        created.plan.implementation_identity()
                .activation_row_mapping !=
            QuantizedGemmActivationRowMapping::PositionRows) {
        return 1;
    }

    auto padded_request = request;
    padded_request.shape = make_device_task_shape(
        QuantizedGemmActivationRowMapping::PaddedSlotRows);
    padded_request.device_tasks = make_device_task_descriptor(
        QuantizedGemmActivationRowMapping::PaddedSlotRows);
    padded_request.implementation.activation_row_mapping =
        QuantizedGemmActivationRowMapping::PaddedSlotRows;
    const auto padded =
        tatara::runtime::create_quantized_gemm_plan(padded_request);
    if (!padded ||
        padded.plan.shape().input_rows != 21 ||
        padded.plan.shape().output_rows != 21 ||
        padded.plan.device_task_capacity() != 7) {
        return 2;
    }

    alignas(64) std::array<std::byte, 1024> workspace;
    workspace.fill(std::byte{0x5a});
    const auto materialized =
        tatara::runtime::materialize_quantized_gemm_workspace(
            created.plan,
            {
                .workspace = std::span<std::byte>{workspace}.first(
                    created.workspace.required_bytes),
                .route_input_rows = {},
                .route_output_rows = {},
            });
    const auto region_bytes = std::as_bytes(created.plan.regions());
    if (!materialized ||
        g_allocation_count != allocations_before ||
        materialized.bytes_written != region_bytes.size() ||
        std::memcmp(
            workspace.data() +
                created.workspace.region_descriptor_offset,
            region_bytes.data(), region_bytes.size()) != 0) {
        return 3;
    }
    for (const auto [offset, bytes] :
         std::array<std::array<std::uint64_t, 2>, 3>{
             std::array<std::uint64_t, 2>{
                 created.workspace.task_descriptor_offset,
                 created.workspace.task_descriptor_bytes},
             std::array<std::uint64_t, 2>{
                 created.workspace
                     .device_task_indirect_argument_offset,
                 created.workspace
                     .device_task_indirect_argument_bytes},
             std::array<std::uint64_t, 2>{
                 created.workspace.device_task_status_offset,
                 created.workspace.device_task_status_bytes},
         }) {
        for (std::uint64_t index = 0; index < bytes; ++index) {
            if (workspace[offset + index] != std::byte{0x5a}) {
                return 4;
            }
        }
    }

    constexpr std::array<std::uint32_t, 1> kForbiddenRoute{0};
    workspace.fill(std::byte{0x6b});
    const auto invalid_materialization =
        tatara::runtime::materialize_quantized_gemm_workspace(
            created.plan,
            {
                .workspace = std::span<std::byte>{workspace}.first(
                    created.workspace.required_bytes),
                .route_input_rows = kForbiddenRoute,
                .route_output_rows = {},
            });
    if (invalid_materialization.error !=
            tatara::runtime::QuantizedGemmMaterializationError::
                InvalidIndirection ||
        g_allocation_count != allocations_before) {
        return 5;
    }
    for (const std::byte value : workspace) {
        if (value != std::byte{0x6b}) {
            return 6;
        }
    }
    return 0;
}

int test_device_padded_task_rejections_and_runtime() {
    const auto regions = make_ragged_regions();
    auto profile = make_profile();
    profile.device_padded_slots_v1_available = true;
    QuantizedGemmCreationRequest request{
        .policy = QuantizedGemmPolicy::NativeRaggedMma,
        .shape = make_device_task_shape(
            QuantizedGemmActivationRowMapping::PositionRows),
        .regions = regions,
        .tasks = {},
        .task_mode =
            QuantizedGemmTaskMode::DevicePaddedSlotsV1,
        .device_tasks = make_device_task_descriptor(
            QuantizedGemmActivationRowMapping::PositionRows),
        .implementation = profile,
        .available_workspace_bytes =
            std::numeric_limits<std::uint64_t>::max(),
    };
    const std::uint64_t allocations_before = g_allocation_count;
    const auto created =
        tatara::runtime::create_quantized_gemm_plan(request);
    if (!created || g_allocation_count != allocations_before) {
        return 1;
    }

    auto mutation = request;
    mutation.task_mode = static_cast<QuantizedGemmTaskMode>(99);
    if (tatara::runtime::create_quantized_gemm_plan(mutation).error !=
        QuantizedGemmCreationError::PolicyUnavailable) {
        return 2;
    }
    mutation = request;
    mutation.implementation.device_padded_slots_v1_available = false;
    if (tatara::runtime::create_quantized_gemm_plan(mutation).error !=
        QuantizedGemmCreationError::PolicyUnavailable) {
        return 3;
    }
    mutation = request;
    ++mutation.device_tasks.version;
    if (tatara::runtime::create_quantized_gemm_plan(mutation).error !=
        QuantizedGemmCreationError::PolicyUnavailable) {
        return 4;
    }
    mutation = request;
    const auto host_tasks = make_ragged_tasks();
    mutation.tasks = host_tasks;
    if (tatara::runtime::create_quantized_gemm_plan(mutation).error !=
        QuantizedGemmCreationError::InvalidTask) {
        return 5;
    }
    mutation = request;
    mutation.task_mode = QuantizedGemmTaskMode::HostCompact;
    if (tatara::runtime::create_quantized_gemm_plan(mutation).error !=
        QuantizedGemmCreationError::InvalidTask) {
        return 6;
    }
    mutation = request;
    mutation.device_tasks.position_capacity = 0;
    if (tatara::runtime::create_quantized_gemm_plan(mutation).error !=
        QuantizedGemmCreationError::InvalidTask) {
        return 36;
    }
    mutation = request;
    mutation.device_tasks.routes_per_position = 0;
    if (tatara::runtime::create_quantized_gemm_plan(mutation).error !=
        QuantizedGemmCreationError::InvalidTask) {
        return 37;
    }
    mutation = request;
    mutation.device_tasks.routes_per_position =
        mutation.shape.expert_count + 1U;
    if (tatara::runtime::create_quantized_gemm_plan(mutation).error !=
        QuantizedGemmCreationError::InvalidTask) {
        return 38;
    }
    mutation = request;
    mutation.device_tasks.padded_slot_stride =
        mutation.device_tasks.routes_per_position;
    if (tatara::runtime::create_quantized_gemm_plan(mutation).error !=
        QuantizedGemmCreationError::InvalidTask) {
        return 7;
    }
    mutation = request;
    mutation.device_tasks.activation_row_mapping =
        static_cast<QuantizedGemmActivationRowMapping>(99);
    if (tatara::runtime::create_quantized_gemm_plan(mutation).error !=
        QuantizedGemmCreationError::InvalidTask) {
        return 26;
    }
    mutation = request;
    mutation.device_tasks.packed_slot_bits = 0;
    if (tatara::runtime::create_quantized_gemm_plan(mutation).error !=
        QuantizedGemmCreationError::InvalidTask) {
        return 39;
    }
    mutation = request;
    mutation.device_tasks.packed_slot_bits =
        std::numeric_limits<std::uint32_t>::digits;
    if (tatara::runtime::create_quantized_gemm_plan(mutation).error !=
        QuantizedGemmCreationError::InvalidTask) {
        return 40;
    }
    mutation = request;
    mutation.device_tasks.packed_slot_bits = 1;
    if (tatara::runtime::create_quantized_gemm_plan(mutation).error !=
        QuantizedGemmCreationError::InvalidTask) {
        return 27;
    }
    mutation = request;
    mutation.device_tasks.position_capacity =
        (std::numeric_limits<std::uint32_t>::max() >>
         mutation.device_tasks.packed_slot_bits) +
        2U;
    mutation.device_tasks.list_capacity_entries =
        mutation.device_tasks.position_capacity;
    mutation.device_tasks.list_expert_stride_entries =
        mutation.device_tasks.list_capacity_entries;
    if (tatara::runtime::create_quantized_gemm_plan(mutation).error !=
        QuantizedGemmCreationError::InvalidTask) {
        return 41;
    }
    mutation = request;
    --mutation.shape.output_rows;
    --mutation.shape.output_storage_bytes;
    if (tatara::runtime::create_quantized_gemm_plan(mutation).error !=
        QuantizedGemmCreationError::InvalidTask) {
        return 28;
    }
    mutation = request;
    mutation.device_tasks.list_capacity_entries = 6;
    if (tatara::runtime::create_quantized_gemm_plan(mutation).error !=
        QuantizedGemmCreationError::InvalidTask) {
        return 8;
    }
    mutation = request;
    mutation.device_tasks.list_expert_stride_entries =
        mutation.device_tasks.list_capacity_entries - 1U;
    if (tatara::runtime::create_quantized_gemm_plan(mutation).error !=
        QuantizedGemmCreationError::InvalidTask) {
        return 42;
    }
    mutation = request;
    --mutation.shape.route_list_count;
    if (tatara::runtime::create_quantized_gemm_plan(mutation).error !=
        QuantizedGemmCreationError::InvalidTask) {
        return 43;
    }
    mutation = request;
    mutation.device_tasks.list_expert_stride_entries =
        std::numeric_limits<std::uint64_t>::max();
    mutation.device_tasks.list_storage_bytes =
        std::numeric_limits<std::uint64_t>::max();
    if (tatara::runtime::create_quantized_gemm_plan(mutation).error !=
        QuantizedGemmCreationError::ArithmeticOverflow) {
        return 9;
    }
    mutation = request;
    mutation.implementation.maximum_tasks = 6;
    if (tatara::runtime::create_quantized_gemm_plan(mutation).error !=
        QuantizedGemmCreationError::DescriptorLimitExceeded) {
        return 10;
    }
    mutation = request;
    --mutation.device_tasks.count_storage_bytes;
    if (tatara::runtime::create_quantized_gemm_plan(mutation).error !=
        QuantizedGemmCreationError::InvalidStorageBounds) {
        return 11;
    }
    mutation = request;
    mutation.device_tasks.list_storage_bytes = 155;
    if (tatara::runtime::create_quantized_gemm_plan(mutation).error !=
        QuantizedGemmCreationError::InvalidStorageBounds) {
        return 12;
    }
    mutation = request;
    mutation.available_workspace_bytes =
        created.workspace.required_bytes - 1U;
    const auto insufficient_workspace =
        tatara::runtime::create_quantized_gemm_plan(mutation);
    if (insufficient_workspace.error !=
            QuantizedGemmCreationError::WorkspaceInsufficient ||
        insufficient_workspace.workspace.required_bytes !=
            created.workspace.required_bytes) {
        return 13;
    }

    alignas(64) std::array<std::byte, 1024> workspace{};
    alignas(64) std::array<std::byte, 4096> activations{};
    alignas(64) std::array<std::byte, 3400> packed_weights{};
    alignas(64) std::array<std::byte, 680> scales{};
    alignas(64) std::array<std::byte, 680> biases{};
    alignas(64) std::array<std::byte, 1024> output{};
    alignas(64) std::array<std::byte, 32> counts{};
    alignas(64) std::array<std::byte, 192> lists{};
    QuantizedGemmRuntimeRequest runtime{
        .policy = QuantizedGemmPolicy::NativeRaggedMma,
        .input_rows = 7,
        .output_rows = 21,
        .workspace = std::span<std::byte>{workspace}.first(
            created.workspace.required_bytes),
        .activations = activations,
        .packed_weights = packed_weights,
        .scales = scales,
        .biases = biases,
        .output = output,
        .route_input_rows = {},
        .route_output_rows = {},
        .device_task_position_count = 7,
        .device_task_counts =
            std::span<const std::byte>{counts}.first(20),
        .device_task_lists =
            std::span<const std::byte>{lists}.first(156),
        .capability =
            {
                .identity = created.plan.implementation_identity(),
                .available = true,
            },
    };
    if (!tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, runtime) ||
        g_allocation_count != allocations_before) {
        return 14;
    }

    auto partial_position_runtime = runtime;
    partial_position_runtime.device_task_position_count = 3;
    partial_position_runtime.input_rows = 3;
    partial_position_runtime.output_rows = 9;
    if (!tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, partial_position_runtime)) {
        return 44;
    }
    const std::uint64_t actual_position_activation_bytes =
        std::uint64_t{partial_position_runtime.input_rows} *
        created.plan.shape().activation_row_stride_elements *
        created.plan.shape().activation_element_bytes;
    auto actual_sized_position_activations = partial_position_runtime;
    actual_sized_position_activations.activations =
        std::span<const std::byte>{activations}.first(
            static_cast<std::size_t>(actual_position_activation_bytes));
    const auto activation_capacity_error =
        tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, actual_sized_position_activations);
    if (activation_capacity_error.error !=
            QuantizedGemmRuntimeError::OperandInsufficient ||
        activation_capacity_error.operand !=
            tatara::runtime::QuantizedGemmOperandKind::Activations ||
        activation_capacity_error.required_operand_bytes !=
            created.plan.required_activation_bytes() ||
        activation_capacity_error.available_operand_bytes !=
            actual_position_activation_bytes) {
        return 45;
    }
    const std::uint64_t actual_position_output_bytes =
        std::uint64_t{partial_position_runtime.output_rows} *
        created.plan.shape().output_row_stride_elements *
        created.plan.shape().output_element_bytes;
    auto actual_sized_position_output = partial_position_runtime;
    actual_sized_position_output.output =
        std::span<std::byte>{output}.first(
            static_cast<std::size_t>(actual_position_output_bytes));
    const auto output_capacity_error =
        tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, actual_sized_position_output);
    if (output_capacity_error.error !=
            QuantizedGemmRuntimeError::OutputInsufficient ||
        output_capacity_error.operand !=
            tatara::runtime::QuantizedGemmOperandKind::Output ||
        output_capacity_error.required_operand_bytes !=
            created.plan.required_output_bytes() ||
        output_capacity_error.available_operand_bytes !=
            actual_position_output_bytes) {
        return 46;
    }
    auto invalid_partial_range = partial_position_runtime;
    invalid_partial_range.device_task_position_count = 8;
    invalid_partial_range.input_rows = 8;
    invalid_partial_range.output_rows = 24;
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, invalid_partial_range)
            .error != QuantizedGemmRuntimeError::InvalidRange) {
        return 47;
    }
    invalid_partial_range = partial_position_runtime;
    --invalid_partial_range.input_rows;
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, invalid_partial_range)
            .error != QuantizedGemmRuntimeError::InvalidRange) {
        return 48;
    }
    invalid_partial_range = partial_position_runtime;
    --invalid_partial_range.output_rows;
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, invalid_partial_range)
            .error != QuantizedGemmRuntimeError::InvalidRange) {
        return 49;
    }

    auto runtime_mutation = runtime;
    runtime_mutation.device_task_position_count = 0;
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, runtime_mutation)
            .error != QuantizedGemmRuntimeError::InvalidRange) {
        return 15;
    }
    runtime_mutation = runtime;
    runtime_mutation.input_rows = 6;
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, runtime_mutation)
            .error != QuantizedGemmRuntimeError::InvalidRange) {
        return 16;
    }
    runtime_mutation = runtime;
    runtime_mutation.device_task_counts =
        runtime.device_task_counts.first(19);
    const auto count_error =
        tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, runtime_mutation);
    if (count_error.error !=
            QuantizedGemmRuntimeError::OperandInsufficient ||
        count_error.operand !=
            tatara::runtime::QuantizedGemmOperandKind::
                DeviceTaskCounts ||
        count_error.required_operand_bytes !=
            created.plan.required_device_task_count_bytes() ||
        count_error.available_operand_bytes != 19) {
        return 17;
    }
    runtime_mutation = runtime;
    runtime_mutation.device_task_lists =
        runtime.device_task_lists.first(155);
    const auto list_error =
        tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, runtime_mutation);
    if (list_error.error !=
            QuantizedGemmRuntimeError::OperandInsufficient ||
        list_error.operand !=
            tatara::runtime::QuantizedGemmOperandKind::
                DeviceTaskLists ||
        list_error.required_operand_bytes !=
            created.plan.required_device_task_list_bytes() ||
        list_error.available_operand_bytes != 155) {
        return 18;
    }
    runtime_mutation = runtime;
    runtime_mutation.device_task_counts =
        std::span<const std::byte>{counts}.subspan(1, 20);
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, runtime_mutation)
            .error !=
        QuantizedGemmRuntimeError::DeviceTaskResourceMisaligned) {
        return 19;
    }
    runtime_mutation = runtime;
    runtime_mutation.device_task_lists =
        std::span<const std::byte>{lists}.subspan(1, 156);
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, runtime_mutation)
            .error !=
        QuantizedGemmRuntimeError::DeviceTaskResourceMisaligned) {
        return 29;
    }
    runtime_mutation = runtime;
    runtime_mutation.device_task_counts =
        std::span<const std::byte>{
            runtime.workspace.data(), 20};
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, runtime_mutation)
            .error !=
        QuantizedGemmRuntimeError::WorkspaceAliasesOperand) {
        return 20;
    }
    runtime_mutation = runtime;
    runtime_mutation.device_task_lists =
        std::span<const std::byte>{
            runtime.workspace.data(), 156};
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, runtime_mutation)
            .error !=
        QuantizedGemmRuntimeError::WorkspaceAliasesOperand) {
        return 30;
    }
    runtime_mutation = runtime;
    runtime_mutation.device_task_counts =
        std::span<const std::byte>{activations}.first(20);
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, runtime_mutation)
            .error != QuantizedGemmRuntimeError::
                DeviceTaskResourceAliasesOperand) {
        return 21;
    }
    runtime_mutation = runtime;
    runtime_mutation.device_task_counts =
        std::span<const std::byte>{packed_weights}.first(20);
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, runtime_mutation)
            .error != QuantizedGemmRuntimeError::
                DeviceTaskResourceAliasesOperand) {
        return 31;
    }
    runtime_mutation = runtime;
    runtime_mutation.device_task_counts =
        std::span<const std::byte>{scales}.first(20);
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, runtime_mutation)
            .error != QuantizedGemmRuntimeError::
                DeviceTaskResourceAliasesOperand) {
        return 32;
    }
    runtime_mutation = runtime;
    runtime_mutation.device_task_counts =
        std::span<const std::byte>{biases}.first(20);
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, runtime_mutation)
            .error != QuantizedGemmRuntimeError::
                DeviceTaskResourceAliasesOperand) {
        return 33;
    }
    runtime_mutation = runtime;
    runtime_mutation.device_task_counts =
        std::span<const std::byte>{output.data(), 20};
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, runtime_mutation)
            .error != QuantizedGemmRuntimeError::
                DeviceTaskResourceAliasesOperand) {
        return 34;
    }
    runtime_mutation = runtime;
    runtime_mutation.device_task_counts =
        std::span<const std::byte>{lists}.first(20);
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, runtime_mutation)
            .error != QuantizedGemmRuntimeError::
                DeviceTaskResourceAliasesOperand) {
        return 22;
    }
    runtime_mutation = runtime;
    ++runtime_mutation.capability.identity
          .device_task_descriptor_version;
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, runtime_mutation)
            .error !=
        QuantizedGemmRuntimeError::CapabilityIdentityMismatch) {
        return 23;
    }
    runtime_mutation = runtime;
    runtime_mutation.capability.identity.task_mode =
        QuantizedGemmTaskMode::HostCompact;
    if (tatara::runtime::validate_quantized_gemm_runtime(
            created.plan, runtime_mutation)
            .error !=
        QuantizedGemmRuntimeError::CapabilityIdentityMismatch) {
        return 24;
    }

    auto padded_profile = profile;
    padded_profile.activation_row_mapping =
        QuantizedGemmActivationRowMapping::PaddedSlotRows;
    const auto padded_created =
        tatara::runtime::create_quantized_gemm_plan({
            .policy = QuantizedGemmPolicy::NativeRaggedMma,
            .shape = make_device_task_shape(
                QuantizedGemmActivationRowMapping::PaddedSlotRows),
            .regions = regions,
            .tasks = {},
            .task_mode =
                QuantizedGemmTaskMode::DevicePaddedSlotsV1,
            .device_tasks = make_device_task_descriptor(
                QuantizedGemmActivationRowMapping::PaddedSlotRows),
            .implementation = padded_profile,
            .available_workspace_bytes =
                std::numeric_limits<std::uint64_t>::max(),
        });
    runtime_mutation = runtime;
    runtime_mutation.device_task_position_count = 3;
    runtime_mutation.input_rows = 9;
    runtime_mutation.output_rows = 9;
    runtime_mutation.capability.identity =
        padded_created.plan.implementation_identity();
    if (!padded_created ||
        !tatara::runtime::validate_quantized_gemm_runtime(
            padded_created.plan, runtime_mutation)) {
        return 35;
    }
    const std::uint64_t actual_padded_activation_bytes =
        std::uint64_t{runtime_mutation.input_rows} *
        padded_created.plan.shape().activation_row_stride_elements *
        padded_created.plan.shape().activation_element_bytes;
    auto actual_sized_padded_activations = runtime_mutation;
    actual_sized_padded_activations.activations =
        std::span<const std::byte>{activations}.first(
            static_cast<std::size_t>(actual_padded_activation_bytes));
    const auto padded_activation_capacity_error =
        tatara::runtime::validate_quantized_gemm_runtime(
            padded_created.plan, actual_sized_padded_activations);
    if (padded_activation_capacity_error.error !=
            QuantizedGemmRuntimeError::OperandInsufficient ||
        padded_activation_capacity_error.operand !=
            tatara::runtime::QuantizedGemmOperandKind::Activations ||
        padded_activation_capacity_error.required_operand_bytes !=
            padded_created.plan.required_activation_bytes() ||
        padded_activation_capacity_error.available_operand_bytes !=
            actual_padded_activation_bytes) {
        return 50;
    }
    auto invalid_padded_rows = runtime_mutation;
    --invalid_padded_rows.input_rows;
    if (tatara::runtime::validate_quantized_gemm_runtime(
            padded_created.plan, invalid_padded_rows)
            .error != QuantizedGemmRuntimeError::InvalidRange) {
        return 51;
    }
    invalid_padded_rows = runtime_mutation;
    --invalid_padded_rows.output_rows;
    if (tatara::runtime::validate_quantized_gemm_runtime(
            padded_created.plan, invalid_padded_rows)
            .error != QuantizedGemmRuntimeError::InvalidRange) {
        return 52;
    }
    return g_allocation_count == allocations_before ? 0 : 25;
}

int test_profile_v3_resources_mapping_and_identity() {
    const auto ragged_regions = make_ragged_regions();
    const auto position_shape = make_device_task_shape(
        QuantizedGemmActivationRowMapping::PositionRows);
    const auto position_tasks = make_device_task_descriptor(
        QuantizedGemmActivationRowMapping::PositionRows);
    const auto fused_profile = make_direct_profile(
        1024, QuantizedGemmActivationRowMapping::PositionRows, 101, 3);
    const QuantizedGemmCreationRequest fused_request{
        .policy = QuantizedGemmPolicy::NativeRaggedMma,
        .shape = position_shape,
        .regions = ragged_regions,
        .tasks = {},
        .task_mode =
            QuantizedGemmTaskMode::DevicePaddedSlotsV1,
        .device_tasks = position_tasks,
        .implementation = fused_profile,
        .available_workspace_bytes =
            std::numeric_limits<std::uint64_t>::max(),
    };
    const std::uint64_t allocations_before = g_allocation_count;
    const auto fused =
        tatara::runtime::create_quantized_gemm_plan(fused_request);
    if (!fused || g_allocation_count != allocations_before ||
        fused.workspace.threadgroup_memory_bytes != 0 ||
        fused.workspace.accumulator_elements_per_threadgroup != 1024 ||
        fused.workspace.row_grid_groups != 6 ||
        fused.workspace.task_descriptor_bytes !=
            6U * sizeof(QuantizedGemmTaskDescriptor) ||
        fused.plan.implementation_identity()
                .required_threadgroup_memory_bytes !=
            0 ||
        fused.plan.implementation_identity()
                .required_accumulator_elements_per_threadgroup !=
            1024 ||
        fused.plan.implementation_identity().activation_row_mapping !=
            QuantizedGemmActivationRowMapping::PositionRows ||
        fused.plan.implementation_identity().specialization_id != 101 ||
        fused.plan.implementation_identity().specialization_version != 3) {
        return 1;
    }

    auto split_request = fused_request;
    split_request.implementation = make_direct_profile(
        512, QuantizedGemmActivationRowMapping::PositionRows, 102, 4);
    const auto split =
        tatara::runtime::create_quantized_gemm_plan(split_request);
    if (!split ||
        split.workspace.threadgroup_memory_bytes != 0 ||
        split.workspace.accumulator_elements_per_threadgroup != 512 ||
        split.plan.implementation_identity().specialization_id != 102 ||
        split.plan.implementation_identity().specialization_version != 4) {
        return 2;
    }
    auto down_request = split_request;
    down_request.shape = make_device_task_shape(
        QuantizedGemmActivationRowMapping::PaddedSlotRows, 4);
    down_request.device_tasks = make_device_task_descriptor(
        QuantizedGemmActivationRowMapping::PaddedSlotRows);
    down_request.implementation = make_direct_profile(
        512, QuantizedGemmActivationRowMapping::PaddedSlotRows, 103, 5, 4);
    const auto down =
        tatara::runtime::create_quantized_gemm_plan(down_request);
    if (!down ||
        down.workspace.threadgroup_memory_bytes != 0 ||
        down.workspace.accumulator_elements_per_threadgroup != 512 ||
        down.plan.implementation_identity().activation_row_mapping !=
            QuantizedGemmActivationRowMapping::PaddedSlotRows ||
        down.plan.shape().output_element_bytes != 4 ||
        down.plan.required_output_bytes() != 21U * 24U * 4U ||
        down.plan.implementation_identity().specialization_id != 103 ||
        down.plan.implementation_identity().specialization_version != 5) {
        return 3;
    }

    auto mapping_mismatch = fused_request;
    mapping_mismatch.shape = make_device_task_shape(
        QuantizedGemmActivationRowMapping::PaddedSlotRows);
    mapping_mismatch.device_tasks = make_device_task_descriptor(
        QuantizedGemmActivationRowMapping::PaddedSlotRows);
    if (tatara::runtime::create_quantized_gemm_plan(mapping_mismatch).error !=
        QuantizedGemmCreationError::PolicyUnavailable) {
        return 4;
    }

    const auto dense_regions =
        make_dense_regions(AffineQuantization::Q4);
    QuantizedGemmCreationRequest staged_request{
        .policy = QuantizedGemmPolicy::NativeDenseMma,
        .shape = make_dense_shape(AffineQuantization::Q4),
        .regions = dense_regions,
        .tasks = {},
        .implementation = make_profile(),
        .available_workspace_bytes =
            std::numeric_limits<std::uint64_t>::max(),
    };
    auto invalid_profile = staged_request;
    ++invalid_profile.implementation.required_threadgroup_memory_bytes;
    if (tatara::runtime::create_quantized_gemm_plan(invalid_profile).error !=
        QuantizedGemmCreationError::InvalidImplementationProfile) {
        return 5;
    }
    invalid_profile = staged_request;
    invalid_profile.implementation.threadgroup_staging_buffer_count = 0;
    if (tatara::runtime::create_quantized_gemm_plan(invalid_profile).error !=
        QuantizedGemmCreationError::InvalidImplementationProfile) {
        return 6;
    }
    invalid_profile = fused_request;
    invalid_profile.implementation.staging_element_bytes = 2;
    if (tatara::runtime::create_quantized_gemm_plan(invalid_profile).error !=
        QuantizedGemmCreationError::InvalidImplementationProfile) {
        return 7;
    }
    invalid_profile = fused_request;
    invalid_profile.implementation
        .required_accumulator_elements_per_threadgroup = 0;
    if (tatara::runtime::create_quantized_gemm_plan(invalid_profile).error !=
        QuantizedGemmCreationError::InvalidImplementationProfile) {
        return 8;
    }
    invalid_profile = fused_request;
    invalid_profile.implementation.maximum_accumulator_elements = 1023;
    if (tatara::runtime::create_quantized_gemm_plan(invalid_profile).error !=
        QuantizedGemmCreationError::AccumulatorLimitExceeded) {
        return 9;
    }
    invalid_profile = fused_request;
    invalid_profile.implementation.specialization_id = 0;
    if (tatara::runtime::create_quantized_gemm_plan(invalid_profile).error !=
        QuantizedGemmCreationError::InvalidImplementationProfile) {
        return 10;
    }
    invalid_profile = fused_request;
    invalid_profile.implementation.specialization_version = 0;
    if (tatara::runtime::create_quantized_gemm_plan(invalid_profile).error !=
        QuantizedGemmCreationError::InvalidImplementationProfile) {
        return 11;
    }

    alignas(64) std::array<std::byte, 1024> workspace{};
    alignas(64) std::array<std::byte, 4096> activations{};
    alignas(64) std::array<std::byte, 3400> packed_weights{};
    alignas(64) std::array<std::byte, 680> scales{};
    alignas(64) std::array<std::byte, 680> biases{};
    alignas(64) std::array<std::byte, 4096> output{};
    alignas(64) std::array<std::byte, 32> counts{};
    alignas(64) std::array<std::byte, 192> lists{};
    QuantizedGemmRuntimeRequest runtime{
        .policy = QuantizedGemmPolicy::NativeRaggedMma,
        .input_rows = 7,
        .output_rows = 21,
        .workspace = std::span<std::byte>{workspace}.first(
            fused.workspace.required_bytes),
        .activations = activations,
        .packed_weights = packed_weights,
        .scales = scales,
        .biases = biases,
        .output = output,
        .route_input_rows = {},
        .route_output_rows = {},
        .device_task_position_count = 7,
        .device_task_counts =
            std::span<const std::byte>{counts}.first(20),
        .device_task_lists =
            std::span<const std::byte>{lists}.first(156),
        .capability =
            {
                .identity = fused.plan.implementation_identity(),
                .available = true,
            },
    };
    if (!tatara::runtime::validate_quantized_gemm_runtime(
            fused.plan, runtime)) {
        return 12;
    }
    auto down_runtime = runtime;
    down_runtime.input_rows = 21;
    down_runtime.capability.identity =
        down.plan.implementation_identity();
    if (!tatara::runtime::validate_quantized_gemm_runtime(
            down.plan, down_runtime)) {
        return 20;
    }
    down_runtime.output = std::span<std::byte>{output}.first(
        fused.plan.required_output_bytes());
    const auto down_output_capacity_error =
        tatara::runtime::validate_quantized_gemm_runtime(
            down.plan, down_runtime);
    if (down_output_capacity_error.error !=
            QuantizedGemmRuntimeError::OperandInsufficient ||
        down_output_capacity_error.operand !=
            tatara::runtime::QuantizedGemmOperandKind::Output ||
        down_output_capacity_error.required_operand_bytes !=
            down.plan.required_output_bytes() ||
        down_output_capacity_error.available_operand_bytes !=
            fused.plan.required_output_bytes()) {
        return 21;
    }

    auto runtime_mismatch = runtime;
    runtime_mismatch.capability.identity =
        split.plan.implementation_identity();
    if (tatara::runtime::validate_quantized_gemm_runtime(
            fused.plan, runtime_mismatch)
            .error !=
        QuantizedGemmRuntimeError::CapabilityIdentityMismatch) {
        return 13;
    }
    runtime_mismatch = runtime;
    ++runtime_mismatch.capability.identity.specialization_id;
    if (tatara::runtime::validate_quantized_gemm_runtime(
            fused.plan, runtime_mismatch)
            .error !=
        QuantizedGemmRuntimeError::CapabilityIdentityMismatch) {
        return 14;
    }
    runtime_mismatch = runtime;
    ++runtime_mismatch.capability.identity.specialization_version;
    if (tatara::runtime::validate_quantized_gemm_runtime(
            fused.plan, runtime_mismatch)
            .error !=
        QuantizedGemmRuntimeError::CapabilityIdentityMismatch) {
        return 15;
    }
    runtime_mismatch = runtime;
    ++runtime_mismatch.capability.identity
          .required_threadgroup_memory_bytes;
    if (tatara::runtime::validate_quantized_gemm_runtime(
            fused.plan, runtime_mismatch)
            .error !=
        QuantizedGemmRuntimeError::CapabilityIdentityMismatch) {
        return 16;
    }
    runtime_mismatch = runtime;
    --runtime_mismatch.capability.identity
          .required_accumulator_elements_per_threadgroup;
    if (tatara::runtime::validate_quantized_gemm_runtime(
            fused.plan, runtime_mismatch)
            .error !=
        QuantizedGemmRuntimeError::CapabilityIdentityMismatch) {
        return 17;
    }
    runtime_mismatch = runtime;
    runtime_mismatch.capability.identity.activation_row_mapping =
        QuantizedGemmActivationRowMapping::PaddedSlotRows;
    if (tatara::runtime::validate_quantized_gemm_runtime(
            fused.plan, runtime_mismatch)
            .error !=
        QuantizedGemmRuntimeError::CapabilityIdentityMismatch) {
        return 18;
    }
    return g_allocation_count == allocations_before ? 0 : 19;
}

} // namespace

void* operator new(std::size_t size) {
    ++g_allocation_count;
    if (void* memory = std::malloc(size)) {
        return memory;
    }
    std::abort();
}

void operator delete(void* memory) noexcept {
    std::free(memory);
}

void* operator new[](std::size_t size) {
    ++g_allocation_count;
    if (void* memory = std::malloc(size)) {
        return memory;
    }
    std::abort();
}

void operator delete[](void* memory) noexcept {
    std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept {
    std::free(memory);
}

void operator delete[](void* memory, std::size_t) noexcept {
    std::free(memory);
}

int main() {
    if (const int result = test_task_descriptor_physical_serialization()) {
        return result;
    }
    if (const int result = test_q4_q8_dense_tails_and_exact_fallback()) {
        return 10 + result;
    }
    if (const int result = test_ragged_tasks_partials_and_workspace()) {
        return 30 + result;
    }
    if (const int result = test_creation_rejections_and_overflow()) {
        return 50 + result;
    }
    if (const int result = test_extent_validation_has_fixed_work()) {
        return 80 + result;
    }
    if (const int result = test_extent_relation_matches_small_oracle()) {
        return 90 + result;
    }
    if (const int result = test_extent_relation_near_u64_bounds()) {
        return 100 + result;
    }
    if (const int result =
            test_runtime_validation_and_policy_immutability()) {
        return 110 + result;
    }
    if (const int result = test_threadgroup_staging_resource_contract()) {
        return 130 + result;
    }
    if (const int result =
            test_device_padded_task_creation_and_materialization()) {
        return 150 + result;
    }
    if (const int result =
            test_device_padded_task_rejections_and_runtime()) {
        return 170 + result;
    }
    if (const int result =
            test_profile_v3_resources_mapping_and_identity()) {
        return 230 + result;
    }
    return 0;
}
