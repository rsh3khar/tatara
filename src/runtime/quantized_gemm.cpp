#include "tatara/runtime/quantized_gemm.h"

#include <cstring>
#include <limits>
#include <type_traits>

namespace tatara::runtime {
namespace {

struct CheckedU64 {
    std::uint64_t value{0};
    bool valid{false};
};

constexpr CheckedU64 checked_add(std::uint64_t left,
                                 std::uint64_t right) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return {};
    }
    return {.value = left + right, .valid = true};
}

constexpr CheckedU64 checked_multiply(std::uint64_t left,
                                      std::uint64_t right) noexcept {
    if (left != 0 &&
        right > std::numeric_limits<std::uint64_t>::max() / left) {
        return {};
    }
    return {.value = left * right, .valid = true};
}

constexpr CheckedU64 checked_ceil_divide(std::uint64_t value,
                                         std::uint64_t divisor) noexcept {
    if (divisor == 0) {
        return {};
    }
    return {.value = value / divisor + (value % divisor != 0 ? 1U : 0U),
            .valid = true};
}

constexpr bool power_of_two(std::uint32_t value) noexcept {
    return value != 0 && (value & (value - 1U)) == 0;
}

constexpr CheckedU64 checked_align(std::uint64_t value,
                                   std::uint32_t alignment) noexcept {
    if (!power_of_two(alignment)) {
        return {};
    }
    const std::uint64_t mask = static_cast<std::uint64_t>(alignment) - 1U;
    const CheckedU64 sum = checked_add(value, mask);
    return sum.valid
               ? CheckedU64{.value = sum.value & ~mask, .valid = true}
               : CheckedU64{};
}

bool identity_digest_present(
    const std::array<std::uint8_t, kQuantizedGemmIdentityBytes>& digest) noexcept {
    std::uint8_t combined = 0;
    for (const std::uint8_t byte : digest) {
        combined = static_cast<std::uint8_t>(combined | byte);
    }
    return combined != 0;
}

bool same_identity(const QuantizedGemmImplementationIdentity& left,
                   const QuantizedGemmImplementationIdentity& right) noexcept {
    return left.implementation_sha256 == right.implementation_sha256 &&
           left.specialization_id == right.specialization_id &&
           left.specialization_version == right.specialization_version &&
           left.policy == right.policy &&
           left.quantization == right.quantization &&
           left.group_size == right.group_size &&
           left.tile_rows == right.tile_rows &&
           left.tile_columns == right.tile_columns &&
           left.tile_reduction_columns == right.tile_reduction_columns &&
           left.threads_per_threadgroup == right.threads_per_threadgroup &&
           left.task_mode == right.task_mode &&
           left.device_task_descriptor_version ==
               right.device_task_descriptor_version &&
           left.required_threadgroup_memory_bytes ==
               right.required_threadgroup_memory_bytes &&
           left.required_accumulator_elements_per_threadgroup ==
               right.required_accumulator_elements_per_threadgroup &&
           left.activation_row_mapping == right.activation_row_mapping;
}

struct StridedExtent {
    std::uint64_t begin{0};
    std::uint64_t stride{0};
    std::uint64_t bytes{0};
    std::uint32_t count{0};
    std::uint64_t final_end{0};
};

enum class ExtentRelation : std::uint8_t {
    Disjoint,
    Overlap,
    Unsupported,
};

struct SignedU64Difference {
    std::uint64_t magnitude{0};
    bool negative{false};
};

constexpr std::uint32_t kMaximumFloorSumIterations =
    static_cast<std::uint32_t>(
        std::numeric_limits<std::uint64_t>::digits) *
    2U;

constexpr SignedU64Difference
signed_difference(std::uint64_t left, std::uint64_t right) noexcept {
    return left >= right
               ? SignedU64Difference{
                     .magnitude = left - right, .negative = false}
               : SignedU64Difference{
                     .magnitude = right - left, .negative = true};
}

CheckedU64 bounded_floor_sum(std::uint64_t count,
                             std::uint64_t modulus,
                             std::uint64_t slope,
                             std::uint64_t intercept) noexcept {
    if (modulus == 0) {
        return {};
    }

    std::uint64_t sum = 0;
    for (std::uint32_t iteration = 0;
         iteration < kMaximumFloorSumIterations; ++iteration) {
        if (slope >= modulus) {
            const std::uint64_t quotient = slope / modulus;
            slope %= modulus;
            std::uint64_t triangular_left = count;
            std::uint64_t triangular_right = count == 0 ? 0 : count - 1U;
            if ((triangular_left & 1U) == 0) {
                triangular_left /= 2U;
            } else {
                triangular_right /= 2U;
            }
            const CheckedU64 triangular =
                checked_multiply(triangular_left, triangular_right);
            const CheckedU64 contribution =
                triangular.valid
                    ? checked_multiply(triangular.value, quotient)
                    : CheckedU64{};
            const CheckedU64 next_sum =
                contribution.valid
                    ? checked_add(sum, contribution.value)
                    : CheckedU64{};
            if (!next_sum.valid) {
                return {};
            }
            sum = next_sum.value;
        }
        if (intercept >= modulus) {
            const CheckedU64 contribution =
                checked_multiply(count, intercept / modulus);
            const CheckedU64 next_sum =
                contribution.valid
                    ? checked_add(sum, contribution.value)
                    : CheckedU64{};
            if (!next_sum.valid) {
                return {};
            }
            sum = next_sum.value;
            intercept %= modulus;
        }
        if (count == 0 || slope == 0) {
            return {.value = sum, .valid = true};
        }

        const CheckedU64 final_slope =
            checked_multiply(count - 1U, slope);
        const CheckedU64 final_numerator =
            final_slope.valid
                ? checked_add(final_slope.value, intercept)
                : CheckedU64{};
        if (!final_numerator.valid) {
            return {};
        }
        if (final_numerator.value < modulus) {
            return {.value = sum, .valid = true};
        }

        const std::uint64_t quotient =
            final_numerator.value / modulus;
        const std::uint64_t remainder =
            final_numerator.value % modulus;
        const bool carry = slope >= modulus - remainder;
        const CheckedU64 next_count =
            carry ? checked_add(quotient, 1U)
                  : CheckedU64{.value = quotient, .valid = true};
        if (!next_count.valid) {
            return {};
        }
        const std::uint64_t next_intercept =
            carry ? slope - (modulus - remainder)
                  : remainder + slope;
        const std::uint64_t previous_modulus = modulus;
        modulus = slope;
        slope = previous_modulus;
        count = next_count.value;
        intercept = next_intercept;
    }
    return {};
}

CheckedU64 first_index_at_least(const SignedU64Difference& base,
                                std::uint64_t increment,
                                std::uint64_t target,
                                std::uint64_t count) noexcept {
    if (count == 0) {
        return {.value = 0, .valid = true};
    }
    if (!base.negative && base.magnitude >= target) {
        return {.value = 0, .valid = true};
    }
    if (increment == 0) {
        return {.value = count, .valid = true};
    }

    CheckedU64 required;
    if (base.negative) {
        required = checked_add(base.magnitude, target);
        if (!required.valid) {
            return {.value = count, .valid = true};
        }
    } else {
        required = {
            .value = target - base.magnitude,
            .valid = true,
        };
    }
    const CheckedU64 index =
        checked_ceil_divide(required.value, increment);
    if (!index.valid) {
        return {};
    }
    return {
        .value = index.value < count ? index.value : count,
        .valid = true,
    };
}

CheckedU64 nonnegative_value_at(const SignedU64Difference& base,
                                std::uint64_t increment,
                                std::uint64_t index) noexcept {
    const CheckedU64 delta = checked_multiply(increment, index);
    if (!delta.valid) {
        return {};
    }
    if (base.negative) {
        return delta.value >= base.magnitude
                   ? CheckedU64{
                         .value = delta.value - base.magnitude,
                         .valid = true}
                   : CheckedU64{};
    }
    return checked_add(base.magnitude, delta.value);
}

CheckedU64 count_start_differences_at_most(
    const StridedExtent& left, const StridedExtent& right,
    const SignedU64Difference& base) noexcept {
    if (left.count == 0 || right.count == 0 ||
        left.stride == 0 || right.stride == 0) {
        return {};
    }

    const CheckedU64 full_threshold = checked_multiply(
        std::uint64_t{right.count - 1U}, right.stride);
    if (!full_threshold.valid) {
        return {};
    }
    const CheckedU64 partial_begin = first_index_at_least(
        base, left.stride, 0, left.count);
    const CheckedU64 full_begin = first_index_at_least(
        base, left.stride, full_threshold.value, left.count);
    if (!partial_begin.valid || !full_begin.valid ||
        full_begin.value < partial_begin.value) {
        return {};
    }

    std::uint64_t total = 0;
    const std::uint64_t partial_count =
        full_begin.value - partial_begin.value;
    if (partial_count != 0) {
        const CheckedU64 partial_intercept = nonnegative_value_at(
            base, left.stride, partial_begin.value);
        if (!partial_intercept.valid ||
            partial_intercept.value >= full_threshold.value) {
            return {};
        }
        const CheckedU64 partial_sum = bounded_floor_sum(
            partial_count, right.stride, left.stride,
            partial_intercept.value);
        const CheckedU64 partial_total =
            partial_sum.valid
                ? checked_add(partial_sum.value, partial_count)
                : CheckedU64{};
        if (!partial_total.valid) {
            return {};
        }
        total = partial_total.value;
    }

    const CheckedU64 full_count = checked_multiply(
        std::uint64_t{left.count} - full_begin.value,
        right.count);
    const CheckedU64 result =
        full_count.valid ? checked_add(total, full_count.value)
                         : CheckedU64{};
    return result;
}

ExtentRelation extent_relation(const StridedExtent& left,
                               const StridedExtent& right) noexcept {
    if (left.final_end <= right.begin || right.final_end <= left.begin) {
        return ExtentRelation::Disjoint;
    }

    const CheckedU64 left_first_end = checked_add(left.begin, left.bytes);
    const CheckedU64 right_first_end = checked_add(right.begin, right.bytes);
    if (!left_first_end.valid || !right_first_end.valid ||
        left.count == 0 || right.count == 0) {
        return ExtentRelation::Unsupported;
    }
    if (left.begin < right_first_end.value &&
        right.begin < left_first_end.value) {
        return ExtentRelation::Overlap;
    }

    if (left.stride == 0 || right.stride == 0 ||
        left.bytes > left.stride || right.bytes > right.stride) {
        return ExtentRelation::Unsupported;
    }

    // Integer half-open intervals overlap when their start difference lies in
    // [1-right.bytes, left.bytes-1].
    const SignedU64Difference upper_base = signed_difference(
        left_first_end.value - 1U, right.begin);
    const SignedU64Difference lower_base = signed_difference(
        left.begin, right_first_end.value);
    const CheckedU64 upper_count =
        count_start_differences_at_most(left, right, upper_base);
    const CheckedU64 lower_count =
        count_start_differences_at_most(left, right, lower_base);
    if (!upper_count.valid || !lower_count.valid ||
        upper_count.value < lower_count.value) {
        return ExtentRelation::Unsupported;
    }
    return upper_count.value != lower_count.value
               ? ExtentRelation::Overlap
               : ExtentRelation::Disjoint;
}

bool valid_policy(QuantizedGemmPolicy policy) noexcept {
    switch (policy) {
    case QuantizedGemmPolicy::ExactRow:
    case QuantizedGemmPolicy::NativeDenseMma:
    case QuantizedGemmPolicy::NativeRaggedMma:
        return true;
    }
    return false;
}

bool valid_task_mode(QuantizedGemmTaskMode mode) noexcept {
    switch (mode) {
    case QuantizedGemmTaskMode::HostCompact:
    case QuantizedGemmTaskMode::DevicePaddedSlotsV1:
        return true;
    }
    return false;
}

bool valid_activation_row_mapping(
    QuantizedGemmActivationRowMapping mapping) noexcept {
    switch (mapping) {
    case QuantizedGemmActivationRowMapping::PositionRows:
    case QuantizedGemmActivationRowMapping::PaddedSlotRows:
        return true;
    }
    return false;
}

bool empty_device_task_descriptor(
    const QuantizedGemmDeviceTaskDescriptor& descriptor) noexcept {
    return descriptor.version == 0 &&
           descriptor.activation_row_mapping ==
               QuantizedGemmActivationRowMapping::PositionRows &&
           descriptor.position_capacity == 0 &&
           descriptor.routes_per_position == 0 &&
           descriptor.padded_slot_stride == 0 &&
           descriptor.packed_slot_bits == 0 &&
           descriptor.list_capacity_entries == 0 &&
           descriptor.list_expert_stride_entries == 0 &&
           descriptor.count_storage_bytes == 0 &&
           descriptor.list_storage_bytes == 0;
}

bool policy_available(
    QuantizedGemmPolicy policy,
    const QuantizedGemmImplementationProfile& profile) noexcept {
    switch (policy) {
    case QuantizedGemmPolicy::ExactRow:
        return profile.exact_row_available;
    case QuantizedGemmPolicy::NativeDenseMma:
        return profile.native_dense_mma_available;
    case QuantizedGemmPolicy::NativeRaggedMma:
        return profile.native_ragged_mma_available;
    }
    return false;
}

bool policy_accepts_workload(QuantizedGemmPolicy policy,
                             QuantizedGemmWorkload workload) noexcept {
    if (policy == QuantizedGemmPolicy::ExactRow) {
        return workload == QuantizedGemmWorkload::Dense ||
               workload == QuantizedGemmWorkload::Ragged;
    }
    if (policy == QuantizedGemmPolicy::NativeDenseMma) {
        return workload == QuantizedGemmWorkload::Dense;
    }
    if (policy == QuantizedGemmPolicy::NativeRaggedMma) {
        return workload == QuantizedGemmWorkload::Ragged;
    }
    return false;
}

bool valid_profile(const QuantizedGemmImplementationProfile& profile) noexcept {
    const bool direct =
        profile.activation_staging_row_stride_elements == 0 &&
        profile.weight_staging_row_stride_elements == 0 &&
        profile.threadgroup_staging_buffer_count == 0 &&
        profile.staging_element_bytes == 0 &&
        profile.required_threadgroup_memory_bytes == 0;
    const bool staged =
        profile.activation_staging_row_stride_elements >=
            profile.tile_reduction_columns &&
        profile.weight_staging_row_stride_elements >=
            profile.tile_columns &&
        profile.threadgroup_staging_buffer_count != 0 &&
        profile.staging_element_bytes != 0 &&
        profile.required_threadgroup_memory_bytes != 0;
    return profile.profile_version == kQuantizedGemmProfileVersion &&
           identity_digest_present(profile.implementation_sha256) &&
           profile.specialization_id != 0 &&
           profile.specialization_version != 0 &&
           profile.supported_group_size != 0 &&
           profile.maximum_regions != 0 && profile.maximum_tasks != 0 &&
           profile.maximum_regions <= kQuantizedGemmRegionCapacity &&
           profile.maximum_tasks <= kQuantizedGemmTaskCapacity &&
           profile.maximum_experts != 0 &&
           profile.maximum_route_entries != 0 && profile.tile_rows != 0 &&
           profile.supported_activation_element_bytes != 0 &&
           profile.supported_output_element_bytes != 0 &&
           profile.supported_quantization_parameter_bytes != 0 &&
           profile.tile_columns != 0 &&
           profile.tile_reduction_columns != 0 &&
           (direct || staged) &&
           profile.threads_per_threadgroup != 0 &&
           profile.maximum_threads_per_threadgroup != 0 &&
           profile.maximum_threadgroup_memory_bytes != 0 &&
           profile.maximum_accumulator_elements != 0 &&
           profile.partial_partition_count != 0 &&
           profile.maximum_partial_partition_count != 0 &&
           profile.partial_partition_count <=
               profile.maximum_partial_partition_count &&
           profile.partial_element_bytes != 0 &&
           power_of_two(profile.workspace_alignment) &&
           profile.workspace_alignment >=
               alignof(QuantizedGemmRegionDescriptor) &&
           profile.workspace_alignment >=
               alignof(QuantizedGemmTaskDescriptor) &&
           (profile.q4_available || profile.q8_available) &&
           profile.exact_row_available &&
           valid_activation_row_mapping(
               profile.activation_row_mapping);
}

QuantizedGemmCreationResult
failure(QuantizedGemmCreationError error,
        const QuantizedGemmCreationRequest& request,
        const QuantizedGemmWorkspacePlan& workspace = {},
        std::uint64_t required_output_bytes = 0) noexcept {
    return {
        .error = error,
        .workspace = workspace,
        .available_workspace_bytes = request.available_workspace_bytes,
        .required_output_bytes = required_output_bytes,
        .plan = {},
    };
}

bool append_workspace_component(std::uint64_t component,
                                std::uint32_t alignment,
                                std::uint64_t& offset,
                                std::uint64_t& total) noexcept {
    if (component == 0) {
        return true;
    }
    const CheckedU64 start = checked_align(total, alignment);
    if (!start.valid) {
        return false;
    }
    const CheckedU64 next = checked_add(start.value, component);
    if (!next.valid) {
        return false;
    }
    offset = start.value;
    total = next.value;
    return true;
}

} // namespace

QuantizedGemmCreationResult
create_quantized_gemm_plan(
    const QuantizedGemmCreationRequest& request) noexcept {
    const QuantizedGemmShapeDescriptor& shape = request.shape;
    const QuantizedGemmImplementationProfile& profile =
        request.implementation;

    if (!valid_profile(profile)) {
        return failure(
            QuantizedGemmCreationError::InvalidImplementationProfile,
            request);
    }
    if (!valid_policy(request.policy) ||
        !policy_accepts_workload(request.policy, shape.workload) ||
        !policy_available(request.policy, profile)) {
        return failure(QuantizedGemmCreationError::PolicyUnavailable,
                       request);
    }
    if (!valid_task_mode(request.task_mode)) {
        return failure(QuantizedGemmCreationError::PolicyUnavailable,
                       request);
    }
    const bool device_tasks =
        request.task_mode ==
        QuantizedGemmTaskMode::DevicePaddedSlotsV1;
    if (!device_tasks &&
        !empty_device_task_descriptor(request.device_tasks)) {
        return failure(QuantizedGemmCreationError::InvalidTask, request);
    }
    if (device_tasks &&
        (request.policy != QuantizedGemmPolicy::NativeRaggedMma ||
         shape.workload != QuantizedGemmWorkload::Ragged ||
         !profile.device_padded_slots_v1_available ||
         request.device_tasks.version !=
             kQuantizedGemmDeviceTaskDescriptorVersion)) {
        return failure(QuantizedGemmCreationError::PolicyUnavailable,
                       request);
    }
    if (device_tasks && !request.tasks.empty()) {
        return failure(QuantizedGemmCreationError::InvalidTask, request);
    }
    if (shape.input_rows == 0 || shape.output_rows == 0 ||
        shape.output_columns == 0 || shape.reduction_columns == 0 ||
        shape.activation_element_bytes == 0 ||
        shape.output_element_bytes == 0 ||
        shape.quantization_parameter_bytes == 0 ||
        shape.activation_element_bytes !=
            profile.supported_activation_element_bytes ||
        shape.output_element_bytes !=
            profile.supported_output_element_bytes ||
        shape.quantization_parameter_bytes !=
            profile.supported_quantization_parameter_bytes ||
        shape.activation_row_stride_elements < shape.reduction_columns ||
        shape.output_row_stride_elements < shape.output_columns) {
        return failure(QuantizedGemmCreationError::InvalidShape, request);
    }
    if (shape.weight_layout !=
        QuantizedWeightLayout::OutputMajorAffineGroups) {
        return failure(QuantizedGemmCreationError::UnsupportedLayout,
                       request);
    }
    std::uint32_t quantization_bits = 0;
    switch (shape.quantization) {
    case AffineQuantization::Q4:
        if (!profile.q4_available) {
            return failure(
                QuantizedGemmCreationError::UnsupportedQuantization,
                request);
        }
        quantization_bits = 4;
        break;
    case AffineQuantization::Q8:
        if (!profile.q8_available) {
            return failure(
                QuantizedGemmCreationError::UnsupportedQuantization,
                request);
        }
        quantization_bits = 8;
        break;
    default:
        return failure(QuantizedGemmCreationError::UnsupportedQuantization,
                       request);
    }
    if (shape.group_size == 0 ||
        shape.group_size != profile.supported_group_size) {
        return failure(QuantizedGemmCreationError::UnsupportedGroup,
                       request);
    }
    if (request.regions.empty() ||
        request.regions.size() > profile.maximum_regions ||
        request.tasks.size() > profile.maximum_tasks) {
        return failure(
            QuantizedGemmCreationError::DescriptorLimitExceeded, request);
    }

    std::uint64_t device_route_capacity = 0;
    std::uint64_t device_task_capacity = 0;
    std::uint64_t required_device_task_count_bytes = 0;
    std::uint64_t required_device_task_list_bytes = 0;
    if (shape.workload == QuantizedGemmWorkload::Dense) {
        if (shape.output_rows != shape.input_rows || shape.expert_count != 0 ||
            shape.route_list_count != 0 || !request.tasks.empty() ||
            device_tasks) {
            return failure(QuantizedGemmCreationError::InvalidShape,
                           request);
        }
    } else if (shape.workload == QuantizedGemmWorkload::Ragged) {
        if (shape.expert_count == 0 ||
            shape.expert_count > profile.maximum_experts ||
            shape.route_list_count == 0 ||
            shape.route_list_count > profile.maximum_route_entries ||
            (!device_tasks &&
             (shape.route_list_count != shape.output_rows ||
              request.tasks.empty()))) {
            return failure(QuantizedGemmCreationError::InvalidShape,
                           request);
        }
    } else {
        return failure(QuantizedGemmCreationError::InvalidShape, request);
    }

    if (device_tasks) {
        const QuantizedGemmDeviceTaskDescriptor& descriptor =
            request.device_tasks;
        if (!valid_activation_row_mapping(
                descriptor.activation_row_mapping) ||
            descriptor.position_capacity == 0 ||
            descriptor.routes_per_position == 0 ||
            descriptor.routes_per_position > shape.expert_count ||
            descriptor.padded_slot_stride <=
                descriptor.routes_per_position ||
            descriptor.packed_slot_bits == 0 ||
            descriptor.packed_slot_bits >=
                std::numeric_limits<std::uint32_t>::digits ||
            descriptor.list_capacity_entries <
                descriptor.position_capacity ||
            descriptor.list_expert_stride_entries <
                descriptor.list_capacity_entries) {
            return failure(QuantizedGemmCreationError::InvalidTask,
                           request);
        }
        if (descriptor.activation_row_mapping !=
            profile.activation_row_mapping) {
            return failure(QuantizedGemmCreationError::PolicyUnavailable,
                           request);
        }
        const std::uint64_t packed_slot_capacity =
            std::uint64_t{1} << descriptor.packed_slot_bits;
        if (descriptor.padded_slot_stride > packed_slot_capacity ||
            std::uint64_t{descriptor.position_capacity - 1U} >
                (std::numeric_limits<std::uint32_t>::max() >>
                 descriptor.packed_slot_bits)) {
            return failure(QuantizedGemmCreationError::InvalidTask,
                           request);
        }

        const CheckedU64 padded_rows = checked_multiply(
            descriptor.position_capacity,
            descriptor.padded_slot_stride);
        const CheckedU64 route_capacity = checked_multiply(
            descriptor.position_capacity,
            descriptor.routes_per_position);
        const CheckedU64 list_address_limit = checked_multiply(
            shape.expert_count,
            descriptor.list_expert_stride_entries);
        const CheckedU64 final_list_expert_offset = checked_multiply(
            shape.expert_count - 1U,
            descriptor.list_expert_stride_entries);
        const CheckedU64 required_list_entries =
            final_list_expert_offset.valid
                ? checked_add(final_list_expert_offset.value,
                              descriptor.list_capacity_entries)
                : CheckedU64{};
        const CheckedU64 required_count_bytes = checked_multiply(
            shape.expert_count, sizeof(std::uint32_t));
        const CheckedU64 required_list_bytes =
            required_list_entries.valid
                ? checked_multiply(required_list_entries.value,
                                   sizeof(std::uint32_t))
                : CheckedU64{};
        const CheckedU64 route_tiles = route_capacity.valid
                                           ? checked_ceil_divide(
                                                 route_capacity.value,
                                                 profile.tile_rows)
                                           : CheckedU64{};
        const std::uint64_t maximum_nonempty_experts =
            route_capacity.valid &&
                    route_capacity.value < shape.expert_count
                ? route_capacity.value
                : shape.expert_count;
        const CheckedU64 task_capacity =
            route_tiles.valid
                ? checked_add(route_tiles.value,
                              maximum_nonempty_experts)
                : CheckedU64{};
        if (!padded_rows.valid || !route_capacity.valid ||
            !list_address_limit.valid ||
            !required_list_entries.valid ||
            !required_count_bytes.valid ||
            !required_list_bytes.valid || !route_tiles.valid ||
            !task_capacity.valid) {
            return failure(QuantizedGemmCreationError::ArithmeticOverflow,
                           request);
        }
        const std::uint64_t expected_input_rows =
            descriptor.activation_row_mapping ==
                    QuantizedGemmActivationRowMapping::PositionRows
                ? descriptor.position_capacity
                : padded_rows.value;
        if (padded_rows.value >
                std::numeric_limits<std::uint32_t>::max() ||
            route_capacity.value >
                std::numeric_limits<std::uint32_t>::max() ||
            expected_input_rows != shape.input_rows ||
            padded_rows.value != shape.output_rows ||
            route_capacity.value != shape.route_list_count ||
            list_address_limit.value >
                std::numeric_limits<std::uint32_t>::max()) {
            return failure(QuantizedGemmCreationError::InvalidTask,
                           request);
        }
        if (task_capacity.value > profile.maximum_tasks ||
            task_capacity.value > kQuantizedGemmTaskCapacity) {
            return failure(
                QuantizedGemmCreationError::DescriptorLimitExceeded,
                request);
        }
        if (required_count_bytes.value >
                descriptor.count_storage_bytes ||
            required_list_bytes.value >
                descriptor.list_storage_bytes) {
            return failure(
                QuantizedGemmCreationError::InvalidStorageBounds,
                request);
        }
        device_route_capacity = route_capacity.value;
        device_task_capacity = task_capacity.value;
        required_device_task_count_bytes =
            required_count_bytes.value;
        required_device_task_list_bytes = required_list_bytes.value;
    }

    const CheckedU64 groups_per_weight_row = checked_ceil_divide(
        shape.reduction_columns, shape.group_size);
    const CheckedU64 packed_row_bits =
        checked_multiply(shape.reduction_columns, quantization_bits);
    if (!packed_row_bits.valid) {
        return failure(QuantizedGemmCreationError::ArithmeticOverflow,
                       request);
    }
    const CheckedU64 minimum_weight_row_bytes =
        checked_ceil_divide(packed_row_bits.value, 8);
    const CheckedU64 minimum_parameter_row_bytes =
        groups_per_weight_row.valid
            ? checked_multiply(groups_per_weight_row.value,
                               shape.quantization_parameter_bytes)
            : CheckedU64{};
    if (!groups_per_weight_row.valid || !minimum_weight_row_bytes.valid ||
        !minimum_parameter_row_bytes.valid) {
        return failure(QuantizedGemmCreationError::ArithmeticOverflow,
                       request);
    }
    if (shape.packed_weight_row_stride_bytes <
            minimum_weight_row_bytes.value ||
        shape.scale_row_stride_bytes <
            minimum_parameter_row_bytes.value ||
        shape.bias_row_stride_bytes <
            minimum_parameter_row_bytes.value) {
        return failure(QuantizedGemmCreationError::InvalidShape, request);
    }
    const CheckedU64 maximum_partitions = checked_ceil_divide(
        shape.reduction_columns, profile.tile_reduction_columns);
    if (!maximum_partitions.valid) {
        return failure(QuantizedGemmCreationError::ArithmeticOverflow,
                       request);
    }
    if (request.policy != QuantizedGemmPolicy::ExactRow &&
        (profile.partial_partition_count > maximum_partitions.value ||
         profile.partial_partition_count >
             groups_per_weight_row.value)) {
        return failure(
            QuantizedGemmCreationError::InvalidImplementationProfile,
            request);
    }

    const CheckedU64 activation_row_bytes =
        checked_multiply(shape.activation_row_stride_elements,
                         shape.activation_element_bytes);
    const CheckedU64 required_activation_bytes =
        activation_row_bytes.valid
            ? checked_multiply(shape.input_rows,
                               activation_row_bytes.value)
            : CheckedU64{};
    const CheckedU64 output_row_bytes =
        checked_multiply(shape.output_row_stride_elements,
                         shape.output_element_bytes);
    const CheckedU64 required_output_bytes =
        output_row_bytes.valid
            ? checked_multiply(shape.output_rows, output_row_bytes.value)
            : CheckedU64{};
    if (!required_activation_bytes.valid || !required_output_bytes.valid) {
        return failure(QuantizedGemmCreationError::ArithmeticOverflow,
                       request);
    }
    if (required_activation_bytes.value > shape.activation_storage_bytes ||
        required_output_bytes.value > shape.output_storage_bytes) {
        return failure(QuantizedGemmCreationError::InvalidStorageBounds,
                       request, {}, required_output_bytes.value);
    }

    std::uint64_t output_cursor = 0;
    std::uint64_t column_grid_groups = 0;
    std::array<StridedExtent, kQuantizedGemmRegionCapacity> weight_extents{};
    std::array<StridedExtent, kQuantizedGemmRegionCapacity> scale_extents{};
    std::array<StridedExtent, kQuantizedGemmRegionCapacity> bias_extents{};
    std::uint64_t required_weight_storage = 0;
    std::uint64_t required_scale_storage = 0;
    std::uint64_t required_bias_storage = 0;
    for (std::size_t region_index = 0;
         region_index < request.regions.size(); ++region_index) {
        const QuantizedGemmRegionDescriptor& region =
            request.regions[region_index];
        if (region.output_column_count == 0 ||
            region.output_column_begin != output_cursor) {
            return failure(QuantizedGemmCreationError::InvalidRegion,
                           request, {}, required_output_bytes.value);
        }
        const CheckedU64 next_output =
            checked_add(output_cursor, region.output_column_count);
        const CheckedU64 region_weight_rows =
            checked_multiply(region.output_column_count - 1U,
                             shape.packed_weight_row_stride_bytes);
        const CheckedU64 region_weight_extent =
            region_weight_rows.valid
                ? checked_add(region_weight_rows.value,
                              minimum_weight_row_bytes.value)
                : CheckedU64{};
        const CheckedU64 region_parameter_rows =
            checked_multiply(region.output_column_count - 1U,
                             shape.scale_row_stride_bytes);
        const CheckedU64 region_scale_extent =
            region_parameter_rows.valid
                ? checked_add(region_parameter_rows.value,
                              minimum_parameter_row_bytes.value)
                : CheckedU64{};
        const CheckedU64 region_bias_rows =
            checked_multiply(region.output_column_count - 1U,
                             shape.bias_row_stride_bytes);
        const CheckedU64 region_bias_extent =
            region_bias_rows.valid
                ? checked_add(region_bias_rows.value,
                              minimum_parameter_row_bytes.value)
                : CheckedU64{};
        if (!next_output.valid || !region_weight_extent.valid ||
            !region_scale_extent.valid || !region_bias_extent.valid) {
            return failure(QuantizedGemmCreationError::ArithmeticOverflow,
                           request, {}, required_output_bytes.value);
        }
        if (next_output.value > shape.output_columns ||
            region.scale_offset_bytes %
                    shape.quantization_parameter_bytes !=
                0 ||
            region.bias_offset_bytes %
                    shape.quantization_parameter_bytes !=
                0) {
            return failure(QuantizedGemmCreationError::InvalidRegion,
                           request, {}, required_output_bytes.value);
        }

        std::uint64_t expert_weight_offset = 0;
        std::uint64_t expert_scale_offset = 0;
        std::uint64_t expert_bias_offset = 0;
        if (shape.workload == QuantizedGemmWorkload::Ragged) {
            if (region.packed_weight_expert_stride_bytes <
                    region_weight_extent.value ||
                region.scale_expert_stride_bytes <
                    region_scale_extent.value ||
                region.bias_expert_stride_bytes <
                    region_bias_extent.value) {
                return failure(
                    QuantizedGemmCreationError::InvalidRegion, request, {},
                    required_output_bytes.value);
            }
            const std::uint64_t last_expert = shape.expert_count - 1U;
            const CheckedU64 weight_offset = checked_multiply(
                last_expert, region.packed_weight_expert_stride_bytes);
            const CheckedU64 scale_offset = checked_multiply(
                last_expert, region.scale_expert_stride_bytes);
            const CheckedU64 bias_offset = checked_multiply(
                last_expert, region.bias_expert_stride_bytes);
            if (!weight_offset.valid || !scale_offset.valid ||
                !bias_offset.valid) {
                return failure(
                    QuantizedGemmCreationError::ArithmeticOverflow, request,
                    {}, required_output_bytes.value);
            }
            expert_weight_offset = weight_offset.value;
            expert_scale_offset = scale_offset.value;
            expert_bias_offset = bias_offset.value;
        } else if (region.packed_weight_expert_stride_bytes != 0 ||
                   region.scale_expert_stride_bytes != 0 ||
                   region.bias_expert_stride_bytes != 0) {
            return failure(QuantizedGemmCreationError::InvalidRegion,
                           request, {}, required_output_bytes.value);
        }

        const CheckedU64 weight_end = checked_add(
            region.packed_weight_offset_bytes, expert_weight_offset);
        const CheckedU64 weight_extent_end =
            weight_end.valid
                ? checked_add(weight_end.value, region_weight_extent.value)
                : CheckedU64{};
        const CheckedU64 scale_end =
            checked_add(region.scale_offset_bytes, expert_scale_offset);
        const CheckedU64 scale_extent_end =
            scale_end.valid
                ? checked_add(scale_end.value, region_scale_extent.value)
                : CheckedU64{};
        const CheckedU64 bias_end =
            checked_add(region.bias_offset_bytes, expert_bias_offset);
        const CheckedU64 bias_extent_end =
            bias_end.valid
                ? checked_add(bias_end.value, region_bias_extent.value)
                : CheckedU64{};
        if (!weight_extent_end.valid || !scale_extent_end.valid ||
            !bias_extent_end.valid) {
            return failure(QuantizedGemmCreationError::ArithmeticOverflow,
                           request, {}, required_output_bytes.value);
        }
        if (weight_extent_end.value > shape.packed_weight_storage_bytes ||
            scale_extent_end.value > shape.scale_storage_bytes ||
            bias_extent_end.value > shape.bias_storage_bytes) {
            return failure(
                QuantizedGemmCreationError::InvalidStorageBounds, request, {},
                required_output_bytes.value);
        }
        const std::uint32_t extent_count =
            shape.workload == QuantizedGemmWorkload::Ragged
                ? shape.expert_count
                : 1U;
        weight_extents[region_index] = {
            .begin = region.packed_weight_offset_bytes,
            .stride = region.packed_weight_expert_stride_bytes,
            .bytes = region_weight_extent.value,
            .count = extent_count,
            .final_end = weight_extent_end.value,
        };
        scale_extents[region_index] = {
            .begin = region.scale_offset_bytes,
            .stride = region.scale_expert_stride_bytes,
            .bytes = region_scale_extent.value,
            .count = extent_count,
            .final_end = scale_extent_end.value,
        };
        bias_extents[region_index] = {
            .begin = region.bias_offset_bytes,
            .stride = region.bias_expert_stride_bytes,
            .bytes = region_bias_extent.value,
            .count = extent_count,
            .final_end = bias_extent_end.value,
        };
        if (required_weight_storage < weight_extent_end.value) {
            required_weight_storage = weight_extent_end.value;
        }
        if (required_scale_storage < scale_extent_end.value) {
            required_scale_storage = scale_extent_end.value;
        }
        if (required_bias_storage < bias_extent_end.value) {
            required_bias_storage = bias_extent_end.value;
        }
        const CheckedU64 region_column_groups = checked_ceil_divide(
            region.output_column_count, profile.tile_columns);
        const CheckedU64 next_column_groups =
            region_column_groups.valid
                ? checked_add(column_grid_groups,
                              region_column_groups.value)
                : CheckedU64{};
        if (!next_column_groups.valid) {
            return failure(QuantizedGemmCreationError::ArithmeticOverflow,
                           request, {}, required_output_bytes.value);
        }
        column_grid_groups = next_column_groups.value;
        output_cursor = next_output.value;
    }
    if (output_cursor != shape.output_columns) {
        return failure(QuantizedGemmCreationError::InvalidRegion, request, {},
                       required_output_bytes.value);
    }
    for (std::size_t left = 0; left < request.regions.size(); ++left) {
        for (std::size_t right = left + 1; right < request.regions.size();
             ++right) {
            const std::array<ExtentRelation, 3> relations{
                extent_relation(weight_extents[left],
                                weight_extents[right]),
                extent_relation(scale_extents[left],
                                scale_extents[right]),
                extent_relation(bias_extents[left],
                                bias_extents[right]),
            };
            if (relations[0] == ExtentRelation::Overlap ||
                relations[1] == ExtentRelation::Overlap ||
                relations[2] == ExtentRelation::Overlap) {
                return failure(QuantizedGemmCreationError::InvalidRegion,
                               request, {},
                               required_output_bytes.value);
            }
            if (relations[0] != ExtentRelation::Disjoint ||
                relations[1] != ExtentRelation::Disjoint ||
                relations[2] != ExtentRelation::Disjoint) {
                return failure(
                    QuantizedGemmCreationError::
                        ExtentValidationUnsupported,
                    request, {}, required_output_bytes.value);
            }
        }
    }

    std::uint64_t row_grid_groups = 0;
    if (shape.workload == QuantizedGemmWorkload::Dense) {
        const CheckedU64 groups =
            checked_ceil_divide(shape.input_rows, profile.tile_rows);
        if (!groups.valid) {
            return failure(QuantizedGemmCreationError::ArithmeticOverflow,
                           request, {}, required_output_bytes.value);
        }
        row_grid_groups = groups.value;
    } else if (!device_tasks) {
        std::uint64_t route_cursor = 0;
        std::uint32_t previous_expert = 0;
        bool first_task = true;
        for (const QuantizedGemmTaskDescriptor& task : request.tasks) {
            if (task.row_count == 0 || task.expert_index >= shape.expert_count ||
                task.route_list_begin != route_cursor ||
                task.output_row_begin != route_cursor ||
                (!first_task && task.expert_index <= previous_expert)) {
                return failure(QuantizedGemmCreationError::InvalidTask,
                               request, {}, required_output_bytes.value);
            }
            const CheckedU64 next_route =
                checked_add(route_cursor, task.row_count);
            const CheckedU64 task_row_groups =
                checked_ceil_divide(task.row_count, profile.tile_rows);
            const CheckedU64 next_row_groups =
                task_row_groups.valid
                    ? checked_add(row_grid_groups, task_row_groups.value)
                    : CheckedU64{};
            if (!next_route.valid || !next_row_groups.valid) {
                return failure(
                    QuantizedGemmCreationError::ArithmeticOverflow, request,
                    {}, required_output_bytes.value);
            }
            if (next_route.value > shape.route_list_count) {
                return failure(QuantizedGemmCreationError::InvalidTask,
                               request, {}, required_output_bytes.value);
            }
            route_cursor = next_route.value;
            row_grid_groups = next_row_groups.value;
            previous_expert = task.expert_index;
            first_task = false;
        }
        if (route_cursor != shape.route_list_count) {
            return failure(QuantizedGemmCreationError::InvalidTask, request,
                           {}, required_output_bytes.value);
        }
    } else {
        row_grid_groups = device_task_capacity;
    }

    QuantizedGemmWorkspacePlan workspace;
    workspace.row_grid_groups = row_grid_groups;
    workspace.column_grid_groups = column_grid_groups;
    const CheckedU64 base_threadgroups =
        checked_multiply(row_grid_groups, column_grid_groups);
    if (!base_threadgroups.valid) {
        return failure(QuantizedGemmCreationError::ArithmeticOverflow,
                       request, workspace, required_output_bytes.value);
    }
    const std::uint32_t partition_count =
        request.policy == QuantizedGemmPolicy::ExactRow
            ? 1U
            : profile.partial_partition_count;
    const CheckedU64 threadgroups =
        checked_multiply(base_threadgroups.value, partition_count);
    if (!threadgroups.valid) {
        return failure(QuantizedGemmCreationError::ArithmeticOverflow,
                       request, workspace, required_output_bytes.value);
    }
    workspace.threadgroup_count = threadgroups.value;
    workspace.reduction_grid_groups =
        partition_count > 1 ? base_threadgroups.value : 0;
    workspace.threads_per_threadgroup = profile.threads_per_threadgroup;
    workspace.partial_partition_count = partition_count;
    workspace.alignment = profile.workspace_alignment;

    if (profile.threads_per_threadgroup >
        profile.maximum_threads_per_threadgroup) {
        return failure(QuantizedGemmCreationError::ThreadLimitExceeded,
                       request, workspace, required_output_bytes.value);
    }
    if (request.policy != QuantizedGemmPolicy::ExactRow) {
        if (profile.required_accumulator_elements_per_threadgroup == 0) {
            return failure(
                QuantizedGemmCreationError::InvalidImplementationProfile,
                request, workspace, required_output_bytes.value);
        }
        workspace.accumulator_elements_per_threadgroup =
            profile.required_accumulator_elements_per_threadgroup;
        if (workspace.accumulator_elements_per_threadgroup >
            profile.maximum_accumulator_elements) {
            return failure(
                QuantizedGemmCreationError::AccumulatorLimitExceeded,
                request, workspace, required_output_bytes.value);
        }

        if (profile.required_threadgroup_memory_bytes != 0) {
            const CheckedU64 activation_tile = checked_multiply(
                profile.tile_rows,
                profile.activation_staging_row_stride_elements);
            const CheckedU64 weight_tile = checked_multiply(
                profile.tile_reduction_columns,
                profile.weight_staging_row_stride_elements);
            const CheckedU64 tile_elements =
                activation_tile.valid && weight_tile.valid
                    ? checked_add(activation_tile.value,
                                  weight_tile.value)
                    : CheckedU64{};
            const CheckedU64 one_stage_threadgroup_memory =
                tile_elements.valid
                    ? checked_multiply(
                          tile_elements.value,
                          profile.staging_element_bytes)
                    : CheckedU64{};
            const CheckedU64 threadgroup_memory =
                one_stage_threadgroup_memory.valid
                    ? checked_multiply(
                          one_stage_threadgroup_memory.value,
                          profile.threadgroup_staging_buffer_count)
                    : CheckedU64{};
            if (!threadgroup_memory.valid) {
                return failure(
                    QuantizedGemmCreationError::ArithmeticOverflow,
                    request, workspace, required_output_bytes.value);
            }
            if (threadgroup_memory.value !=
                profile.required_threadgroup_memory_bytes) {
                return failure(
                    QuantizedGemmCreationError::
                        InvalidImplementationProfile,
                    request, workspace, required_output_bytes.value);
            }
        }
        workspace.threadgroup_memory_bytes =
            profile.required_threadgroup_memory_bytes;
        if (workspace.threadgroup_memory_bytes >
            profile.maximum_threadgroup_memory_bytes) {
            return failure(
                QuantizedGemmCreationError::ThreadgroupMemoryExceeded,
                request, workspace, required_output_bytes.value);
        }
    }

    if (request.policy != QuantizedGemmPolicy::ExactRow) {
        const CheckedU64 region_bytes =
            checked_multiply(request.regions.size(),
                             sizeof(QuantizedGemmRegionDescriptor));
        const std::uint64_t task_count =
            device_tasks ? device_task_capacity
                         : request.tasks.size();
        const CheckedU64 task_bytes =
            checked_multiply(task_count,
                             sizeof(QuantizedGemmTaskDescriptor));
        const CheckedU64 route_index_bytes = checked_multiply(
            shape.route_list_count, sizeof(std::uint32_t));
        if (!region_bytes.valid || !task_bytes.valid ||
            !route_index_bytes.valid) {
            return failure(QuantizedGemmCreationError::ArithmeticOverflow,
                           request, workspace,
                           required_output_bytes.value);
        }
        workspace.region_descriptor_bytes = region_bytes.value;
        workspace.task_descriptor_bytes = task_bytes.value;
        if (device_tasks) {
            workspace.device_task_indirect_argument_bytes =
                3U * sizeof(std::uint32_t);
            workspace.device_task_status_bytes =
                sizeof(QuantizedGemmDeviceTaskStatus);
        } else if (
            shape.workload == QuantizedGemmWorkload::Ragged) {
            workspace.route_input_index_bytes = route_index_bytes.value;
            workspace.route_output_index_bytes = route_index_bytes.value;
        }
        if (partition_count > 1) {
            const CheckedU64 partial_rows =
                checked_multiply(shape.output_rows, shape.output_columns);
            const CheckedU64 partitioned =
                partial_rows.valid
                    ? checked_multiply(partial_rows.value, partition_count)
                    : CheckedU64{};
            const CheckedU64 partial_bytes =
                partitioned.valid
                    ? checked_multiply(partitioned.value,
                                       profile.partial_element_bytes)
                    : CheckedU64{};
            if (!partial_bytes.valid) {
                return failure(
                    QuantizedGemmCreationError::ArithmeticOverflow,
                    request, workspace, required_output_bytes.value);
            }
            workspace.partial_bytes = partial_bytes.value;
        }

        std::uint64_t total = 0;
        if (!append_workspace_component(
                workspace.region_descriptor_bytes, workspace.alignment,
                workspace.region_descriptor_offset, total) ||
            !append_workspace_component(
                workspace.task_descriptor_bytes, workspace.alignment,
                workspace.task_descriptor_offset, total) ||
            !append_workspace_component(
                workspace.device_task_indirect_argument_bytes,
                workspace.alignment,
                workspace.device_task_indirect_argument_offset, total) ||
            !append_workspace_component(
                workspace.device_task_status_bytes, workspace.alignment,
                workspace.device_task_status_offset, total) ||
            !append_workspace_component(
                workspace.route_input_index_bytes, workspace.alignment,
                workspace.route_input_index_offset, total) ||
            !append_workspace_component(
                workspace.route_output_index_bytes, workspace.alignment,
                workspace.route_output_index_offset, total) ||
            !append_workspace_component(workspace.partial_bytes,
                                        workspace.alignment,
                                        workspace.partial_offset, total)) {
            return failure(
                QuantizedGemmCreationError::ArithmeticOverflow, request,
                workspace, required_output_bytes.value);
        }
        const CheckedU64 aligned_total =
            checked_align(total, workspace.alignment);
        if (!aligned_total.valid) {
            return failure(
                QuantizedGemmCreationError::ArithmeticOverflow, request,
                workspace, required_output_bytes.value);
        }
        workspace.required_bytes = aligned_total.value;
    }

    if (workspace.required_bytes > request.available_workspace_bytes) {
        return failure(QuantizedGemmCreationError::WorkspaceInsufficient,
                       request, workspace, required_output_bytes.value);
    }

    QuantizedGemmCreationResult result{
        .error = QuantizedGemmCreationError::None,
        .workspace = workspace,
        .available_workspace_bytes = request.available_workspace_bytes,
        .required_output_bytes = required_output_bytes.value,
        .plan = {},
    };
    result.plan.valid_ = true;
    result.plan.selected_policy_ = request.policy;
    result.plan.task_mode_ = request.task_mode;
    result.plan.shape_ = shape;
    result.plan.device_tasks_ = request.device_tasks;
    result.plan.region_count_ =
        static_cast<std::uint16_t>(request.regions.size());
    result.plan.task_count_ =
        static_cast<std::uint16_t>(request.tasks.size());
    for (std::size_t index = 0; index < request.regions.size(); ++index) {
        result.plan.regions_[index] = request.regions[index];
    }
    for (std::size_t index = 0; index < request.tasks.size(); ++index) {
        result.plan.tasks_[index] = request.tasks[index];
    }
    result.plan.workspace_ = workspace;
    result.plan.implementation_identity_ = {
        .implementation_sha256 = profile.implementation_sha256,
        .specialization_id = profile.specialization_id,
        .specialization_version = profile.specialization_version,
        .policy = request.policy,
        .quantization = shape.quantization,
        .group_size = shape.group_size,
        .tile_rows = profile.tile_rows,
        .tile_columns = profile.tile_columns,
        .tile_reduction_columns = profile.tile_reduction_columns,
        .threads_per_threadgroup = profile.threads_per_threadgroup,
        .task_mode = request.task_mode,
        .device_task_descriptor_version =
            device_tasks ? request.device_tasks.version : 0U,
        .required_threadgroup_memory_bytes =
            workspace.threadgroup_memory_bytes,
        .required_accumulator_elements_per_threadgroup =
            workspace.accumulator_elements_per_threadgroup,
        .activation_row_mapping =
            device_tasks ? profile.activation_row_mapping
                         : QuantizedGemmActivationRowMapping::
                               PositionRows,
    };
    result.plan.required_activation_bytes_ =
        required_activation_bytes.value;
    result.plan.required_packed_weight_bytes_ = required_weight_storage;
    result.plan.required_scale_bytes_ = required_scale_storage;
    result.plan.required_bias_bytes_ = required_bias_storage;
    result.plan.required_output_bytes_ = required_output_bytes.value;
    result.plan.device_route_capacity_ = device_route_capacity;
    result.plan.device_task_capacity_ = device_task_capacity;
    result.plan.required_device_task_count_bytes_ =
        required_device_task_count_bytes;
    result.plan.required_device_task_list_bytes_ =
        required_device_task_list_bytes;
    return result;
}

namespace {

bool valid_runtime_indirection(
    const QuantizedGemmRequestPlan& plan,
    std::span<const std::uint32_t> route_input_rows,
    std::span<const std::uint32_t> route_output_rows) noexcept {
    if (plan.task_mode() ==
        QuantizedGemmTaskMode::DevicePaddedSlotsV1) {
        return route_input_rows.empty() && route_output_rows.empty();
    }
    if (plan.shape().workload == QuantizedGemmWorkload::Dense) {
        return route_input_rows.empty() && route_output_rows.empty();
    }
    if (route_input_rows.size() != plan.shape().route_list_count ||
        route_output_rows.size() != plan.shape().route_list_count) {
        return false;
    }
    for (std::size_t index = 0; index < route_input_rows.size(); ++index) {
        // The scatter map is the canonical permutation 0..R-1; accepting an
        // arbitrary permutation would require separately planned uniqueness scratch.
        if (route_input_rows[index] >= plan.shape().input_rows ||
            route_output_rows[index] != index) {
            return false;
        }
    }
    return true;
}

bool valid_workspace_view(const QuantizedGemmRequestPlan& plan,
                          std::span<std::byte> workspace) noexcept {
    return workspace.size() >= plan.workspace().required_bytes &&
           (plan.workspace().required_bytes == 0 ||
            reinterpret_cast<std::uintptr_t>(workspace.data()) %
                    plan.workspace().alignment ==
                0);
}

bool component_in_workspace(std::uint64_t offset, std::uint64_t bytes,
                            std::uint64_t workspace_bytes) noexcept {
    if (bytes == 0) {
        return offset == kNoQuantizedGemmWorkspaceOffset;
    }
    if (offset == kNoQuantizedGemmWorkspaceOffset) {
        return false;
    }
    const CheckedU64 end = checked_add(offset, bytes);
    return end.valid && end.value <= workspace_bytes;
}

bool byte_ranges_overlap(const std::byte* left_data,
                         std::size_t left_size,
                         const std::byte* right_data,
                         std::size_t right_size) noexcept {
    if (left_size == 0 || right_size == 0) {
        return false;
    }
    const std::uintptr_t left_begin =
        reinterpret_cast<std::uintptr_t>(left_data);
    const std::uintptr_t right_begin =
        reinterpret_cast<std::uintptr_t>(right_data);
    const CheckedU64 left_end =
        checked_add(left_begin, left_size);
    const CheckedU64 right_end =
        checked_add(right_begin, right_size);
    if (!left_end.valid || !right_end.valid) {
        return true;
    }
    return left_begin < right_end.value &&
           right_begin < left_end.value;
}

bool spans_overlap(std::span<const std::byte> source,
                   std::span<std::byte> destination) noexcept {
    return byte_ranges_overlap(source.data(), source.size(),
                               destination.data(),
                               destination.size());
}

bool spans_overlap(std::span<const std::byte> left,
                   std::span<const std::byte> right) noexcept {
    return byte_ranges_overlap(left.data(), left.size(), right.data(),
                               right.size());
}

} // namespace

QuantizedGemmRuntimeValidation validate_quantized_gemm_runtime(
    const QuantizedGemmRequestPlan& plan,
    const QuantizedGemmRuntimeRequest& request) noexcept {
    QuantizedGemmRuntimeValidation result{
        .error = QuantizedGemmRuntimeError::InvalidPlan,
        .required_workspace_bytes = plan.workspace().required_bytes,
        .available_workspace_bytes =
            static_cast<std::uint64_t>(request.workspace.size()),
        .required_output_bytes = plan.required_output_bytes(),
        .available_output_bytes =
            static_cast<std::uint64_t>(request.output.size()),
    };
    if (!plan.valid()) {
        return result;
    }
    if (request.policy != plan.selected_policy()) {
        result.error = QuantizedGemmRuntimeError::PolicyChangeProhibited;
        return result;
    }
    const bool device_tasks =
        plan.task_mode() ==
        QuantizedGemmTaskMode::DevicePaddedSlotsV1;
    if (!device_tasks) {
        if (request.input_rows != plan.shape().input_rows ||
            request.output_rows != plan.shape().output_rows ||
            request.device_task_position_count != 0 ||
            !request.device_task_counts.empty() ||
            !request.device_task_lists.empty()) {
            result.error = QuantizedGemmRuntimeError::InvalidRange;
            return result;
        }
    } else {
        const QuantizedGemmDeviceTaskDescriptor& descriptor =
            plan.device_tasks();
        if (request.device_task_position_count == 0 ||
            request.device_task_position_count >
                descriptor.position_capacity) {
            result.error = QuantizedGemmRuntimeError::InvalidRange;
            return result;
        }
        const CheckedU64 padded_rows = checked_multiply(
            request.device_task_position_count,
            descriptor.padded_slot_stride);
        const std::uint64_t expected_input_rows =
            descriptor.activation_row_mapping ==
                    QuantizedGemmActivationRowMapping::PositionRows
                ? request.device_task_position_count
                : padded_rows.value;
        if (!padded_rows.valid ||
            expected_input_rows >
                std::numeric_limits<std::uint32_t>::max() ||
            padded_rows.value >
                std::numeric_limits<std::uint32_t>::max() ||
            request.input_rows != expected_input_rows ||
            request.output_rows != padded_rows.value) {
            result.error = QuantizedGemmRuntimeError::InvalidRange;
            return result;
        }
    }
    if (!valid_runtime_indirection(plan, request.route_input_rows,
                                   request.route_output_rows)) {
        result.error = QuantizedGemmRuntimeError::InvalidIndirection;
        return result;
    }
    if (request.workspace.size() < plan.workspace().required_bytes) {
        result.error = QuantizedGemmRuntimeError::WorkspaceInsufficient;
        return result;
    }
    if (!valid_workspace_view(plan, request.workspace)) {
        result.error = QuantizedGemmRuntimeError::WorkspaceMisaligned;
        return result;
    }

    const auto require_operand =
        [&result](QuantizedGemmOperandKind operand, std::uint64_t required,
                  std::size_t available) noexcept {
            if (available >= required) {
                return false;
            }
            result.error =
                operand == QuantizedGemmOperandKind::Output
                    ? QuantizedGemmRuntimeError::OutputInsufficient
                    : QuantizedGemmRuntimeError::OperandInsufficient;
            result.operand = operand;
            result.required_operand_bytes = required;
            result.available_operand_bytes = available;
            return true;
        };
    if (require_operand(QuantizedGemmOperandKind::Activations,
                        plan.required_activation_bytes(),
                        request.activations.size()) ||
        require_operand(QuantizedGemmOperandKind::PackedWeights,
                        plan.required_packed_weight_bytes(),
                        request.packed_weights.size()) ||
        require_operand(QuantizedGemmOperandKind::Scales,
                        plan.required_scale_bytes(),
                        request.scales.size()) ||
        require_operand(QuantizedGemmOperandKind::Biases,
                        plan.required_bias_bytes(),
                        request.biases.size()) ||
        require_operand(QuantizedGemmOperandKind::Output,
                        plan.required_output_bytes(),
                        request.output.size()) ||
        (device_tasks &&
         require_operand(
             QuantizedGemmOperandKind::DeviceTaskCounts,
             plan.required_device_task_count_bytes(),
             request.device_task_counts.size())) ||
        (device_tasks &&
         require_operand(
             QuantizedGemmOperandKind::DeviceTaskLists,
             plan.required_device_task_list_bytes(),
             request.device_task_lists.size()))) {
        return result;
    }
    if (plan.workspace().required_bytes != 0 &&
        (spans_overlap(request.activations, request.workspace) ||
         spans_overlap(request.packed_weights, request.workspace) ||
         spans_overlap(request.scales, request.workspace) ||
         spans_overlap(request.biases, request.workspace) ||
         spans_overlap(std::span<const std::byte>{request.output.data(),
                                                 request.output.size()},
                       request.workspace) ||
         (device_tasks &&
          (spans_overlap(request.device_task_counts,
                         request.workspace) ||
           spans_overlap(request.device_task_lists,
                         request.workspace))))) {
        result.error =
            QuantizedGemmRuntimeError::WorkspaceAliasesOperand;
        return result;
    }
    if (device_tasks) {
        if (reinterpret_cast<std::uintptr_t>(
                request.device_task_counts.data()) %
                    alignof(std::uint32_t) !=
                0 ||
            reinterpret_cast<std::uintptr_t>(
                request.device_task_lists.data()) %
                    alignof(std::uint32_t) !=
                0) {
            result.error =
                QuantizedGemmRuntimeError::DeviceTaskResourceMisaligned;
            return result;
        }
        const std::span<const std::byte> output_bytes{
            request.output.data(), request.output.size()};
        const auto aliases_operand =
            [&request, output_bytes](
                std::span<const std::byte> device_resource) noexcept {
                return spans_overlap(device_resource,
                                     request.activations) ||
                       spans_overlap(device_resource,
                                     request.packed_weights) ||
                       spans_overlap(device_resource, request.scales) ||
                       spans_overlap(device_resource, request.biases) ||
                       spans_overlap(device_resource, output_bytes);
            };
        if (spans_overlap(request.device_task_counts,
                          request.device_task_lists) ||
            aliases_operand(request.device_task_counts) ||
            aliases_operand(request.device_task_lists)) {
            result.error = QuantizedGemmRuntimeError::
                DeviceTaskResourceAliasesOperand;
            return result;
        }
    }
    if (!request.capability.available) {
        result.error = QuantizedGemmRuntimeError::CapabilityUnavailable;
        return result;
    }
    if (!same_identity(request.capability.identity,
                       plan.implementation_identity())) {
        result.error =
            QuantizedGemmRuntimeError::CapabilityIdentityMismatch;
        return result;
    }
    result.error = QuantizedGemmRuntimeError::None;
    return result;
}

QuantizedGemmMaterializationResult materialize_quantized_gemm_workspace(
    const QuantizedGemmRequestPlan& plan,
    const QuantizedGemmMaterializationRequest& request) noexcept {
    QuantizedGemmMaterializationResult result{
        .error = QuantizedGemmMaterializationError::InvalidPlan,
        .required_workspace_bytes = plan.workspace().required_bytes,
        .available_workspace_bytes =
            static_cast<std::uint64_t>(request.workspace.size()),
        .bytes_written = 0,
    };
    if (!plan.valid()) {
        return result;
    }
    if (plan.selected_policy() == QuantizedGemmPolicy::ExactRow) {
        if (!valid_runtime_indirection(plan, request.route_input_rows,
                                       request.route_output_rows)) {
            result.error =
                QuantizedGemmMaterializationError::InvalidIndirection;
            return result;
        }
        result.error = QuantizedGemmMaterializationError::None;
        return result;
    }
    if (request.workspace.size() < plan.workspace().required_bytes) {
        result.error =
            QuantizedGemmMaterializationError::WorkspaceInsufficient;
        return result;
    }
    if (!valid_workspace_view(plan, request.workspace)) {
        result.error =
            QuantizedGemmMaterializationError::WorkspaceMisaligned;
        return result;
    }
    if (!valid_runtime_indirection(plan, request.route_input_rows,
                                   request.route_output_rows)) {
        result.error =
            QuantizedGemmMaterializationError::InvalidIndirection;
        return result;
    }
    const QuantizedGemmWorkspacePlan& workspace = plan.workspace();
    if (!component_in_workspace(workspace.region_descriptor_offset,
                                workspace.region_descriptor_bytes,
                                workspace.required_bytes) ||
        !component_in_workspace(workspace.task_descriptor_offset,
                                workspace.task_descriptor_bytes,
                                workspace.required_bytes) ||
        !component_in_workspace(
            workspace.device_task_indirect_argument_offset,
            workspace.device_task_indirect_argument_bytes,
            workspace.required_bytes) ||
        !component_in_workspace(workspace.device_task_status_offset,
                                workspace.device_task_status_bytes,
                                workspace.required_bytes) ||
        !component_in_workspace(workspace.route_input_index_offset,
                                workspace.route_input_index_bytes,
                                workspace.required_bytes) ||
        !component_in_workspace(workspace.route_output_index_offset,
                                workspace.route_output_index_bytes,
                                workspace.required_bytes) ||
        !component_in_workspace(workspace.partial_offset,
                                workspace.partial_bytes,
                                workspace.required_bytes)) {
        return result;
    }

    const std::span<const std::byte> region_source =
        std::as_bytes(plan.regions());
    const std::span<const std::byte> task_source =
        std::as_bytes(plan.tasks());
    const std::span<const std::byte> input_source =
        std::as_bytes(request.route_input_rows);
    const std::span<const std::byte> output_source =
        std::as_bytes(request.route_output_rows);
    const bool device_tasks =
        plan.task_mode() ==
        QuantizedGemmTaskMode::DevicePaddedSlotsV1;
    if (region_source.size() != workspace.region_descriptor_bytes ||
        (!device_tasks &&
         task_source.size() != workspace.task_descriptor_bytes) ||
        (device_tasks && !task_source.empty()) ||
        input_source.size() != workspace.route_input_index_bytes ||
        output_source.size() != workspace.route_output_index_bytes) {
        return result;
    }
    if (spans_overlap(region_source, request.workspace) ||
        spans_overlap(task_source, request.workspace) ||
        spans_overlap(input_source, request.workspace) ||
        spans_overlap(output_source, request.workspace)) {
        result.error =
            QuantizedGemmMaterializationError::SourceAliasesWorkspace;
        return result;
    }

    static_assert(
        std::is_trivially_copyable_v<QuantizedGemmRegionDescriptor>);
    static_assert(
        std::is_trivially_copyable_v<QuantizedGemmTaskDescriptor>);
    if (workspace.region_descriptor_bytes != 0) {
        std::memcpy(request.workspace.data() +
                        workspace.region_descriptor_offset,
                    region_source.data(),
                    workspace.region_descriptor_bytes);
    }
    if (!device_tasks && workspace.task_descriptor_bytes != 0) {
        std::memcpy(
            request.workspace.data() + workspace.task_descriptor_offset,
            task_source.data(), workspace.task_descriptor_bytes);
    }
    if (workspace.route_input_index_bytes != 0) {
        std::memcpy(
            request.workspace.data() + workspace.route_input_index_offset,
            input_source.data(), workspace.route_input_index_bytes);
    }
    if (workspace.route_output_index_bytes != 0) {
        std::memcpy(
            request.workspace.data() + workspace.route_output_index_offset,
            output_source.data(), workspace.route_output_index_bytes);
    }
    result.bytes_written =
        workspace.region_descriptor_bytes +
        (device_tasks ? 0U : workspace.task_descriptor_bytes) +
        workspace.route_input_index_bytes +
        workspace.route_output_index_bytes;
    result.error = QuantizedGemmMaterializationError::None;
    return result;
}

} // namespace tatara::runtime
