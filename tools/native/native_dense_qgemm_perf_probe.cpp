#include "tatara/backend/metal/commands.h"
#include "tatara/backend/metal/pipeline.h"
#include "tatara/backend/metal/resources.h"
#include "tatara/generated/kernel_library.h"
#include "tatara/generated/model_plan.h"
#include "tatara/runtime/prefill_geometry.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace tatara::backend::metal;
using namespace tatara::runtime;

constexpr std::uint32_t kQ4ValuesPerWord = 8;
constexpr std::uint32_t kGroupSize = 64;
constexpr std::uint32_t kSimdgroupThreads = 32;
constexpr std::uint32_t kN1Threads = 128;
constexpr std::uint32_t kN1TileRows = 32;
constexpr std::uint32_t kN1TileColumns = 32;
constexpr std::uint32_t kExactBundleThreads = 128;
constexpr std::uint32_t kExactOutputThreads = 64;
constexpr std::size_t kGuardElements = 16;
constexpr std::uint16_t kOutputCanary = 0x5A5A;
constexpr std::uint16_t kBodyPoison = 0x7FC1;
constexpr std::uint32_t kMaximumBfloatUlp = 2;
constexpr float kMaximumNormalizedError = 0.02F;

constexpr int kExitUsage = 40;
constexpr int kExitGeometry = 41;
constexpr int kExitExtent = 42;
constexpr int kExitCpuContract = 43;
constexpr int kExitDevice = 44;
constexpr int kExitQueue = 45;
constexpr int kExitLibrary = 46;
constexpr int kExitFunction = 47;
constexpr int kExitExactPipeline = 48;
constexpr int kExitN1Pipeline = 49;
constexpr int kExitAllocation = 50;
constexpr int kExitInitialization = 51;
constexpr int kExitCommandBuffer = 52;
constexpr int kExitComputePass = 53;
constexpr int kExitEncode = 54;
constexpr int kExitEndPass = 55;
constexpr int kExitCommit = 56;
constexpr int kExitExecution = 57;
constexpr int kExitTiming = 58;
constexpr int kExitCanary = 59;
constexpr int kExitExactNondeterminism = 60;
constexpr int kExitN1Nondeterminism = 61;
constexpr int kExitStagedOracle = 62;
constexpr int kExitNumericalFamily = 63;
constexpr int kExitSampleAccounting = 64;
constexpr int kExitControlDrift = 65;
constexpr int kExitSteelPipeline = 66;
constexpr int kExitSteelGdnFused2Pipeline = 67;
constexpr int kExitSteelGdnFused2Mismatch = 68;
constexpr int kExitSteelGdnBn64Pipeline = 69;
constexpr int kExitSteelGdnBn64Mismatch = 70;
constexpr int kExitSteelGdnBk64Pipeline = 71;
constexpr int kExitSteelGdnBk64Mismatch = 72;
constexpr int kExitSteelGdnBm64Pipeline = 73;
constexpr int kExitSteelGdnBm64Mismatch = 74;
constexpr int kExitSteelGdnBm64Bk64Pipeline = 75;
constexpr int kExitSteelGdnBm64Bk64Mismatch = 76;
constexpr int kExitSteelGdnBm48Pipeline = 77;
constexpr int kExitSteelGdnBm48Mismatch = 78;
constexpr int kExitSteelGdnBm96Pipeline = 79;
constexpr int kExitSteelGdnBm96Mismatch = 80;
constexpr int kExitSteelGdnBm128Pipeline = 81;
constexpr int kExitSteelGdnBm128Mismatch = 82;
constexpr int kExitSteelGdnBm64Wm2Wn2Pipeline = 83;
constexpr int kExitSteelGdnBm64Wm2Wn2Mismatch = 84;

constexpr PrefillPolicy kGeometryPolicy{
    .schedule = PrefillSchedule::LayerMajor,
    .context_capacity = 16384,
    .maximum_block_rows = 2048,
    .first_chunk_rows = 256,
    .query_tile_rows = 256,
    .attention_partition = 256,
    .exact_rows_per_threadgroup = 16,
    .gdn_gate_hoist = true,
};

constexpr auto kGeometryResult = make_prefill_geometry(
    tatara::model::qwen36::generated::kModelPlan, kGeometryPolicy);
static_assert(kGeometryResult);
constexpr PrefillGeometry kGeometry = kGeometryResult.geometry;

// This is a workload fixture, not a model topology constant: the frozen
// 3,925-row prefix is chunked by the admitted first/max policy.
constexpr std::array<std::uint32_t, 3> kWorkloadRows{
    kGeometryPolicy.first_chunk_rows,
    kGeometryPolicy.maximum_block_rows,
    3925U - kGeometryPolicy.first_chunk_rows -
        kGeometryPolicy.maximum_block_rows,
};
static_assert(
    kWorkloadRows[0] + kWorkloadRows[1] + kWorkloadRows[2] == 3925U);
static_assert(kWorkloadRows[2] > 0);
static_assert(kWorkloadRows[2] <= kGeometryPolicy.maximum_block_rows);

constexpr std::string_view kWarmupSchedule =
    "ABBAABBAABBAABBAABBAABBAABBAABBAABBAABBAABBAABBAABBAABBAABBAABBA";
constexpr std::string_view kMeasuredSchedule =
    "ABABBABAABABBABAABABBABAABBAABABBA";

consteval std::size_t count_arm(std::string_view schedule, char arm) {
    std::size_t count = 0;
    for (const char value : schedule) {
        count += static_cast<std::size_t>(value == arm);
    }
    return count;
}

constexpr std::size_t kCorrectnessCommandBuffersPerCase = 2;
constexpr std::size_t kWarmupCommandBuffersPerCase = kWarmupSchedule.size();
constexpr std::size_t kMeasuredCommandBuffersPerCase =
    kMeasuredSchedule.size();
constexpr std::size_t kMeasuredSamplesPerArm = 17;
constexpr std::size_t kCommandBuffersPerCase =
    kCorrectnessCommandBuffersPerCase + kWarmupCommandBuffersPerCase +
    kMeasuredCommandBuffersPerCase;
constexpr std::size_t kCaseCount = 9;
constexpr std::size_t kTotalCommandBuffers =
    kCaseCount * kCommandBuffersPerCase;

static_assert(count_arm(kWarmupSchedule, 'A') == 32);
static_assert(count_arm(kWarmupSchedule, 'B') == 32);
static_assert(count_arm(kMeasuredSchedule, 'A') == kMeasuredSamplesPerArm);
static_assert(count_arm(kMeasuredSchedule, 'B') == kMeasuredSamplesPerArm);
static_assert(kMeasuredSchedule.front() == 'A');
static_assert(kMeasuredSchedule.back() == 'A');
static_assert(kCommandBuffersPerCase == 100);
static_assert(kTotalCommandBuffers == 900);
static_assert(kN1Threads ==
              tatara::backend::metal::generated::
                  kKernelLibraryNativeDenseQgemmN1Threads);
static_assert(
    kN1TileRows ==
    tatara::backend::metal::generated::
        kKernelLibraryNativeDenseQgemmN1TileRows);
static_assert(
    kN1TileColumns ==
    tatara::backend::metal::generated::
        kKernelLibraryNativeDenseQgemmN1TileColumns);
static_assert(
    kGroupSize ==
    tatara::backend::metal::generated::kKernelLibraryGroupSize);

enum class OperationKind : std::uint8_t {
    GdnInput,
    AttentionInput,
    Output,
};

enum class Arm : std::uint8_t {
    Exact,
    N1,
    Steel,
    SteelGdnFused2,
    SteelGdnBm64,
    SteelGdnBm64Wm2Wn2,
    SteelGdnBm64Bk64,
    SteelGdnBm48,
    SteelGdnBm96,
    SteelGdnBm128,
    SteelGdnBn64,
    SteelGdnBk64,
};

constexpr bool is_gdn_treatment(Arm arm) noexcept {
    return arm == Arm::SteelGdnFused2 ||
           arm == Arm::SteelGdnBm64 ||
           arm == Arm::SteelGdnBm64Wm2Wn2 ||
           arm == Arm::SteelGdnBm64Bk64 ||
           arm == Arm::SteelGdnBm48 ||
           arm == Arm::SteelGdnBm96 ||
           arm == Arm::SteelGdnBm128 ||
           arm == Arm::SteelGdnBn64 ||
           arm == Arm::SteelGdnBk64;
}

constexpr int gdn_mismatch_exit(Arm arm) noexcept {
    switch (arm) {
    case Arm::SteelGdnBm64:
        return kExitSteelGdnBm64Mismatch;
    case Arm::SteelGdnBm64Wm2Wn2:
        return kExitSteelGdnBm64Wm2Wn2Mismatch;
    case Arm::SteelGdnBm64Bk64:
        return kExitSteelGdnBm64Bk64Mismatch;
    case Arm::SteelGdnBm48:
        return kExitSteelGdnBm48Mismatch;
    case Arm::SteelGdnBm96:
        return kExitSteelGdnBm96Mismatch;
    case Arm::SteelGdnBm128:
        return kExitSteelGdnBm128Mismatch;
    case Arm::SteelGdnBn64:
        return kExitSteelGdnBn64Mismatch;
    case Arm::SteelGdnBk64:
        return kExitSteelGdnBk64Mismatch;
    default:
        return kExitSteelGdnFused2Mismatch;
    }
}

constexpr std::string_view gdn_component_name(Arm arm) noexcept {
    switch (arm) {
    case Arm::SteelGdnBm64:
        return "tatara_mlx_steel_gdn_bm64_component";
    case Arm::SteelGdnBm64Wm2Wn2:
        return "tatara_mlx_steel_gdn_bm64_wm2_wn2_component";
    case Arm::SteelGdnBm64Bk64:
        return "tatara_mlx_steel_gdn_bm64_bk64_component";
    case Arm::SteelGdnBm48:
        return "tatara_mlx_steel_gdn_bm48_component";
    case Arm::SteelGdnBm96:
        return "tatara_mlx_steel_gdn_bm96_component";
    case Arm::SteelGdnBm128:
        return "tatara_mlx_steel_gdn_bm128_component";
    case Arm::SteelGdnBn64:
        return "tatara_mlx_steel_gdn_bn64_component";
    case Arm::SteelGdnBk64:
        return "tatara_mlx_steel_gdn_bk64_component";
    default:
        return "tatara_mlx_steel_gdn_fused2_component";
    }
}

constexpr std::string_view gdn_perf_name(Arm arm) noexcept {
    switch (arm) {
    case Arm::SteelGdnBm64:
        return "tatara_mlx_steel_gdn_bm64_perf";
    case Arm::SteelGdnBm64Wm2Wn2:
        return "tatara_mlx_steel_gdn_bm64_wm2_wn2_perf";
    case Arm::SteelGdnBm64Bk64:
        return "tatara_mlx_steel_gdn_bm64_bk64_perf";
    case Arm::SteelGdnBm48:
        return "tatara_mlx_steel_gdn_bm48_perf";
    case Arm::SteelGdnBm96:
        return "tatara_mlx_steel_gdn_bm96_perf";
    case Arm::SteelGdnBm128:
        return "tatara_mlx_steel_gdn_bm128_perf";
    case Arm::SteelGdnBn64:
        return "tatara_mlx_steel_gdn_bn64_perf";
    case Arm::SteelGdnBk64:
        return "tatara_mlx_steel_gdn_bk64_perf";
    default:
        return "tatara_mlx_steel_gdn_fused2_perf";
    }
}

constexpr std::string_view gdn_treatment_dispatch_label(
    Arm arm) noexcept {
    if (arm == Arm::SteelGdnFused2) {
        return "treatment_dispatches=1";
    }
    if (arm == Arm::SteelGdnBm48 ||
        arm == Arm::SteelGdnBm96) {
        return "treatment_dispatches=8";
    }
    return "treatment_dispatches=4";
}

struct OperationSpec {
    OperationKind kind;
    std::string_view name;
    std::uint32_t columns;
    std::uint32_t reduction;
    std::array<std::uint32_t, 4> region_columns;
    std::uint32_t region_count;
};

struct CaseSpec {
    OperationSpec operation;
    std::uint32_t rows;
};

constexpr std::array<OperationSpec, 3> make_operations() {
    const std::uint32_t gdn_qkv =
        kGeometry.gdn_qk_values + kGeometry.gdn_value_values;
    const std::uint32_t gdn_z = kGeometry.gdn_value_values;
    const std::uint32_t gdn_parameter = kGeometry.recurrent_heads;
    const std::uint32_t attention_qgate =
        2U * kGeometry.query_heads *
        kGeometry.attention_head_dimension;
    const std::uint32_t attention_kv =
        kGeometry.key_value_heads *
        kGeometry.attention_head_dimension;
    return {{
        {
            .kind = OperationKind::GdnInput,
            .name = "gdn-input",
            .columns = kGeometry.gdn_projection_rows,
            .reduction = kGeometry.hidden,
            .region_columns = {
                gdn_qkv,
                gdn_z,
                gdn_parameter,
                gdn_parameter,
            },
            .region_count = 4,
        },
        {
            .kind = OperationKind::AttentionInput,
            .name = "attention-input",
            .columns = kGeometry.attention_projection_rows,
            .reduction = kGeometry.hidden,
            .region_columns = {
                attention_qgate,
                attention_kv,
                attention_kv,
                0,
            },
            .region_count = 3,
        },
        {
            .kind = OperationKind::Output,
            .name = "output",
            .columns = kGeometry.hidden,
            .reduction = kGeometry.gdn_value_values,
            .region_columns = {kGeometry.hidden, 0, 0, 0},
            .region_count = 1,
        },
    }};
}

constexpr std::array<OperationSpec, 3> kOperations = make_operations();
static_assert(
    kGeometry.gdn_value_values == kGeometry.attention_vector_values);
static_assert(
    kOperations[0].region_columns[0] +
        kOperations[0].region_columns[1] +
        kOperations[0].region_columns[2] +
        kOperations[0].region_columns[3] ==
    kOperations[0].columns);
static_assert(
    kOperations[1].region_columns[0] +
        kOperations[1].region_columns[1] +
        kOperations[1].region_columns[2] ==
    kOperations[1].columns);
static_assert(kOperations[0].region_count == 4);
static_assert(kOperations[1].region_count == 3);
static_assert(kOperations[2].region_count == 1);

constexpr std::array<CaseSpec, kCaseCount> make_cases() {
    std::array<CaseSpec, kCaseCount> cases{};
    std::size_t index = 0;
    for (const OperationSpec& operation : kOperations) {
        for (const std::uint32_t rows : kWorkloadRows) {
            cases[index++] = {.operation = operation, .rows = rows};
        }
    }
    return cases;
}

constexpr std::array<CaseSpec, kCaseCount> kCases = make_cases();

bool checked_add(
    std::uint64_t left, std::uint64_t right,
    std::uint64_t& result) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

bool checked_multiply(
    std::uint64_t left, std::uint64_t right,
    std::uint64_t& result) noexcept {
    if (left != 0 &&
        right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

std::uint64_t ceil_div(
    std::uint64_t numerator, std::uint64_t denominator) noexcept {
    return numerator / denominator +
           static_cast<std::uint64_t>(numerator % denominator != 0);
}

std::uint16_t bfloat16(float value) noexcept {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t exponent = bits & 0x7F800000U;
    if (exponent != 0x7F800000U) {
        bits += 0x7FFFU + ((bits >> 16U) & 1U);
    }
    return static_cast<std::uint16_t>(bits >> 16U);
}

float from_bfloat16(std::uint16_t value) noexcept {
    return std::bit_cast<float>(
        static_cast<std::uint32_t>(value) << 16U);
}

std::uint32_t ordered_bfloat(std::uint16_t value) noexcept {
    if ((value & 0x8000U) != 0U) {
        return 0x8000U -
               static_cast<std::uint32_t>(value & 0x7FFFU);
    }
    return 0x8000U + static_cast<std::uint32_t>(value);
}

std::uint32_t bfloat_ulp_distance(
    std::uint16_t left, std::uint16_t right) noexcept {
    const std::uint32_t ordered_left = ordered_bfloat(left);
    const std::uint32_t ordered_right = ordered_bfloat(right);
    return ordered_left > ordered_right
               ? ordered_left - ordered_right
               : ordered_right - ordered_left;
}

std::uint64_t hash_bfloat(
    std::span<const std::uint16_t> values) noexcept {
    constexpr std::uint64_t kOffset = 1469598103934665603ULL;
    constexpr std::uint64_t kPrime = 1099511628211ULL;
    std::uint64_t hash = kOffset;
    for (const std::uint16_t value : values) {
        hash ^= static_cast<std::uint8_t>(value & 0xFFU);
        hash *= kPrime;
        hash ^= static_cast<std::uint8_t>(value >> 8U);
        hash *= kPrime;
    }
    return hash;
}

struct ResourceBudget {
    std::uint64_t activation_bytes = 0;
    std::uint64_t weight_bytes = 0;
    std::uint64_t output_bytes = 0;
    std::uint64_t gpu_bytes = 0;
    std::uint64_t with_host_snapshot_bytes = 0;
};

bool make_resource_budget(
    const CaseSpec& spec, ResourceBudget& budget) noexcept {
    std::uint64_t activation_values = 0;
    std::uint64_t packed_bytes = 0;
    std::uint64_t parameter_bytes = 0;
    std::uint64_t output_values = 0;
    std::uint64_t output_storage_values = 0;
    std::uint64_t output_bytes = 0;
    std::uint64_t host_snapshot_bytes = 0;
    std::uint64_t weights = 0;
    std::uint64_t gpu = 0;
    if (!checked_multiply(
            spec.rows, spec.operation.reduction, activation_values) ||
        !checked_multiply(activation_values, sizeof(std::uint16_t),
                          budget.activation_bytes) ||
        !checked_multiply(
            spec.operation.columns, spec.operation.reduction,
            packed_bytes) ||
        packed_bytes % 2U != 0 ||
        !checked_multiply(
            spec.operation.columns, spec.operation.reduction,
            parameter_bytes) ||
        parameter_bytes % 32U != 0 ||
        !checked_add(
            packed_bytes / 2U, parameter_bytes / 16U, weights) ||
        !checked_multiply(
            spec.rows, spec.operation.columns, output_values) ||
        !checked_add(
            output_values, 2U * kGuardElements,
            output_storage_values) ||
        !checked_multiply(
            output_storage_values, sizeof(std::uint16_t),
            output_bytes) ||
        !checked_multiply(
            output_values, sizeof(std::uint16_t),
            host_snapshot_bytes) ||
        !checked_add(budget.activation_bytes, weights, gpu) ||
        !checked_add(gpu, output_bytes, gpu) ||
        !checked_add(
            gpu, host_snapshot_bytes,
            budget.with_host_snapshot_bytes)) {
        return false;
    }
    budget.weight_bytes = weights;
    budget.output_bytes = output_bytes;
    budget.gpu_bytes = gpu;
    return true;
}

bool validate_cpu_contract(
    std::uint64_t& maximum_with_host_bytes) noexcept {
    maximum_with_host_bytes = 0;
    for (const CaseSpec& spec : kCases) {
        if (spec.rows == 0 || spec.operation.columns == 0 ||
            spec.operation.reduction == 0 ||
            spec.operation.reduction % kGroupSize != 0 ||
            spec.operation.region_count == 0 ||
            spec.operation.region_count >
                spec.operation.region_columns.size()) {
            return false;
        }
        std::uint32_t region_sum = 0;
        for (std::uint32_t index = 0;
             index < spec.operation.region_count; ++index) {
            if (spec.operation.region_columns[index] == 0 ||
                spec.operation.region_columns[index] >
                    std::numeric_limits<std::uint32_t>::max() -
                        region_sum) {
                return false;
            }
            region_sum += spec.operation.region_columns[index];
        }
        if (region_sum != spec.operation.columns) {
            return false;
        }
        ResourceBudget budget;
        if (!make_resource_budget(spec, budget)) {
            return false;
        }
        maximum_with_host_bytes = std::max(
            maximum_with_host_bytes,
            budget.with_host_snapshot_bytes);
    }
    return true;
}

struct RegionResources {
    std::uint32_t columns = 0;
    std::uint32_t column_begin = 0;
    MetalBuffer packed;
    MetalBuffer scales;
    MetalBuffer biases;
};

struct CaseResources {
    CaseSpec spec;
    MetalBuffer activations;
    std::array<RegionResources, 4> regions;
    MetalBuffer output;
    std::vector<std::uint16_t> exact_snapshot;
};

std::uint64_t hash_buffer(
    const MetalBuffer& buffer, std::uint64_t hash) noexcept {
    constexpr std::uint64_t kPrime = 1099511628211ULL;
    const auto* bytes =
        static_cast<const std::uint8_t*>(buffer.contents());
    for (std::uint64_t index = 0;
         index < buffer.size_bytes(); ++index) {
        hash ^= bytes[index];
        hash *= kPrime;
    }
    return hash;
}

std::uint64_t input_identity(
    const CaseResources& resources) noexcept {
    constexpr std::uint64_t kOffset = 1469598103934665603ULL;
    std::uint64_t hash =
        hash_buffer(resources.activations, kOffset);
    for (std::uint32_t index = 0;
         index < resources.spec.operation.region_count; ++index) {
        const RegionResources& region = resources.regions[index];
        hash = hash_buffer(region.packed, hash);
        hash = hash_buffer(region.scales, hash);
        hash = hash_buffer(region.biases, hash);
    }
    return hash;
}

std::uint32_t quantized_value(
    std::uint32_t column, std::uint32_t reduction) noexcept {
    return (column * 11U + reduction * 5U + 3U) & 15U;
}

bool move_buffer(
    MetalBufferResult result, MetalBuffer& destination) noexcept {
    if (!result || !result.buffer) {
        return false;
    }
    destination = std::move(*result.buffer);
    return true;
}

int create_case_resources(
    const MetalDevice& device, const CaseSpec& spec,
    CaseResources& resources) {
    resources.spec = spec;
    ResourceBudget budget;
    if (!make_resource_budget(spec, budget)) {
        return kExitExtent;
    }
    std::uint64_t activation_values = 0;
    std::uint64_t output_values = 0;
    std::uint64_t output_storage_values = 0;
    if (!checked_multiply(
            spec.rows, spec.operation.reduction,
            activation_values) ||
        !checked_multiply(
            spec.rows, spec.operation.columns, output_values) ||
        !checked_add(
            output_values, 2U * kGuardElements,
            output_storage_values) ||
        output_values >
            std::numeric_limits<std::size_t>::max()) {
        return kExitExtent;
    }
    if (!move_buffer(
            create_shared_buffer(
                device, activation_values * sizeof(std::uint16_t)),
            resources.activations) ||
        !move_buffer(
            create_shared_buffer(
                device,
                output_storage_values * sizeof(std::uint16_t)),
            resources.output)) {
        return kExitAllocation;
    }
    try {
        resources.exact_snapshot.resize(
            static_cast<std::size_t>(output_values));
    } catch (...) {
        return kExitAllocation;
    }

    auto* activations = static_cast<std::uint16_t*>(
        resources.activations.contents());
    for (std::uint32_t row = 0; row < spec.rows; ++row) {
        for (std::uint32_t reduction = 0;
             reduction < spec.operation.reduction; ++reduction) {
            const std::int32_t band = static_cast<std::int32_t>(
                (row * 19U + reduction * 7U) % 37U) -
                                      18;
            const float value =
                static_cast<float>(band) / 19.0F + 0.03125F;
            activations[
                static_cast<std::size_t>(row) *
                    spec.operation.reduction +
                reduction] = bfloat16(value);
        }
    }

    std::uint32_t column_begin = 0;
    const std::uint32_t words_per_row =
        spec.operation.reduction / kQ4ValuesPerWord;
    const std::uint32_t groups_per_row =
        spec.operation.reduction / kGroupSize;
    for (std::uint32_t region_index = 0;
         region_index < spec.operation.region_count;
         ++region_index) {
        RegionResources& region = resources.regions[region_index];
        region.columns =
            spec.operation.region_columns[region_index];
        region.column_begin = column_begin;
        const std::uint64_t packed_words =
            std::uint64_t{region.columns} * words_per_row;
        const std::uint64_t parameter_values =
            std::uint64_t{region.columns} * groups_per_row;
        if (!move_buffer(
                create_shared_buffer(
                    device,
                    packed_words * sizeof(std::uint32_t)),
                region.packed) ||
            !move_buffer(
                create_shared_buffer(
                    device,
                    parameter_values * sizeof(std::uint16_t)),
                region.scales) ||
            !move_buffer(
                create_shared_buffer(
                    device,
                    parameter_values * sizeof(std::uint16_t)),
                region.biases)) {
            return kExitAllocation;
        }
        std::memset(
            region.packed.contents(), 0,
            static_cast<std::size_t>(
                packed_words * sizeof(std::uint32_t)));
        auto* packed =
            static_cast<std::uint32_t*>(region.packed.contents());
        auto* scales =
            static_cast<std::uint16_t*>(region.scales.contents());
        auto* biases =
            static_cast<std::uint16_t*>(region.biases.contents());
        for (std::uint32_t local_column = 0;
             local_column < region.columns; ++local_column) {
            const std::uint32_t global_column =
                column_begin + local_column;
            for (std::uint32_t reduction = 0;
                 reduction < spec.operation.reduction; ++reduction) {
                const std::size_t word_index =
                    static_cast<std::size_t>(local_column) *
                        words_per_row +
                    reduction / kQ4ValuesPerWord;
                packed[word_index] |=
                    quantized_value(global_column, reduction)
                    << (4U *
                        (reduction % kQ4ValuesPerWord));
            }
            for (std::uint32_t group = 0;
                 group < groups_per_row; ++group) {
                const std::size_t parameter_index =
                    static_cast<std::size_t>(local_column) *
                        groups_per_row +
                    group;
                scales[parameter_index] = bfloat16(
                    0.071F +
                    0.013F *
                        static_cast<float>(global_column % 7U) +
                    0.019F *
                        static_cast<float>(group % 5U));
                biases[parameter_index] = bfloat16(
                    -0.137F +
                    0.017F *
                        static_cast<float>(global_column % 5U) -
                    0.011F *
                        static_cast<float>(group % 3U));
            }
        }
        column_begin += region.columns;
    }
    if (column_begin != spec.operation.columns) {
        return kExitInitialization;
    }

    auto* output =
        static_cast<std::uint16_t*>(resources.output.contents());
    std::fill(
        output,
        output +
            static_cast<std::size_t>(output_storage_values),
        kBodyPoison);
    std::fill(output, output + kGuardElements, kOutputCanary);
    std::fill(
        output + kGuardElements +
            static_cast<std::size_t>(output_values),
        output + static_cast<std::size_t>(output_storage_values),
        kOutputCanary);
    return 0;
}

std::span<std::uint16_t> output_storage(
    CaseResources& resources) noexcept {
    return {
        static_cast<std::uint16_t*>(
            resources.output.contents()),
        static_cast<std::size_t>(
            resources.output.size_bytes() /
            sizeof(std::uint16_t)),
    };
}

std::span<const std::uint16_t> output_body(
    const CaseResources& resources) noexcept {
    const auto* values = static_cast<const std::uint16_t*>(
        resources.output.contents());
    return {
        values + kGuardElements,
        static_cast<std::size_t>(resources.spec.rows) *
            resources.spec.operation.columns,
    };
}

void prepare_output(
    CaseResources& resources, bool poison_body) noexcept {
    std::span<std::uint16_t> storage = output_storage(resources);
    std::fill(
        storage.begin(),
        storage.begin() +
            static_cast<std::ptrdiff_t>(kGuardElements),
        kOutputCanary);
    std::fill(
        storage.end() -
            static_cast<std::ptrdiff_t>(kGuardElements),
        storage.end(), kOutputCanary);
    if (poison_body) {
        std::fill(
            storage.begin() +
                static_cast<std::ptrdiff_t>(kGuardElements),
            storage.end() -
                static_cast<std::ptrdiff_t>(kGuardElements),
            kBodyPoison);
    }
}

bool canaries_intact(
    const CaseResources& resources) noexcept {
    const std::span<const std::uint16_t> storage{
        static_cast<const std::uint16_t*>(
            resources.output.contents()),
        static_cast<std::size_t>(
            resources.output.size_bytes() /
            sizeof(std::uint16_t)),
    };
    for (std::size_t index = 0; index < kGuardElements;
         ++index) {
        if (storage[index] != kOutputCanary ||
            storage[storage.size() - kGuardElements + index] !=
                kOutputCanary) {
            return false;
        }
    }
    return true;
}

bool body_is_finite(
    const CaseResources& resources) noexcept {
    for (const std::uint16_t bits : output_body(resources)) {
        if (!std::isfinite(from_bfloat16(bits))) {
            return false;
        }
    }
    return true;
}

struct Pipelines {
    MetalComputePipeline exact_gdn;
    MetalComputePipeline exact_attention;
    MetalComputePipeline exact_output;
    MetalComputePipeline n1;
    MetalComputePipeline steel;
    MetalComputePipeline steel_gdn_fused2;
    MetalComputePipeline steel_gdn_bm64;
    MetalComputePipeline steel_gdn_bm64_wm2_wn2;
    MetalComputePipeline steel_gdn_bm64_bk64;
    MetalComputePipeline steel_gdn_bm48;
    MetalComputePipeline steel_gdn_bm96;
    MetalComputePipeline steel_gdn_bm128;
    MetalComputePipeline steel_gdn_bn64;
    MetalComputePipeline steel_gdn_bk64;
};

int create_named_pipeline(
    const MetalDevice& device, const MetalLibrary& library,
    std::string_view name, int pipeline_exit,
    MetalComputePipeline& destination) {
    auto function = create_function(library, name);
    if (!function || !function.function) {
        std::cerr << "tatara_native_dense_qgemm_perf_error"
                  << " category=function-lookup"
                  << " function=" << name << '\n';
        return kExitFunction;
    }
    auto pipeline =
        create_compute_pipeline(device, *function.function);
    if (!pipeline || !pipeline.pipeline) {
        std::cerr << "tatara_native_dense_qgemm_perf_error"
                  << " category=pipeline"
                  << " function=" << name << '\n';
        return pipeline_exit;
    }
    destination = std::move(*pipeline.pipeline);
    return 0;
}

int create_pipelines(
    const MetalDevice& device, const MetalLibrary& library,
    Pipelines& pipelines) {
    if (int result = create_named_pipeline(
            device, library, "gdn_project_blk",
            kExitExactPipeline, pipelines.exact_gdn);
        result != 0) {
        return result;
    }
    if (int result = create_named_pipeline(
            device, library, "attn_project_blk",
            kExitExactPipeline, pipelines.exact_attention);
        result != 0) {
        return result;
    }
    if (int result = create_named_pipeline(
            device, library, "outproj_blk",
            kExitExactPipeline, pipelines.exact_output);
        result != 0) {
        return result;
    }
    if (int result = create_named_pipeline(
            device, library, "native_dense_qgemm_q4_bf16_n1",
            kExitN1Pipeline, pipelines.n1);
        result != 0) {
        return result;
    }
    if constexpr (
        tatara::backend::metal::generated::
            kKernelLibraryMlxSteelEnabled) {
        if (int result = create_named_pipeline(
            device, library,
            tatara::backend::metal::generated::
                kKernelLibraryMlxSteelDenseKernelName,
            kExitSteelPipeline, pipelines.steel);
            result != 0) {
            return result;
        }
        if (int result = create_named_pipeline(
            device, library,
            tatara::backend::metal::generated::
                kKernelLibraryMlxSteelGdnFused2KernelName,
            kExitSteelGdnFused2Pipeline,
            pipelines.steel_gdn_fused2);
            result != 0) {
            return result;
        }
        if (int result = create_named_pipeline(
            device, library,
            tatara::backend::metal::generated::
                kKernelLibraryMlxSteelGdnBm64KernelName,
            kExitSteelGdnBm64Pipeline,
            pipelines.steel_gdn_bm64);
            result != 0) {
            return result;
        }
        if (int result = create_named_pipeline(
            device, library,
            tatara::backend::metal::generated::
                kKernelLibraryMlxSteelGdnBm64Wm2Wn2KernelName,
            kExitSteelGdnBm64Wm2Wn2Pipeline,
            pipelines.steel_gdn_bm64_wm2_wn2);
            result != 0) {
            return result;
        }
        if (int result = create_named_pipeline(
            device, library,
            tatara::backend::metal::generated::
                kKernelLibraryMlxSteelGdnBm64Bk64KernelName,
            kExitSteelGdnBm64Bk64Pipeline,
            pipelines.steel_gdn_bm64_bk64);
            result != 0) {
            return result;
        }
        if (int result = create_named_pipeline(
            device, library,
            tatara::backend::metal::generated::
                kKernelLibraryMlxSteelGdnBm48KernelName,
            kExitSteelGdnBm48Pipeline,
            pipelines.steel_gdn_bm48);
            result != 0) {
            return result;
        }
        if (int result = create_named_pipeline(
            device, library,
            tatara::backend::metal::generated::
                kKernelLibraryMlxSteelGdnBm96KernelName,
            kExitSteelGdnBm96Pipeline,
            pipelines.steel_gdn_bm96);
            result != 0) {
            return result;
        }
        if (int result = create_named_pipeline(
            device, library,
            tatara::backend::metal::generated::
                kKernelLibraryMlxSteelGdnBm128KernelName,
            kExitSteelGdnBm128Pipeline,
            pipelines.steel_gdn_bm128);
            result != 0) {
            return result;
        }
        if (int result = create_named_pipeline(
            device, library,
            tatara::backend::metal::generated::
                kKernelLibraryMlxSteelGdnBn64KernelName,
            kExitSteelGdnBn64Pipeline,
            pipelines.steel_gdn_bn64);
            result != 0) {
            return result;
        }
        return create_named_pipeline(
            device, library,
            tatara::backend::metal::generated::
                kKernelLibraryMlxSteelGdnBk64KernelName,
            kExitSteelGdnBk64Pipeline,
            pipelines.steel_gdn_bk64);
    }
    return 0;
}

bool encoded(MetalCommandError error) noexcept {
    return error == MetalCommandError::None;
}

int encode_exact(
    MetalComputePass& pass, const Pipelines& pipelines,
    const CaseResources& resources) {
    const CaseSpec& spec = resources.spec;
    const std::uint64_t output_offset =
        kGuardElements * sizeof(std::uint16_t);
    if (spec.operation.kind == OperationKind::GdnInput) {
        if (!encoded(set_compute_pipeline(
                pass, pipelines.exact_gdn)) ||
            !encoded(set_buffer(
                pass, resources.activations, 0, 0))) {
            return kExitEncode;
        }
        for (std::uint32_t index = 0; index < 4; ++index) {
            const RegionResources& region =
                resources.regions[index];
            const std::uint32_t base = 1U + 3U * index;
            if (!encoded(set_buffer(
                    pass, region.packed, 0, base)) ||
                !encoded(set_buffer(
                    pass, region.scales, 0, base + 1U)) ||
                !encoded(set_buffer(
                    pass, region.biases, 0, base + 2U))) {
                return kExitEncode;
            }
        }
        if (!encoded(set_buffer(
                pass, resources.output, output_offset, 13)) ||
            !encoded(set_bytes(
                pass, &spec.rows, sizeof(spec.rows), 14)) ||
            !encoded(dispatch_threadgroups(
                pass,
                {
                    .width = ceil_div(
                        spec.operation.columns,
                        kExactBundleThreads /
                            kSimdgroupThreads),
                    .height = 1,
                    .depth = 1,
                },
                {
                    .width = kExactBundleThreads,
                    .height = 1,
                    .depth = 1,
                }))) {
            return kExitEncode;
        }
        return 0;
    }
    if (spec.operation.kind == OperationKind::AttentionInput) {
        if (!encoded(set_compute_pipeline(
                pass, pipelines.exact_attention)) ||
            !encoded(set_buffer(
                pass, resources.activations, 0, 0))) {
            return kExitEncode;
        }
        for (std::uint32_t index = 0; index < 3; ++index) {
            const RegionResources& region =
                resources.regions[index];
            const std::uint32_t base = 1U + 3U * index;
            if (!encoded(set_buffer(
                    pass, region.packed, 0, base)) ||
                !encoded(set_buffer(
                    pass, region.scales, 0, base + 1U)) ||
                !encoded(set_buffer(
                    pass, region.biases, 0, base + 2U))) {
                return kExitEncode;
            }
        }
        if (!encoded(set_buffer(
                pass, resources.output, output_offset, 10)) ||
            !encoded(set_bytes(
                pass, &spec.rows, sizeof(spec.rows), 11)) ||
            !encoded(dispatch_threadgroups(
                pass,
                {
                    .width = ceil_div(
                        spec.operation.columns,
                        kExactBundleThreads /
                            kSimdgroupThreads),
                    .height = 1,
                    .depth = 1,
                },
                {
                    .width = kExactBundleThreads,
                    .height = 1,
                    .depth = 1,
                }))) {
            return kExitEncode;
        }
        return 0;
    }

    const RegionResources& region = resources.regions[0];
    if (!encoded(set_compute_pipeline(
            pass, pipelines.exact_output)) ||
        !encoded(set_buffer(
            pass, resources.activations, 0, 0)) ||
        !encoded(set_buffer(pass, region.packed, 0, 1)) ||
        !encoded(set_buffer(pass, region.scales, 0, 2)) ||
        !encoded(set_buffer(pass, region.biases, 0, 3)) ||
        !encoded(set_buffer(
            pass, resources.output, output_offset, 4)) ||
        !encoded(set_bytes(
            pass, &spec.rows, sizeof(spec.rows), 5)) ||
        !encoded(set_bytes(
            pass, &spec.operation.reduction,
            sizeof(spec.operation.reduction), 6)) ||
        !encoded(dispatch_threadgroups(
            pass,
            {
                .width = ceil_div(
                    spec.operation.columns,
                    kExactOutputThreads /
                        kSimdgroupThreads),
                .height = 1,
                .depth = 1,
            },
            {
                .width = kExactOutputThreads,
                .height = 1,
                .depth = 1,
            }))) {
        return kExitEncode;
    }
    return 0;
}

int encode_n1(
    MetalComputePass& pass, const Pipelines& pipelines,
    const CaseResources& resources) {
    const CaseSpec& spec = resources.spec;
    if (!encoded(set_compute_pipeline(pass, pipelines.n1))) {
        return kExitEncode;
    }
    const std::uint64_t activation_stride =
        spec.operation.reduction;
    const std::uint64_t packed_stride_words =
        spec.operation.reduction / kQ4ValuesPerWord;
    const std::uint64_t parameter_stride =
        spec.operation.reduction / kGroupSize;
    const std::uint64_t output_stride =
        spec.operation.columns;
    for (std::uint32_t index = 0;
         index < spec.operation.region_count; ++index) {
        const RegionResources& region = resources.regions[index];
        const std::uint64_t output_offset =
            (kGuardElements + region.column_begin) *
            sizeof(std::uint16_t);
        if (!encoded(set_buffer(
                pass, resources.activations, 0, 0)) ||
            !encoded(set_buffer(pass, region.packed, 0, 1)) ||
            !encoded(set_buffer(pass, region.scales, 0, 2)) ||
            !encoded(set_buffer(pass, region.biases, 0, 3)) ||
            !encoded(set_buffer(
                pass, resources.output, output_offset, 4)) ||
            !encoded(set_bytes(
                pass, &spec.rows, sizeof(spec.rows), 5)) ||
            !encoded(set_bytes(
                pass, &region.columns,
                sizeof(region.columns), 6)) ||
            !encoded(set_bytes(
                pass, &spec.operation.reduction,
                sizeof(spec.operation.reduction), 7)) ||
            !encoded(set_bytes(
                pass, &activation_stride,
                sizeof(activation_stride), 8)) ||
            !encoded(set_bytes(
                pass, &packed_stride_words,
                sizeof(packed_stride_words), 9)) ||
            !encoded(set_bytes(
                pass, &parameter_stride,
                sizeof(parameter_stride), 10)) ||
            !encoded(set_bytes(
                pass, &output_stride,
                sizeof(output_stride), 11)) ||
            !encoded(dispatch_threadgroups(
                pass,
                {
                    .width = ceil_div(
                        region.columns, kN1TileColumns),
                    .height = ceil_div(
                        spec.rows, kN1TileRows),
                    .depth = 1,
                },
                {
                    .width = kN1Threads,
                    .height = 1,
                    .depth = 1,
                }))) {
            return kExitEncode;
        }
    }
    return 0;
}

int encode_steel(
    MetalComputePass& pass, const Pipelines& pipelines,
    const CaseResources& resources) {
    const CaseSpec& spec = resources.spec;
    if (spec.rows % kN1TileRows != 0 ||
        spec.operation.columns % kN1TileColumns != 0 ||
        spec.operation.reduction % 32U != 0 ||
        !encoded(set_compute_pipeline(pass, pipelines.steel))) {
        return kExitEncode;
    }
    const std::int32_t rows =
        static_cast<std::int32_t>(spec.rows);
    const std::int32_t reduction =
        static_cast<std::int32_t>(spec.operation.reduction);
    const std::int32_t output_stride =
        static_cast<std::int32_t>(spec.operation.columns);
    for (std::uint32_t index = 0;
         index < spec.operation.region_count; ++index) {
        const RegionResources& region = resources.regions[index];
        if (region.columns % kN1TileColumns != 0) {
            return kExitEncode;
        }
        const std::int32_t columns =
            static_cast<std::int32_t>(region.columns);
        const std::uint64_t output_offset =
            (kGuardElements + region.column_begin) *
            sizeof(std::uint16_t);
        if (!encoded(set_buffer(
                pass, region.packed, 0, 0)) ||
            !encoded(set_buffer(
                pass, region.scales, 0, 1)) ||
            !encoded(set_buffer(
                pass, region.biases, 0, 2)) ||
            !encoded(set_buffer(
                pass, resources.activations, 0, 3)) ||
            !encoded(set_buffer(
                pass, resources.output, output_offset, 4)) ||
            !encoded(set_bytes(
                pass, &reduction, sizeof(reduction), 5)) ||
            !encoded(set_bytes(
                pass, &columns, sizeof(columns), 6)) ||
            !encoded(set_bytes(
                pass, &rows, sizeof(rows), 7)) ||
            !encoded(set_bytes(
                pass, &output_stride,
                sizeof(output_stride), 8)) ||
            !encoded(dispatch_threadgroups(
                pass,
                {
                    .width = region.columns / kN1TileColumns,
                    .height = spec.rows / kN1TileRows,
                    .depth = 1,
                },
                {
                    .width = kN1Threads / 4U,
                    .height = 2,
                    .depth = 2,
                }))) {
            return kExitEncode;
        }
    }
    return 0;
}

int encode_steel_gdn_fused2(
    MetalComputePass& pass, const Pipelines& pipelines,
    const CaseResources& resources) {
    const CaseSpec& spec = resources.spec;
    if (spec.operation.kind != OperationKind::GdnInput ||
        spec.operation.region_count != 4U ||
        spec.operation.columns != kGeometry.gdn_projection_rows ||
        spec.operation.columns % (2U * kN1TileColumns) != 0U ||
        spec.operation.reduction != kGeometry.hidden ||
        !encoded(set_compute_pipeline(
            pass, pipelines.steel_gdn_fused2))) {
        return kExitEncode;
    }
    for (std::uint32_t index = 0; index < 4U; ++index) {
        const RegionResources& region = resources.regions[index];
        const std::uint32_t base = 3U * index;
        if (!encoded(set_buffer(
                pass, region.packed, 0, base)) ||
            !encoded(set_buffer(
                pass, region.scales, 0, base + 1U)) ||
            !encoded(set_buffer(
                pass, region.biases, 0, base + 2U))) {
            return kExitEncode;
        }
    }
    const std::uint64_t output_offset =
        kGuardElements * sizeof(std::uint16_t);
    const std::uint32_t rows = spec.rows;
    const std::uint32_t reduction = spec.operation.reduction;
    const std::uint32_t output_stride = spec.operation.columns;
    if (!encoded(set_buffer(
            pass, resources.activations, 0, 12)) ||
        !encoded(set_buffer(
            pass, resources.output, output_offset, 13)) ||
        !encoded(set_bytes(
            pass, &rows, sizeof(rows), 14)) ||
        !encoded(set_bytes(
            pass, &reduction, sizeof(reduction), 15)) ||
        !encoded(set_bytes(
            pass, &output_stride, sizeof(output_stride), 16)) ||
        !encoded(dispatch_threadgroups(
            pass,
            {
                .width =
                    spec.operation.columns /
                    (2U * kN1TileColumns),
                .height = ceil_div(
                    spec.rows, kN1TileRows),
                .depth = 1,
            },
            {
                .width = kN1Threads / 4U,
                .height = 2,
                .depth = 2,
            }))) {
        return kExitEncode;
    }
    return 0;
}

int encode_steel_gdn_bm64(
    MetalComputePass& pass, const Pipelines& pipelines,
    const CaseResources& resources) {
    const CaseSpec& spec = resources.spec;
    constexpr std::uint32_t kTileRows = 64U;
    if (spec.operation.kind != OperationKind::GdnInput ||
        spec.operation.region_count != 4U ||
        spec.operation.columns != kGeometry.gdn_projection_rows ||
        spec.operation.reduction != kGeometry.hidden ||
        spec.rows % kTileRows != 0U ||
        !encoded(set_compute_pipeline(
            pass, pipelines.steel_gdn_bm64))) {
        return kExitEncode;
    }
    const std::int32_t rows =
        static_cast<std::int32_t>(spec.rows);
    const std::int32_t reduction =
        static_cast<std::int32_t>(spec.operation.reduction);
    const std::int32_t output_stride =
        static_cast<std::int32_t>(spec.operation.columns);
    for (std::uint32_t index = 0; index < 4U; ++index) {
        const RegionResources& region = resources.regions[index];
        if (region.columns % kN1TileColumns != 0U) {
            return kExitEncode;
        }
        const std::int32_t columns =
            static_cast<std::int32_t>(region.columns);
        const std::uint64_t output_offset =
            (kGuardElements + region.column_begin) *
            sizeof(std::uint16_t);
        if (!encoded(set_buffer(
                pass, region.packed, 0, 0)) ||
            !encoded(set_buffer(
                pass, region.scales, 0, 1)) ||
            !encoded(set_buffer(
                pass, region.biases, 0, 2)) ||
            !encoded(set_buffer(
                pass, resources.activations, 0, 3)) ||
            !encoded(set_buffer(
                pass, resources.output, output_offset, 4)) ||
            !encoded(set_bytes(
                pass, &reduction, sizeof(reduction), 5)) ||
            !encoded(set_bytes(
                pass, &columns, sizeof(columns), 6)) ||
            !encoded(set_bytes(
                pass, &rows, sizeof(rows), 7)) ||
            !encoded(set_bytes(
                pass, &output_stride,
                sizeof(output_stride), 8)) ||
            !encoded(dispatch_threadgroups(
                pass,
                {
                    .width =
                        region.columns / kN1TileColumns,
                    .height = spec.rows / kTileRows,
                    .depth = 1,
                },
                {
                    .width = kSimdgroupThreads,
                    .height = 4,
                    .depth = 1,
                }))) {
            return kExitEncode;
        }
    }
    return 0;
}

int encode_steel_gdn_bm64_wm2_wn2(
    MetalComputePass& pass, const Pipelines& pipelines,
    const CaseResources& resources) {
    const CaseSpec& spec = resources.spec;
    constexpr std::uint32_t kTileRows = 64U;
    if (spec.operation.kind != OperationKind::GdnInput ||
        spec.operation.region_count != 4U ||
        spec.operation.columns != kGeometry.gdn_projection_rows ||
        spec.operation.reduction != kGeometry.hidden ||
        spec.rows % kTileRows != 0U ||
        !encoded(set_compute_pipeline(
            pass, pipelines.steel_gdn_bm64_wm2_wn2))) {
        return kExitEncode;
    }
    const std::int32_t rows =
        static_cast<std::int32_t>(spec.rows);
    const std::int32_t reduction =
        static_cast<std::int32_t>(spec.operation.reduction);
    const std::int32_t output_stride =
        static_cast<std::int32_t>(spec.operation.columns);
    for (std::uint32_t index = 0; index < 4U; ++index) {
        const RegionResources& region = resources.regions[index];
        if (region.columns % kN1TileColumns != 0U) {
            return kExitEncode;
        }
        const std::int32_t columns =
            static_cast<std::int32_t>(region.columns);
        const std::uint64_t output_offset =
            (kGuardElements + region.column_begin) *
            sizeof(std::uint16_t);
        if (!encoded(set_buffer(
                pass, region.packed, 0, 0)) ||
            !encoded(set_buffer(
                pass, region.scales, 0, 1)) ||
            !encoded(set_buffer(
                pass, region.biases, 0, 2)) ||
            !encoded(set_buffer(
                pass, resources.activations, 0, 3)) ||
            !encoded(set_buffer(
                pass, resources.output, output_offset, 4)) ||
            !encoded(set_bytes(
                pass, &reduction, sizeof(reduction), 5)) ||
            !encoded(set_bytes(
                pass, &columns, sizeof(columns), 6)) ||
            !encoded(set_bytes(
                pass, &rows, sizeof(rows), 7)) ||
            !encoded(set_bytes(
                pass, &output_stride,
                sizeof(output_stride), 8)) ||
            !encoded(dispatch_threadgroups(
                pass,
                {
                    .width =
                        region.columns / kN1TileColumns,
                    .height = spec.rows / kTileRows,
                    .depth = 1,
                },
                {
                    .width = kSimdgroupThreads,
                    .height = 2,
                    .depth = 2,
                }))) {
            return kExitEncode;
        }
    }
    return 0;
}

int encode_steel_gdn_bm64_bk64(
    MetalComputePass& pass, const Pipelines& pipelines,
    const CaseResources& resources) {
    const CaseSpec& spec = resources.spec;
    constexpr std::uint32_t kTileRows = 64U;
    if (spec.operation.kind != OperationKind::GdnInput ||
        spec.operation.region_count != 4U ||
        spec.operation.columns != kGeometry.gdn_projection_rows ||
        spec.operation.reduction != kGeometry.hidden ||
        spec.operation.reduction % kGroupSize != 0U ||
        spec.rows % kTileRows != 0U ||
        !encoded(set_compute_pipeline(
            pass, pipelines.steel_gdn_bm64_bk64))) {
        return kExitEncode;
    }
    const std::int32_t rows =
        static_cast<std::int32_t>(spec.rows);
    const std::int32_t reduction =
        static_cast<std::int32_t>(spec.operation.reduction);
    const std::int32_t output_stride =
        static_cast<std::int32_t>(spec.operation.columns);
    for (std::uint32_t index = 0; index < 4U; ++index) {
        const RegionResources& region = resources.regions[index];
        if (region.columns % kN1TileColumns != 0U) {
            return kExitEncode;
        }
        const std::int32_t columns =
            static_cast<std::int32_t>(region.columns);
        const std::uint64_t output_offset =
            (kGuardElements + region.column_begin) *
            sizeof(std::uint16_t);
        if (!encoded(set_buffer(
                pass, region.packed, 0, 0)) ||
            !encoded(set_buffer(
                pass, region.scales, 0, 1)) ||
            !encoded(set_buffer(
                pass, region.biases, 0, 2)) ||
            !encoded(set_buffer(
                pass, resources.activations, 0, 3)) ||
            !encoded(set_buffer(
                pass, resources.output, output_offset, 4)) ||
            !encoded(set_bytes(
                pass, &reduction, sizeof(reduction), 5)) ||
            !encoded(set_bytes(
                pass, &columns, sizeof(columns), 6)) ||
            !encoded(set_bytes(
                pass, &rows, sizeof(rows), 7)) ||
            !encoded(set_bytes(
                pass, &output_stride,
                sizeof(output_stride), 8)) ||
            !encoded(dispatch_threadgroups(
                pass,
                {
                    .width =
                        region.columns / kN1TileColumns,
                    .height = spec.rows / kTileRows,
                    .depth = 1,
                },
                {
                    .width = kSimdgroupThreads,
                    .height = 4,
                    .depth = 1,
                }))) {
            return kExitEncode;
        }
    }
    return 0;
}

int encode_steel_gdn_bm128(
    MetalComputePass& pass, const Pipelines& pipelines,
    const CaseResources& resources) {
    const CaseSpec& spec = resources.spec;
    constexpr std::uint32_t kTileRows = 128U;
    if (spec.operation.kind != OperationKind::GdnInput ||
        spec.operation.region_count != 4U ||
        spec.operation.columns != kGeometry.gdn_projection_rows ||
        spec.operation.reduction != kGeometry.hidden ||
        spec.rows % kTileRows != 0U ||
        !encoded(set_compute_pipeline(
            pass, pipelines.steel_gdn_bm128))) {
        return kExitEncode;
    }
    const std::int32_t rows =
        static_cast<std::int32_t>(spec.rows);
    const std::int32_t reduction =
        static_cast<std::int32_t>(spec.operation.reduction);
    const std::int32_t output_stride =
        static_cast<std::int32_t>(spec.operation.columns);
    for (std::uint32_t index = 0; index < 4U; ++index) {
        const RegionResources& region = resources.regions[index];
        if (region.columns % kN1TileColumns != 0U) {
            return kExitEncode;
        }
        const std::int32_t columns =
            static_cast<std::int32_t>(region.columns);
        const std::uint64_t output_offset =
            (kGuardElements + region.column_begin) *
            sizeof(std::uint16_t);
        if (!encoded(set_buffer(
                pass, region.packed, 0, 0)) ||
            !encoded(set_buffer(
                pass, region.scales, 0, 1)) ||
            !encoded(set_buffer(
                pass, region.biases, 0, 2)) ||
            !encoded(set_buffer(
                pass, resources.activations, 0, 3)) ||
            !encoded(set_buffer(
                pass, resources.output, output_offset, 4)) ||
            !encoded(set_bytes(
                pass, &reduction, sizeof(reduction), 5)) ||
            !encoded(set_bytes(
                pass, &columns, sizeof(columns), 6)) ||
            !encoded(set_bytes(
                pass, &rows, sizeof(rows), 7)) ||
            !encoded(set_bytes(
                pass, &output_stride,
                sizeof(output_stride), 8)) ||
            !encoded(dispatch_threadgroups(
                pass,
                {
                    .width =
                        region.columns / kN1TileColumns,
                    .height = spec.rows / kTileRows,
                    .depth = 1,
                },
                {
                    .width = kSimdgroupThreads,
                    .height = 8,
                    .depth = 1,
                }))) {
            return kExitEncode;
        }
    }
    return 0;
}

int encode_steel_gdn_bm48(
    MetalComputePass& pass, const Pipelines& pipelines,
    const CaseResources& resources) {
    const CaseSpec& spec = resources.spec;
    constexpr std::uint32_t kBodyTileRows = 48U;
    constexpr std::uint32_t kTailTileRows = 32U;
    const std::uint32_t body_rows =
        (spec.rows / kBodyTileRows) * kBodyTileRows;
    const std::uint32_t tail_rows = spec.rows - body_rows;
    if (spec.operation.kind != OperationKind::GdnInput ||
        spec.operation.region_count != 4U ||
        spec.operation.columns != kGeometry.gdn_projection_rows ||
        spec.operation.reduction != kGeometry.hidden ||
        body_rows == 0U || body_rows % kBodyTileRows != 0U ||
        tail_rows != kTailTileRows) {
        return kExitEncode;
    }
    const std::int32_t body_rows_i32 =
        static_cast<std::int32_t>(body_rows);
    const std::int32_t tail_rows_i32 =
        static_cast<std::int32_t>(tail_rows);
    const std::int32_t reduction =
        static_cast<std::int32_t>(spec.operation.reduction);
    const std::int32_t output_stride =
        static_cast<std::int32_t>(spec.operation.columns);
    const std::uint64_t tail_input_offset =
        std::uint64_t{body_rows} * spec.operation.reduction *
        sizeof(std::uint16_t);
    for (std::uint32_t index = 0; index < 4U; ++index) {
        const RegionResources& region = resources.regions[index];
        if (region.columns % kN1TileColumns != 0U) {
            return kExitEncode;
        }
        const std::int32_t columns =
            static_cast<std::int32_t>(region.columns);
        const std::uint64_t body_output_offset =
            (kGuardElements + region.column_begin) *
            sizeof(std::uint16_t);
        if (!encoded(set_compute_pipeline(
                pass, pipelines.steel_gdn_bm48)) ||
            !encoded(set_buffer(
                pass, region.packed, 0, 0)) ||
            !encoded(set_buffer(
                pass, region.scales, 0, 1)) ||
            !encoded(set_buffer(
                pass, region.biases, 0, 2)) ||
            !encoded(set_buffer(
                pass, resources.activations, 0, 3)) ||
            !encoded(set_buffer(
                pass, resources.output,
                body_output_offset, 4)) ||
            !encoded(set_bytes(
                pass, &reduction, sizeof(reduction), 5)) ||
            !encoded(set_bytes(
                pass, &columns, sizeof(columns), 6)) ||
            !encoded(set_bytes(
                pass, &body_rows_i32,
                sizeof(body_rows_i32), 7)) ||
            !encoded(set_bytes(
                pass, &output_stride,
                sizeof(output_stride), 8)) ||
            !encoded(dispatch_threadgroups(
                pass,
                {
                    .width =
                        region.columns / kN1TileColumns,
                    .height = body_rows / kBodyTileRows,
                    .depth = 1,
                },
                {
                    .width = kSimdgroupThreads,
                    .height = 3,
                    .depth = 1,
                }))) {
            return kExitEncode;
        }
        const std::uint64_t tail_output_offset =
            (kGuardElements +
             std::uint64_t{body_rows} * spec.operation.columns +
             region.column_begin) *
            sizeof(std::uint16_t);
        if (!encoded(set_compute_pipeline(
                pass, pipelines.steel)) ||
            !encoded(set_buffer(
                pass, resources.activations,
                tail_input_offset, 3)) ||
            !encoded(set_buffer(
                pass, resources.output,
                tail_output_offset, 4)) ||
            !encoded(set_bytes(
                pass, &tail_rows_i32,
                sizeof(tail_rows_i32), 7)) ||
            !encoded(dispatch_threadgroups(
                pass,
                {
                    .width =
                        region.columns / kN1TileColumns,
                    .height = tail_rows / kTailTileRows,
                    .depth = 1,
                },
                {
                    .width = kSimdgroupThreads,
                    .height = 2,
                    .depth = 2,
                }))) {
            return kExitEncode;
        }
    }
    return 0;
}

int encode_steel_gdn_bm96(
    MetalComputePass& pass, const Pipelines& pipelines,
    const CaseResources& resources) {
    const CaseSpec& spec = resources.spec;
    constexpr std::uint32_t kBodyTileRows = 96U;
    constexpr std::uint32_t kTailTileRows = 32U;
    const std::uint32_t body_rows =
        (spec.rows / kBodyTileRows) * kBodyTileRows;
    const std::uint32_t tail_rows = spec.rows - body_rows;
    if (spec.operation.kind != OperationKind::GdnInput ||
        spec.operation.region_count != 4U ||
        spec.operation.columns != kGeometry.gdn_projection_rows ||
        spec.operation.reduction != kGeometry.hidden ||
        body_rows == 0U || body_rows % kBodyTileRows != 0U ||
        tail_rows != kTailTileRows) {
        return kExitEncode;
    }
    const std::int32_t body_rows_i32 =
        static_cast<std::int32_t>(body_rows);
    const std::int32_t tail_rows_i32 =
        static_cast<std::int32_t>(tail_rows);
    const std::int32_t reduction =
        static_cast<std::int32_t>(spec.operation.reduction);
    const std::int32_t output_stride =
        static_cast<std::int32_t>(spec.operation.columns);
    const std::uint64_t tail_input_offset =
        std::uint64_t{body_rows} * spec.operation.reduction *
        sizeof(std::uint16_t);
    for (std::uint32_t index = 0; index < 4U; ++index) {
        const RegionResources& region = resources.regions[index];
        if (region.columns % kN1TileColumns != 0U) {
            return kExitEncode;
        }
        const std::int32_t columns =
            static_cast<std::int32_t>(region.columns);
        const std::uint64_t body_output_offset =
            (kGuardElements + region.column_begin) *
            sizeof(std::uint16_t);
        if (!encoded(set_compute_pipeline(
                pass, pipelines.steel_gdn_bm96)) ||
            !encoded(set_buffer(
                pass, region.packed, 0, 0)) ||
            !encoded(set_buffer(
                pass, region.scales, 0, 1)) ||
            !encoded(set_buffer(
                pass, region.biases, 0, 2)) ||
            !encoded(set_buffer(
                pass, resources.activations, 0, 3)) ||
            !encoded(set_buffer(
                pass, resources.output,
                body_output_offset, 4)) ||
            !encoded(set_bytes(
                pass, &reduction, sizeof(reduction), 5)) ||
            !encoded(set_bytes(
                pass, &columns, sizeof(columns), 6)) ||
            !encoded(set_bytes(
                pass, &body_rows_i32,
                sizeof(body_rows_i32), 7)) ||
            !encoded(set_bytes(
                pass, &output_stride,
                sizeof(output_stride), 8)) ||
            !encoded(dispatch_threadgroups(
                pass,
                {
                    .width =
                        region.columns / kN1TileColumns,
                    .height = body_rows / kBodyTileRows,
                    .depth = 1,
                },
                {
                    .width = kSimdgroupThreads,
                    .height = 6,
                    .depth = 1,
                }))) {
            return kExitEncode;
        }
        const std::uint64_t tail_output_offset =
            (kGuardElements +
             std::uint64_t{body_rows} * spec.operation.columns +
             region.column_begin) *
            sizeof(std::uint16_t);
        if (!encoded(set_compute_pipeline(
                pass, pipelines.steel)) ||
            !encoded(set_buffer(
                pass, resources.activations,
                tail_input_offset, 3)) ||
            !encoded(set_buffer(
                pass, resources.output,
                tail_output_offset, 4)) ||
            !encoded(set_bytes(
                pass, &tail_rows_i32,
                sizeof(tail_rows_i32), 7)) ||
            !encoded(dispatch_threadgroups(
                pass,
                {
                    .width =
                        region.columns / kN1TileColumns,
                    .height = tail_rows / kTailTileRows,
                    .depth = 1,
                },
                {
                    .width = kSimdgroupThreads,
                    .height = 2,
                    .depth = 2,
                }))) {
            return kExitEncode;
        }
    }
    return 0;
}

int encode_steel_gdn_bn64(
    MetalComputePass& pass, const Pipelines& pipelines,
    const CaseResources& resources) {
    const CaseSpec& spec = resources.spec;
    constexpr std::uint32_t kWideTileColumns = 64U;
    if (spec.operation.kind != OperationKind::GdnInput ||
        spec.operation.region_count != 4U ||
        spec.operation.columns != kGeometry.gdn_projection_rows ||
        spec.operation.reduction != kGeometry.hidden ||
        spec.rows % kN1TileRows != 0U) {
        return kExitEncode;
    }
    const std::int32_t rows =
        static_cast<std::int32_t>(spec.rows);
    const std::int32_t reduction =
        static_cast<std::int32_t>(spec.operation.reduction);
    const std::int32_t output_stride =
        static_cast<std::int32_t>(spec.operation.columns);
    for (std::uint32_t index = 0; index < 4U; ++index) {
        const RegionResources& region = resources.regions[index];
        const bool wide = index < 2U;
        const std::uint32_t tile_columns =
            wide ? kWideTileColumns : kN1TileColumns;
        if (region.columns % tile_columns != 0U ||
            !encoded(set_compute_pipeline(
                pass,
                wide ? pipelines.steel_gdn_bn64
                     : pipelines.steel))) {
            return kExitEncode;
        }
        const std::int32_t columns =
            static_cast<std::int32_t>(region.columns);
        const std::uint64_t output_offset =
            (kGuardElements + region.column_begin) *
            sizeof(std::uint16_t);
        if (!encoded(set_buffer(
                pass, region.packed, 0, 0)) ||
            !encoded(set_buffer(
                pass, region.scales, 0, 1)) ||
            !encoded(set_buffer(
                pass, region.biases, 0, 2)) ||
            !encoded(set_buffer(
                pass, resources.activations, 0, 3)) ||
            !encoded(set_buffer(
                pass, resources.output, output_offset, 4)) ||
            !encoded(set_bytes(
                pass, &reduction, sizeof(reduction), 5)) ||
            !encoded(set_bytes(
                pass, &columns, sizeof(columns), 6)) ||
            !encoded(set_bytes(
                pass, &rows, sizeof(rows), 7)) ||
            !encoded(set_bytes(
                pass, &output_stride,
                sizeof(output_stride), 8)) ||
            !encoded(dispatch_threadgroups(
                pass,
                {
                    .width = region.columns / tile_columns,
                    .height = spec.rows / kN1TileRows,
                    .depth = 1,
                },
                {
                    .width = kSimdgroupThreads,
                    .height = 2,
                    .depth = wide ? 4U : 2U,
                }))) {
            return kExitEncode;
        }
    }
    return 0;
}

int encode_steel_gdn_bk64(
    MetalComputePass& pass, const Pipelines& pipelines,
    const CaseResources& resources) {
    const CaseSpec& spec = resources.spec;
    if (spec.operation.kind != OperationKind::GdnInput ||
        spec.operation.region_count != 4U ||
        spec.operation.columns != kGeometry.gdn_projection_rows ||
        spec.operation.reduction != kGeometry.hidden ||
        spec.rows % kN1TileRows != 0U ||
        !encoded(set_compute_pipeline(
            pass, pipelines.steel_gdn_bk64))) {
        return kExitEncode;
    }
    const std::int32_t rows =
        static_cast<std::int32_t>(spec.rows);
    const std::int32_t reduction =
        static_cast<std::int32_t>(spec.operation.reduction);
    const std::int32_t output_stride =
        static_cast<std::int32_t>(spec.operation.columns);
    for (std::uint32_t index = 0; index < 4U; ++index) {
        const RegionResources& region = resources.regions[index];
        if (region.columns % kN1TileColumns != 0U) {
            return kExitEncode;
        }
        const std::int32_t columns =
            static_cast<std::int32_t>(region.columns);
        const std::uint64_t output_offset =
            (kGuardElements + region.column_begin) *
            sizeof(std::uint16_t);
        if (!encoded(set_buffer(
                pass, region.packed, 0, 0)) ||
            !encoded(set_buffer(
                pass, region.scales, 0, 1)) ||
            !encoded(set_buffer(
                pass, region.biases, 0, 2)) ||
            !encoded(set_buffer(
                pass, resources.activations, 0, 3)) ||
            !encoded(set_buffer(
                pass, resources.output, output_offset, 4)) ||
            !encoded(set_bytes(
                pass, &reduction, sizeof(reduction), 5)) ||
            !encoded(set_bytes(
                pass, &columns, sizeof(columns), 6)) ||
            !encoded(set_bytes(
                pass, &rows, sizeof(rows), 7)) ||
            !encoded(set_bytes(
                pass, &output_stride,
                sizeof(output_stride), 8)) ||
            !encoded(dispatch_threadgroups(
                pass,
                {
                    .width =
                        region.columns / kN1TileColumns,
                    .height = spec.rows / kN1TileRows,
                    .depth = 1,
                },
                {
                    .width = kSimdgroupThreads,
                    .height = 2,
                    .depth = 2,
                }))) {
            return kExitEncode;
        }
    }
    return 0;
}

struct RunTiming {
    double gpu_milliseconds = 0.0;
    double wall_milliseconds = 0.0;
};

struct RunResult {
    int exit_code = 0;
    RunTiming timing;
};

double milliseconds_between(
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end) noexcept {
    return std::chrono::duration<double, std::milli>(
               end - begin)
        .count();
}

RunResult run_arm(
    const MetalCommandQueue& queue, const Pipelines& pipelines,
    CaseResources& resources, Arm arm, bool poison_body,
    bool require_timing, std::uint32_t& submissions) {
    prepare_output(resources, poison_body);
    const auto wall_begin = std::chrono::steady_clock::now();
    auto command = create_command_buffer(queue);
    if (!command || !command.command_buffer) {
        return {.exit_code = kExitCommandBuffer};
    }
    auto pass =
        begin_compute_pass(std::move(*command.command_buffer));
    if (!pass || !pass.compute_pass) {
        return {.exit_code = kExitComputePass};
    }
    int encode_result = kExitEncode;
    switch (arm) {
    case Arm::Exact:
        encode_result = encode_exact(
            *pass.compute_pass, pipelines, resources);
        break;
    case Arm::N1:
        encode_result = encode_n1(
            *pass.compute_pass, pipelines, resources);
        break;
    case Arm::Steel:
        encode_result = encode_steel(
            *pass.compute_pass, pipelines, resources);
        break;
    case Arm::SteelGdnFused2:
        encode_result = encode_steel_gdn_fused2(
            *pass.compute_pass, pipelines, resources);
        break;
    case Arm::SteelGdnBm64:
        encode_result = encode_steel_gdn_bm64(
            *pass.compute_pass, pipelines, resources);
        break;
    case Arm::SteelGdnBm64Wm2Wn2:
        encode_result = encode_steel_gdn_bm64_wm2_wn2(
            *pass.compute_pass, pipelines, resources);
        break;
    case Arm::SteelGdnBm64Bk64:
        encode_result = encode_steel_gdn_bm64_bk64(
            *pass.compute_pass, pipelines, resources);
        break;
    case Arm::SteelGdnBm48:
        encode_result = encode_steel_gdn_bm48(
            *pass.compute_pass, pipelines, resources);
        break;
    case Arm::SteelGdnBm96:
        encode_result = encode_steel_gdn_bm96(
            *pass.compute_pass, pipelines, resources);
        break;
    case Arm::SteelGdnBm128:
        encode_result = encode_steel_gdn_bm128(
            *pass.compute_pass, pipelines, resources);
        break;
    case Arm::SteelGdnBn64:
        encode_result = encode_steel_gdn_bn64(
            *pass.compute_pass, pipelines, resources);
        break;
    case Arm::SteelGdnBk64:
        encode_result = encode_steel_gdn_bk64(
            *pass.compute_pass, pipelines, resources);
        break;
    }
    if (encode_result != 0) {
        return {.exit_code = encode_result};
    }
    auto ended =
        end_compute_pass(std::move(*pass.compute_pass));
    if (!ended || !ended.command_buffer) {
        return {.exit_code = kExitEndPass};
    }
    auto pending = commit(std::move(*ended.command_buffer));
    if (!pending || !pending.pending_execution) {
        return {.exit_code = kExitCommit};
    }
    ++submissions;
    auto completed = wait_until_completed_timed(
        std::move(*pending.pending_execution));
    const auto wall_end = std::chrono::steady_clock::now();
    if (!completed) {
        return {.exit_code = kExitExecution};
    }
    if (!canaries_intact(resources)) {
        return {.exit_code = kExitCanary};
    }
    const double gpu_start =
        completed.timing.gpu_start_seconds;
    const double gpu_end =
        completed.timing.gpu_end_seconds;
    const bool valid_timing =
        std::isfinite(gpu_start) && std::isfinite(gpu_end) &&
        gpu_end > gpu_start;
    if (require_timing && !valid_timing) {
        return {.exit_code = kExitTiming};
    }
    return {
        .exit_code = 0,
        .timing =
            {
                .gpu_milliseconds =
                    valid_timing
                        ? (gpu_end - gpu_start) * 1000.0
                        : 0.0,
                .wall_milliseconds =
                    milliseconds_between(wall_begin, wall_end),
            },
    };
}

struct NumericReport {
    float maximum_absolute = 0.0F;
    float maximum_normalized = 0.0F;
    std::uint32_t maximum_bfloat_ulp = 0;
};

bool compare_family(
    const CaseResources& resources,
    NumericReport& report) noexcept {
    const std::span<const std::uint16_t> actual =
        output_body(resources);
    if (actual.size() != resources.exact_snapshot.size()) {
        return false;
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float exact =
            from_bfloat16(resources.exact_snapshot[index]);
        const float n1 = from_bfloat16(actual[index]);
        if (!std::isfinite(exact) || !std::isfinite(n1)) {
            return false;
        }
        const float absolute = std::fabs(exact - n1);
        const float normalized =
            absolute / std::fmax(1.0F, std::fabs(exact));
        report.maximum_absolute =
            std::fmax(report.maximum_absolute, absolute);
        report.maximum_normalized =
            std::fmax(report.maximum_normalized, normalized);
        report.maximum_bfloat_ulp = std::max(
            report.maximum_bfloat_ulp,
            bfloat_ulp_distance(
                resources.exact_snapshot[index],
                actual[index]));
    }
    return report.maximum_normalized <=
           kMaximumNormalizedError;
}

const RegionResources* find_region(
    const CaseResources& resources,
    std::uint32_t column) noexcept {
    for (std::uint32_t index = 0;
         index < resources.spec.operation.region_count; ++index) {
        const RegionResources& region = resources.regions[index];
        if (column >= region.column_begin &&
            column - region.column_begin < region.columns) {
            return &region;
        }
    }
    return nullptr;
}

bool staged_oracle_matches(
    const CaseResources& resources) noexcept {
    const CaseSpec& spec = resources.spec;
    const auto* activations = static_cast<const std::uint16_t*>(
        resources.activations.contents());
    const std::span<const std::uint16_t> actual =
        output_body(resources);
    for (std::uint32_t region_index = 0;
         region_index < spec.operation.region_count;
         ++region_index) {
        const RegionResources& region =
            resources.regions[region_index];
        for (std::uint32_t boundary = 0; boundary < 2;
             ++boundary) {
            const std::uint32_t row =
                (region_index * 17U + boundary *
                     (spec.rows - 1U)) %
                spec.rows;
            const std::uint32_t column =
                boundary == 0
                    ? region.column_begin
                    : region.column_begin + region.columns - 1U;
            const RegionResources* located =
                find_region(resources, column);
            if (located == nullptr) {
                return false;
            }
            const std::uint32_t local_column =
                column - located->column_begin;
            const auto* packed =
                static_cast<const std::uint32_t*>(
                    located->packed.contents());
            const auto* scales =
                static_cast<const std::uint16_t*>(
                    located->scales.contents());
            const auto* biases =
                static_cast<const std::uint16_t*>(
                    located->biases.contents());
            const std::uint32_t words_per_row =
                spec.operation.reduction / kQ4ValuesPerWord;
            const std::uint32_t groups_per_row =
                spec.operation.reduction / kGroupSize;
            float accumulator = 0.0F;
            for (std::uint32_t reduction = 0;
                 reduction < spec.operation.reduction; ++reduction) {
                const std::uint32_t word =
                    packed[
                        static_cast<std::size_t>(local_column) *
                            words_per_row +
                        reduction / kQ4ValuesPerWord];
                const std::uint32_t quantized =
                    (word >>
                     (4U *
                      (reduction % kQ4ValuesPerWord))) &
                    15U;
                const std::size_t parameter_index =
                    static_cast<std::size_t>(local_column) *
                        groups_per_row +
                    reduction / kGroupSize;
                const float dequantized = std::fma(
                    static_cast<float>(quantized),
                    from_bfloat16(scales[parameter_index]),
                    from_bfloat16(biases[parameter_index]));
                const float staged =
                    from_bfloat16(bfloat16(dequantized));
                const float activation = from_bfloat16(
                    activations[
                        static_cast<std::size_t>(row) *
                            spec.operation.reduction +
                        reduction]);
                accumulator =
                    std::fma(activation, staged, accumulator);
            }
            const std::uint16_t expected =
                bfloat16(accumulator);
            const std::uint16_t observed =
                actual[
                    static_cast<std::size_t>(row) *
                        spec.operation.columns +
                    column];
            const float expected_value =
                from_bfloat16(expected);
            const float observed_value =
                from_bfloat16(observed);
            const float normalized =
                std::fabs(expected_value - observed_value) /
                std::fmax(1.0F, std::fabs(expected_value));
            if (!std::isfinite(observed_value) ||
                normalized > kMaximumNormalizedError ||
                bfloat_ulp_distance(expected, observed) >
                    kMaximumBfloatUlp) {
                return false;
            }
        }
    }
    return true;
}

struct Samples {
    std::array<double, kMeasuredSamplesPerArm> gpu{};
    std::array<double, kMeasuredSamplesPerArm> wall{};
    std::size_t count = 0;
};

struct Distribution {
    double p10 = 0.0;
    double p50 = 0.0;
    double p90 = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

template <std::size_t Size>
Distribution summarize(std::array<double, Size> values) {
    static_assert(Size == kMeasuredSamplesPerArm);
    std::sort(values.begin(), values.end());
    return {
        .p10 = values[1],
        .p50 = values[8],
        .p90 = values[15],
        .minimum = values.front(),
        .maximum = values.back(),
    };
}

bool control_ranges_overlap(const Samples& exact) noexcept {
    if (exact.count != kMeasuredSamplesPerArm) {
        return false;
    }
    const auto early_begin = exact.gpu.begin();
    const auto early_end = early_begin + 4;
    const auto late_begin = exact.gpu.end() - 4;
    const auto late_end = exact.gpu.end();
    const auto [early_min, early_max] =
        std::minmax_element(early_begin, early_end);
    const auto [late_min, late_max] =
        std::minmax_element(late_begin, late_end);
    return *early_max >= *late_min && *late_max >= *early_min;
}

struct CaseReport {
    CaseSpec spec;
    ResourceBudget budget;
    NumericReport numeric;
    std::uint64_t exact_hash = 0;
    std::uint64_t n1_hash = 0;
    Distribution exact_gpu;
    Distribution n1_gpu;
    Distribution exact_wall;
    Distribution n1_wall;
};

int execute_case(
    const MetalDevice& device, const MetalCommandQueue& queue,
    const Pipelines& pipelines, const CaseSpec& spec,
    CaseReport& report, std::uint32_t& total_submissions) {
    CaseResources resources;
    const int resource_result =
        create_case_resources(device, spec, resources);
    if (resource_result != 0) {
        return resource_result;
    }
    report.spec = spec;
    if (!make_resource_budget(spec, report.budget)) {
        return kExitExtent;
    }
    std::uint32_t case_submissions = 0;

    RunResult exact = run_arm(
        queue, pipelines, resources, Arm::Exact, true, false,
        case_submissions);
    if (exact.exit_code != 0) {
        return exact.exit_code;
    }
    if (!body_is_finite(resources)) {
        return kExitNumericalFamily;
    }
    const std::span<const std::uint16_t> exact_body =
        output_body(resources);
    std::memcpy(
        resources.exact_snapshot.data(), exact_body.data(),
        exact_body.size_bytes());
    report.exact_hash = hash_bfloat(exact_body);

    RunResult n1 = run_arm(
        queue, pipelines, resources, Arm::N1, true, false,
        case_submissions);
    if (n1.exit_code != 0) {
        return n1.exit_code;
    }
    if (!body_is_finite(resources) ||
        !staged_oracle_matches(resources)) {
        return kExitStagedOracle;
    }
    report.n1_hash = hash_bfloat(output_body(resources));
    if (!compare_family(resources, report.numeric)) {
        return kExitNumericalFamily;
    }

    bool exact_determinism_checked = false;
    bool n1_determinism_checked = false;
    for (const char scheduled : kWarmupSchedule) {
        const Arm arm =
            scheduled == 'A' ? Arm::Exact : Arm::N1;
        RunResult warmup = run_arm(
            queue, pipelines, resources, arm, false, true,
            case_submissions);
        if (warmup.exit_code != 0) {
            return warmup.exit_code;
        }
        if (arm == Arm::Exact &&
            !exact_determinism_checked) {
            exact_determinism_checked = true;
            if (hash_bfloat(output_body(resources)) !=
                report.exact_hash) {
                return kExitExactNondeterminism;
            }
        } else if (
            arm == Arm::N1 && !n1_determinism_checked) {
            n1_determinism_checked = true;
            if (hash_bfloat(output_body(resources)) !=
                report.n1_hash) {
                return kExitN1Nondeterminism;
            }
        }
    }
    if (!exact_determinism_checked ||
        !n1_determinism_checked) {
        return kExitSampleAccounting;
    }

    Samples exact_samples;
    Samples n1_samples;
    for (const char scheduled : kMeasuredSchedule) {
        const Arm arm =
            scheduled == 'A' ? Arm::Exact : Arm::N1;
        RunResult measured = run_arm(
            queue, pipelines, resources, arm, false, true,
            case_submissions);
        if (measured.exit_code != 0) {
            return measured.exit_code;
        }
        Samples& samples =
            arm == Arm::Exact ? exact_samples : n1_samples;
        if (samples.count >= samples.gpu.size()) {
            return kExitSampleAccounting;
        }
        samples.gpu[samples.count] =
            measured.timing.gpu_milliseconds;
        samples.wall[samples.count] =
            measured.timing.wall_milliseconds;
        ++samples.count;
    }
    if (exact_samples.count != kMeasuredSamplesPerArm ||
        n1_samples.count != kMeasuredSamplesPerArm ||
        case_submissions != kCommandBuffersPerCase) {
        return kExitSampleAccounting;
    }
    if (!control_ranges_overlap(exact_samples)) {
        return kExitControlDrift;
    }

    report.exact_gpu = summarize(exact_samples.gpu);
    report.n1_gpu = summarize(n1_samples.gpu);
    report.exact_wall = summarize(exact_samples.wall);
    report.n1_wall = summarize(n1_samples.wall);
    total_submissions += case_submissions;
    std::cout
        << "tatara_native_dense_qgemm_perf_progress"
        << " operation=" << spec.operation.name
        << " rows=" << spec.rows
        << " submissions=" << case_submissions << '\n'
        << std::flush;
    return 0;
}

int execute_steel_case(
    const MetalDevice& device, const MetalCommandQueue& queue,
    const Pipelines& pipelines, const CaseSpec& spec,
    CaseReport& report, std::uint32_t& total_submissions) {
    if (spec.rows != kGeometryPolicy.maximum_block_rows) {
        return kExitCpuContract;
    }
    CaseResources resources;
    const int resource_result =
        create_case_resources(device, spec, resources);
    if (resource_result != 0) {
        return resource_result;
    }
    report.spec = spec;
    if (!make_resource_budget(spec, report.budget)) {
        return kExitExtent;
    }
    std::uint32_t case_submissions = 0;

    RunResult n1 = run_arm(
        queue, pipelines, resources, Arm::N1, true, false,
        case_submissions);
    if (n1.exit_code != 0 ||
        !body_is_finite(resources) ||
        !staged_oracle_matches(resources)) {
        return n1.exit_code != 0
                   ? n1.exit_code
                   : kExitStagedOracle;
    }
    const std::span<const std::uint16_t> n1_body =
        output_body(resources);
    std::memcpy(
        resources.exact_snapshot.data(), n1_body.data(),
        n1_body.size_bytes());
    report.exact_hash = hash_bfloat(n1_body);

    RunResult steel = run_arm(
        queue, pipelines, resources, Arm::Steel, true, false,
        case_submissions);
    if (steel.exit_code != 0) {
        return steel.exit_code;
    }
    if (!body_is_finite(resources) ||
        !compare_family(resources, report.numeric)) {
        return kExitNumericalFamily;
    }
    report.n1_hash = hash_bfloat(output_body(resources));

    bool n1_determinism_checked = false;
    bool steel_determinism_checked = false;
    for (const char scheduled : kWarmupSchedule) {
        const Arm arm =
            scheduled == 'A' ? Arm::N1 : Arm::Steel;
        RunResult warmup = run_arm(
            queue, pipelines, resources, arm, false, true,
            case_submissions);
        if (warmup.exit_code != 0) {
            return warmup.exit_code;
        }
        const std::uint64_t actual_hash =
            hash_bfloat(output_body(resources));
        if (arm == Arm::N1 && !n1_determinism_checked) {
            n1_determinism_checked = true;
            if (actual_hash != report.exact_hash) {
                return kExitN1Nondeterminism;
            }
        } else if (
            arm == Arm::Steel && !steel_determinism_checked) {
            steel_determinism_checked = true;
            if (actual_hash != report.n1_hash) {
                return kExitExactNondeterminism;
            }
        }
    }
    if (!n1_determinism_checked ||
        !steel_determinism_checked) {
        return kExitSampleAccounting;
    }

    Samples n1_samples;
    Samples steel_samples;
    for (const char scheduled : kMeasuredSchedule) {
        const Arm arm =
            scheduled == 'A' ? Arm::N1 : Arm::Steel;
        RunResult measured = run_arm(
            queue, pipelines, resources, arm, false, true,
            case_submissions);
        if (measured.exit_code != 0) {
            return measured.exit_code;
        }
        Samples& samples =
            arm == Arm::N1 ? n1_samples : steel_samples;
        if (samples.count >= samples.gpu.size()) {
            return kExitSampleAccounting;
        }
        samples.gpu[samples.count] =
            measured.timing.gpu_milliseconds;
        samples.wall[samples.count] =
            measured.timing.wall_milliseconds;
        ++samples.count;
    }
    if (n1_samples.count != kMeasuredSamplesPerArm ||
        steel_samples.count != kMeasuredSamplesPerArm ||
        case_submissions != kCommandBuffersPerCase) {
        return kExitSampleAccounting;
    }
    if (!control_ranges_overlap(n1_samples)) {
        return kExitControlDrift;
    }

    report.exact_gpu = summarize(n1_samples.gpu);
    report.n1_gpu = summarize(steel_samples.gpu);
    report.exact_wall = summarize(n1_samples.wall);
    report.n1_wall = summarize(steel_samples.wall);
    total_submissions += case_submissions;
    std::cout
        << "tatara_mlx_steel_dense_perf_progress"
        << " operation=" << spec.operation.name
        << " rows=" << spec.rows
        << " submissions=" << case_submissions << '\n'
        << std::flush;
    return 0;
}

int execute_gdn_fused2_component(
    const MetalDevice& device, const MetalCommandQueue& queue,
    const Pipelines& pipelines, const CaseSpec& spec,
    Arm treatment) {
    if (spec.operation.kind != OperationKind::GdnInput ||
        spec.rows != kGeometryPolicy.maximum_block_rows ||
        !is_gdn_treatment(treatment)) {
        return kExitCpuContract;
    }
    const int mismatch_exit = gdn_mismatch_exit(treatment);
    CaseResources resources;
    const int resource_result =
        create_case_resources(device, spec, resources);
    if (resource_result != 0) {
        return resource_result;
    }
    const std::uint64_t original_inputs =
        input_identity(resources);
    std::uint32_t submissions = 0;
    RunResult control = run_arm(
        queue, pipelines, resources, Arm::Steel, true, false,
        submissions);
    if (control.exit_code != 0 ||
        !body_is_finite(resources) ||
        input_identity(resources) != original_inputs) {
        return control.exit_code != 0
                   ? control.exit_code
                   : mismatch_exit;
    }
    const auto control_body = output_body(resources);
    std::memcpy(
        resources.exact_snapshot.data(), control_body.data(),
        control_body.size_bytes());
    const std::uint64_t control_hash =
        hash_bfloat(control_body);

    RunResult treatment_run = run_arm(
        queue, pipelines, resources, treatment,
        true, false, submissions);
    const auto treatment_body = output_body(resources);
    const auto first_mismatch = std::mismatch(
        treatment_body.begin(), treatment_body.end(),
        resources.exact_snapshot.begin());
    const bool byte_exact =
        first_mismatch.first == treatment_body.end();
    if (treatment_run.exit_code != 0 ||
        !body_is_finite(resources) ||
        input_identity(resources) != original_inputs ||
        treatment_body.size() !=
            resources.exact_snapshot.size() ||
        !byte_exact ||
        submissions != 2U) {
        if (!byte_exact) {
            const std::size_t offset = static_cast<std::size_t>(
                first_mismatch.first - treatment_body.begin());
            std::cerr
                << "tatara_native_dense_qgemm_perf_mismatch"
                << " element=" << offset
                << " row=" << offset / spec.operation.columns
                << " column=" << offset % spec.operation.columns
                << " expected=" << *first_mismatch.second
                << " actual=" << *first_mismatch.first << '\n';
        }
        return treatment_run.exit_code != 0
                   ? treatment_run.exit_code
                   : mismatch_exit;
    }
    std::cout
        << gdn_component_name(treatment)
        << ' ' << gdn_treatment_dispatch_label(treatment)
        << " rows=" << spec.rows
        << " columns=" << spec.operation.columns
        << " reduction=" << spec.operation.reduction
        << " control_dispatches=4"
        << " control_hash=" << control_hash
        << " treatment_hash="
        << hash_bfloat(treatment_body)
        << " byte_exact=yes"
        << " command_buffers=" << submissions << '\n';
    return 0;
}

int execute_gdn_fused2_case(
    const MetalDevice& device, const MetalCommandQueue& queue,
    const Pipelines& pipelines, const CaseSpec& spec,
    CaseReport& report, std::uint32_t& total_submissions,
    Arm treatment, bool reverse_order = false) {
    if (spec.operation.kind != OperationKind::GdnInput ||
        spec.rows != kGeometryPolicy.maximum_block_rows ||
        !is_gdn_treatment(treatment)) {
        return kExitCpuContract;
    }
    const int mismatch_exit = gdn_mismatch_exit(treatment);
    CaseResources resources;
    const int resource_result =
        create_case_resources(device, spec, resources);
    if (resource_result != 0) {
        return resource_result;
    }
    report.spec = spec;
    if (!make_resource_budget(spec, report.budget)) {
        return kExitExtent;
    }
    const std::uint64_t original_inputs =
        input_identity(resources);
    std::uint32_t case_submissions = 0;

    RunResult control = run_arm(
        queue, pipelines, resources, Arm::Steel, true, false,
        case_submissions);
    if (control.exit_code != 0 ||
        !body_is_finite(resources) ||
        input_identity(resources) != original_inputs) {
        return control.exit_code != 0
                   ? control.exit_code
                   : mismatch_exit;
    }
    const auto control_body = output_body(resources);
    std::memcpy(
        resources.exact_snapshot.data(), control_body.data(),
        control_body.size_bytes());
    report.exact_hash = hash_bfloat(control_body);

    RunResult treatment_run = run_arm(
        queue, pipelines, resources, treatment,
        true, false, case_submissions);
    const auto treatment_body = output_body(resources);
    if (treatment_run.exit_code != 0 ||
        !body_is_finite(resources) ||
        input_identity(resources) != original_inputs ||
        treatment_body.size() !=
            resources.exact_snapshot.size() ||
        !std::equal(
            treatment_body.begin(), treatment_body.end(),
            resources.exact_snapshot.begin())) {
        return treatment_run.exit_code != 0
                   ? treatment_run.exit_code
                   : mismatch_exit;
    }
    report.n1_hash = hash_bfloat(treatment_body);

    bool control_determinism_checked = false;
    bool treatment_determinism_checked = false;
    for (const char scheduled : kWarmupSchedule) {
        const bool scheduled_control =
            (scheduled == 'A') != reverse_order;
        const Arm arm =
            scheduled_control ? Arm::Steel : treatment;
        RunResult warmup = run_arm(
            queue, pipelines, resources, arm, false, true,
            case_submissions);
        if (warmup.exit_code != 0) {
            return warmup.exit_code;
        }
        const std::uint64_t actual_hash =
            hash_bfloat(output_body(resources));
        if (arm == Arm::Steel &&
            !control_determinism_checked) {
            control_determinism_checked = true;
            if (actual_hash != report.exact_hash) {
                return mismatch_exit;
            }
        } else if (
            arm == treatment &&
            !treatment_determinism_checked) {
            treatment_determinism_checked = true;
            if (actual_hash != report.n1_hash) {
                return mismatch_exit;
            }
        }
    }
    if (!control_determinism_checked ||
        !treatment_determinism_checked) {
        return kExitSampleAccounting;
    }

    Samples control_samples;
    Samples treatment_samples;
    for (const char scheduled : kMeasuredSchedule) {
        const bool scheduled_control =
            (scheduled == 'A') != reverse_order;
        const Arm arm =
            scheduled_control ? Arm::Steel : treatment;
        RunResult measured = run_arm(
            queue, pipelines, resources, arm, false, true,
            case_submissions);
        if (measured.exit_code != 0) {
            return measured.exit_code;
        }
        Samples& samples =
            arm == Arm::Steel ? control_samples
                              : treatment_samples;
        if (samples.count >= samples.gpu.size()) {
            return kExitSampleAccounting;
        }
        samples.gpu[samples.count] =
            measured.timing.gpu_milliseconds;
        samples.wall[samples.count] =
            measured.timing.wall_milliseconds;
        ++samples.count;
    }
    if (control_samples.count != kMeasuredSamplesPerArm ||
        treatment_samples.count != kMeasuredSamplesPerArm ||
        case_submissions != kCommandBuffersPerCase ||
        input_identity(resources) != original_inputs) {
        return kExitSampleAccounting;
    }
    if (!control_ranges_overlap(control_samples)) {
        return kExitControlDrift;
    }
    report.exact_gpu = summarize(control_samples.gpu);
    report.n1_gpu = summarize(treatment_samples.gpu);
    report.exact_wall = summarize(control_samples.wall);
    report.n1_wall = summarize(treatment_samples.wall);
    total_submissions += case_submissions;
    return 0;
}

void print_distribution(
    std::string_view prefix,
    const Distribution& distribution) {
    std::cout << ' ' << prefix << "_p10=" << distribution.p10
              << ' ' << prefix << "_p50=" << distribution.p50
              << ' ' << prefix << "_p90=" << distribution.p90
              << ' ' << prefix << "_min=" << distribution.minimum
              << ' ' << prefix << "_max=" << distribution.maximum;
}

void print_report(const CaseReport& report) {
    std::cout << "tatara_native_dense_qgemm_perf_case"
              << " operation=" << report.spec.operation.name
              << " rows=" << report.spec.rows
              << " columns=" << report.spec.operation.columns
              << " reduction=" << report.spec.operation.reduction
              << " regions=" << report.spec.operation.region_count
              << " gpu_bytes=" << report.budget.gpu_bytes
              << " with_host_bytes="
              << report.budget.with_host_snapshot_bytes
              << " max_abs=" << report.numeric.maximum_absolute
              << " max_norm=" << report.numeric.maximum_normalized
              << " max_bfloat_ulp="
              << report.numeric.maximum_bfloat_ulp
              << " exact_hash=" << report.exact_hash
              << " n1_hash=" << report.n1_hash;
    print_distribution("exact_gpu_ms", report.exact_gpu);
    print_distribution("n1_gpu_ms", report.n1_gpu);
    print_distribution("exact_wall_ms", report.exact_wall);
    print_distribution("n1_wall_ms", report.n1_wall);
    std::cout << " gpu_p50_speedup="
              << report.exact_gpu.p50 / report.n1_gpu.p50
              << '\n';
}

void print_steel_report(const CaseReport& report) {
    std::cout << "tatara_mlx_steel_dense_perf_case"
              << " operation=" << report.spec.operation.name
              << " rows=" << report.spec.rows
              << " columns=" << report.spec.operation.columns
              << " reduction=" << report.spec.operation.reduction
              << " regions=" << report.spec.operation.region_count
              << " max_abs=" << report.numeric.maximum_absolute
              << " max_norm=" << report.numeric.maximum_normalized
              << " max_bfloat_ulp="
              << report.numeric.maximum_bfloat_ulp
              << " n1_hash=" << report.exact_hash
              << " steel_hash=" << report.n1_hash;
    print_distribution("n1_gpu_ms", report.exact_gpu);
    print_distribution("steel_gpu_ms", report.n1_gpu);
    print_distribution("n1_wall_ms", report.exact_wall);
    print_distribution("steel_wall_ms", report.n1_wall);
    std::cout << " steel_gpu_p50_speedup="
              << report.exact_gpu.p50 / report.n1_gpu.p50
              << '\n';
}

void print_gdn_treatment_report(
    const CaseReport& report, Arm treatment) {
    std::cout
        << gdn_perf_name(treatment)
        << ' ' << gdn_treatment_dispatch_label(treatment)
        << " rows=" << report.spec.rows
        << " columns=" << report.spec.operation.columns
        << " reduction=" << report.spec.operation.reduction
        << " control_dispatches=4"
        << " control_hash=" << report.exact_hash
        << " treatment_hash=" << report.n1_hash;
    print_distribution(
        "control_gpu_ms", report.exact_gpu);
    print_distribution(
        "treatment_gpu_ms", report.n1_gpu);
    print_distribution(
        "control_wall_ms", report.exact_wall);
    print_distribution(
        "treatment_wall_ms", report.n1_wall);
    std::cout << " gpu_p50_speedup="
              << report.exact_gpu.p50 / report.n1_gpu.p50
              << " samples_per_arm=" << kMeasuredSamplesPerArm
              << " command_buffers=" << kCommandBuffersPerCase
              << '\n';
}

int fail(int code, std::string_view category) {
    std::cerr << "tatara_native_dense_qgemm_perf_error"
              << " category=" << category
              << " code=" << code << '\n';
    return code;
}

std::string_view category_for_exit(int code) noexcept {
    switch (code) {
    case kExitGeometry:
        return "geometry";
    case kExitExtent:
        return "extent";
    case kExitCpuContract:
        return "cpu-contract";
    case kExitDevice:
        return "device";
    case kExitQueue:
        return "queue";
    case kExitLibrary:
        return "library";
    case kExitFunction:
        return "function";
    case kExitExactPipeline:
        return "exact-pipeline";
    case kExitN1Pipeline:
        return "n1-pipeline";
    case kExitAllocation:
        return "allocation";
    case kExitInitialization:
        return "initialization";
    case kExitCommandBuffer:
        return "command-buffer";
    case kExitComputePass:
        return "compute-pass";
    case kExitEncode:
        return "encode";
    case kExitEndPass:
        return "end-pass";
    case kExitCommit:
        return "commit";
    case kExitExecution:
        return "execution";
    case kExitTiming:
        return "timing";
    case kExitCanary:
        return "canary";
    case kExitExactNondeterminism:
        return "exact-nondeterminism";
    case kExitN1Nondeterminism:
        return "n1-nondeterminism";
    case kExitStagedOracle:
        return "staged-oracle";
    case kExitNumericalFamily:
        return "numerical-family";
    case kExitSampleAccounting:
        return "sample-accounting";
    case kExitControlDrift:
        return "control-drift";
    case kExitSteelPipeline:
        return "steel-pipeline";
    case kExitSteelGdnFused2Pipeline:
        return "steel-gdn-fused2-pipeline";
    case kExitSteelGdnFused2Mismatch:
        return "steel-gdn-fused2-mismatch";
    case kExitSteelGdnBm64Pipeline:
        return "steel-gdn-bm64-pipeline";
    case kExitSteelGdnBm64Mismatch:
        return "steel-gdn-bm64-mismatch";
    case kExitSteelGdnBm64Bk64Pipeline:
        return "steel-gdn-bm64-bk64-pipeline";
    case kExitSteelGdnBm64Bk64Mismatch:
        return "steel-gdn-bm64-bk64-mismatch";
    case kExitSteelGdnBm48Pipeline:
        return "steel-gdn-bm48-pipeline";
    case kExitSteelGdnBm48Mismatch:
        return "steel-gdn-bm48-mismatch";
    case kExitSteelGdnBm96Pipeline:
        return "steel-gdn-bm96-pipeline";
    case kExitSteelGdnBm96Mismatch:
        return "steel-gdn-bm96-mismatch";
    case kExitSteelGdnBm128Pipeline:
        return "steel-gdn-bm128-pipeline";
    case kExitSteelGdnBm128Mismatch:
        return "steel-gdn-bm128-mismatch";
    case kExitSteelGdnBm64Wm2Wn2Pipeline:
        return "steel-gdn-bm64-wm2-wn2-pipeline";
    case kExitSteelGdnBm64Wm2Wn2Mismatch:
        return "steel-gdn-bm64-wm2-wn2-mismatch";
    case kExitSteelGdnBn64Pipeline:
        return "steel-gdn-bn64-pipeline";
    case kExitSteelGdnBn64Mismatch:
        return "steel-gdn-bn64-mismatch";
    case kExitSteelGdnBk64Pipeline:
        return "steel-gdn-bk64-pipeline";
    case kExitSteelGdnBk64Mismatch:
        return "steel-gdn-bk64-mismatch";
    default:
        return "unknown";
    }
}

} // namespace

int main(int argument_count, char** arguments) {
    if (argument_count != 2) {
        std::cerr
            << "usage: tatara_native_dense_qgemm_perf_probe"
            << " --cpu-only|--compile-only|--gpu|--gpu-steel|"
               "--gpu-steel-gdn-fused2-component|"
               "--benchmark-steel-gdn-fused2|"
               "--gpu-steel-gdn-bm64-component|"
               "--benchmark-steel-gdn-bm64|"
               "--gpu-steel-gdn-bm64-wm2-wn2-component|"
               "--benchmark-steel-gdn-bm64-wm2-wn2|"
               "--benchmark-steel-gdn-bm64-wm2-wn2-reverse|"
               "--gpu-steel-gdn-bm64-bk64-component|"
               "--benchmark-steel-gdn-bm64-bk64|"
               "--gpu-steel-gdn-bm48-component|"
               "--benchmark-steel-gdn-bm48|"
               "--gpu-steel-gdn-bm96-component|"
               "--benchmark-steel-gdn-bm96|"
               "--gpu-steel-gdn-bm128-component|"
               "--benchmark-steel-gdn-bm128|"
               "--gpu-steel-gdn-bn64-component|"
               "--benchmark-steel-gdn-bn64|"
               "--gpu-steel-gdn-bk64-component|"
               "--benchmark-steel-gdn-bk64\n";
        return kExitUsage;
    }
    const std::string_view mode(arguments[1]);
    if (mode != "--cpu-only" && mode != "--compile-only" &&
        mode != "--gpu" && mode != "--gpu-steel" &&
        mode != "--gpu-steel-gdn-fused2-component" &&
        mode != "--benchmark-steel-gdn-fused2" &&
        mode != "--gpu-steel-gdn-bm64-component" &&
        mode != "--benchmark-steel-gdn-bm64" &&
        mode != "--gpu-steel-gdn-bm64-wm2-wn2-component" &&
        mode != "--benchmark-steel-gdn-bm64-wm2-wn2" &&
        mode != "--benchmark-steel-gdn-bm64-wm2-wn2-reverse" &&
        mode != "--gpu-steel-gdn-bm64-bk64-component" &&
        mode != "--benchmark-steel-gdn-bm64-bk64" &&
        mode != "--gpu-steel-gdn-bm48-component" &&
        mode != "--benchmark-steel-gdn-bm48" &&
        mode != "--gpu-steel-gdn-bm96-component" &&
        mode != "--benchmark-steel-gdn-bm96" &&
        mode != "--gpu-steel-gdn-bm128-component" &&
        mode != "--benchmark-steel-gdn-bm128" &&
        mode != "--gpu-steel-gdn-bn64-component" &&
        mode != "--benchmark-steel-gdn-bn64" &&
        mode != "--gpu-steel-gdn-bk64-component" &&
        mode != "--benchmark-steel-gdn-bk64") {
        std::cerr
            << "usage: tatara_native_dense_qgemm_perf_probe"
            << " --cpu-only|--compile-only|--gpu|--gpu-steel|"
               "--gpu-steel-gdn-fused2-component|"
               "--benchmark-steel-gdn-fused2|"
               "--gpu-steel-gdn-bm64-component|"
               "--benchmark-steel-gdn-bm64|"
               "--gpu-steel-gdn-bm64-wm2-wn2-component|"
               "--benchmark-steel-gdn-bm64-wm2-wn2|"
               "--benchmark-steel-gdn-bm64-wm2-wn2-reverse|"
               "--gpu-steel-gdn-bm64-bk64-component|"
               "--benchmark-steel-gdn-bm64-bk64|"
               "--gpu-steel-gdn-bm48-component|"
               "--benchmark-steel-gdn-bm48|"
               "--gpu-steel-gdn-bm96-component|"
               "--benchmark-steel-gdn-bm96|"
               "--gpu-steel-gdn-bm128-component|"
               "--benchmark-steel-gdn-bm128|"
               "--gpu-steel-gdn-bn64-component|"
               "--benchmark-steel-gdn-bn64|"
               "--gpu-steel-gdn-bk64-component|"
               "--benchmark-steel-gdn-bk64\n";
        return kExitUsage;
    }

    if (!kGeometryResult) {
        return fail(kExitGeometry, "geometry");
    }
    std::uint64_t maximum_with_host_bytes = 0;
    if (!validate_cpu_contract(maximum_with_host_bytes)) {
        return fail(kExitCpuContract, "cpu-contract");
    }
    if (mode == "--cpu-only") {
        std::cout
            << "native dense QGEMM perf fixture: PASS_CPU\n"
            << "  cases: " << kCases.size() << '\n'
            << "  command buffers per GPU case: "
            << kCommandBuffersPerCase << '\n'
            << "  bounded GPU command buffers: "
            << kTotalCommandBuffers << '\n'
            << "  measured samples per arm: "
            << kMeasuredSamplesPerArm << '\n'
            << "  maximum bytes with host snapshot: "
            << maximum_with_host_bytes << '\n'
            << "  command buffers submitted: 0\n"
            << "  evidence class: component-only"
            << " synthetic-resident projection evidence;"
            << " not model, engine, serving, TTFT or"
            << " tokens-per-second evidence\n";
        return 0;
    }

    auto device = create_system_device();
    if (!device || !device.device) {
        return fail(kExitDevice, "device");
    }
    auto queue = create_command_queue(*device.device);
    if (!queue || !queue.command_queue) {
        return fail(kExitQueue, "queue");
    }
    auto library = create_library_with_source(
        *device.device,
        tatara::backend::metal::generated::
            kernel_library_source());
    if (!library || !library.library) {
        std::cerr << library.failure_description << '\n';
        return fail(kExitLibrary, "library");
    }
    Pipelines pipelines;
    if (const int pipeline_result = create_pipelines(
            *device.device, *library.library, pipelines);
        pipeline_result != 0) {
        return fail(
            pipeline_result,
            category_for_exit(pipeline_result));
    }
    if (mode == "--compile-only") {
        constexpr std::size_t kPipelineCount =
            tatara::backend::metal::generated::
                kKernelLibraryMlxSteelEnabled
                ? 14U
                : 4U;
        std::cout
            << "native dense QGEMM perf fixture:"
            << " PASS_COMPILE_ONLY\n"
            << "  device: " << device.device->name() << '\n'
            << "  pipelines: " << kPipelineCount << '\n'
            << "  command buffers submitted: 0\n";
        return 0;
    }

    if (mode == "--gpu-steel-gdn-fused2-component") {
        if (!tatara::backend::metal::generated::
                kKernelLibraryMlxSteelEnabled) {
            return fail(
                kExitSteelGdnFused2Pipeline,
                "steel-not-enabled");
        }
        const CaseSpec spec{
            .operation = kOperations[0],
            .rows = kGeometryPolicy.maximum_block_rows,
        };
        const int result = execute_gdn_fused2_component(
            *device.device, *queue.command_queue, pipelines, spec,
            Arm::SteelGdnFused2);
        if (result != 0) {
            return fail(result, category_for_exit(result));
        }
        std::cout
            << "native MLX Steel GDN fused-pair component:"
            << " PASS_GPU\n"
            << "  device: " << device.device->name() << '\n'
            << "  command buffers submitted: 2\n"
            << "  timing samples: 0\n";
        return 0;
    }

    if (mode == "--benchmark-steel-gdn-fused2") {
        if (!tatara::backend::metal::generated::
                kKernelLibraryMlxSteelEnabled) {
            return fail(
                kExitSteelGdnFused2Pipeline,
                "steel-not-enabled");
        }
        const CaseSpec spec{
            .operation = kOperations[0],
            .rows = kGeometryPolicy.maximum_block_rows,
        };
        CaseReport report{};
        std::uint32_t submissions = 0;
        const int result = execute_gdn_fused2_case(
            *device.device, *queue.command_queue, pipelines, spec,
            report, submissions, Arm::SteelGdnFused2);
        if (result != 0) {
            return fail(result, category_for_exit(result));
        }
        if (submissions != kCommandBuffersPerCase) {
            return fail(
                kExitSampleAccounting, "sample-accounting");
        }
        std::cout << std::fixed << std::setprecision(6);
        print_gdn_treatment_report(
            report, Arm::SteelGdnFused2);
        std::cout
            << "native MLX Steel GDN fused-pair benchmark:"
            << " PASS_GPU\n"
            << "  device: " << device.device->name() << '\n'
            << "  command buffers submitted: "
            << submissions << '\n'
            << "  evidence class: component-only"
               " synthetic-resident same-binary Steel control/"
               "fused-pair treatment; not engine or"
               " tokens-per-second evidence\n";
        return 0;
    }

    if (mode == "--gpu-steel-gdn-bm64-component") {
        if (!tatara::backend::metal::generated::
                kKernelLibraryMlxSteelEnabled) {
            return fail(
                kExitSteelGdnBm64Pipeline,
                "steel-not-enabled");
        }
        const CaseSpec spec{
            .operation = kOperations[0],
            .rows = kGeometryPolicy.maximum_block_rows,
        };
        const int result = execute_gdn_fused2_component(
            *device.device, *queue.command_queue, pipelines, spec,
            Arm::SteelGdnBm64);
        if (result != 0) {
            return fail(result, category_for_exit(result));
        }
        std::cout
            << "native MLX Steel GDN BM64 component:"
            << " PASS_GPU\n"
            << "  device: " << device.device->name() << '\n'
            << "  command buffers submitted: 2\n"
            << "  timing samples: 0\n";
        return 0;
    }

    if (mode == "--benchmark-steel-gdn-bm64") {
        if (!tatara::backend::metal::generated::
                kKernelLibraryMlxSteelEnabled) {
            return fail(
                kExitSteelGdnBm64Pipeline,
                "steel-not-enabled");
        }
        const CaseSpec spec{
            .operation = kOperations[0],
            .rows = kGeometryPolicy.maximum_block_rows,
        };
        CaseReport report{};
        std::uint32_t submissions = 0;
        const int result = execute_gdn_fused2_case(
            *device.device, *queue.command_queue, pipelines, spec,
            report, submissions, Arm::SteelGdnBm64);
        if (result != 0) {
            return fail(result, category_for_exit(result));
        }
        if (submissions != kCommandBuffersPerCase) {
            return fail(
                kExitSampleAccounting, "sample-accounting");
        }
        std::cout << std::fixed << std::setprecision(6);
        print_gdn_treatment_report(
            report, Arm::SteelGdnBm64);
        std::cout
            << "native MLX Steel GDN BM64 benchmark:"
            << " PASS_GPU\n"
            << "  device: " << device.device->name() << '\n'
            << "  command buffers submitted: "
            << submissions << '\n'
            << "  evidence class: component-only"
               " synthetic-resident same-binary Steel control/"
               "BM64 treatment; not engine or"
               " tokens-per-second evidence\n";
        return 0;
    }

    if (mode == "--gpu-steel-gdn-bm64-wm2-wn2-component") {
        if (!tatara::backend::metal::generated::
                kKernelLibraryMlxSteelEnabled) {
            return fail(
                kExitSteelGdnBm64Wm2Wn2Pipeline,
                "steel-not-enabled");
        }
        const CaseSpec spec{
            .operation = kOperations[0],
            .rows = kGeometryPolicy.maximum_block_rows,
        };
        const int result = execute_gdn_fused2_component(
            *device.device, *queue.command_queue, pipelines, spec,
            Arm::SteelGdnBm64Wm2Wn2);
        if (result != 0) {
            return fail(result, category_for_exit(result));
        }
        std::cout
            << "native MLX Steel GDN BM64/WM2/WN2 component:"
            << " PASS_GPU\n"
            << "  device: " << device.device->name() << '\n'
            << "  command buffers submitted: 2\n"
            << "  timing samples: 0\n";
        return 0;
    }

    if (mode == "--benchmark-steel-gdn-bm64-wm2-wn2" ||
        mode ==
            "--benchmark-steel-gdn-bm64-wm2-wn2-reverse") {
        if (!tatara::backend::metal::generated::
                kKernelLibraryMlxSteelEnabled) {
            return fail(
                kExitSteelGdnBm64Wm2Wn2Pipeline,
                "steel-not-enabled");
        }
        const CaseSpec spec{
            .operation = kOperations[0],
            .rows = kGeometryPolicy.maximum_block_rows,
        };
        CaseReport report{};
        std::uint32_t submissions = 0;
        const int result = execute_gdn_fused2_case(
            *device.device, *queue.command_queue, pipelines, spec,
            report, submissions, Arm::SteelGdnBm64Wm2Wn2,
            mode ==
                "--benchmark-steel-gdn-bm64-wm2-wn2-reverse");
        if (result != 0) {
            return fail(result, category_for_exit(result));
        }
        if (submissions != kCommandBuffersPerCase) {
            return fail(
                kExitSampleAccounting, "sample-accounting");
        }
        std::cout << std::fixed << std::setprecision(6);
        print_gdn_treatment_report(
            report, Arm::SteelGdnBm64Wm2Wn2);
        std::cout
            << "native MLX Steel GDN BM64/WM2/WN2 benchmark:"
            << " PASS_GPU\n"
            << "  schedule: "
            << (mode ==
                        "--benchmark-steel-gdn-bm64-wm2-wn2-reverse"
                    ? "reverse"
                    : "forward")
            << '\n'
            << "  device: " << device.device->name() << '\n'
            << "  command buffers submitted: "
            << submissions << '\n'
            << "  evidence class: component-only"
               " synthetic-resident same-binary Steel control/"
               "BM64/WM2/WN2 treatment; not engine or"
               " tokens-per-second evidence\n";
        return 0;
    }

    if (mode == "--gpu-steel-gdn-bm64-bk64-component") {
        if (!tatara::backend::metal::generated::
                kKernelLibraryMlxSteelEnabled) {
            return fail(
                kExitSteelGdnBm64Bk64Pipeline,
                "steel-not-enabled");
        }
        const CaseSpec spec{
            .operation = kOperations[0],
            .rows = kGeometryPolicy.maximum_block_rows,
        };
        const int result = execute_gdn_fused2_component(
            *device.device, *queue.command_queue, pipelines, spec,
            Arm::SteelGdnBm64Bk64);
        if (result != 0) {
            return fail(result, category_for_exit(result));
        }
        std::cout
            << "native MLX Steel GDN BM64/BK64 component:"
            << " PASS_GPU\n"
            << "  device: " << device.device->name() << '\n'
            << "  command buffers submitted: 2\n"
            << "  timing samples: 0\n";
        return 0;
    }

    if (mode == "--benchmark-steel-gdn-bm64-bk64") {
        if (!tatara::backend::metal::generated::
                kKernelLibraryMlxSteelEnabled) {
            return fail(
                kExitSteelGdnBm64Bk64Pipeline,
                "steel-not-enabled");
        }
        const CaseSpec spec{
            .operation = kOperations[0],
            .rows = kGeometryPolicy.maximum_block_rows,
        };
        CaseReport report{};
        std::uint32_t submissions = 0;
        const int result = execute_gdn_fused2_case(
            *device.device, *queue.command_queue, pipelines, spec,
            report, submissions, Arm::SteelGdnBm64Bk64);
        if (result != 0) {
            return fail(result, category_for_exit(result));
        }
        if (submissions != kCommandBuffersPerCase) {
            return fail(
                kExitSampleAccounting, "sample-accounting");
        }
        std::cout << std::fixed << std::setprecision(6);
        print_gdn_treatment_report(
            report, Arm::SteelGdnBm64Bk64);
        std::cout
            << "native MLX Steel GDN BM64/BK64 benchmark:"
            << " PASS_GPU\n"
            << "  device: " << device.device->name() << '\n'
            << "  command buffers submitted: "
            << submissions << '\n'
            << "  evidence class: component-only"
               " synthetic-resident same-binary Steel control/"
               "BM64/BK64 treatment; not engine or"
               " tokens-per-second evidence\n";
        return 0;
    }

    if (mode == "--gpu-steel-gdn-bm48-component") {
        if (!tatara::backend::metal::generated::
                kKernelLibraryMlxSteelEnabled) {
            return fail(
                kExitSteelGdnBm48Pipeline,
                "steel-not-enabled");
        }
        const CaseSpec spec{
            .operation = kOperations[0],
            .rows = kGeometryPolicy.maximum_block_rows,
        };
        const int result = execute_gdn_fused2_component(
            *device.device, *queue.command_queue, pipelines, spec,
            Arm::SteelGdnBm48);
        if (result != 0) {
            return fail(result, category_for_exit(result));
        }
        std::cout
            << "native MLX Steel GDN BM48 component:"
            << " PASS_GPU\n"
            << "  device: " << device.device->name() << '\n'
            << "  command buffers submitted: 2\n"
            << "  timing samples: 0\n";
        return 0;
    }

    if (mode == "--benchmark-steel-gdn-bm48") {
        if (!tatara::backend::metal::generated::
                kKernelLibraryMlxSteelEnabled) {
            return fail(
                kExitSteelGdnBm48Pipeline,
                "steel-not-enabled");
        }
        const CaseSpec spec{
            .operation = kOperations[0],
            .rows = kGeometryPolicy.maximum_block_rows,
        };
        CaseReport report{};
        std::uint32_t submissions = 0;
        const int result = execute_gdn_fused2_case(
            *device.device, *queue.command_queue, pipelines, spec,
            report, submissions, Arm::SteelGdnBm48);
        if (result != 0) {
            return fail(result, category_for_exit(result));
        }
        if (submissions != kCommandBuffersPerCase) {
            return fail(
                kExitSampleAccounting, "sample-accounting");
        }
        std::cout << std::fixed << std::setprecision(6);
        print_gdn_treatment_report(
            report, Arm::SteelGdnBm48);
        std::cout
            << "native MLX Steel GDN BM48 benchmark:"
            << " PASS_GPU\n"
            << "  device: " << device.device->name() << '\n'
            << "  command buffers submitted: "
            << submissions << '\n'
            << "  evidence class: component-only"
               " synthetic-resident same-binary Steel control/"
               "BM48 treatment; not engine or"
               " tokens-per-second evidence\n";
        return 0;
    }

    if (mode == "--gpu-steel-gdn-bm96-component") {
        if (!tatara::backend::metal::generated::
                kKernelLibraryMlxSteelEnabled) {
            return fail(
                kExitSteelGdnBm96Pipeline,
                "steel-not-enabled");
        }
        const CaseSpec spec{
            .operation = kOperations[0],
            .rows = kGeometryPolicy.maximum_block_rows,
        };
        const int result = execute_gdn_fused2_component(
            *device.device, *queue.command_queue, pipelines, spec,
            Arm::SteelGdnBm96);
        if (result != 0) {
            return fail(result, category_for_exit(result));
        }
        std::cout
            << "native MLX Steel GDN BM96 component:"
            << " PASS_GPU\n"
            << "  device: " << device.device->name() << '\n'
            << "  command buffers submitted: 2\n"
            << "  timing samples: 0\n";
        return 0;
    }

    if (mode == "--benchmark-steel-gdn-bm96") {
        if (!tatara::backend::metal::generated::
                kKernelLibraryMlxSteelEnabled) {
            return fail(
                kExitSteelGdnBm96Pipeline,
                "steel-not-enabled");
        }
        const CaseSpec spec{
            .operation = kOperations[0],
            .rows = kGeometryPolicy.maximum_block_rows,
        };
        CaseReport report{};
        std::uint32_t submissions = 0;
        const int result = execute_gdn_fused2_case(
            *device.device, *queue.command_queue, pipelines, spec,
            report, submissions, Arm::SteelGdnBm96);
        if (result != 0) {
            return fail(result, category_for_exit(result));
        }
        if (submissions != kCommandBuffersPerCase) {
            return fail(
                kExitSampleAccounting, "sample-accounting");
        }
        std::cout << std::fixed << std::setprecision(6);
        print_gdn_treatment_report(
            report, Arm::SteelGdnBm96);
        std::cout
            << "native MLX Steel GDN BM96 benchmark:"
            << " PASS_GPU\n"
            << "  device: " << device.device->name() << '\n'
            << "  command buffers submitted: "
            << submissions << '\n'
            << "  evidence class: component-only"
               " synthetic-resident same-binary Steel control/"
               "BM96 treatment; not engine or"
               " tokens-per-second evidence\n";
        return 0;
    }

    if (mode == "--gpu-steel-gdn-bm128-component") {
        if (!tatara::backend::metal::generated::
                kKernelLibraryMlxSteelEnabled) {
            return fail(
                kExitSteelGdnBm128Pipeline,
                "steel-not-enabled");
        }
        const CaseSpec spec{
            .operation = kOperations[0],
            .rows = kGeometryPolicy.maximum_block_rows,
        };
        const int result = execute_gdn_fused2_component(
            *device.device, *queue.command_queue, pipelines, spec,
            Arm::SteelGdnBm128);
        if (result != 0) {
            return fail(result, category_for_exit(result));
        }
        std::cout
            << "native MLX Steel GDN BM128 component:"
            << " PASS_GPU\n"
            << "  device: " << device.device->name() << '\n'
            << "  command buffers submitted: 2\n"
            << "  timing samples: 0\n";
        return 0;
    }

    if (mode == "--benchmark-steel-gdn-bm128") {
        if (!tatara::backend::metal::generated::
                kKernelLibraryMlxSteelEnabled) {
            return fail(
                kExitSteelGdnBm128Pipeline,
                "steel-not-enabled");
        }
        const CaseSpec spec{
            .operation = kOperations[0],
            .rows = kGeometryPolicy.maximum_block_rows,
        };
        CaseReport report{};
        std::uint32_t submissions = 0;
        const int result = execute_gdn_fused2_case(
            *device.device, *queue.command_queue, pipelines, spec,
            report, submissions, Arm::SteelGdnBm128);
        if (result != 0) {
            return fail(result, category_for_exit(result));
        }
        if (submissions != kCommandBuffersPerCase) {
            return fail(
                kExitSampleAccounting, "sample-accounting");
        }
        std::cout << std::fixed << std::setprecision(6);
        print_gdn_treatment_report(
            report, Arm::SteelGdnBm128);
        std::cout
            << "native MLX Steel GDN BM128 benchmark:"
            << " PASS_GPU\n"
            << "  device: " << device.device->name() << '\n'
            << "  command buffers submitted: "
            << submissions << '\n'
            << "  evidence class: component-only"
               " synthetic-resident same-binary Steel control/"
               "BM128 treatment; not engine or"
               " tokens-per-second evidence\n";
        return 0;
    }

    if (mode == "--gpu-steel-gdn-bn64-component") {
        if (!tatara::backend::metal::generated::
                kKernelLibraryMlxSteelEnabled) {
            return fail(
                kExitSteelGdnBn64Pipeline,
                "steel-not-enabled");
        }
        const CaseSpec spec{
            .operation = kOperations[0],
            .rows = kGeometryPolicy.maximum_block_rows,
        };
        const int result = execute_gdn_fused2_component(
            *device.device, *queue.command_queue, pipelines, spec,
            Arm::SteelGdnBn64);
        if (result != 0) {
            return fail(result, category_for_exit(result));
        }
        std::cout
            << "native MLX Steel GDN BN64 component:"
            << " PASS_GPU\n"
            << "  device: " << device.device->name() << '\n'
            << "  command buffers submitted: 2\n"
            << "  timing samples: 0\n";
        return 0;
    }

    if (mode == "--benchmark-steel-gdn-bn64") {
        if (!tatara::backend::metal::generated::
                kKernelLibraryMlxSteelEnabled) {
            return fail(
                kExitSteelGdnBn64Pipeline,
                "steel-not-enabled");
        }
        const CaseSpec spec{
            .operation = kOperations[0],
            .rows = kGeometryPolicy.maximum_block_rows,
        };
        CaseReport report{};
        std::uint32_t submissions = 0;
        const int result = execute_gdn_fused2_case(
            *device.device, *queue.command_queue, pipelines, spec,
            report, submissions, Arm::SteelGdnBn64);
        if (result != 0) {
            return fail(result, category_for_exit(result));
        }
        if (submissions != kCommandBuffersPerCase) {
            return fail(
                kExitSampleAccounting, "sample-accounting");
        }
        std::cout << std::fixed << std::setprecision(6);
        print_gdn_treatment_report(
            report, Arm::SteelGdnBn64);
        std::cout
            << "native MLX Steel GDN BN64 benchmark:"
            << " PASS_GPU\n"
            << "  device: " << device.device->name() << '\n'
            << "  command buffers submitted: "
            << submissions << '\n'
            << "  evidence class: component-only"
               " synthetic-resident same-binary Steel control/"
               "BN64 treatment; not engine or"
               " tokens-per-second evidence\n";
        return 0;
    }

    if (mode == "--gpu-steel-gdn-bk64-component") {
        if (!tatara::backend::metal::generated::
                kKernelLibraryMlxSteelEnabled) {
            return fail(
                kExitSteelGdnBk64Pipeline,
                "steel-not-enabled");
        }
        const CaseSpec spec{
            .operation = kOperations[0],
            .rows = kGeometryPolicy.maximum_block_rows,
        };
        const int result = execute_gdn_fused2_component(
            *device.device, *queue.command_queue, pipelines, spec,
            Arm::SteelGdnBk64);
        if (result != 0) {
            return fail(result, category_for_exit(result));
        }
        std::cout
            << "native MLX Steel GDN BK64 component:"
            << " PASS_GPU\n"
            << "  device: " << device.device->name() << '\n'
            << "  command buffers submitted: 2\n"
            << "  timing samples: 0\n";
        return 0;
    }

    if (mode == "--benchmark-steel-gdn-bk64") {
        if (!tatara::backend::metal::generated::
                kKernelLibraryMlxSteelEnabled) {
            return fail(
                kExitSteelGdnBk64Pipeline,
                "steel-not-enabled");
        }
        const CaseSpec spec{
            .operation = kOperations[0],
            .rows = kGeometryPolicy.maximum_block_rows,
        };
        CaseReport report{};
        std::uint32_t submissions = 0;
        const int result = execute_gdn_fused2_case(
            *device.device, *queue.command_queue, pipelines, spec,
            report, submissions, Arm::SteelGdnBk64);
        if (result != 0) {
            return fail(result, category_for_exit(result));
        }
        if (submissions != kCommandBuffersPerCase) {
            return fail(
                kExitSampleAccounting, "sample-accounting");
        }
        std::cout << std::fixed << std::setprecision(6);
        print_gdn_treatment_report(
            report, Arm::SteelGdnBk64);
        std::cout
            << "native MLX Steel GDN BK64 benchmark:"
            << " PASS_GPU\n"
            << "  device: " << device.device->name() << '\n'
            << "  command buffers submitted: "
            << submissions << '\n'
            << "  evidence class: component-only"
               " synthetic-resident same-binary Steel control/"
               "BK64 treatment; not engine or"
               " tokens-per-second evidence\n";
        return 0;
    }

    if (mode == "--gpu-steel") {
        if (!tatara::backend::metal::generated::
                kKernelLibraryMlxSteelEnabled) {
            return fail(kExitSteelPipeline, "steel-not-enabled");
        }
        std::array<CaseReport, kOperations.size()> reports{};
        std::uint32_t submissions = 0;
        for (std::size_t index = 0;
             index < kOperations.size(); ++index) {
            const CaseSpec spec{
                .operation = kOperations[index],
                .rows = kGeometryPolicy.maximum_block_rows,
            };
            const int result = execute_steel_case(
                *device.device, *queue.command_queue, pipelines,
                spec, reports[index], submissions);
            if (result != 0) {
                return fail(result, category_for_exit(result));
            }
        }
        constexpr std::uint32_t kExpectedSubmissions =
            static_cast<std::uint32_t>(
                kOperations.size() * kCommandBuffersPerCase);
        if (submissions != kExpectedSubmissions) {
            return fail(
                kExitSampleAccounting, "sample-accounting");
        }
        std::cout << std::fixed << std::setprecision(6);
        for (const CaseReport& report : reports) {
            print_steel_report(report);
        }
        std::cout
            << "native MLX Steel dense QGEMM perf fixture:"
            << " PASS_GPU\n"
            << "  device: " << device.device->name() << '\n'
            << "  cases: " << reports.size() << '\n'
            << "  command buffers submitted: " << submissions << '\n'
            << "  evidence class: component-only"
            << " synthetic-resident same-binary N1/Steel evidence;"
            << " not engine or tokens-per-second evidence\n";
        return 0;
    }

    std::array<CaseReport, kCaseCount> reports{};
    std::uint32_t submissions = 0;
    for (std::size_t index = 0; index < kCases.size(); ++index) {
        const int result = execute_case(
            *device.device, *queue.command_queue, pipelines,
            kCases[index], reports[index], submissions);
        if (result != 0) {
            return fail(result, category_for_exit(result));
        }
    }
    if (submissions != kTotalCommandBuffers) {
        return fail(
            kExitSampleAccounting, "sample-accounting");
    }

    std::cout << std::fixed << std::setprecision(6);
    for (const CaseReport& report : reports) {
        print_report(report);
    }
    std::cout
        << "native dense QGEMM perf fixture: PASS_GPU\n"
        << "  device: " << device.device->name() << '\n'
        << "  cases: " << reports.size() << '\n'
        << "  command buffers submitted: " << submissions << '\n'
        << "  evidence class: component-only"
        << " synthetic-resident projection evidence;"
        << " not model, engine, serving, TTFT or"
        << " tokens-per-second evidence\n";
    return 0;
}
