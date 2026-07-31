#include "tatara/backend/metal/commands.h"
#include "tatara/backend/metal/pipeline.h"
#include "tatara/backend/metal/resources.h"
#include "tatara/generated/kernel_library.h"
#include "tatara/runtime/quantized_gemm.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace tatara::backend::metal;
using tatara::runtime::QuantizedGemmDeviceTaskStatus;
using tatara::runtime::QuantizedGemmTaskDescriptor;
namespace generated = tatara::backend::metal::generated;

constexpr std::uint32_t kExperts = generated::kKernelLibraryMoeExperts;
constexpr std::uint32_t kActiveExperts =
    generated::kKernelLibraryMoeActiveExperts;
constexpr std::uint32_t kHidden = generated::kKernelLibraryHidden;
constexpr std::uint32_t kExpertDimension =
    generated::kKernelLibraryMoeExpertDimension;
constexpr std::uint32_t kGroupSize = generated::kKernelLibraryGroupSize;
constexpr std::uint32_t kTileRows =
    generated::kKernelLibraryNativeRoutedQgemmR1TileRows;
constexpr std::uint32_t kTileColumns =
    generated::kKernelLibraryNativeRoutedQgemmR1TileColumns;
constexpr std::uint32_t kBn64TileColumns = 2U * kTileColumns;
constexpr std::uint32_t kThreads =
    generated::kKernelLibraryNativeRoutedQgemmR1Threads;
constexpr std::uint32_t kBm32TileRows = 32U;
constexpr std::uint32_t kBm32Threads = 128U;
constexpr std::uint32_t kTaskCapacityLimit =
    generated::kKernelLibraryNativeRoutedQgemmR1TaskCapacity;
constexpr std::uint32_t kPackedSlotBits =
    std::bit_width(kActiveExperts);
constexpr std::uint32_t kSlotCount = kActiveExperts + 1U;
constexpr std::uint32_t kQ4ValuesPerWord = 8U;
constexpr std::uint32_t kExactThreads = 128U;
constexpr std::uint32_t kSimdgroupWidth = 32U;
constexpr std::uint32_t kHighPositionCount = 2048U;
constexpr std::uint32_t kHighPositionCapacity = 2048U;
constexpr std::uint32_t kHighListStride = 2048U;
constexpr std::uint32_t kHighLowCount = 63U;
constexpr std::uint32_t kHighHighCount = 65U;
constexpr std::uint32_t kHighExpectedTasks = 1152U;
constexpr std::uint32_t kHighBm32ExpectedTasks = 640U;
constexpr std::uint32_t kHighPlannedTaskCapacity = 1280U;

constexpr std::uint32_t kNumericPositionCount = 17U;
constexpr std::uint32_t kNumericPositionCapacity = 32U;
constexpr std::uint32_t kNumericListStride = 32U;
constexpr std::uint32_t kNumericExpectedTasks = 10U;
constexpr std::uint32_t kNumericBm32ExpectedTasks = 9U;
constexpr std::uint32_t kNumericPlannedTaskCapacity = 272U;
constexpr std::uint32_t kNumericSharedExpectedTasks = 12U;
constexpr std::uint32_t kNumericSharedBm32ExpectedTasks = 10U;
constexpr std::uint32_t kNumericSharedPlannedTaskCapacity = 275U;

constexpr std::uint32_t kUpColumnGroups =
    (kExpertDimension + kTileColumns - 1U) / kTileColumns;
constexpr std::uint32_t kDownColumnGroups =
    (kHidden + kTileColumns - 1U) / kTileColumns;
constexpr std::uint32_t kBn64UpColumnGroups =
    (kExpertDimension + kBn64TileColumns - 1U) /
    kBn64TileColumns;
constexpr std::uint32_t kBn64DownColumnGroups =
    (kHidden + kBn64TileColumns - 1U) /
    kBn64TileColumns;
constexpr std::uint64_t kExpectedComputeThreadgroups =
    std::uint64_t{kNumericExpectedTasks} *
    (3U * kUpColumnGroups + kDownColumnGroups);

constexpr std::size_t kGuardBytes = 64U;
constexpr std::byte kGuardCanary{0xA5};
constexpr std::uint32_t kListCanary = 0xD4C3B2A1U;
constexpr std::uint32_t kTaskCanary = 0xC7B6A594U;
constexpr std::uint32_t kWordCanary = 0xE3D2C1B0U;
constexpr std::uint16_t kHiddenCanary = 0x7FC1U;
constexpr std::uint32_t kPartialCanary = 0x7FC01234U;
constexpr std::uint32_t kMaximumBfloatUlp = 2U;
constexpr float kMaximumNormalizedError = 0.02F;

constexpr std::string_view kProducerName =
    "native_routed_qgemm_r1_build_tasks";
constexpr std::string_view kFusedName =
    "native_routed_qgemm_r1_fused_upgate_swiglu";
constexpr std::string_view kGateName = "native_routed_qgemm_r1_gate";
constexpr std::string_view kUpName =
    "native_routed_qgemm_r1_up_swiglu";
constexpr std::string_view kDownName =
    "native_routed_qgemm_r1_down_partial";
constexpr std::string_view kSteelDownName =
    generated::kKernelLibraryMlxSteelRoutedDownKernelName;
constexpr std::string_view kSteelUpgateName =
    generated::kKernelLibraryMlxSteelRoutedUpgateKernelName;
constexpr std::string_view kSteelBm32DownName =
    generated::kKernelLibraryMlxSteelRoutedBm32DownKernelName;
constexpr std::string_view kSteelBm32UpgateName =
    generated::kKernelLibraryMlxSteelRoutedBm32UpgateKernelName;
constexpr std::string_view kSteelBk64DownName =
    generated::kKernelLibraryMlxSteelRoutedBk64DownKernelName;
constexpr std::string_view kSteelBk64UpgateName =
    generated::kKernelLibraryMlxSteelRoutedBk64UpgateKernelName;
constexpr std::string_view kSteelBn64DownName =
    generated::kKernelLibraryMlxSteelRoutedBn64DownKernelName;
constexpr std::string_view kSteelBn64UpgateName =
    generated::kKernelLibraryMlxSteelRoutedBn64UpgateKernelName;

constexpr int kExitUsage = 40;
constexpr int kExitCpuContract = 41;
constexpr int kExitDevice = 42;
constexpr int kExitQueue = 43;
constexpr int kExitLibrary = 44;
constexpr int kExitFunction = 45;
constexpr int kExitProducerPipeline = 46;
constexpr int kExitFusedPipeline = 47;
constexpr int kExitGatePipeline = 48;
constexpr int kExitUpPipeline = 49;
constexpr int kExitDownPipeline = 50;
constexpr int kExitBuffer = 51;
constexpr int kExitInitialization = 52;
constexpr int kExitCommandBuffer = 53;
constexpr int kExitComputePass = 54;
constexpr int kExitEncode = 55;
constexpr int kExitEndPass = 56;
constexpr int kExitCommit = 57;
constexpr int kExitExecution = 58;
constexpr int kExitProducerNotProduced = 59;
constexpr int kExitProducerCount = 60;
constexpr int kExitProducerConservation = 61;
constexpr int kExitProducerCapacity = 62;
constexpr int kExitProducerSlot = 63;
constexpr int kExitProducerUnknown = 64;
constexpr int kExitProducerPartition = 65;
constexpr int kExitProducerGrid = 66;
constexpr int kExitCanary = 67;
constexpr int kExitFusedNumerics = 68;
constexpr int kExitSplitNumerics = 69;
constexpr int kExitDownNumerics = 70;
constexpr int kExitFamilyDiscrimination = 71;
constexpr int kExitSubmissionAccounting = 72;
constexpr int kExitBenchmarkPipeline = 73;
constexpr int kExitTiming = 74;
constexpr int kExitControlDrift = 75;
constexpr int kExitSampleAccounting = 76;
constexpr int kExitSharedHiddenNumerics = 77;
constexpr int kExitSharedDownNumerics = 78;
constexpr int kExitSteelUnavailable = 79;

constexpr std::string_view kBenchmarkWarmupSchedule =
    "ABBAABBAABBAABBAABBAABBAABBAABBA"
    "ABBAABBAABBAABBAABBAABBAABBAABBA";
constexpr std::string_view kBenchmarkMeasuredSchedule =
    "ABABBABAABABBABAABABBABAABBAABABBA";
constexpr std::size_t kBenchmarkSamplesPerArm = 17U;

static_assert(kExperts % 2U == 0U);
static_assert(kExperts > kActiveExperts);
static_assert(kActiveExperts + 1U <= (1U << kPackedSlotBits));
static_assert(sizeof(QuantizedGemmTaskDescriptor) ==
              generated::kKernelLibraryNativeRoutedQgemmR1TaskBytes);
static_assert(kThreads == 64U);
static_assert(kTileRows == 16U);
static_assert(kBm32Threads == 2U * kThreads);
static_assert(kBm32TileRows == 2U * kTileRows);
static_assert(kTileColumns == 32U);
static_assert(kHighPositionCount * kActiveExperts ==
              (kHighLowCount + kHighHighCount) * (kExperts / 2U));
static_assert(kHighExpectedTasks == 1152U);
static_assert(kHighPlannedTaskCapacity == 1280U);
static_assert(kNumericExpectedTasks == 10U);
static_assert(kNumericPlannedTaskCapacity == 272U);
static_assert(kExpectedComputeThreadgroups == 1120U);
consteval std::size_t count_arm(std::string_view schedule, char arm) {
    std::size_t count = 0U;
    for (const char value : schedule) {
        count += static_cast<std::size_t>(value == arm);
    }
    return count;
}

static_assert(count_arm(kBenchmarkWarmupSchedule, 'A') == 32U);
static_assert(count_arm(kBenchmarkWarmupSchedule, 'B') == 32U);
static_assert(
    count_arm(kBenchmarkMeasuredSchedule, 'A') ==
    kBenchmarkSamplesPerArm);
static_assert(
    count_arm(kBenchmarkMeasuredSchedule, 'B') ==
    kBenchmarkSamplesPerArm);
static_assert(kBenchmarkMeasuredSchedule.front() == 'A');
static_assert(kBenchmarkMeasuredSchedule.back() == 'A');

std::uint32_t ceil_div(std::uint32_t value,
                       std::uint32_t divisor) noexcept {
    return (value + divisor - 1U) / divisor;
}

bool checked_add(std::size_t left, std::size_t right,
                 std::size_t& result) noexcept {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

std::uint16_t bfloat16(float value) noexcept {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7F800000U) != 0x7F800000U) {
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

std::uint32_t bfloat_ulp_distance(std::uint16_t left,
                                  std::uint16_t right) noexcept {
    const std::uint32_t ordered_left = ordered_bfloat(left);
    const std::uint32_t ordered_right = ordered_bfloat(right);
    return ordered_left > ordered_right
               ? ordered_left - ordered_right
               : ordered_right - ordered_left;
}

std::uint32_t pack_route(std::uint32_t position,
                         std::uint32_t slot) noexcept {
    return (position << kPackedSlotBits) | slot;
}

struct RouteFixture {
    std::string_view name;
    std::uint32_t position_count = 0;
    std::uint32_t position_capacity = 0;
    std::uint32_t list_stride = 0;
    std::uint32_t planned_task_capacity = 0;
    std::uint32_t task_tile_rows = kTileRows;
    bool include_shared_expert = false;
    std::vector<std::uint32_t> counts;
    std::vector<std::uint32_t> lists;
    std::vector<std::uint32_t> expert_for_position_slot;
    std::vector<QuantizedGemmTaskDescriptor> tasks;
};

bool finalize_routes(
    std::string_view name, std::uint32_t position_count,
    std::uint32_t position_capacity, std::uint32_t list_stride,
    std::uint32_t planned_task_capacity,
    std::span<const std::uint32_t> expert_for_position_slot,
    RouteFixture& fixture) {
    if (position_count == 0U || position_count > position_capacity ||
        position_capacity > list_stride ||
        expert_for_position_slot.size() !=
            std::size_t{position_count} * kActiveExperts) {
        return false;
    }
    std::vector<std::vector<std::uint32_t>> per_expert(kExperts);
    for (std::uint32_t position = 0; position < position_count;
         ++position) {
        for (std::uint32_t slot = 0; slot < kActiveExperts; ++slot) {
            const std::size_t index =
                std::size_t{position} * kActiveExperts + slot;
            const std::uint32_t expert = expert_for_position_slot[index];
            if (expert >= kExperts) {
                return false;
            }
            per_expert[expert].push_back(pack_route(position, slot));
        }
    }

    fixture = {};
    fixture.name = name;
    fixture.position_count = position_count;
    fixture.position_capacity = position_capacity;
    fixture.list_stride = list_stride;
    fixture.planned_task_capacity = planned_task_capacity;
    fixture.counts.resize(kExperts);
    fixture.lists.assign(
        std::size_t{kExperts} * list_stride, kListCanary);
    fixture.expert_for_position_slot.assign(
        expert_for_position_slot.begin(),
        expert_for_position_slot.end());

    std::uint32_t expected_task_count = 0U;
    for (std::uint32_t expert = 0; expert < kExperts; ++expert) {
        const auto& routes = per_expert[expert];
        if (routes.size() > position_capacity) {
            return false;
        }
        fixture.counts[expert] =
            static_cast<std::uint32_t>(routes.size());
        std::copy(
            routes.begin(), routes.end(),
            fixture.lists.begin() +
                static_cast<std::ptrdiff_t>(
                    std::size_t{expert} * list_stride));
        expected_task_count +=
            ceil_div(fixture.counts[expert], kTileRows);
        for (std::uint32_t local_begin = 0U;
             local_begin < fixture.counts[expert];
             local_begin += kTileRows) {
            fixture.tasks.push_back({
                .expert_index = expert,
                .route_list_begin =
                    expert * list_stride + local_begin,
                .row_count = std::min(
                    kTileRows,
                    fixture.counts[expert] - local_begin),
                .output_row_begin = 0U,
            });
        }
    }
    return fixture.tasks.size() == expected_task_count &&
           fixture.tasks.size() <= planned_task_capacity;
}

bool make_high_water_routes(RouteFixture& fixture) {
    std::vector<std::uint32_t> experts(
        std::size_t{kHighPositionCount} * kActiveExperts);
    for (std::uint32_t position = 0U;
         position < kHighPositionCount; ++position) {
        for (std::uint32_t slot = 0U; slot < kActiveExperts; ++slot) {
            const std::uint32_t route_index =
                position * kActiveExperts + slot;
            const std::uint32_t base_expert =
                route_index % kExperts;
            const std::uint32_t cycle = route_index / kExperts;
            experts[route_index] =
                cycle == 0U && base_expert < kExperts / 2U
                    ? base_expert + kExperts / 2U
                    : base_expert;
        }
    }
    return finalize_routes(
        "high-water", kHighPositionCount,
        kHighPositionCapacity, kHighListStride,
        kHighPlannedTaskCapacity, experts, fixture);
}

bool make_numeric_routes(RouteFixture& fixture) {
    std::vector<std::uint32_t> experts(
        std::size_t{kNumericPositionCount} * kActiveExperts);
    for (std::uint32_t position = 0U;
         position < kNumericPositionCount; ++position) {
        experts[std::size_t{position} * kActiveExperts] = 0U;
        const std::uint32_t omitted =
            1U + position % kActiveExperts;
        std::uint32_t slot = 1U;
        for (std::uint32_t expert = 1U;
             expert <= kActiveExperts; ++expert) {
            if (expert != omitted) {
                experts[std::size_t{position} * kActiveExperts +
                        slot++] = expert;
            }
        }
        if (slot != kActiveExperts) {
            return false;
        }
    }
    return finalize_routes(
        "numeric", kNumericPositionCount,
        kNumericPositionCapacity, kNumericListStride,
        kNumericPlannedTaskCapacity, experts, fixture);
}

bool make_numeric_shared_routes(
    const RouteFixture& numeric, RouteFixture& fixture) {
    if (numeric.include_shared_expert ||
        numeric.counts.size() != kExperts ||
        numeric.lists.size() !=
            std::size_t{kExperts} * numeric.list_stride) {
        return false;
    }
    fixture = numeric;
    fixture.name = "numeric-shared";
    fixture.planned_task_capacity =
        kNumericSharedPlannedTaskCapacity;
    fixture.include_shared_expert = true;
    fixture.counts.push_back(numeric.position_count);
    fixture.lists.resize(
        std::size_t{kExperts + 1U} * numeric.list_stride,
        kListCanary);
    for (std::uint32_t position = 0U;
         position < numeric.position_count; ++position) {
        fixture.lists[
            std::size_t{kExperts} * numeric.list_stride + position] =
            pack_route(position, kActiveExperts);
    }
    for (std::uint32_t local_begin = 0U;
         local_begin < numeric.position_count;
         local_begin += kTileRows) {
        fixture.tasks.push_back({
            .expert_index = kExperts,
            .route_list_begin =
                kExperts * numeric.list_stride + local_begin,
            .row_count = std::min(
                kTileRows, numeric.position_count - local_begin),
            .output_row_begin = 0U,
        });
    }
    return fixture.tasks.size() == kNumericSharedExpectedTasks &&
           fixture.tasks.size() <= fixture.planned_task_capacity;
}



bool valid_task_partition(
    const RouteFixture& fixture,
    std::span<const QuantizedGemmTaskDescriptor> tasks) {
    std::size_t task_index = 0U;
    for (std::uint32_t expert = 0U;
         expert < fixture.counts.size(); ++expert) {
        std::uint32_t local_cursor = 0U;
        while (local_cursor < fixture.counts[expert]) {
            if (task_index >= tasks.size()) {
                return false;
            }
            const auto& task = tasks[task_index++];
            const std::uint32_t expected_rows = std::min(
                fixture.task_tile_rows,
                fixture.counts[expert] - local_cursor);
            if (task.expert_index >= fixture.counts.size() ||
                task.expert_index != expert ||
                task.route_list_begin !=
                    expert * fixture.list_stride + local_cursor ||
                task.row_count != expected_rows ||
                task.output_row_begin != 0U) {
                return false;
            }
            local_cursor += expected_rows;
        }
    }
    return task_index == tasks.size();
}

bool make_bm32_route_fixture(
    const RouteFixture& source, RouteFixture& fixture) {
    if (source.counts.empty() || source.lists.empty() ||
        source.task_tile_rows != kTileRows) {
        return false;
    }
    fixture = source;
    fixture.task_tile_rows = kBm32TileRows;
    fixture.tasks.clear();
    for (std::uint32_t expert = 0U;
         expert < fixture.counts.size(); ++expert) {
        for (std::uint32_t local_begin = 0U;
             local_begin < fixture.counts[expert];
             local_begin += kBm32TileRows) {
            fixture.tasks.push_back({
                .expert_index = expert,
                .route_list_begin =
                    expert * fixture.list_stride + local_begin,
                .row_count = std::min(
                    kBm32TileRows,
                    fixture.counts[expert] - local_begin),
                .output_row_begin = 0U,
            });
        }
    }
    return fixture.tasks.size() <= fixture.planned_task_capacity &&
           valid_task_partition(fixture, fixture.tasks);
}

bool validate_route_fixture(const RouteFixture& fixture) {
    const std::uint32_t expert_count =
        kExperts +
        static_cast<std::uint32_t>(fixture.include_shared_expert);
    const std::uint32_t route_slots =
        kActiveExperts +
        static_cast<std::uint32_t>(fixture.include_shared_expert);
    if (fixture.counts.size() != expert_count ||
        fixture.lists.size() !=
            std::size_t{expert_count} * fixture.list_stride ||
        fixture.tasks.size() > fixture.planned_task_capacity ||
        !valid_task_partition(fixture, fixture.tasks)) {
        return false;
    }
    std::vector<std::uint8_t> seen(
        std::size_t{fixture.position_count} * route_slots, 0U);
    std::uint64_t route_count = 0U;
    for (std::uint32_t expert = 0U; expert < expert_count; ++expert) {
        if (fixture.counts[expert] > fixture.position_count ||
            fixture.counts[expert] > fixture.position_capacity) {
            return false;
        }
        for (std::uint32_t local = 0U;
             local < fixture.counts[expert]; ++local) {
            const std::uint32_t packed =
                fixture.lists[
                    std::size_t{expert} * fixture.list_stride + local];
            const std::uint32_t position =
                packed >> kPackedSlotBits;
            const std::uint32_t slot =
                packed & ((1U << kPackedSlotBits) - 1U);
            if (position >= fixture.position_count ||
                slot >= route_slots ||
                (expert == kExperts) !=
                    (slot == kActiveExperts)) {
                return false;
            }
            const std::size_t index =
                std::size_t{position} * route_slots + slot;
            if (seen[index] != 0U ||
                (slot < kActiveExperts &&
                 fixture.expert_for_position_slot[
                     std::size_t{position} * kActiveExperts + slot] !=
                     expert)) {
                return false;
            }
            seen[index] = 1U;
            ++route_count;
        }
        for (std::uint32_t local = fixture.counts[expert];
             local < fixture.list_stride; ++local) {
            if (fixture.lists[
                    std::size_t{expert} * fixture.list_stride + local] !=
                kListCanary) {
                return false;
            }
        }
    }
    return route_count ==
               std::uint64_t{fixture.position_count} * route_slots &&
           std::all_of(
               seen.begin(), seen.end(),
               [](std::uint8_t value) { return value == 1U; });
}

bool validate_frozen_route_cases(
    const RouteFixture& high, const RouteFixture& numeric,
    const RouteFixture& numeric_shared) {
    if (!validate_route_fixture(high) ||
        !validate_route_fixture(numeric) ||
        !validate_route_fixture(numeric_shared) ||
        high.tasks.size() != kHighExpectedTasks ||
        high.planned_task_capacity != kHighPlannedTaskCapacity ||
        numeric.tasks.size() != kNumericExpectedTasks ||
        numeric.planned_task_capacity !=
            kNumericPlannedTaskCapacity ||
        numeric_shared.tasks.size() !=
            kNumericSharedExpectedTasks ||
        numeric_shared.planned_task_capacity !=
            kNumericSharedPlannedTaskCapacity ||
        numeric_shared.counts[kExperts] !=
            kNumericPositionCount) {
        return false;
    }
    for (std::uint32_t expert = 0U; expert < kExperts; ++expert) {
        const std::uint32_t expected =
            expert < kExperts / 2U ? kHighLowCount : kHighHighCount;
        if (high.counts[expert] != expected) {
            return false;
        }
    }
    if (numeric.counts[0] != kNumericPositionCount ||
        numeric.counts[1] != kNumericPositionCount - 3U) {
        return false;
    }
    for (std::uint32_t expert = 2U;
         expert <= kActiveExperts; ++expert) {
        if (numeric.counts[expert] !=
            kNumericPositionCount - 2U) {
            return false;
        }
    }
    for (std::uint32_t expert = kActiveExperts + 1U;
         expert < kExperts; ++expert) {
        if (numeric.counts[expert] != 0U) {
            return false;
        }
    }
    const std::size_t shared_begin =
        std::size_t{kExperts} * numeric_shared.list_stride;
    for (std::uint32_t position = 0U;
         position < numeric_shared.position_count; ++position) {
        if (numeric_shared.lists[shared_begin + position] !=
            pack_route(position, kActiveExperts)) {
            return false;
        }
    }
    return numeric_shared.tasks[kNumericExpectedTasks].row_count ==
               kTileRows &&
           numeric_shared.tasks[kNumericExpectedTasks + 1U].row_count ==
               1U;
}

bool selected_up_column(std::uint32_t column) noexcept {
    const std::uint32_t tile = column / kTileColumns;
    return column % kTileColumns ==
           (tile * 7U + 3U) % kTileColumns;
}

bool selected_down_column(std::uint32_t column) noexcept {
    const std::uint32_t tile = column / kTileColumns;
    return column % kTileColumns ==
           (tile * 11U + 5U) % kTileColumns;
}

std::size_t packed_index(std::uint32_t expert,
                         std::uint32_t output_column,
                         std::uint32_t reduction_column,
                         std::uint32_t output_columns,
                         std::uint32_t reduction_columns) noexcept {
    return (std::size_t{expert} * output_columns + output_column) *
               (reduction_columns / kQ4ValuesPerWord) +
           reduction_column / kQ4ValuesPerWord;
}

std::size_t parameter_index(
    std::uint32_t expert, std::uint32_t output_column,
    std::uint32_t reduction_column, std::uint32_t output_columns,
    std::uint32_t reduction_columns) noexcept {
    return (std::size_t{expert} * output_columns + output_column) *
               (reduction_columns / kGroupSize) +
           reduction_column / kGroupSize;
}

void set_q4(std::vector<std::uint32_t>& packed,
            std::uint32_t expert, std::uint32_t output_column,
            std::uint32_t reduction_column,
            std::uint32_t output_columns,
            std::uint32_t reduction_columns,
            std::uint32_t value) {
    const std::size_t index =
        packed_index(expert, output_column, reduction_column,
                     output_columns, reduction_columns);
    const std::uint32_t shift =
        4U * (reduction_column % kQ4ValuesPerWord);
    packed[index] &= ~(15U << shift);
    packed[index] |= (value & 15U) << shift;
}

float q4_weight(
    const std::vector<std::uint32_t>& packed,
    const std::vector<std::uint16_t>& scales,
    const std::vector<std::uint16_t>& biases,
    std::uint32_t expert, std::uint32_t output_column,
    std::uint32_t reduction_column,
    std::uint32_t output_columns,
    std::uint32_t reduction_columns) {
    const std::uint32_t word =
        packed[packed_index(
            expert, output_column, reduction_column,
            output_columns, reduction_columns)];
    const std::uint32_t quantized =
        (word >>
         (4U * (reduction_column % kQ4ValuesPerWord))) &
        15U;
    const std::size_t metadata =
        parameter_index(
            expert, output_column, reduction_column,
            output_columns, reduction_columns);
    const float weight =
        static_cast<float>(quantized) *
            from_bfloat16(scales[metadata]) +
        from_bfloat16(biases[metadata]);
    return weight;
}

struct NumericFixture {
    std::vector<std::uint16_t> input;
    std::vector<std::uint32_t> gate_packed;
    std::vector<std::uint16_t> gate_scales;
    std::vector<std::uint16_t> gate_biases;
    std::vector<std::uint32_t> up_packed;
    std::vector<std::uint16_t> up_scales;
    std::vector<std::uint16_t> up_biases;
    std::vector<std::uint32_t> down_packed;
    std::vector<std::uint16_t> down_scales;
    std::vector<std::uint16_t> down_biases;
    std::vector<std::uint16_t> fused_expected;
    std::vector<std::uint16_t> split_expected;
    std::vector<std::uint16_t> staged_expected;
    std::vector<std::uint16_t> assimilated_expected;
    std::vector<float> down_expected;
    std::vector<float> staged_down_expected;
    std::vector<float> assimilated_down_expected;
    std::size_t family_differences = 0U;
    std::size_t direct_staged_differences = 0U;
};

float swiglu(float gate, float up) noexcept {
    return (gate / (1.0F + std::exp(-gate))) * up;
}

float sparse_up_dot(
    const NumericFixture& fixture,
    const std::vector<std::uint32_t>& packed,
    const std::vector<std::uint16_t>& scales,
    const std::vector<std::uint16_t>& biases,
    std::uint32_t position, std::uint32_t expert,
    std::uint32_t output_column) {
    if (!selected_up_column(output_column)) {
        return 0.0F;
    }
    float accumulator = 0.0F;
    for (std::uint32_t group = 0U;
         group < kHidden / kGroupSize; ++group) {
        const std::uint32_t reduction =
            group * kGroupSize +
            (position * 11U + group * 13U + 5U) % kGroupSize;
        const float activation = from_bfloat16(
            fixture.input[
                std::size_t{position} * kHidden + reduction]);
        const float weight = q4_weight(
            packed, scales, biases, expert, output_column,
            reduction, kExpertDimension, kHidden);
        accumulator = std::fma(activation, weight, accumulator);
    }
    return accumulator;
}

float sparse_down_dot(
    const NumericFixture& fixture,
    std::span<const std::uint16_t> hidden,
    std::uint32_t position, std::uint32_t slot,
    std::uint32_t hidden_slots, std::uint32_t expert,
    std::uint32_t output_column) {
    if (!selected_down_column(output_column)) {
        return 0.0F;
    }
    float accumulator = 0.0F;
    for (std::uint32_t reduction = 0U;
         reduction < kExpertDimension; ++reduction) {
        if (!selected_up_column(reduction)) {
            continue;
        }
        const float activation = from_bfloat16(
            hidden[
                (std::size_t{position} * hidden_slots + slot) *
                    kExpertDimension +
                reduction]);
        const float weight = q4_weight(
            fixture.down_packed, fixture.down_scales,
            fixture.down_biases, expert, output_column, reduction,
            kHidden, kExpertDimension);
        accumulator = std::fma(activation, weight, accumulator);
    }
    return accumulator;
}

bool make_numeric_fixture(
    const RouteFixture& routes, NumericFixture& fixture) {
    if (routes.position_count != kNumericPositionCount ||
        routes.position_capacity != kNumericPositionCapacity) {
        return false;
    }
    const std::uint32_t stored_experts = kActiveExperts + 1U;
    fixture = {};
    fixture.input.assign(
        std::size_t{kNumericPositionCapacity} * kHidden,
        bfloat16(0.0F));

    const std::size_t up_packed_count =
        std::size_t{stored_experts} * kExpertDimension *
        (kHidden / kQ4ValuesPerWord);
    const std::size_t up_parameter_count =
        std::size_t{stored_experts} * kExpertDimension *
        (kHidden / kGroupSize);
    const std::size_t down_packed_count =
        std::size_t{stored_experts} * kHidden *
        (kExpertDimension / kQ4ValuesPerWord);
    const std::size_t down_parameter_count =
        std::size_t{stored_experts} * kHidden *
        (kExpertDimension / kGroupSize);

    fixture.gate_packed.assign(up_packed_count, 0U);
    fixture.gate_scales.assign(up_parameter_count, bfloat16(0.0F));
    fixture.gate_biases.assign(up_parameter_count, bfloat16(0.0F));
    fixture.up_packed.assign(up_packed_count, 0U);
    fixture.up_scales.assign(up_parameter_count, bfloat16(0.0F));
    fixture.up_biases.assign(up_parameter_count, bfloat16(0.0F));
    fixture.down_packed.assign(down_packed_count, 0U);
    fixture.down_scales.assign(
        down_parameter_count, bfloat16(0.0F));
    fixture.down_biases.assign(
        down_parameter_count, bfloat16(0.0F));

    for (std::uint32_t position = 0U;
         position < kNumericPositionCount; ++position) {
        for (std::uint32_t group = 0U;
             group < kHidden / kGroupSize; ++group) {
            const std::uint32_t reduction =
                group * kGroupSize +
                (position * 11U + group * 13U + 5U) % kGroupSize;
            fixture.input[
                std::size_t{position} * kHidden + reduction] =
                bfloat16(
                    0.25F +
                    static_cast<float>(
                        (position * 5U + group * 3U) % 11U) *
                        0.03125F);
        }
    }

    for (std::uint32_t expert = 0U;
         expert < stored_experts; ++expert) {
        for (std::uint32_t column = 0U;
             column < kExpertDimension; ++column) {
            if (!selected_up_column(column)) {
                continue;
            }
            for (std::uint32_t group = 0U;
                 group < kHidden / kGroupSize; ++group) {
                const std::size_t metadata = parameter_index(
                    expert, column, group * kGroupSize,
                    kExpertDimension, kHidden);
                fixture.gate_scales[metadata] = bfloat16(
                    0.006F +
                    0.0007F *
                        static_cast<float>(
                            (expert + column + group) % 7U));
                fixture.gate_biases[metadata] = bfloat16(
                    -0.031F +
                    0.002F *
                        static_cast<float>((column + group) % 5U));
                fixture.up_scales[metadata] = bfloat16(
                    0.0045F +
                    0.0006F *
                        static_cast<float>(
                            (expert * 2U + column + group) % 9U));
                fixture.up_biases[metadata] = bfloat16(
                    -0.017F +
                    0.0015F *
                        static_cast<float>(
                            (expert + column + 2U * group) % 7U));
                for (std::uint32_t inner = 0U;
                     inner < kGroupSize; ++inner) {
                    const std::uint32_t reduction =
                        group * kGroupSize + inner;
                    set_q4(
                        fixture.gate_packed, expert, column,
                        reduction, kExpertDimension, kHidden,
                        (expert * 3U + column * 5U +
                         reduction * 7U + 1U) &
                            15U);
                    set_q4(
                        fixture.up_packed, expert, column,
                        reduction, kExpertDimension, kHidden,
                        (expert * 5U + column * 3U +
                         reduction * 11U + 2U) &
                            15U);
                }
            }
        }

        for (std::uint32_t column = 0U; column < kHidden; ++column) {
            if (!selected_down_column(column)) {
                continue;
            }
            for (std::uint32_t group = 0U;
                 group < kExpertDimension / kGroupSize; ++group) {
                const std::size_t metadata = parameter_index(
                    expert, column, group * kGroupSize,
                    kHidden, kExpertDimension);
                fixture.down_scales[metadata] = bfloat16(
                    0.005F +
                    0.0005F *
                        static_cast<float>(
                            (expert + column + group) % 8U));
                fixture.down_biases[metadata] = bfloat16(
                    -0.013F +
                    0.001F *
                        static_cast<float>(
                            (expert + 2U * column + group) % 6U));
                for (std::uint32_t inner = 0U;
                     inner < kGroupSize; ++inner) {
                    const std::uint32_t reduction =
                        group * kGroupSize + inner;
                    set_q4(
                        fixture.down_packed, expert, column,
                        reduction, kHidden, kExpertDimension,
                        (expert * 7U + column * 3U +
                         reduction * 5U + 4U) &
                            15U);
                }
            }
        }
    }

    const std::size_t hidden_values =
        std::size_t{kNumericPositionCount} * kActiveExperts *
        kExpertDimension;
    const std::size_t partial_values =
        std::size_t{kNumericPositionCount} * kActiveExperts *
        kHidden;
    const std::size_t assimilated_hidden_values =
        std::size_t{kNumericPositionCount} * kSlotCount *
        kExpertDimension;
    const std::size_t assimilated_partial_values =
        std::size_t{kNumericPositionCount} * kSlotCount * kHidden;
    fixture.fused_expected.resize(hidden_values);
    fixture.split_expected.resize(hidden_values);
    fixture.staged_expected.resize(hidden_values);
    fixture.assimilated_expected.resize(assimilated_hidden_values);
    fixture.down_expected.resize(partial_values);
    fixture.staged_down_expected.resize(partial_values);
    fixture.assimilated_down_expected.resize(
        assimilated_partial_values);

    for (std::uint32_t position = 0U;
         position < kNumericPositionCount; ++position) {
        for (std::uint32_t slot = 0U;
             slot < kActiveExperts; ++slot) {
            const std::uint32_t expert =
                routes.expert_for_position_slot[
                    std::size_t{position} * kActiveExperts + slot];
            for (std::uint32_t column = 0U;
                 column < kExpertDimension; ++column) {
                const float gate = sparse_up_dot(
                    fixture, fixture.gate_packed,
                    fixture.gate_scales, fixture.gate_biases,
                    position, expert, column);
                const float up = sparse_up_dot(
                    fixture, fixture.up_packed,
                    fixture.up_scales, fixture.up_biases,
                    position, expert, column);
                const float staged_gate = sparse_up_dot(
                    fixture, fixture.gate_packed,
                    fixture.gate_scales, fixture.gate_biases,
                    position, expert, column);
                const float staged_up = sparse_up_dot(
                    fixture, fixture.up_packed,
                    fixture.up_scales, fixture.up_biases,
                    position, expert, column);
                const std::size_t index =
                    (std::size_t{position} * kActiveExperts + slot) *
                        kExpertDimension +
                    column;
                fixture.fused_expected[index] =
                    bfloat16(swiglu(gate, up));
                fixture.split_expected[index] =
                    bfloat16(swiglu(
                        from_bfloat16(bfloat16(gate)), up));
                const std::uint16_t staged =
                    bfloat16(swiglu(staged_gate, staged_up));
                fixture.staged_expected[index] = staged;
                fixture.assimilated_expected[
                    (std::size_t{position} * kSlotCount + slot) *
                        kExpertDimension +
                    column] = staged;
                fixture.family_differences +=
                    static_cast<std::size_t>(
                        fixture.fused_expected[index] !=
                        fixture.split_expected[index]);
                fixture.direct_staged_differences +=
                    static_cast<std::size_t>(
                        fixture.fused_expected[index] != staged);
            }
        }
    }

    for (std::uint32_t position = 0U;
         position < kNumericPositionCount; ++position) {
        for (std::uint32_t slot = 0U;
             slot < kActiveExperts; ++slot) {
            const std::uint32_t expert =
                routes.expert_for_position_slot[
                    std::size_t{position} * kActiveExperts + slot];
            for (std::uint32_t column = 0U; column < kHidden; ++column) {
                fixture.down_expected[
                    (std::size_t{position} * kActiveExperts + slot) *
                        kHidden +
                    column] =
                    sparse_down_dot(
                        fixture, fixture.split_expected,
                        position, slot, kActiveExperts, expert,
                        column);
                fixture.staged_down_expected[
                    (std::size_t{position} * kActiveExperts + slot) *
                        kHidden +
                    column] =
                    sparse_down_dot(
                        fixture, fixture.staged_expected,
                        position, slot, kActiveExperts, expert,
                        column);
                fixture.assimilated_down_expected[
                    (std::size_t{position} * kSlotCount + slot) *
                        kHidden +
                    column] =
                    fixture.staged_down_expected[
                        (std::size_t{position} * kActiveExperts + slot) *
                            kHidden +
                        column];
            }
        }
    }
    constexpr std::uint32_t kSharedFixtureWeightExpert =
        kActiveExperts;
    for (std::uint32_t position = 0U;
         position < kNumericPositionCount; ++position) {
        for (std::uint32_t column = 0U;
             column < kExpertDimension; ++column) {
            const float gate = sparse_up_dot(
                fixture, fixture.gate_packed,
                fixture.gate_scales, fixture.gate_biases,
                position, kSharedFixtureWeightExpert, column);
            const float up = sparse_up_dot(
                fixture, fixture.up_packed,
                fixture.up_scales, fixture.up_biases,
                position, kSharedFixtureWeightExpert, column);
            fixture.assimilated_expected[
                (std::size_t{position} * kSlotCount +
                 kActiveExperts) *
                    kExpertDimension +
                column] = bfloat16(swiglu(gate, up));
        }
    }
    for (std::uint32_t position = 0U;
         position < kNumericPositionCount; ++position) {
        for (std::uint32_t column = 0U; column < kHidden; ++column) {
            fixture.assimilated_down_expected[
                (std::size_t{position} * kSlotCount +
                 kActiveExperts) *
                    kHidden +
                column] = sparse_down_dot(
                    fixture, fixture.assimilated_expected,
                    position, kActiveExperts, kSlotCount,
                    kSharedFixtureWeightExpert, column);
        }
    }
    return fixture.family_differences != 0U &&
           fixture.direct_staged_differences == 0U;
}

bool numeric_fixture_is_finite(const NumericFixture& fixture) {
    for (const std::uint16_t value : fixture.fused_expected) {
        if (!std::isfinite(from_bfloat16(value))) {
            return false;
        }
    }
    for (const std::uint16_t value : fixture.split_expected) {
        if (!std::isfinite(from_bfloat16(value))) {
            return false;
        }
    }
    for (const std::uint16_t value : fixture.staged_expected) {
        if (!std::isfinite(from_bfloat16(value))) {
            return false;
        }
    }
    for (const std::uint16_t value : fixture.assimilated_expected) {
        if (!std::isfinite(from_bfloat16(value))) {
            return false;
        }
    }
    return std::all_of(
               fixture.down_expected.begin(),
               fixture.down_expected.end(),
               [](float value) { return std::isfinite(value); }) &&
           std::all_of(
               fixture.staged_down_expected.begin(),
               fixture.staged_down_expected.end(),
               [](float value) { return std::isfinite(value); }) &&
           std::all_of(
               fixture.assimilated_down_expected.begin(),
               fixture.assimilated_down_expected.end(),
               [](float value) { return std::isfinite(value); });
}

struct GuardedBuffer {
    MetalBuffer buffer;
    std::size_t body_bytes = 0U;
    std::vector<std::byte> initial;
    std::vector<std::byte> captured;
};

enum class GuardedBufferError : std::uint8_t {
    None,
    Allocation,
    Initialization,
};

GuardedBufferError initialize_guarded_buffer(
    const MetalDevice& device, std::span<const std::byte> body,
    GuardedBuffer& output) {
    std::size_t with_prefix = 0U;
    std::size_t total = 0U;
    if (body.empty() ||
        !checked_add(kGuardBytes, body.size(), with_prefix) ||
        !checked_add(with_prefix, kGuardBytes, total)) {
        return GuardedBufferError::Initialization;
    }
    auto created = create_shared_buffer(device, total);
    if (!created) {
        return GuardedBufferError::Allocation;
    }
    output = {};
    output.body_bytes = body.size();
    output.initial.assign(total, kGuardCanary);
    std::copy(
        body.begin(), body.end(),
        output.initial.begin() +
            static_cast<std::ptrdiff_t>(kGuardBytes));
    output.buffer = std::move(*created.buffer);
    if (!output.buffer || output.buffer.size_bytes() != total ||
        output.buffer.contents() == nullptr) {
        return GuardedBufferError::Initialization;
    }
    std::memcpy(
        output.buffer.contents(), output.initial.data(),
        output.initial.size());
    return GuardedBufferError::None;
}

template <typename Value>
GuardedBufferError initialize_guarded_values(
    const MetalDevice& device, std::span<const Value> body,
    GuardedBuffer& output) {
    return initialize_guarded_buffer(
        device, std::as_bytes(body), output);
}

std::span<const std::byte> current_bytes(
    const GuardedBuffer& buffer) {
    return {
        static_cast<const std::byte*>(buffer.buffer.contents()),
        buffer.initial.size()};
}

template <typename Value>
std::span<const Value> current_body(
    const GuardedBuffer& buffer) {
    return {
        reinterpret_cast<const Value*>(
            static_cast<const std::byte*>(buffer.buffer.contents()) +
            kGuardBytes),
        buffer.body_bytes / sizeof(Value)};
}

bool guards_intact(const GuardedBuffer& buffer) {
    const auto current = current_bytes(buffer);
    if (current.size() != buffer.initial.size()) {
        return false;
    }
    return std::all_of(
               current.begin(),
               current.begin() +
                   static_cast<std::ptrdiff_t>(kGuardBytes),
               [](std::byte value) {
                   return value == kGuardCanary;
               }) &&
           std::all_of(
               current.end() -
                   static_cast<std::ptrdiff_t>(kGuardBytes),
               current.end(),
               [](std::byte value) {
                   return value == kGuardCanary;
               });
}

bool unchanged_from_initial(const GuardedBuffer& buffer) {
    const auto current = current_bytes(buffer);
    return current.size() == buffer.initial.size() &&
           std::equal(
               current.begin(), current.end(),
               buffer.initial.begin());
}

void capture_buffer(GuardedBuffer& buffer) {
    const auto current = current_bytes(buffer);
    buffer.captured.assign(current.begin(), current.end());
}

bool unchanged_from_capture(const GuardedBuffer& buffer) {
    const auto current = current_bytes(buffer);
    return !buffer.captured.empty() &&
           current.size() == buffer.captured.size() &&
           std::equal(
               current.begin(), current.end(),
               buffer.captured.begin());
}

int guarded_error(GuardedBufferError error) noexcept {
    switch (error) {
    case GuardedBufferError::None:
        return 0;
    case GuardedBufferError::Allocation:
        return kExitBuffer;
    case GuardedBufferError::Initialization:
        return kExitInitialization;
    }
    return kExitInitialization;
}

struct ProducerBuffers {
    GuardedBuffer counts;
    GuardedBuffer lists;
    GuardedBuffer tasks;
    GuardedBuffer grid;
    GuardedBuffer status;
};

int initialize_producer_buffers(
    const MetalDevice& device, const RouteFixture& fixture,
    ProducerBuffers& buffers) {
    std::vector<QuantizedGemmTaskDescriptor> task_poison(
        fixture.planned_task_capacity,
        {
            .expert_index = kTaskCanary,
            .route_list_begin = kTaskCanary,
            .row_count = kTaskCanary,
            .output_row_begin = kTaskCanary,
        });
    constexpr std::array<std::uint32_t, 3> kGridPoison{
        kWordCanary, kWordCanary, kWordCanary};
    constexpr std::array<std::uint32_t, 1> kStatusPoison{
        kWordCanary};
    const std::array<GuardedBufferError, 5> results{
        initialize_guarded_values(
            device, std::span<const std::uint32_t>{fixture.counts},
            buffers.counts),
        initialize_guarded_values(
            device, std::span<const std::uint32_t>{fixture.lists},
            buffers.lists),
        initialize_guarded_values(
            device,
            std::span<const QuantizedGemmTaskDescriptor>{task_poison},
            buffers.tasks),
        initialize_guarded_values(
            device, std::span<const std::uint32_t>{kGridPoison},
            buffers.grid),
        initialize_guarded_values(
            device, std::span<const std::uint32_t>{kStatusPoison},
            buffers.status),
    };
    for (const GuardedBufferError result : results) {
        if (result != GuardedBufferError::None) {
            return guarded_error(result);
        }
    }
    return 0;
}

int initialize_resolved_producer_buffers(
    const MetalDevice& device, const RouteFixture& fixture,
    std::uint32_t column_groups, ProducerBuffers& buffers) {
    std::vector<QuantizedGemmTaskDescriptor> tasks(
        fixture.planned_task_capacity,
        {
            .expert_index = kTaskCanary,
            .route_list_begin = kTaskCanary,
            .row_count = kTaskCanary,
            .output_row_begin = kTaskCanary,
        });
    if (fixture.tasks.size() > tasks.size() ||
        column_groups == 0U) {
        return kExitInitialization;
    }
    std::copy(
        fixture.tasks.begin(), fixture.tasks.end(), tasks.begin());
    const std::array<std::uint32_t, 3> grid{
        static_cast<std::uint32_t>(fixture.tasks.size()),
        column_groups,
        1U,
    };
    const std::array<std::uint32_t, 1> status{
        static_cast<std::uint32_t>(
            QuantizedGemmDeviceTaskStatus::Ready),
    };
    const std::array<GuardedBufferError, 5> results{
        initialize_guarded_values(
            device, std::span<const std::uint32_t>{fixture.counts},
            buffers.counts),
        initialize_guarded_values(
            device, std::span<const std::uint32_t>{fixture.lists},
            buffers.lists),
        initialize_guarded_values(
            device,
            std::span<const QuantizedGemmTaskDescriptor>{tasks},
            buffers.tasks),
        initialize_guarded_values(
            device, std::span<const std::uint32_t>{grid},
            buffers.grid),
        initialize_guarded_values(
            device, std::span<const std::uint32_t>{status},
            buffers.status),
    };
    for (const GuardedBufferError result : results) {
        if (result != GuardedBufferError::None) {
            return guarded_error(result);
        }
    }
    return 0;
}

struct ComputeBuffers {
    GuardedBuffer input;
    GuardedBuffer gate_packed;
    GuardedBuffer gate_scales;
    GuardedBuffer gate_biases;
    GuardedBuffer up_packed;
    GuardedBuffer up_scales;
    GuardedBuffer up_biases;
    GuardedBuffer down_packed;
    GuardedBuffer down_scales;
    GuardedBuffer down_biases;
    GuardedBuffer fused_hidden;
    GuardedBuffer split_hidden;
    GuardedBuffer partials;
    GuardedBuffer staged_hidden;
    GuardedBuffer staged_partials;
    GuardedBuffer assimilated_hidden;
    GuardedBuffer assimilated_partials;
    GuardedBuffer bm32_hidden;
    GuardedBuffer bm32_partials;
    GuardedBuffer bk64_hidden;
    GuardedBuffer bk64_partials;
    GuardedBuffer bn64_hidden;
    GuardedBuffer bn64_partials;
};

int initialize_compute_buffers(
    const MetalDevice& device, const NumericFixture& fixture,
    ComputeBuffers& buffers) {
    const std::size_t hidden_count =
        std::size_t{kNumericPositionCapacity} * kSlotCount *
        kExpertDimension;
    const std::size_t partial_count =
        std::size_t{kNumericPositionCapacity} * kSlotCount * kHidden;
    std::vector<std::uint16_t> hidden_poison(
        hidden_count, kHiddenCanary);
    std::vector<std::uint32_t> partial_poison(
        partial_count, kPartialCanary);

    const std::array<GuardedBufferError, 23> results{
        initialize_guarded_values(
            device, std::span<const std::uint16_t>{fixture.input},
            buffers.input),
        initialize_guarded_values(
            device, std::span<const std::uint32_t>{fixture.gate_packed},
            buffers.gate_packed),
        initialize_guarded_values(
            device, std::span<const std::uint16_t>{fixture.gate_scales},
            buffers.gate_scales),
        initialize_guarded_values(
            device, std::span<const std::uint16_t>{fixture.gate_biases},
            buffers.gate_biases),
        initialize_guarded_values(
            device, std::span<const std::uint32_t>{fixture.up_packed},
            buffers.up_packed),
        initialize_guarded_values(
            device, std::span<const std::uint16_t>{fixture.up_scales},
            buffers.up_scales),
        initialize_guarded_values(
            device, std::span<const std::uint16_t>{fixture.up_biases},
            buffers.up_biases),
        initialize_guarded_values(
            device, std::span<const std::uint32_t>{fixture.down_packed},
            buffers.down_packed),
        initialize_guarded_values(
            device, std::span<const std::uint16_t>{fixture.down_scales},
            buffers.down_scales),
        initialize_guarded_values(
            device, std::span<const std::uint16_t>{fixture.down_biases},
            buffers.down_biases),
        initialize_guarded_values(
            device, std::span<const std::uint16_t>{hidden_poison},
            buffers.fused_hidden),
        initialize_guarded_values(
            device, std::span<const std::uint16_t>{hidden_poison},
            buffers.split_hidden),
        initialize_guarded_values(
            device, std::span<const std::uint32_t>{partial_poison},
            buffers.partials),
        initialize_guarded_values(
            device, std::span<const std::uint16_t>{hidden_poison},
            buffers.staged_hidden),
        initialize_guarded_values(
            device, std::span<const std::uint32_t>{partial_poison},
            buffers.staged_partials),
        initialize_guarded_values(
            device, std::span<const std::uint16_t>{hidden_poison},
            buffers.assimilated_hidden),
        initialize_guarded_values(
            device, std::span<const std::uint32_t>{partial_poison},
            buffers.assimilated_partials),
        initialize_guarded_values(
            device, std::span<const std::uint16_t>{hidden_poison},
            buffers.bm32_hidden),
        initialize_guarded_values(
            device, std::span<const std::uint32_t>{partial_poison},
            buffers.bm32_partials),
        initialize_guarded_values(
            device, std::span<const std::uint16_t>{hidden_poison},
            buffers.bk64_hidden),
        initialize_guarded_values(
            device, std::span<const std::uint32_t>{partial_poison},
            buffers.bk64_partials),
        initialize_guarded_values(
            device, std::span<const std::uint16_t>{hidden_poison},
            buffers.bn64_hidden),
        initialize_guarded_values(
            device, std::span<const std::uint32_t>{partial_poison},
            buffers.bn64_partials),
    };
    for (const GuardedBufferError result : results) {
        if (result != GuardedBufferError::None) {
            return guarded_error(result);
        }
    }
    return 0;
}

struct GpuResources {
    ProducerBuffers high;
    ProducerBuffers high_bn64;
    ProducerBuffers high_bm32;
    ProducerBuffers numeric_up;
    ProducerBuffers numeric_down;
    ProducerBuffers numeric_bn64_up;
    ProducerBuffers numeric_bn64_down;
    ProducerBuffers numeric_bm32_up;
    ProducerBuffers numeric_bm32_down;
    ProducerBuffers numeric_shared_up;
    ProducerBuffers numeric_shared_down;
    ProducerBuffers numeric_shared_bn64_up;
    ProducerBuffers numeric_shared_bn64_down;
    ProducerBuffers numeric_shared_bm32_up;
    ProducerBuffers numeric_shared_bm32_down;
    ComputeBuffers compute;
};

int initialize_gpu_resources(
    const MetalDevice& device, const RouteFixture& high,
    const RouteFixture& numeric,
    const RouteFixture& numeric_shared,
    const RouteFixture& high_bm32,
    const RouteFixture& numeric_bm32,
    const RouteFixture& numeric_shared_bm32,
    const NumericFixture& fixture,
    GpuResources& resources) {
    int result = initialize_producer_buffers(
        device, high, resources.high);
    if (result == 0) {
        result = initialize_resolved_producer_buffers(
            device, high, kBn64UpColumnGroups,
            resources.high_bn64);
    }
    if (result == 0) {
        result = initialize_resolved_producer_buffers(
            device, high_bm32, kUpColumnGroups,
            resources.high_bm32);
    }
    if (result == 0) {
        result = initialize_producer_buffers(
            device, numeric, resources.numeric_up);
    }
    if (result == 0) {
        result = initialize_producer_buffers(
            device, numeric, resources.numeric_down);
    }
    if (result == 0) {
        result = initialize_resolved_producer_buffers(
            device, numeric, kBn64UpColumnGroups,
            resources.numeric_bn64_up);
    }
    if (result == 0) {
        result = initialize_resolved_producer_buffers(
            device, numeric, kBn64DownColumnGroups,
            resources.numeric_bn64_down);
    }
    if (result == 0) {
        result = initialize_resolved_producer_buffers(
            device, numeric_bm32, kUpColumnGroups,
            resources.numeric_bm32_up);
    }
    if (result == 0) {
        result = initialize_resolved_producer_buffers(
            device, numeric_bm32, kDownColumnGroups,
            resources.numeric_bm32_down);
    }
    if (result == 0) {
        result = initialize_producer_buffers(
            device, numeric_shared, resources.numeric_shared_up);
    }
    if (result == 0) {
        result = initialize_producer_buffers(
            device, numeric_shared, resources.numeric_shared_down);
    }
    if (result == 0) {
        result = initialize_resolved_producer_buffers(
            device, numeric_shared, kBn64UpColumnGroups,
            resources.numeric_shared_bn64_up);
    }
    if (result == 0) {
        result = initialize_resolved_producer_buffers(
            device, numeric_shared, kBn64DownColumnGroups,
            resources.numeric_shared_bn64_down);
    }
    if (result == 0) {
        result = initialize_resolved_producer_buffers(
            device, numeric_shared_bm32, kUpColumnGroups,
            resources.numeric_shared_bm32_up);
    }
    if (result == 0) {
        result = initialize_resolved_producer_buffers(
            device, numeric_shared_bm32, kDownColumnGroups,
            resources.numeric_shared_bm32_down);
    }
    if (result == 0) {
        result = initialize_compute_buffers(
            device, fixture, resources.compute);
    }
    return result;
}

struct Pipelines {
    MetalComputePipeline producer;
    MetalComputePipeline fused;
    MetalComputePipeline gate;
    MetalComputePipeline up;
    MetalComputePipeline down;
    MetalComputePipeline staged_fused;
    MetalComputePipeline staged_down;
    MetalComputePipeline steel_upgate;
    MetalComputePipeline steel_down;
    MetalComputePipeline steel_bm32_upgate;
    MetalComputePipeline steel_bm32_down;
    MetalComputePipeline steel_bk64_upgate;
    MetalComputePipeline steel_bk64_down;
    MetalComputePipeline steel_bn64_upgate;
    MetalComputePipeline steel_bn64_down;
};

struct BenchmarkPipelines {
    MetalComputePipeline exact_upgate;
    MetalComputePipeline exact_down;
};

int load_pipeline(
    const MetalDevice& device, const MetalLibrary& library,
    std::string_view name, int pipeline_exit,
    MetalComputePipeline& output) {
    auto function = create_function(library, name);
    if (!function) {
        return kExitFunction;
    }
    auto pipeline =
        create_compute_pipeline(device, *function.function);
    if (!pipeline) {
        std::cerr << pipeline.failure_description << '\n';
        return pipeline_exit;
    }
    output = std::move(*pipeline.pipeline);
    return 0;
}

int load_pipelines(
    const MetalDevice& device, const MetalLibrary& library,
    Pipelines& pipelines) {
    int result = load_pipeline(
        device, library, kProducerName, kExitProducerPipeline,
        pipelines.producer);
    if (result == 0) {
        result = load_pipeline(
            device, library, kFusedName, kExitFusedPipeline,
            pipelines.fused);
    }
    if (result == 0) {
        result = load_pipeline(
            device, library, kGateName, kExitGatePipeline,
            pipelines.gate);
    }
    if (result == 0) {
        result = load_pipeline(
            device, library, kUpName, kExitUpPipeline,
            pipelines.up);
    }
    if (result == 0) {
        result = load_pipeline(
            device, library, kDownName, kExitDownPipeline,
            pipelines.down);
    }
    if (result == 0) {
        result = load_pipeline(
            device, library,
            "native_routed_qgemm_r2_fused_upgate_swiglu",
            kExitFusedPipeline, pipelines.staged_fused);
    }
    if (result == 0) {
        result = load_pipeline(
            device, library,
            "native_routed_qgemm_r2_down_partial",
            kExitDownPipeline, pipelines.staged_down);
    }
    if constexpr (generated::kKernelLibraryMlxSteelEnabled) {
        if (result == 0) {
            result = load_pipeline(
                device, library, kSteelUpgateName,
                kExitFusedPipeline, pipelines.steel_upgate);
        }
        if (result == 0) {
            result = load_pipeline(
                device, library, kSteelDownName,
                kExitDownPipeline, pipelines.steel_down);
        }
        if (result == 0) {
            result = load_pipeline(
                device, library, kSteelBm32UpgateName,
                kExitFusedPipeline,
                pipelines.steel_bm32_upgate);
        }
        if (result == 0) {
            result = load_pipeline(
                device, library, kSteelBm32DownName,
                kExitDownPipeline,
                pipelines.steel_bm32_down);
        }
        if (result == 0) {
            result = load_pipeline(
                device, library, kSteelBk64UpgateName,
                kExitFusedPipeline,
                pipelines.steel_bk64_upgate);
        }
        if (result == 0) {
            result = load_pipeline(
                device, library, kSteelBk64DownName,
                kExitDownPipeline,
                pipelines.steel_bk64_down);
        }
        if (result == 0) {
            result = load_pipeline(
                device, library, kSteelBn64UpgateName,
                kExitFusedPipeline,
                pipelines.steel_bn64_upgate);
        }
        if (result == 0) {
            result = load_pipeline(
                device, library, kSteelBn64DownName,
                kExitDownPipeline,
                pipelines.steel_bn64_down);
        }
    }
    return result;
}

int load_benchmark_pipelines(
    const MetalDevice& device, const MetalLibrary& library,
    BenchmarkPipelines& pipelines) {
    int result = load_pipeline(
        device, library, "block_upgate", kExitBenchmarkPipeline,
        pipelines.exact_upgate);
    if (result == 0) {
        result = load_pipeline(
            device, library, "block_down_partial",
            kExitBenchmarkPipeline, pipelines.exact_down);
    }
    return result;
}

bool encoded(MetalCommandError error) noexcept {
    return error == MetalCommandError::None;
}

bool encode_producer_dispatch(
    MetalComputePass& pass, const MetalComputePipeline& pipeline,
    const ProducerBuffers& buffers, const RouteFixture& fixture,
    std::uint32_t column_groups,
    std::uint32_t& direct_dispatches) {
    const std::uint32_t position_count = fixture.position_count;
    const std::uint32_t include_shared_expert =
        static_cast<std::uint32_t>(fixture.include_shared_expert);
    const std::uint32_t expert_count =
        kExperts + include_shared_expert;
    const std::uint32_t active_expert_count =
        kActiveExperts + include_shared_expert;
    const std::uint32_t route_list_expert_stride =
        fixture.list_stride;
    const std::uint32_t position_capacity =
        fixture.position_capacity;
    const std::uint64_t route_list_total_extent =
        std::uint64_t{expert_count} * fixture.list_stride;
    const std::uint32_t planned_task_capacity =
        fixture.planned_task_capacity;
    const std::uint32_t packed_slot_bits = kPackedSlotBits;
    if (!encoded(set_compute_pipeline(pass, pipeline)) ||
        !encoded(set_buffer(
            pass, buffers.counts.buffer, kGuardBytes, 0)) ||
        !encoded(set_buffer(
            pass, buffers.lists.buffer, kGuardBytes, 1)) ||
        !encoded(set_buffer(
            pass, buffers.tasks.buffer, kGuardBytes, 2)) ||
        !encoded(set_buffer(
            pass, buffers.grid.buffer, kGuardBytes, 3)) ||
        !encoded(set_buffer(
            pass, buffers.status.buffer, kGuardBytes, 4)) ||
        !encoded(set_bytes(
            pass, &position_count, sizeof(position_count), 5)) ||
        !encoded(set_bytes(
            pass, &expert_count, sizeof(expert_count), 6)) ||
        !encoded(set_bytes(
            pass, &active_expert_count,
            sizeof(active_expert_count), 7)) ||
        !encoded(set_bytes(
            pass, &route_list_expert_stride,
            sizeof(route_list_expert_stride), 8)) ||
        !encoded(set_bytes(
            pass, &position_capacity,
            sizeof(position_capacity), 9)) ||
        !encoded(set_bytes(
            pass, &route_list_total_extent,
            sizeof(route_list_total_extent), 10)) ||
        !encoded(set_bytes(
            pass, &planned_task_capacity,
            sizeof(planned_task_capacity), 11)) ||
        !encoded(set_bytes(
            pass, &column_groups, sizeof(column_groups), 12)) ||
        !encoded(set_bytes(
            pass, &packed_slot_bits,
            sizeof(packed_slot_bits), 13)) ||
        !encoded(set_bytes(
            pass, &include_shared_expert,
            sizeof(include_shared_expert), 14)) ||
        !encoded(dispatch_threadgroups(
            pass, {.width = 1, .height = 1, .depth = 1},
            {.width = 1, .height = 1, .depth = 1}))) {
        return false;
    }
    ++direct_dispatches;
    return true;
}

int run_producers(
    const MetalCommandQueue& queue,
    const MetalComputePipeline& producer,
    const RouteFixture& high, const RouteFixture& numeric,
    const RouteFixture* numeric_shared,
    GpuResources& resources, std::uint32_t& command_buffers,
    std::uint32_t& direct_dispatches) {
    auto command = create_command_buffer(queue);
    if (!command) {
        return kExitCommandBuffer;
    }
    auto pass = begin_compute_pass(std::move(*command.command_buffer));
    if (!pass) {
        return kExitComputePass;
    }
    if (!encode_producer_dispatch(
            *pass.compute_pass, producer, resources.high, high,
            kUpColumnGroups, direct_dispatches) ||
        !encode_producer_dispatch(
            *pass.compute_pass, producer, resources.numeric_up,
            numeric, kUpColumnGroups, direct_dispatches) ||
        !encode_producer_dispatch(
            *pass.compute_pass, producer, resources.numeric_down,
            numeric, kDownColumnGroups, direct_dispatches) ||
        (numeric_shared != nullptr &&
         (!encode_producer_dispatch(
              *pass.compute_pass, producer,
              resources.numeric_shared_up, *numeric_shared,
              kUpColumnGroups, direct_dispatches) ||
          !encode_producer_dispatch(
              *pass.compute_pass, producer,
              resources.numeric_shared_down, *numeric_shared,
              kDownColumnGroups, direct_dispatches)))) {
        return kExitEncode;
    }
    auto ended = end_compute_pass(std::move(*pass.compute_pass));
    if (!ended) {
        return kExitEndPass;
    }
    auto pending = commit(std::move(*ended.command_buffer));
    if (!pending) {
        return kExitCommit;
    }
    ++command_buffers;
    auto completed =
        wait_until_completed(std::move(*pending.pending_execution));
    if (!completed) {
        std::cerr << "R1 producer execution failed: "
                  << completed.failure_description.view() << '\n';
        return kExitExecution;
    }
    return 0;
}

int producer_status_exit(std::uint32_t raw) noexcept {
    switch (static_cast<QuantizedGemmDeviceTaskStatus>(raw)) {
    case QuantizedGemmDeviceTaskStatus::NotProduced:
        return kExitProducerNotProduced;
    case QuantizedGemmDeviceTaskStatus::Ready:
        return 0;
    case QuantizedGemmDeviceTaskStatus::CountOutOfRange:
        return kExitProducerCount;
    case QuantizedGemmDeviceTaskStatus::RouteConservationFailure:
        return kExitProducerConservation;
    case QuantizedGemmDeviceTaskStatus::TaskCapacityExceeded:
        return kExitProducerCapacity;
    case QuantizedGemmDeviceTaskStatus::PackedSlotOutOfRange:
        return kExitProducerSlot;
    }
    return kExitProducerUnknown;
}

bool producer_guards_and_read_only_intact(
    const ProducerBuffers& buffers) {
    return guards_intact(buffers.counts) &&
           guards_intact(buffers.lists) &&
           guards_intact(buffers.tasks) &&
           guards_intact(buffers.grid) &&
           guards_intact(buffers.status) &&
           unchanged_from_initial(buffers.counts) &&
           unchanged_from_initial(buffers.lists);
}

int validate_producer_output(
    const RouteFixture& fixture, std::uint32_t column_groups,
    const ProducerBuffers& buffers) {
    const auto status = current_body<std::uint32_t>(buffers.status);
    if (status.size() != 1U) {
        return kExitCanary;
    }
    const int status_result = producer_status_exit(status[0]);
    if (status_result != 0) {
        return status_result;
    }
    if (!producer_guards_and_read_only_intact(buffers)) {
        return kExitCanary;
    }

    const auto tasks =
        current_body<QuantizedGemmTaskDescriptor>(buffers.tasks);
    if (tasks.size() != fixture.planned_task_capacity ||
        !valid_task_partition(
            fixture, tasks.first(fixture.tasks.size()))) {
        return kExitProducerPartition;
    }
    for (std::size_t index = 0U;
         index < fixture.tasks.size(); ++index) {
        if (tasks[index].expert_index !=
                fixture.tasks[index].expert_index ||
            tasks[index].route_list_begin !=
                fixture.tasks[index].route_list_begin ||
            tasks[index].row_count !=
                fixture.tasks[index].row_count ||
            tasks[index].output_row_begin != 0U) {
            return kExitProducerPartition;
        }
    }
    for (std::size_t index = fixture.tasks.size();
         index < tasks.size(); ++index) {
        if (tasks[index].expert_index != kTaskCanary ||
            tasks[index].route_list_begin != kTaskCanary ||
            tasks[index].row_count != kTaskCanary ||
            tasks[index].output_row_begin != kTaskCanary) {
            return kExitCanary;
        }
    }

    const auto grid = current_body<std::uint32_t>(buffers.grid);
    if (grid.size() != 3U ||
        grid[0] != fixture.tasks.size() ||
        grid[1] != column_groups || grid[2] != 1U) {
        return kExitProducerGrid;
    }
    return 0;
}

void capture_producer(ProducerBuffers& buffers) {
    capture_buffer(buffers.counts);
    capture_buffer(buffers.lists);
    capture_buffer(buffers.tasks);
    capture_buffer(buffers.grid);
    capture_buffer(buffers.status);
}

bool producer_unchanged_after_compute(
    const ProducerBuffers& buffers) {
    return unchanged_from_capture(buffers.counts) &&
           unchanged_from_capture(buffers.lists) &&
           unchanged_from_capture(buffers.tasks) &&
           unchanged_from_capture(buffers.grid) &&
           unchanged_from_capture(buffers.status);
}

bool numeric_task_bodies_identical(
    const ProducerBuffers& up, const ProducerBuffers& down) {
    const auto up_tasks =
        current_body<QuantizedGemmTaskDescriptor>(up.tasks);
    const auto down_tasks =
        current_body<QuantizedGemmTaskDescriptor>(down.tasks);
    return up_tasks.size() == down_tasks.size() &&
           std::equal(
               up_tasks.begin(), up_tasks.end(),
               down_tasks.begin(),
               [](const auto& left, const auto& right) {
                   return left.expert_index == right.expert_index &&
                          left.route_list_begin ==
                              right.route_list_begin &&
                          left.row_count == right.row_count &&
                          left.output_row_begin ==
                              right.output_row_begin;
               });
}

int run_compute(
    const MetalCommandQueue& queue, const Pipelines& pipelines,
    const RouteFixture& numeric, GpuResources& resources,
    std::uint32_t& command_buffers,
    std::uint32_t& indirect_dispatches,
    std::uint32_t& barriers,
    std::uint64_t& indirect_threadgroups) {
    auto command = create_command_buffer(queue);
    if (!command) {
        return kExitCommandBuffer;
    }
    auto pass = begin_compute_pass(std::move(*command.command_buffer));
    if (!pass) {
        return kExitComputePass;
    }
    const std::uint32_t route_list_expert_stride =
        numeric.list_stride;
    const std::uint32_t route_list_capacity_per_expert =
        numeric.position_capacity;
    const std::uint64_t route_list_total_extent =
        std::uint64_t{kExperts} * numeric.list_stride;
    const std::uint32_t input_rows = numeric.position_count;

    if (!encoded(set_compute_pipeline(
            *pass.compute_pass, pipelines.fused)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.compute.input.buffer,
            kGuardBytes, 0)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.numeric_up.lists.buffer,
            kGuardBytes, 1)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.numeric_up.tasks.buffer,
            kGuardBytes, 2)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.compute.gate_packed.buffer,
            kGuardBytes, 3)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.compute.gate_scales.buffer,
            kGuardBytes, 4)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.compute.gate_biases.buffer,
            kGuardBytes, 5)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.compute.up_packed.buffer,
            kGuardBytes, 6)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.compute.up_scales.buffer,
            kGuardBytes, 7)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.compute.up_biases.buffer,
            kGuardBytes, 8)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.compute.fused_hidden.buffer,
            kGuardBytes, 9)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.numeric_up.grid.buffer,
            kGuardBytes, 10)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &route_list_expert_stride,
            sizeof(route_list_expert_stride), 11)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &route_list_capacity_per_expert,
            sizeof(route_list_capacity_per_expert), 12)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &route_list_total_extent,
            sizeof(route_list_total_extent), 13)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &input_rows, sizeof(input_rows), 14)) ||
        !encoded(dispatch_threadgroups_indirect(
            *pass.compute_pass, resources.numeric_up.grid.buffer,
            kGuardBytes,
            {.width = kThreads, .height = 1, .depth = 1}))) {
        return kExitEncode;
    }
    ++indirect_dispatches;
    indirect_threadgroups +=
        std::uint64_t{kNumericExpectedTasks} * kUpColumnGroups;

    if (!encoded(set_compute_pipeline(
            *pass.compute_pass, pipelines.gate)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.compute.input.buffer,
            kGuardBytes, 0)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.numeric_up.lists.buffer,
            kGuardBytes, 1)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.numeric_up.tasks.buffer,
            kGuardBytes, 2)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.compute.gate_packed.buffer,
            kGuardBytes, 3)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.compute.gate_scales.buffer,
            kGuardBytes, 4)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.compute.gate_biases.buffer,
            kGuardBytes, 5)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.compute.split_hidden.buffer,
            kGuardBytes, 6)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.numeric_up.grid.buffer,
            kGuardBytes, 7)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &route_list_expert_stride,
            sizeof(route_list_expert_stride), 8)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &route_list_capacity_per_expert,
            sizeof(route_list_capacity_per_expert), 9)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &route_list_total_extent,
            sizeof(route_list_total_extent), 10)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &input_rows, sizeof(input_rows), 11)) ||
        !encoded(dispatch_threadgroups_indirect(
            *pass.compute_pass, resources.numeric_up.grid.buffer,
            kGuardBytes,
            {.width = kThreads, .height = 1, .depth = 1})) ||
        !encoded(memory_barrier(*pass.compute_pass))) {
        return kExitEncode;
    }
    ++indirect_dispatches;
    ++barriers;
    indirect_threadgroups +=
        std::uint64_t{kNumericExpectedTasks} * kUpColumnGroups;

    if (!encoded(set_compute_pipeline(
            *pass.compute_pass, pipelines.up)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.compute.input.buffer,
            kGuardBytes, 0)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.numeric_up.lists.buffer,
            kGuardBytes, 1)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.numeric_up.tasks.buffer,
            kGuardBytes, 2)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.compute.up_packed.buffer,
            kGuardBytes, 3)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.compute.up_scales.buffer,
            kGuardBytes, 4)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.compute.up_biases.buffer,
            kGuardBytes, 5)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.compute.split_hidden.buffer,
            kGuardBytes, 6)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.numeric_up.grid.buffer,
            kGuardBytes, 7)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &route_list_expert_stride,
            sizeof(route_list_expert_stride), 8)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &route_list_capacity_per_expert,
            sizeof(route_list_capacity_per_expert), 9)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &route_list_total_extent,
            sizeof(route_list_total_extent), 10)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &input_rows, sizeof(input_rows), 11)) ||
        !encoded(dispatch_threadgroups_indirect(
            *pass.compute_pass, resources.numeric_up.grid.buffer,
            kGuardBytes,
            {.width = kThreads, .height = 1, .depth = 1})) ||
        !encoded(memory_barrier(*pass.compute_pass))) {
        return kExitEncode;
    }
    ++indirect_dispatches;
    ++barriers;
    indirect_threadgroups +=
        std::uint64_t{kNumericExpectedTasks} * kUpColumnGroups;

    if (!encoded(set_compute_pipeline(
            *pass.compute_pass, pipelines.down)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.compute.split_hidden.buffer,
            kGuardBytes, 0)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.numeric_down.lists.buffer,
            kGuardBytes, 1)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.numeric_down.tasks.buffer,
            kGuardBytes, 2)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.compute.down_packed.buffer,
            kGuardBytes, 3)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.compute.down_scales.buffer,
            kGuardBytes, 4)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.compute.down_biases.buffer,
            kGuardBytes, 5)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.compute.partials.buffer,
            kGuardBytes, 6)) ||
        !encoded(set_buffer(
            *pass.compute_pass, resources.numeric_down.grid.buffer,
            kGuardBytes, 7)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &route_list_expert_stride,
            sizeof(route_list_expert_stride), 8)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &route_list_capacity_per_expert,
            sizeof(route_list_capacity_per_expert), 9)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &route_list_total_extent,
            sizeof(route_list_total_extent), 10)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &input_rows, sizeof(input_rows), 11)) ||
        !encoded(dispatch_threadgroups_indirect(
            *pass.compute_pass, resources.numeric_down.grid.buffer,
            kGuardBytes,
            {.width = kThreads, .height = 1, .depth = 1}))) {
        return kExitEncode;
    }
    ++indirect_dispatches;
    indirect_threadgroups +=
        std::uint64_t{kNumericExpectedTasks} * kDownColumnGroups;

    auto ended = end_compute_pass(std::move(*pass.compute_pass));
    if (!ended) {
        return kExitEndPass;
    }
    auto pending = commit(std::move(*ended.command_buffer));
    if (!pending) {
        return kExitCommit;
    }
    ++command_buffers;
    auto completed =
        wait_until_completed(std::move(*pending.pending_execution));
    if (!completed) {
        std::cerr << "R1 compute execution failed: "
                  << completed.failure_description.view() << '\n';
        return kExitExecution;
    }
    return 0;
}

int run_staged_compute(
    const MetalCommandQueue& queue, const Pipelines& pipelines,
    const RouteFixture& routes,
    const ProducerBuffers& producer_up,
    const ProducerBuffers& producer_down,
    ComputeBuffers& compute,
    GuardedBuffer& hidden,
    GuardedBuffer& partials,
    std::uint32_t& command_buffers,
    bool steel_upgate = false,
    bool steel_down = false,
    bool steel_bm32 = false,
    bool steel_bk64 = false,
    bool steel_bn64 = false) {
    auto command = create_command_buffer(queue);
    if (!command) {
        return kExitCommandBuffer;
    }
    auto pass =
        begin_compute_pass(std::move(*command.command_buffer));
    if (!pass) {
        return kExitComputePass;
    }
    const std::uint32_t route_list_expert_stride =
        routes.list_stride;
    const std::uint32_t route_list_capacity_per_expert =
        routes.position_capacity;
    const std::uint64_t route_list_total_extent =
        std::uint64_t{routes.counts.size()} * routes.list_stride;
    const std::uint32_t input_rows = routes.position_count;
    const std::uint32_t include_shared_expert =
        static_cast<std::uint32_t>(routes.include_shared_expert);
    const MetalSize compute_threads{
        .width = steel_bm32 ? kBm32Threads : kThreads,
        .height = 1U,
        .depth = 1U,
    };
    const std::size_t shared_up_packed_offset =
        kGuardBytes +
        include_shared_expert *
            std::size_t{kActiveExperts} * kExpertDimension *
            (kHidden / kQ4ValuesPerWord) * sizeof(std::uint32_t);
    const std::size_t shared_up_parameter_offset =
        kGuardBytes +
        include_shared_expert *
            std::size_t{kActiveExperts} * kExpertDimension *
            (kHidden / kGroupSize) * sizeof(std::uint16_t);
    const std::size_t shared_down_packed_offset =
        kGuardBytes +
        include_shared_expert *
            std::size_t{kActiveExperts} * kHidden *
            (kExpertDimension / kQ4ValuesPerWord) *
            sizeof(std::uint32_t);
    const std::size_t shared_down_parameter_offset =
        kGuardBytes +
        include_shared_expert *
            std::size_t{kActiveExperts} * kHidden *
            (kExpertDimension / kGroupSize) *
            sizeof(std::uint16_t);
    if (!encoded(set_compute_pipeline(
            *pass.compute_pass,
            steel_bm32 ? pipelines.steel_bm32_upgate
            : steel_bk64 ? pipelines.steel_bk64_upgate
            : steel_bn64 ? pipelines.steel_bn64_upgate
            : steel_upgate ? pipelines.steel_upgate
                           : pipelines.staged_fused)) ||
        !encoded(set_buffer(
            *pass.compute_pass, compute.input.buffer,
            kGuardBytes, 0U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, producer_up.lists.buffer,
            kGuardBytes, 1U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, producer_up.tasks.buffer,
            kGuardBytes, 2U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, compute.gate_packed.buffer,
            kGuardBytes, 3U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, compute.gate_scales.buffer,
            kGuardBytes, 4U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, compute.gate_biases.buffer,
            kGuardBytes, 5U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, compute.up_packed.buffer,
            kGuardBytes, 6U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, compute.up_scales.buffer,
            kGuardBytes, 7U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, compute.up_biases.buffer,
            kGuardBytes, 8U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, hidden.buffer, kGuardBytes, 9U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, producer_up.grid.buffer,
            kGuardBytes, 10U)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &route_list_expert_stride,
            sizeof(route_list_expert_stride), 11U)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &route_list_capacity_per_expert,
            sizeof(route_list_capacity_per_expert), 12U)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &route_list_total_extent,
            sizeof(route_list_total_extent), 13U)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &input_rows,
            sizeof(input_rows), 14U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, compute.gate_packed.buffer,
            shared_up_packed_offset, 15U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, compute.gate_scales.buffer,
            shared_up_parameter_offset, 16U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, compute.gate_biases.buffer,
            shared_up_parameter_offset, 17U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, compute.up_packed.buffer,
            shared_up_packed_offset, 18U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, compute.up_scales.buffer,
            shared_up_parameter_offset, 19U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, compute.up_biases.buffer,
            shared_up_parameter_offset, 20U)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &include_shared_expert,
            sizeof(include_shared_expert), 21U)) ||
        !encoded(dispatch_threadgroups_indirect(
            *pass.compute_pass,
            producer_up.grid.buffer,
            kGuardBytes,
            compute_threads)) ||
        !encoded(memory_barrier(*pass.compute_pass)) ||
        !encoded(set_compute_pipeline(
            *pass.compute_pass,
            steel_bm32 ? pipelines.steel_bm32_down
            : steel_bk64 ? pipelines.steel_bk64_down
            : steel_bn64 ? pipelines.steel_bn64_down
            : steel_down ? pipelines.steel_down
                         : pipelines.staged_down)) ||
        !encoded(set_buffer(
            *pass.compute_pass, hidden.buffer, kGuardBytes, 0U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, producer_down.lists.buffer,
            kGuardBytes, 1U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, producer_down.tasks.buffer,
            kGuardBytes, 2U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, compute.down_packed.buffer,
            kGuardBytes, 3U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, compute.down_scales.buffer,
            kGuardBytes, 4U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, compute.down_biases.buffer,
            kGuardBytes, 5U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, partials.buffer, kGuardBytes, 6U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, producer_down.grid.buffer,
            kGuardBytes, 7U)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &route_list_expert_stride,
            sizeof(route_list_expert_stride), 8U)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &route_list_capacity_per_expert,
            sizeof(route_list_capacity_per_expert), 9U)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &route_list_total_extent,
            sizeof(route_list_total_extent), 10U)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &input_rows,
            sizeof(input_rows), 11U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, compute.down_packed.buffer,
            shared_down_packed_offset, 12U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, compute.down_scales.buffer,
            shared_down_parameter_offset, 13U)) ||
        !encoded(set_buffer(
            *pass.compute_pass, compute.down_biases.buffer,
            shared_down_parameter_offset, 14U)) ||
        !encoded(set_bytes(
            *pass.compute_pass, &include_shared_expert,
            sizeof(include_shared_expert), 15U)) ||
        !encoded(dispatch_threadgroups_indirect(
            *pass.compute_pass,
            producer_down.grid.buffer,
            kGuardBytes,
            compute_threads))) {
        return kExitEncode;
    }
    auto ended =
        end_compute_pass(std::move(*pass.compute_pass));
    if (!ended) {
        return kExitEndPass;
    }
    auto pending = commit(std::move(*ended.command_buffer));
    if (!pending) {
        return kExitCommit;
    }
    ++command_buffers;
    auto completed =
        wait_until_completed(std::move(*pending.pending_execution));
    if (!completed) {
        std::cerr << "R2 compute execution failed: "
                  << completed.failure_description.view() << '\n';
        return kExitExecution;
    }
    return 0;
}

struct BenchmarkBuffers {
    MetalBuffer input;
    MetalBuffer gate_packed;
    MetalBuffer gate_scales;
    MetalBuffer gate_biases;
    MetalBuffer up_packed;
    MetalBuffer up_scales;
    MetalBuffer up_biases;
    MetalBuffer down_packed;
    MetalBuffer down_scales;
    MetalBuffer down_biases;
    MetalBuffer hidden;
    MetalBuffer partials;
    MetalBuffer active;
    MetalBuffer down_grid;
    MetalBuffer down_grid_bm32;
    MetalBuffer down_grid_bn64;
    std::uint64_t total_bytes = 0U;
};

int allocate_zero_buffer(
    const MetalDevice& device, std::uint64_t bytes,
    MetalBuffer& output) {
    if (bytes == 0U ||
        bytes > std::numeric_limits<std::size_t>::max()) {
        return kExitInitialization;
    }
    auto created = create_shared_buffer(device, bytes);
    if (!created || !created.buffer) {
        return kExitBuffer;
    }
    output = std::move(*created.buffer);
    if (!output || output.size_bytes() != bytes ||
        output.contents() == nullptr) {
        return kExitInitialization;
    }
    std::memset(
        output.contents(), 0, static_cast<std::size_t>(bytes));
    return 0;
}

int initialize_benchmark_buffers(
    const MetalDevice& device, BenchmarkBuffers& buffers) {
    const std::uint64_t input_bytes =
        std::uint64_t{kHighPositionCapacity} * kHidden *
        sizeof(std::uint16_t);
    const std::uint64_t up_packed_bytes =
        std::uint64_t{kExperts} * kExpertDimension *
        (kHidden / kQ4ValuesPerWord) * sizeof(std::uint32_t);
    const std::uint64_t up_parameter_bytes =
        std::uint64_t{kExperts} * kExpertDimension *
        (kHidden / kGroupSize) * sizeof(std::uint16_t);
    const std::uint64_t down_packed_bytes =
        std::uint64_t{kExperts} * kHidden *
        (kExpertDimension / kQ4ValuesPerWord) *
        sizeof(std::uint32_t);
    const std::uint64_t down_parameter_bytes =
        std::uint64_t{kExperts} * kHidden *
        (kExpertDimension / kGroupSize) *
        sizeof(std::uint16_t);
    const std::uint64_t hidden_bytes =
        std::uint64_t{kHighPositionCapacity} * kSlotCount *
        kExpertDimension * sizeof(std::uint16_t);
    const std::uint64_t partial_bytes =
        std::uint64_t{kHighPositionCapacity} * kSlotCount *
        kHidden * sizeof(float);
    const std::uint64_t active_bytes =
        std::uint64_t{kExperts} * sizeof(std::uint32_t);
    constexpr std::uint64_t kGridBytes =
        3U * sizeof(std::uint32_t);

    const std::array<std::uint64_t, 16> sizes{
        input_bytes,
        up_packed_bytes,
        up_parameter_bytes,
        up_parameter_bytes,
        up_packed_bytes,
        up_parameter_bytes,
        up_parameter_bytes,
        down_packed_bytes,
        down_parameter_bytes,
        down_parameter_bytes,
        hidden_bytes,
        partial_bytes,
        active_bytes,
        kGridBytes,
        kGridBytes,
        kGridBytes,
    };
    for (const std::uint64_t bytes : sizes) {
        if (bytes >
            std::numeric_limits<std::uint64_t>::max() -
                buffers.total_bytes) {
            return kExitInitialization;
        }
        buffers.total_bytes += bytes;
    }

    const std::array<std::pair<std::uint64_t, MetalBuffer*>, 16>
        allocations{{
            {input_bytes, &buffers.input},
            {up_packed_bytes, &buffers.gate_packed},
            {up_parameter_bytes, &buffers.gate_scales},
            {up_parameter_bytes, &buffers.gate_biases},
            {up_packed_bytes, &buffers.up_packed},
            {up_parameter_bytes, &buffers.up_scales},
            {up_parameter_bytes, &buffers.up_biases},
            {down_packed_bytes, &buffers.down_packed},
            {down_parameter_bytes, &buffers.down_scales},
            {down_parameter_bytes, &buffers.down_biases},
            {hidden_bytes, &buffers.hidden},
            {partial_bytes, &buffers.partials},
            {active_bytes, &buffers.active},
            {kGridBytes, &buffers.down_grid},
            {kGridBytes, &buffers.down_grid_bm32},
            {kGridBytes, &buffers.down_grid_bn64},
        }};
    for (const auto& [bytes, destination] : allocations) {
        const int result =
            allocate_zero_buffer(device, bytes, *destination);
        if (result != 0) {
            return result;
        }
    }

    auto* active =
        static_cast<std::uint32_t*>(buffers.active.contents());
    for (std::uint32_t expert = 0U; expert < kExperts; ++expert) {
        active[expert] = expert;
    }
    auto* grid =
        static_cast<std::uint32_t*>(buffers.down_grid.contents());
    grid[0] = kHighExpectedTasks;
    grid[1] = kDownColumnGroups;
    grid[2] = 1U;
    auto* bm32_grid = static_cast<std::uint32_t*>(
        buffers.down_grid_bm32.contents());
    bm32_grid[0] = kHighBm32ExpectedTasks;
    bm32_grid[1] = kDownColumnGroups;
    bm32_grid[2] = 1U;
    auto* bn64_grid = static_cast<std::uint32_t*>(
        buffers.down_grid_bn64.contents());
    bn64_grid[0] = kHighExpectedTasks;
    bn64_grid[1] = kBn64DownColumnGroups;
    bn64_grid[2] = 1U;
    return 0;
}

enum class BenchmarkStage : std::uint8_t {
    Combined,
    Upgate,
    Down,
};

int encode_benchmark_exact(
    MetalComputePass& pass,
    const BenchmarkPipelines& pipelines,
    const ProducerBuffers& routes,
    const BenchmarkBuffers& buffers, BenchmarkStage stage) {
    const std::uint32_t list_capacity = kHighListStride;
    constexpr std::uint32_t kRowsPerExactThreadgroup =
        kExactThreads / kSimdgroupWidth;
    if (stage != BenchmarkStage::Down) {
        if (!encoded(set_compute_pipeline(
                pass, pipelines.exact_upgate)) ||
            !encoded(set_buffer(pass, buffers.input, 0U, 0U)) ||
            !encoded(set_buffer(
                pass, routes.counts.buffer, kGuardBytes, 1U)) ||
            !encoded(set_buffer(
                pass, routes.lists.buffer, kGuardBytes, 2U)) ||
            !encoded(set_buffer(
                pass, buffers.gate_packed, 0U, 3U)) ||
            !encoded(set_buffer(
                pass, buffers.gate_scales, 0U, 4U)) ||
            !encoded(set_buffer(
                pass, buffers.gate_biases, 0U, 5U)) ||
            !encoded(set_buffer(pass, buffers.up_packed, 0U, 6U)) ||
            !encoded(set_buffer(pass, buffers.up_scales, 0U, 7U)) ||
            !encoded(set_buffer(pass, buffers.up_biases, 0U, 8U)) ||
            !encoded(set_buffer(pass, buffers.hidden, 0U, 9U)) ||
            !encoded(set_buffer(pass, buffers.active, 0U, 10U)) ||
            !encoded(set_buffer(
                pass, buffers.gate_packed, 0U, 11U)) ||
            !encoded(set_buffer(
                pass, buffers.gate_scales, 0U, 12U)) ||
            !encoded(set_buffer(
                pass, buffers.gate_biases, 0U, 13U)) ||
            !encoded(set_buffer(
                pass, buffers.up_packed, 0U, 14U)) ||
            !encoded(set_buffer(
                pass, buffers.up_scales, 0U, 15U)) ||
            !encoded(set_buffer(
                pass, buffers.up_biases, 0U, 16U)) ||
            !encoded(set_bytes(
                pass, &list_capacity, sizeof(list_capacity), 17U)) ||
            !encoded(dispatch_threadgroups(
                pass,
                {
                    .width = kExperts,
                    .height = ceil_div(
                        kExpertDimension,
                        kRowsPerExactThreadgroup),
                    .depth = 1U,
                },
                {
                    .width = kExactThreads,
                    .height = 1U,
                    .depth = 1U,
                }))) {
            return kExitEncode;
        }
        if (stage == BenchmarkStage::Combined &&
            !encoded(memory_barrier(pass))) {
            return kExitEncode;
        }
    }
    if (stage != BenchmarkStage::Upgate) {
        if (!encoded(set_compute_pipeline(
                pass, pipelines.exact_down)) ||
            !encoded(set_buffer(pass, buffers.hidden, 0U, 0U)) ||
            !encoded(set_buffer(
                pass, routes.counts.buffer, kGuardBytes, 1U)) ||
            !encoded(set_buffer(
                pass, routes.lists.buffer, kGuardBytes, 2U)) ||
            !encoded(set_buffer(
                pass, buffers.down_packed, 0U, 3U)) ||
            !encoded(set_buffer(
                pass, buffers.down_scales, 0U, 4U)) ||
            !encoded(set_buffer(
                pass, buffers.down_biases, 0U, 5U)) ||
            !encoded(set_buffer(pass, buffers.partials, 0U, 6U)) ||
            !encoded(set_buffer(pass, buffers.active, 0U, 7U)) ||
            !encoded(set_buffer(
                pass, buffers.down_packed, 0U, 8U)) ||
            !encoded(set_buffer(
                pass, buffers.down_scales, 0U, 9U)) ||
            !encoded(set_buffer(
                pass, buffers.down_biases, 0U, 10U)) ||
            !encoded(set_bytes(
                pass, &list_capacity, sizeof(list_capacity), 11U)) ||
            !encoded(dispatch_threadgroups(
                pass,
                {
                    .width = kExperts,
                    .height = ceil_div(
                        kHidden, kRowsPerExactThreadgroup),
                    .depth = 1U,
                },
                {
                    .width = kExactThreads,
                    .height = 1U,
                    .depth = 1U,
                }))) {
            return kExitEncode;
        }
    }
    return 0;
}

enum class BenchmarkArm : std::uint8_t {
    Exact,
    R1,
    R2,
    SteelDown,
    Steel,
    SteelBm32,
    SteelBk64,
    SteelBn64,
};

int encode_benchmark_native(
    MetalComputePass& pass, const Pipelines& pipelines,
    const ProducerBuffers& routes,
    const BenchmarkBuffers& buffers, BenchmarkStage stage,
    BenchmarkArm arm) {
    const bool staged = arm != BenchmarkArm::R1;
    const bool steel_upgate = arm == BenchmarkArm::Steel;
    const bool steel_bm32 = arm == BenchmarkArm::SteelBm32;
    const bool steel_bk64 = arm == BenchmarkArm::SteelBk64;
    const bool steel_bn64 = arm == BenchmarkArm::SteelBn64;
    const bool steel_down =
        arm == BenchmarkArm::SteelDown ||
        arm == BenchmarkArm::Steel || steel_bm32 ||
        steel_bk64 || steel_bn64;
    const std::uint32_t compute_threads =
        steel_bm32 ? kBm32Threads : kThreads;
    const std::uint32_t route_list_expert_stride = kHighListStride;
    const std::uint32_t route_list_capacity_per_expert =
        kHighPositionCapacity;
    const std::uint64_t route_list_total_extent =
        std::uint64_t{kExperts} * kHighListStride;
    const std::uint32_t input_rows = kHighPositionCount;
    const std::uint32_t include_shared_expert = 0U;
    const MetalComputePipeline& upgate_pipeline =
        steel_bm32 ? pipelines.steel_bm32_upgate
        : steel_bk64 ? pipelines.steel_bk64_upgate
        : steel_bn64 ? pipelines.steel_bn64_upgate
        : steel_upgate ? pipelines.steel_upgate
                     : staged ? pipelines.staged_fused
                              : pipelines.fused;
    const MetalComputePipeline& down_pipeline =
        steel_bm32 ? pipelines.steel_bm32_down
        : steel_bk64 ? pipelines.steel_bk64_down
        : steel_bn64 ? pipelines.steel_bn64_down
        : steel_down ? pipelines.steel_down
                   : staged ? pipelines.staged_down
                            : pipelines.down;
    if (stage != BenchmarkStage::Down) {
        if (!encoded(set_compute_pipeline(
                pass, upgate_pipeline)) ||
            !encoded(set_buffer(pass, buffers.input, 0U, 0U)) ||
            !encoded(set_buffer(
                pass, routes.lists.buffer, kGuardBytes, 1U)) ||
            !encoded(set_buffer(
                pass, routes.tasks.buffer, kGuardBytes, 2U)) ||
            !encoded(set_buffer(
                pass, buffers.gate_packed, 0U, 3U)) ||
            !encoded(set_buffer(
                pass, buffers.gate_scales, 0U, 4U)) ||
            !encoded(set_buffer(
                pass, buffers.gate_biases, 0U, 5U)) ||
            !encoded(set_buffer(
                pass, buffers.up_packed, 0U, 6U)) ||
            !encoded(set_buffer(
                pass, buffers.up_scales, 0U, 7U)) ||
            !encoded(set_buffer(
                pass, buffers.up_biases, 0U, 8U)) ||
            !encoded(set_buffer(pass, buffers.hidden, 0U, 9U)) ||
            !encoded(set_buffer(
                pass, routes.grid.buffer, kGuardBytes, 10U)) ||
            !encoded(set_bytes(
                pass, &route_list_expert_stride,
                sizeof(route_list_expert_stride), 11U)) ||
            !encoded(set_bytes(
                pass, &route_list_capacity_per_expert,
                sizeof(route_list_capacity_per_expert), 12U)) ||
            !encoded(set_bytes(
                pass, &route_list_total_extent,
                sizeof(route_list_total_extent), 13U)) ||
            !encoded(set_bytes(
                pass, &input_rows, sizeof(input_rows), 14U)) ||
            (staged &&
             (!encoded(set_buffer(
                  pass, buffers.gate_packed, 0U, 15U)) ||
              !encoded(set_buffer(
                  pass, buffers.gate_scales, 0U, 16U)) ||
              !encoded(set_buffer(
                  pass, buffers.gate_biases, 0U, 17U)) ||
              !encoded(set_buffer(
                  pass, buffers.up_packed, 0U, 18U)) ||
              !encoded(set_buffer(
                  pass, buffers.up_scales, 0U, 19U)) ||
              !encoded(set_buffer(
                  pass, buffers.up_biases, 0U, 20U)) ||
              !encoded(set_bytes(
                  pass, &include_shared_expert,
                  sizeof(include_shared_expert), 21U)))) ||
            !encoded(dispatch_threadgroups_indirect(
                pass, routes.grid.buffer, kGuardBytes,
                {.width = compute_threads,
                 .height = 1U,
                 .depth = 1U}))) {
            return kExitEncode;
        }
        if (stage == BenchmarkStage::Combined &&
            !encoded(memory_barrier(pass))) {
            return kExitEncode;
        }
    }
    if (stage != BenchmarkStage::Upgate) {
        if (!encoded(set_compute_pipeline(
                pass, down_pipeline)) ||
            !encoded(set_buffer(pass, buffers.hidden, 0U, 0U)) ||
            !encoded(set_buffer(
                pass, routes.lists.buffer, kGuardBytes, 1U)) ||
            !encoded(set_buffer(
                pass, routes.tasks.buffer, kGuardBytes, 2U)) ||
            !encoded(set_buffer(
                pass, buffers.down_packed, 0U, 3U)) ||
            !encoded(set_buffer(
                pass, buffers.down_scales, 0U, 4U)) ||
            !encoded(set_buffer(
                pass, buffers.down_biases, 0U, 5U)) ||
            !encoded(set_buffer(pass, buffers.partials, 0U, 6U)) ||
            !encoded(set_buffer(
                pass,
                steel_bm32 ? buffers.down_grid_bm32
                : steel_bn64 ? buffers.down_grid_bn64
                           : buffers.down_grid,
                0U, 7U)) ||
            !encoded(set_bytes(
                pass, &route_list_expert_stride,
                sizeof(route_list_expert_stride), 8U)) ||
            !encoded(set_bytes(
                pass, &route_list_capacity_per_expert,
                sizeof(route_list_capacity_per_expert), 9U)) ||
            !encoded(set_bytes(
                pass, &route_list_total_extent,
                sizeof(route_list_total_extent), 10U)) ||
            !encoded(set_bytes(
                pass, &input_rows, sizeof(input_rows), 11U)) ||
            (staged &&
             (!encoded(set_buffer(
                  pass, buffers.down_packed, 0U, 12U)) ||
              !encoded(set_buffer(
                  pass, buffers.down_scales, 0U, 13U)) ||
              !encoded(set_buffer(
                  pass, buffers.down_biases, 0U, 14U)) ||
              !encoded(set_bytes(
                  pass, &include_shared_expert,
                  sizeof(include_shared_expert), 15U)))) ||
            !encoded(dispatch_threadgroups_indirect(
                pass,
                steel_bm32 ? buffers.down_grid_bm32
                : steel_bn64 ? buffers.down_grid_bn64
                           : buffers.down_grid,
                0U,
                {.width = compute_threads,
                 .height = 1U,
                 .depth = 1U}))) {
            return kExitEncode;
        }
    }
    return 0;
}

struct BenchmarkRun {
    int exit_code = 0;
    double gpu_milliseconds = 0.0;
    double wall_milliseconds = 0.0;
};

BenchmarkRun run_benchmark_arm(
    const MetalCommandQueue& queue, const Pipelines& pipelines,
    const BenchmarkPipelines& benchmark_pipelines,
    const ProducerBuffers& routes,
    const BenchmarkBuffers& buffers, BenchmarkArm arm,
    BenchmarkStage stage) {
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
    int encode_result = 0;
    if (arm == BenchmarkArm::Exact) {
        encode_result = encode_benchmark_exact(
            *pass.compute_pass, benchmark_pipelines, routes,
            buffers, stage);
    } else {
        encode_result = encode_benchmark_native(
            *pass.compute_pass, pipelines, routes, buffers,
            stage, arm);
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
    auto completed = wait_until_completed_timed(
        std::move(*pending.pending_execution));
    const auto wall_end = std::chrono::steady_clock::now();
    if (!completed) {
        std::cerr << "R1 benchmark execution failed: "
                  << completed.failure_description.view() << '\n';
        return {.exit_code = kExitExecution};
    }
    const double gpu_start = completed.timing.gpu_start_seconds;
    const double gpu_end = completed.timing.gpu_end_seconds;
    if (!std::isfinite(gpu_start) || !std::isfinite(gpu_end) ||
        gpu_end <= gpu_start) {
        return {.exit_code = kExitTiming};
    }
    return {
        .exit_code = 0,
        .gpu_milliseconds = (gpu_end - gpu_start) * 1000.0,
        .wall_milliseconds =
            std::chrono::duration<double, std::milli>(
                wall_end - wall_begin)
                .count(),
    };
}

struct BenchmarkSamples {
    std::array<double, kBenchmarkSamplesPerArm> gpu{};
    std::array<double, kBenchmarkSamplesPerArm> wall{};
    std::size_t count = 0U;
};

struct BenchmarkDistribution {
    double p10 = 0.0;
    double p50 = 0.0;
    double p90 = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

BenchmarkDistribution summarize_benchmark(
    std::array<double, kBenchmarkSamplesPerArm> values) {
    std::sort(values.begin(), values.end());
    return {
        .p10 = values[1],
        .p50 = values[8],
        .p90 = values[15],
        .minimum = values.front(),
        .maximum = values.back(),
    };
}

bool benchmark_control_ranges_overlap(
    const BenchmarkSamples& exact) {
    if (exact.count != kBenchmarkSamplesPerArm) {
        return false;
    }
    const auto [early_min, early_max] = std::minmax_element(
        exact.gpu.begin(), exact.gpu.begin() + 4);
    const auto [late_min, late_max] = std::minmax_element(
        exact.gpu.end() - 4, exact.gpu.end());
    return *early_max >= *late_min && *late_max >= *early_min;
}

void print_benchmark_distribution(
    std::string_view name,
    const BenchmarkDistribution& distribution) {
    std::cout << ' ' << name << "_p10=" << distribution.p10
              << ' ' << name << "_p50=" << distribution.p50
              << ' ' << name << "_p90=" << distribution.p90
              << ' ' << name << "_min=" << distribution.minimum
              << ' ' << name << "_max=" << distribution.maximum;
}

struct BenchmarkPairReport {
    BenchmarkDistribution control_gpu;
    BenchmarkDistribution treatment_gpu;
    BenchmarkDistribution control_wall;
    BenchmarkDistribution treatment_wall;
};

int measure_benchmark_pair(
    const MetalCommandQueue& queue, const Pipelines& pipelines,
    const BenchmarkPipelines& benchmark_pipelines,
    const ProducerBuffers& control_routes,
    const ProducerBuffers& treatment_routes,
    const BenchmarkBuffers& buffers, BenchmarkStage stage,
    BenchmarkArm control_arm, BenchmarkArm treatment_arm,
    BenchmarkPairReport& report) {
    for (const char scheduled : kBenchmarkWarmupSchedule) {
        const bool control_scheduled = scheduled == 'A';
        const BenchmarkRun warmup = run_benchmark_arm(
            queue, pipelines, benchmark_pipelines,
            control_scheduled ? control_routes : treatment_routes,
            buffers,
            control_scheduled ? control_arm : treatment_arm, stage);
        if (warmup.exit_code != 0) {
            return warmup.exit_code;
        }
    }

    BenchmarkSamples control;
    BenchmarkSamples treatment;
    for (const char scheduled : kBenchmarkMeasuredSchedule) {
        const BenchmarkArm arm =
            scheduled == 'A' ? control_arm : treatment_arm;
        const BenchmarkRun measured = run_benchmark_arm(
            queue, pipelines, benchmark_pipelines,
            scheduled == 'A' ? control_routes : treatment_routes,
            buffers,
            arm, stage);
        if (measured.exit_code != 0) {
            return measured.exit_code;
        }
        BenchmarkSamples& samples =
            scheduled == 'A' ? control : treatment;
        if (samples.count >= kBenchmarkSamplesPerArm) {
            return kExitSampleAccounting;
        }
        samples.gpu[samples.count] = measured.gpu_milliseconds;
        samples.wall[samples.count] = measured.wall_milliseconds;
        ++samples.count;
    }
    if (control.count != kBenchmarkSamplesPerArm ||
        treatment.count != kBenchmarkSamplesPerArm) {
        return kExitSampleAccounting;
    }
    if (!benchmark_control_ranges_overlap(control)) {
        return kExitControlDrift;
    }
    report = {
        .control_gpu = summarize_benchmark(control.gpu),
        .treatment_gpu = summarize_benchmark(treatment.gpu),
        .control_wall = summarize_benchmark(control.wall),
        .treatment_wall = summarize_benchmark(treatment.wall),
    };
    return 0;
}

void print_benchmark_pair(
    std::string_view treatment, std::string_view stage,
    const BenchmarkPairReport& report,
    std::uint64_t resident_bytes,
    std::uint32_t control_tasks,
    std::uint32_t treatment_tasks) {
    std::cout
        << "tatara_native_routed_qgemm_perf"
        << " treatment=" << treatment
        << " stage=" << stage
        << " rows=" << kHighPositionCount
        << " routed_rows="
        << kHighPositionCount * kActiveExperts
        << " experts=" << kExperts
        << " control_tasks=" << control_tasks
        << " treatment_tasks=" << treatment_tasks
        << " resident_bytes=" << resident_bytes;
    print_benchmark_distribution(
        "control_gpu_ms", report.control_gpu);
    print_benchmark_distribution(
        "treatment_gpu_ms", report.treatment_gpu);
    print_benchmark_distribution(
        "control_wall_ms", report.control_wall);
    print_benchmark_distribution(
        "treatment_wall_ms", report.treatment_wall);
    std::cout << " gpu_p50_speedup="
              << report.control_gpu.p50 /
                     report.treatment_gpu.p50
              << " samples_per_arm=" << kBenchmarkSamplesPerArm
              << " command_buffers="
              << kBenchmarkWarmupSchedule.size() +
                     kBenchmarkMeasuredSchedule.size()
              << '\n';
}

int run_benchmark(
    const MetalDevice& device, const MetalCommandQueue& queue,
    const MetalLibrary& library, const Pipelines& pipelines,
    const ProducerBuffers& control_routes,
    const ProducerBuffers& treatment_routes,
    BenchmarkArm treatment_arm) {
    BenchmarkPipelines benchmark_pipelines;
    const int pipeline_result = load_benchmark_pipelines(
        device, library, benchmark_pipelines);
    if (pipeline_result != 0) {
        return pipeline_result;
    }
    BenchmarkBuffers buffers;
    const int buffer_result =
        initialize_benchmark_buffers(device, buffers);
    if (buffer_result != 0) {
        return buffer_result;
    }

    BenchmarkPairReport combined;
    BenchmarkPairReport upgate;
    BenchmarkPairReport down;
    const BenchmarkArm control_arm =
        treatment_arm == BenchmarkArm::SteelBm32
            ? BenchmarkArm::Steel
        : treatment_arm == BenchmarkArm::SteelBk64
            ? BenchmarkArm::Steel
        : treatment_arm == BenchmarkArm::SteelBn64
            ? BenchmarkArm::Steel
        : (treatment_arm == BenchmarkArm::SteelDown ||
         treatment_arm == BenchmarkArm::Steel)
            ? BenchmarkArm::R2
            : BenchmarkArm::Exact;
    int result = measure_benchmark_pair(
        queue, pipelines, benchmark_pipelines,
        control_routes, treatment_routes, buffers,
        BenchmarkStage::Combined, control_arm, treatment_arm,
        combined);
    if (result == 0) {
        result = measure_benchmark_pair(
            queue, pipelines, benchmark_pipelines,
            control_routes, treatment_routes, buffers,
            BenchmarkStage::Upgate, control_arm, treatment_arm,
            upgate);
    }
    if (result == 0) {
        result = measure_benchmark_pair(
            queue, pipelines, benchmark_pipelines,
            control_routes, treatment_routes, buffers,
            BenchmarkStage::Down, control_arm, treatment_arm,
            down);
    }
    if (result != 0) {
        return result;
    }
    const std::string_view treatment =
        treatment_arm == BenchmarkArm::R1
            ? "r1"
            : treatment_arm == BenchmarkArm::R2
                  ? "r2"
                  : treatment_arm == BenchmarkArm::SteelDown
                        ? "steel_down_vs_r2"
                  : treatment_arm == BenchmarkArm::Steel
                        ? "steel_vs_r2"
                  : treatment_arm == BenchmarkArm::SteelBk64
                        ? "steel_bk64_vs_bk32"
                  : treatment_arm == BenchmarkArm::SteelBn64
                        ? "steel_bn64_vs_bn32"
                        : "steel_bm32_vs_bm16";
    const std::uint32_t treatment_tasks =
        treatment_arm == BenchmarkArm::SteelBm32
            ? kHighBm32ExpectedTasks
            : kHighExpectedTasks;
    print_benchmark_pair(
        treatment, "combined", combined, buffers.total_bytes,
        kHighExpectedTasks, treatment_tasks);
    print_benchmark_pair(
        treatment, "upgate", upgate, buffers.total_bytes,
        kHighExpectedTasks, treatment_tasks);
    print_benchmark_pair(
        treatment, "down", down, buffers.total_bytes,
        kHighExpectedTasks, treatment_tasks);
    return 0;
}

struct NumericReport {
    float maximum_absolute = 0.0F;
    float maximum_normalized = 0.0F;
    std::uint32_t maximum_bfloat_ulp = 0U;
    bool finite = true;
};

bool hidden_poison_partitions_intact(
    std::span<const std::uint16_t> actual,
    bool shared_slot_written = false) {
    if (actual.size() !=
        std::size_t{kNumericPositionCapacity} * kSlotCount *
            kExpertDimension) {
        return false;
    }
    for (std::uint32_t position = 0U;
         position < kNumericPositionCapacity; ++position) {
        for (std::uint32_t slot = 0U; slot < kSlotCount; ++slot) {
            const bool must_remain_poison =
                position >= kNumericPositionCount ||
                (!shared_slot_written &&
                 slot == kActiveExperts);
            if (!must_remain_poison) {
                continue;
            }
            const std::size_t begin =
                (std::size_t{position} * kSlotCount + slot) *
                kExpertDimension;
            for (std::uint32_t column = 0U;
                 column < kExpertDimension; ++column) {
                if (actual[begin + column] != kHiddenCanary) {
                    return false;
                }
            }
        }
    }
    return true;
}

NumericReport compare_hidden(
    std::span<const std::uint16_t> actual,
    std::span<const std::uint16_t> expected,
    std::uint32_t expected_slots = kActiveExperts) {
    NumericReport report;
    if (expected.size() !=
        std::size_t{kNumericPositionCount} * expected_slots *
            kExpertDimension) {
        report.finite = false;
        return report;
    }
    for (std::uint32_t position = 0U;
         position < kNumericPositionCount; ++position) {
        for (std::uint32_t slot = 0U;
             slot < expected_slots; ++slot) {
            for (std::uint32_t column = 0U;
                 column < kExpertDimension; ++column) {
                const std::size_t actual_index =
                    (std::size_t{position} * kSlotCount + slot) *
                        kExpertDimension +
                    column;
                const std::size_t expected_index =
                    (std::size_t{position} * expected_slots + slot) *
                        kExpertDimension +
                    column;
                const std::uint16_t actual_bits = actual[actual_index];
                const std::uint16_t expected_bits =
                    expected[expected_index];
                const float actual_value =
                    from_bfloat16(actual_bits);
                const float expected_value =
                    from_bfloat16(expected_bits);
                report.finite =
                    report.finite && std::isfinite(actual_value) &&
                    std::isfinite(expected_value);
                const float absolute =
                    std::fabs(actual_value - expected_value);
                const float normalized =
                    absolute /
                    std::fmax(1.0F, std::fabs(expected_value));
                report.maximum_absolute =
                    std::fmax(report.maximum_absolute, absolute);
                report.maximum_normalized =
                    std::fmax(report.maximum_normalized, normalized);
                report.maximum_bfloat_ulp = std::max(
                    report.maximum_bfloat_ulp,
                    bfloat_ulp_distance(
                        actual_bits, expected_bits));
            }
        }
    }
    return report;
}

bool partial_poison_partitions_intact(
    std::span<const std::uint32_t> actual,
    bool shared_slot_written = false) {
    if (actual.size() !=
        std::size_t{kNumericPositionCapacity} * kSlotCount * kHidden) {
        return false;
    }
    for (std::uint32_t position = 0U;
         position < kNumericPositionCapacity; ++position) {
        for (std::uint32_t slot = 0U; slot < kSlotCount; ++slot) {
            const bool must_remain_poison =
                position >= kNumericPositionCount ||
                (!shared_slot_written &&
                 slot == kActiveExperts);
            if (!must_remain_poison) {
                continue;
            }
            const std::size_t begin =
                (std::size_t{position} * kSlotCount + slot) * kHidden;
            for (std::uint32_t column = 0U; column < kHidden; ++column) {
                if (actual[begin + column] != kPartialCanary) {
                    return false;
                }
            }
        }
    }
    return true;
}

NumericReport compare_partials(
    std::span<const std::uint32_t> actual,
    std::span<const float> expected,
    std::uint32_t expected_slots = kActiveExperts) {
    NumericReport report;
    if (expected.size() !=
        std::size_t{kNumericPositionCount} * expected_slots *
            kHidden) {
        report.finite = false;
        return report;
    }
    for (std::uint32_t position = 0U;
         position < kNumericPositionCount; ++position) {
        for (std::uint32_t slot = 0U;
             slot < expected_slots; ++slot) {
            for (std::uint32_t column = 0U; column < kHidden; ++column) {
                const std::size_t actual_index =
                    (std::size_t{position} * kSlotCount + slot) *
                        kHidden +
                    column;
                const std::size_t expected_index =
                    (std::size_t{position} * expected_slots + slot) *
                        kHidden +
                    column;
                const float actual_value =
                    std::bit_cast<float>(actual[actual_index]);
                const float expected_value = expected[expected_index];
                report.finite =
                    report.finite && std::isfinite(actual_value) &&
                    std::isfinite(expected_value);
                const float absolute =
                    std::fabs(actual_value - expected_value);
                const float normalized =
                    absolute /
                    std::fmax(1.0F, std::fabs(expected_value));
                report.maximum_absolute =
                    std::fmax(report.maximum_absolute, absolute);
                report.maximum_normalized =
                    std::fmax(report.maximum_normalized, normalized);
            }
        }
    }
    return report;
}

bool compute_read_only_intact(const ComputeBuffers& buffers) {
    return unchanged_from_initial(buffers.input) &&
           unchanged_from_initial(buffers.gate_packed) &&
           unchanged_from_initial(buffers.gate_scales) &&
           unchanged_from_initial(buffers.gate_biases) &&
           unchanged_from_initial(buffers.up_packed) &&
           unchanged_from_initial(buffers.up_scales) &&
           unchanged_from_initial(buffers.up_biases) &&
           unchanged_from_initial(buffers.down_packed) &&
           unchanged_from_initial(buffers.down_scales) &&
           unchanged_from_initial(buffers.down_biases);
}

bool all_compute_guards_intact(const ComputeBuffers& buffers) {
    return guards_intact(buffers.input) &&
           guards_intact(buffers.gate_packed) &&
           guards_intact(buffers.gate_scales) &&
           guards_intact(buffers.gate_biases) &&
           guards_intact(buffers.up_packed) &&
           guards_intact(buffers.up_scales) &&
           guards_intact(buffers.up_biases) &&
           guards_intact(buffers.down_packed) &&
           guards_intact(buffers.down_scales) &&
           guards_intact(buffers.down_biases) &&
           guards_intact(buffers.fused_hidden) &&
           guards_intact(buffers.split_hidden) &&
           guards_intact(buffers.partials) &&
           guards_intact(buffers.staged_hidden) &&
           guards_intact(buffers.staged_partials) &&
           guards_intact(buffers.assimilated_hidden) &&
           guards_intact(buffers.assimilated_partials) &&
           guards_intact(buffers.bm32_hidden) &&
           guards_intact(buffers.bm32_partials) &&
           guards_intact(buffers.bk64_hidden) &&
           guards_intact(buffers.bk64_partials) &&
           guards_intact(buffers.bn64_hidden) &&
           guards_intact(buffers.bn64_partials);
}

int validate_compute_output(
    const NumericFixture& fixture, const GpuResources& resources) {
    if (!producer_unchanged_after_compute(resources.high) ||
        !producer_unchanged_after_compute(resources.numeric_up) ||
        !producer_unchanged_after_compute(resources.numeric_down) ||
        !all_compute_guards_intact(resources.compute) ||
        !compute_read_only_intact(resources.compute)) {
        return kExitCanary;
    }

    const auto fused =
        current_body<std::uint16_t>(resources.compute.fused_hidden);
    const auto split =
        current_body<std::uint16_t>(resources.compute.split_hidden);
    const auto partials =
        current_body<std::uint32_t>(resources.compute.partials);
    if (!hidden_poison_partitions_intact(fused) ||
        !hidden_poison_partitions_intact(split) ||
        !partial_poison_partitions_intact(partials)) {
        return kExitCanary;
    }

    const NumericReport fused_report =
        compare_hidden(fused, fixture.fused_expected);
    if (!fused_report.finite ||
        fused_report.maximum_normalized > kMaximumNormalizedError ||
        fused_report.maximum_bfloat_ulp > kMaximumBfloatUlp) {
        return kExitFusedNumerics;
    }
    const NumericReport split_report =
        compare_hidden(split, fixture.split_expected);
    if (!split_report.finite ||
        split_report.maximum_normalized > kMaximumNormalizedError ||
        split_report.maximum_bfloat_ulp > kMaximumBfloatUlp) {
        return kExitSplitNumerics;
    }
    const NumericReport down_report =
        compare_partials(partials, fixture.down_expected);
    if (!down_report.finite ||
        down_report.maximum_normalized > kMaximumNormalizedError) {
        return kExitDownNumerics;
    }

    std::size_t actual_family_differences = 0U;
    for (std::uint32_t position = 0U;
         position < kNumericPositionCount; ++position) {
        for (std::uint32_t slot = 0U;
             slot < kActiveExperts; ++slot) {
            for (std::uint32_t column = 0U;
                 column < kExpertDimension; ++column) {
                const std::size_t expected_index =
                    (std::size_t{position} * kActiveExperts + slot) *
                        kExpertDimension +
                    column;
                if (fixture.fused_expected[expected_index] ==
                    fixture.split_expected[expected_index]) {
                    continue;
                }
                const std::size_t actual_index =
                    (std::size_t{position} * kSlotCount + slot) *
                        kExpertDimension +
                    column;
                actual_family_differences +=
                    static_cast<std::size_t>(
                        fused[actual_index] != split[actual_index]);
            }
        }
    }
    if (fixture.family_differences == 0U ||
        actual_family_differences == 0U) {
        return kExitFamilyDiscrimination;
    }

    std::cout
        << "tatara_native_routed_qgemm_numeric"
        << " fused_max_norm=" << fused_report.maximum_normalized
        << " fused_max_bfloat_ulp="
        << fused_report.maximum_bfloat_ulp
        << " split_max_norm=" << split_report.maximum_normalized
        << " split_max_bfloat_ulp="
        << split_report.maximum_bfloat_ulp
        << " down_max_norm=" << down_report.maximum_normalized
        << " family_discriminators=" << actual_family_differences
        << '\n';
    return 0;
}

int validate_staged_compute_output(
    const NumericFixture& fixture, const GpuResources& resources) {
    if (!producer_unchanged_after_compute(resources.high) ||
        !producer_unchanged_after_compute(resources.numeric_up) ||
        !producer_unchanged_after_compute(resources.numeric_down) ||
        !all_compute_guards_intact(resources.compute) ||
        !compute_read_only_intact(resources.compute)) {
        return kExitCanary;
    }
    const auto hidden = current_body<std::uint16_t>(
        resources.compute.staged_hidden);
    const auto partials = current_body<std::uint32_t>(
        resources.compute.staged_partials);
    if (!hidden_poison_partitions_intact(hidden) ||
        !partial_poison_partitions_intact(partials)) {
        return kExitCanary;
    }
    const NumericReport hidden_report =
        compare_hidden(hidden, fixture.staged_expected);
    const NumericReport down_report =
        compare_partials(partials, fixture.staged_down_expected);
    if (!hidden_report.finite ||
        hidden_report.maximum_normalized >
            kMaximumNormalizedError ||
        hidden_report.maximum_bfloat_ulp >
            kMaximumBfloatUlp) {
        return kExitFusedNumerics;
    }
    if (!down_report.finite ||
        down_report.maximum_normalized >
            kMaximumNormalizedError) {
        return kExitDownNumerics;
    }
    std::cout
        << "tatara_native_routed_qgemm_r2_numeric"
        << " hidden_max_norm="
        << hidden_report.maximum_normalized
        << " hidden_max_bfloat_ulp="
        << hidden_report.maximum_bfloat_ulp
        << " down_max_norm="
        << down_report.maximum_normalized << '\n';
    return 0;
}

int validate_bm32_compute_output(
    const NumericFixture& fixture, const GpuResources& resources,
    bool shared) {
    const ProducerBuffers& up =
        shared ? resources.numeric_shared_bm32_up
               : resources.numeric_bm32_up;
    const ProducerBuffers& down =
        shared ? resources.numeric_shared_bm32_down
               : resources.numeric_bm32_down;
    if (!producer_unchanged_after_compute(resources.high_bm32) ||
        !producer_unchanged_after_compute(up) ||
        !producer_unchanged_after_compute(down) ||
        !all_compute_guards_intact(resources.compute) ||
        !compute_read_only_intact(resources.compute)) {
        return kExitCanary;
    }
    const auto hidden = current_body<std::uint16_t>(
        resources.compute.bm32_hidden);
    const auto partials = current_body<std::uint32_t>(
        resources.compute.bm32_partials);
    if (!hidden_poison_partitions_intact(hidden, shared) ||
        !partial_poison_partitions_intact(partials, shared)) {
        return kExitCanary;
    }
    const NumericReport hidden_report = compare_hidden(
        hidden,
        shared ? std::span<const std::uint16_t>{
                     fixture.assimilated_expected}
               : std::span<const std::uint16_t>{
                     fixture.staged_expected},
        shared ? kSlotCount : kActiveExperts);
    const NumericReport down_report = compare_partials(
        partials,
        shared ? std::span<const float>{
                     fixture.assimilated_down_expected}
               : std::span<const float>{
                     fixture.staged_down_expected},
        shared ? kSlotCount : kActiveExperts);
    if (!hidden_report.finite ||
        hidden_report.maximum_normalized >
            kMaximumNormalizedError ||
        hidden_report.maximum_bfloat_ulp >
            kMaximumBfloatUlp) {
        return shared ? kExitSharedHiddenNumerics
                      : kExitFusedNumerics;
    }
    if (!down_report.finite ||
        down_report.maximum_normalized >
            kMaximumNormalizedError) {
        return shared ? kExitSharedDownNumerics
                      : kExitDownNumerics;
    }
    std::cout
        << "tatara_native_routed_qgemm_bm32_numeric"
        << " shared=" << (shared ? "yes" : "no")
        << " hidden_max_norm="
        << hidden_report.maximum_normalized
        << " hidden_max_bfloat_ulp="
        << hidden_report.maximum_bfloat_ulp
        << " down_max_norm="
        << down_report.maximum_normalized << '\n';
    return 0;
}

int validate_bk64_compute_output(
    const GpuResources& resources, bool shared) {
    const ProducerBuffers& up =
        shared ? resources.numeric_shared_up
               : resources.numeric_up;
    const ProducerBuffers& down =
        shared ? resources.numeric_shared_down
               : resources.numeric_down;
    if (!producer_unchanged_after_compute(resources.high) ||
        !producer_unchanged_after_compute(up) ||
        !producer_unchanged_after_compute(down) ||
        !all_compute_guards_intact(resources.compute) ||
        !compute_read_only_intact(resources.compute)) {
        return kExitCanary;
    }
    const auto control_hidden = current_body<std::uint16_t>(
        resources.compute.staged_hidden);
    const auto treatment_hidden =
        current_body<std::uint16_t>(
            resources.compute.bk64_hidden);
    const auto control_partials = current_body<std::uint32_t>(
        resources.compute.staged_partials);
    const auto treatment_partials =
        current_body<std::uint32_t>(
            resources.compute.bk64_partials);
    if (!hidden_poison_partitions_intact(
            control_hidden, shared) ||
        !hidden_poison_partitions_intact(
            treatment_hidden, shared) ||
        !partial_poison_partitions_intact(
            control_partials, shared) ||
        !partial_poison_partitions_intact(
            treatment_partials, shared)) {
        return kExitCanary;
    }
    if (!std::equal(
            control_hidden.begin(), control_hidden.end(),
            treatment_hidden.begin(),
            treatment_hidden.end())) {
        return shared ? kExitSharedHiddenNumerics
                      : kExitFusedNumerics;
    }
    if (!std::equal(
            control_partials.begin(), control_partials.end(),
            treatment_partials.begin(),
            treatment_partials.end())) {
        return shared ? kExitSharedDownNumerics
                      : kExitDownNumerics;
    }
    std::cout
        << "tatara_native_routed_qgemm_bk64_numeric"
        << " shared=" << (shared ? "yes" : "no")
        << " hidden_byte_identical=yes"
        << " down_byte_identical=yes\n";
    return 0;
}

int validate_bn64_compute_output(
    const GpuResources& resources, bool shared) {
    const ProducerBuffers& up =
        shared ? resources.numeric_shared_bn64_up
               : resources.numeric_bn64_up;
    const ProducerBuffers& down =
        shared ? resources.numeric_shared_bn64_down
               : resources.numeric_bn64_down;
    if (!producer_unchanged_after_compute(resources.high_bn64) ||
        !producer_unchanged_after_compute(up) ||
        !producer_unchanged_after_compute(down) ||
        !all_compute_guards_intact(resources.compute) ||
        !compute_read_only_intact(resources.compute)) {
        return kExitCanary;
    }
    const auto control_hidden = current_body<std::uint16_t>(
        resources.compute.staged_hidden);
    const auto treatment_hidden =
        current_body<std::uint16_t>(
            resources.compute.bn64_hidden);
    const auto control_partials = current_body<std::uint32_t>(
        resources.compute.staged_partials);
    const auto treatment_partials =
        current_body<std::uint32_t>(
            resources.compute.bn64_partials);
    if (!hidden_poison_partitions_intact(
            control_hidden, shared) ||
        !hidden_poison_partitions_intact(
            treatment_hidden, shared) ||
        !partial_poison_partitions_intact(
            control_partials, shared) ||
        !partial_poison_partitions_intact(
            treatment_partials, shared)) {
        return kExitCanary;
    }
    if (!std::equal(
            control_hidden.begin(), control_hidden.end(),
            treatment_hidden.begin(),
            treatment_hidden.end())) {
        return shared ? kExitSharedHiddenNumerics
                      : kExitFusedNumerics;
    }
    if (!std::equal(
            control_partials.begin(), control_partials.end(),
            treatment_partials.begin(),
            treatment_partials.end())) {
        return shared ? kExitSharedDownNumerics
                      : kExitDownNumerics;
    }
    std::cout
        << "tatara_native_routed_qgemm_bn64_numeric"
        << " shared=" << (shared ? "yes" : "no")
        << " hidden_byte_identical=yes"
        << " down_byte_identical=yes\n";
    return 0;
}

int validate_assimilated_compute_output(
    const NumericFixture& fixture, const GpuResources& resources) {
    if (!producer_unchanged_after_compute(resources.high) ||
        !producer_unchanged_after_compute(resources.numeric_up) ||
        !producer_unchanged_after_compute(resources.numeric_down) ||
        !producer_unchanged_after_compute(
            resources.numeric_shared_up) ||
        !producer_unchanged_after_compute(
            resources.numeric_shared_down) ||
        !all_compute_guards_intact(resources.compute) ||
        !compute_read_only_intact(resources.compute)) {
        return kExitCanary;
    }
    const auto hidden = current_body<std::uint16_t>(
        resources.compute.assimilated_hidden);
    const auto partials = current_body<std::uint32_t>(
        resources.compute.assimilated_partials);
    if (!hidden_poison_partitions_intact(hidden, true) ||
        !partial_poison_partitions_intact(partials, true)) {
        return kExitCanary;
    }
    const NumericReport hidden_report = compare_hidden(
        hidden, fixture.assimilated_expected, kSlotCount);
    const NumericReport down_report = compare_partials(
        partials, fixture.assimilated_down_expected, kSlotCount);
    if (!hidden_report.finite ||
        hidden_report.maximum_normalized >
            kMaximumNormalizedError ||
        hidden_report.maximum_bfloat_ulp >
            kMaximumBfloatUlp) {
        return kExitSharedHiddenNumerics;
    }
    if (!down_report.finite ||
        down_report.maximum_normalized >
            kMaximumNormalizedError) {
        return kExitSharedDownNumerics;
    }
    std::cout
        << "tatara_native_routed_qgemm_r2_shared_numeric"
        << " hidden_max_norm="
        << hidden_report.maximum_normalized
        << " hidden_max_bfloat_ulp="
        << hidden_report.maximum_bfloat_ulp
        << " down_max_norm="
        << down_report.maximum_normalized << '\n';
    return 0;
}

bool cpu_contract(
    const RouteFixture& high, const RouteFixture& numeric,
    const RouteFixture& numeric_shared,
    const NumericFixture& fixture) {
    const std::uint64_t high_route_capacity =
        std::uint64_t{kHighPositionCapacity} * kActiveExperts;
    const std::uint64_t numeric_route_capacity =
        std::uint64_t{kNumericPositionCapacity} * kActiveExperts;
    const std::uint32_t high_capacity =
        ceil_div(
            static_cast<std::uint32_t>(high_route_capacity),
            kTileRows) +
        std::min<std::uint32_t>(
            kExperts,
            static_cast<std::uint32_t>(high_route_capacity));
    const std::uint32_t numeric_capacity =
        ceil_div(
            static_cast<std::uint32_t>(numeric_route_capacity),
            kTileRows) +
        std::min<std::uint32_t>(
            kExperts,
            static_cast<std::uint32_t>(numeric_route_capacity));
    const std::uint64_t numeric_shared_route_capacity =
        std::uint64_t{kNumericPositionCapacity} * kSlotCount;
    const std::uint32_t numeric_shared_capacity =
        ceil_div(
            static_cast<std::uint32_t>(
                numeric_shared_route_capacity),
            kTileRows) +
        std::min<std::uint32_t>(
            kExperts + 1U,
            static_cast<std::uint32_t>(
                numeric_shared_route_capacity));
    return validate_frozen_route_cases(
               high, numeric, numeric_shared) &&
           high_capacity == kHighPlannedTaskCapacity &&
           numeric_capacity == kNumericPlannedTaskCapacity &&
           numeric_shared_capacity ==
               kNumericSharedPlannedTaskCapacity &&
           high.planned_task_capacity <= kTaskCapacityLimit &&
           numeric.planned_task_capacity <= kTaskCapacityLimit &&
           numeric_shared.planned_task_capacity <=
               kTaskCapacityLimit &&
           kUpColumnGroups == 16U &&
           kDownColumnGroups == 64U &&
           kExpectedComputeThreadgroups == 1120U &&
           numeric_fixture_is_finite(fixture) &&
           fixture.family_differences != 0U &&
           fixture.direct_staged_differences == 0U;
}

} // namespace

int main(int argument_count, char** arguments) {
    if (argument_count != 2) {
        std::cerr
            << "usage: tatara_native_routed_qgemm_probe "
               "--cpu-only|--compile-only|--gpu|--gpu-r2|"
               "--gpu-r2-shared|--gpu-steel-down|--gpu-steel|"
               "--gpu-steel-bm32|--gpu-steel-bm32-shared|"
               "--gpu-steel-bk64|--gpu-steel-bk64-shared|"
               "--gpu-steel-bn64|--gpu-steel-bn64-shared|"
               "--benchmark|--benchmark-r2|"
               "--benchmark-steel-down|--benchmark-steel|"
               "--benchmark-steel-bm32|--benchmark-steel-bk64|"
               "--benchmark-steel-bn64\n";
        return kExitUsage;
    }
    const std::string_view mode(arguments[1]);
    if (mode != "--cpu-only" && mode != "--compile-only" &&
        mode != "--gpu" && mode != "--gpu-r2" &&
        mode != "--gpu-r2-shared" &&
        mode != "--gpu-steel-down" &&
        mode != "--gpu-steel" &&
        mode != "--gpu-steel-bm32" &&
        mode != "--gpu-steel-bm32-shared" &&
        mode != "--gpu-steel-bk64" &&
        mode != "--gpu-steel-bk64-shared" &&
        mode != "--gpu-steel-bn64" &&
        mode != "--gpu-steel-bn64-shared" &&
        mode != "--benchmark" && mode != "--benchmark-r2" &&
        mode != "--benchmark-steel-down" &&
        mode != "--benchmark-steel" &&
        mode != "--benchmark-steel-bm32" &&
        mode != "--benchmark-steel-bk64" &&
        mode != "--benchmark-steel-bn64") {
        std::cerr
            << "usage: tatara_native_routed_qgemm_probe "
               "--cpu-only|--compile-only|--gpu|--gpu-r2|"
               "--gpu-r2-shared|--gpu-steel-down|--gpu-steel|"
               "--gpu-steel-bm32|--gpu-steel-bm32-shared|"
               "--gpu-steel-bk64|--gpu-steel-bk64-shared|"
               "--gpu-steel-bn64|--gpu-steel-bn64-shared|"
               "--benchmark|--benchmark-r2|"
               "--benchmark-steel-down|--benchmark-steel|"
               "--benchmark-steel-bm32|--benchmark-steel-bk64|"
               "--benchmark-steel-bn64\n";
        return kExitUsage;
    }
    if ((mode == "--gpu-steel-down" || mode == "--gpu-steel" ||
         mode == "--gpu-steel-bm32" ||
         mode == "--gpu-steel-bm32-shared" ||
         mode == "--gpu-steel-bk64" ||
         mode == "--gpu-steel-bk64-shared" ||
         mode == "--gpu-steel-bn64" ||
         mode == "--gpu-steel-bn64-shared" ||
         mode == "--benchmark-steel-down" ||
         mode == "--benchmark-steel" ||
         mode == "--benchmark-steel-bm32" ||
         mode == "--benchmark-steel-bk64" ||
         mode == "--benchmark-steel-bn64") &&
        !generated::kKernelLibraryMlxSteelEnabled) {
        return kExitSteelUnavailable;
    }

    RouteFixture high;
    RouteFixture numeric;
    RouteFixture numeric_shared;
    RouteFixture high_bm32;
    RouteFixture numeric_bm32;
    RouteFixture numeric_shared_bm32;
    NumericFixture numeric_fixture;
    if (!make_high_water_routes(high) ||
        !make_numeric_routes(numeric) ||
        !make_numeric_shared_routes(numeric, numeric_shared) ||
        !make_bm32_route_fixture(high, high_bm32) ||
        !make_bm32_route_fixture(numeric, numeric_bm32) ||
        !make_bm32_route_fixture(
            numeric_shared, numeric_shared_bm32) ||
        high_bm32.tasks.size() !=
            kHighBm32ExpectedTasks ||
        numeric_bm32.tasks.size() !=
            kNumericBm32ExpectedTasks ||
        numeric_shared_bm32.tasks.size() !=
            kNumericSharedBm32ExpectedTasks ||
        !validate_route_fixture(high_bm32) ||
        !validate_route_fixture(numeric_bm32) ||
        !validate_route_fixture(numeric_shared_bm32) ||
        !make_numeric_fixture(numeric, numeric_fixture) ||
        !cpu_contract(
            high, numeric, numeric_shared, numeric_fixture)) {
        return kExitCpuContract;
    }
    if (mode == "--cpu-only") {
        std::cout
            << "native routed QGEMM fixture: PASS_CPU\n"
            << "  high-water tasks: " << high.tasks.size() << '\n'
            << "  numeric tasks: " << numeric.tasks.size() << '\n'
            << "  numeric shared tasks: "
            << numeric_shared.tasks.size() << '\n'
            << "  BM32 high/numeric/shared tasks: "
            << high_bm32.tasks.size() << '/'
            << numeric_bm32.tasks.size() << '/'
            << numeric_shared_bm32.tasks.size() << '\n'
            << "  family discriminators: "
            << numeric_fixture.family_differences << '\n'
            << "  direct/staged discriminators: "
            << numeric_fixture.direct_staged_differences << '\n'
            << "  command buffers submitted: 0\n";
        return 0;
    }

    auto device = create_system_device();
    if (!device) {
        return kExitDevice;
    }
    auto library = create_library_with_source(
        *device.device, generated::kernel_library_source());
    if (!library) {
        std::cerr << library.failure_description << '\n';
        return kExitLibrary;
    }
    Pipelines pipelines;
    const int pipeline_result =
        load_pipelines(*device.device, *library.library, pipelines);
    if (pipeline_result != 0) {
        return pipeline_result;
    }
    if (mode == "--compile-only") {
        std::cout
            << "native routed QGEMM fixture: PASS_COMPILE\n"
            << "  pipelines compiled: "
            << (generated::kKernelLibraryMlxSteelEnabled ? 15 : 7)
            << '\n'
            << "  command buffers submitted: 0\n";
        return 0;
    }

    auto queue = create_command_queue(*device.device);
    if (!queue) {
        return kExitQueue;
    }
    GpuResources resources;
    const int initialization_result = initialize_gpu_resources(
        *device.device, high, numeric, numeric_shared,
        high_bm32, numeric_bm32, numeric_shared_bm32,
        numeric_fixture, resources);
    if (initialization_result != 0) {
        return initialization_result;
    }

    std::uint32_t command_buffers = 0U;
    std::uint32_t direct_dispatches = 0U;
    std::uint32_t indirect_dispatches = 0U;
    std::uint32_t barriers = 0U;
    std::uint64_t indirect_threadgroups = 0U;

    const bool run_shared_fixture =
        mode == "--gpu-r2-shared" ||
        mode == "--gpu-steel-bm32-shared" ||
        mode == "--gpu-steel-bk64-shared" ||
        mode == "--gpu-steel-bn64-shared";
    const int producer_result = run_producers(
        *queue.command_queue, pipelines.producer, high, numeric,
        run_shared_fixture ? &numeric_shared : nullptr,
        resources, command_buffers, direct_dispatches);
    if (producer_result != 0) {
        return producer_result;
    }

    int validation_result = validate_producer_output(
        high, kUpColumnGroups, resources.high);
    if (validation_result == 0) {
        validation_result = validate_producer_output(
            high, kBn64UpColumnGroups, resources.high_bn64);
    }
    if (validation_result == 0 &&
        !numeric_task_bodies_identical(
            resources.high, resources.high_bn64)) {
        validation_result = kExitProducerPartition;
    }
    if (validation_result == 0) {
        validation_result = validate_producer_output(
            numeric, kUpColumnGroups, resources.numeric_up);
    }
    if (validation_result == 0) {
        validation_result = validate_producer_output(
            numeric, kDownColumnGroups, resources.numeric_down);
    }
    if (validation_result == 0 &&
        !numeric_task_bodies_identical(
            resources.numeric_up, resources.numeric_down)) {
        validation_result = kExitProducerPartition;
    }
    if (validation_result == 0) {
        validation_result = validate_producer_output(
            numeric, kBn64UpColumnGroups,
            resources.numeric_bn64_up);
    }
    if (validation_result == 0) {
        validation_result = validate_producer_output(
            numeric, kBn64DownColumnGroups,
            resources.numeric_bn64_down);
    }
    if (validation_result == 0 &&
        (!numeric_task_bodies_identical(
             resources.numeric_up,
             resources.numeric_bn64_up) ||
         !numeric_task_bodies_identical(
             resources.numeric_down,
             resources.numeric_bn64_down))) {
        validation_result = kExitProducerPartition;
    }
    if (validation_result == 0 && run_shared_fixture) {
        validation_result = validate_producer_output(
            numeric_shared, kUpColumnGroups,
            resources.numeric_shared_up);
    }
    if (validation_result == 0 && run_shared_fixture) {
        validation_result = validate_producer_output(
            numeric_shared, kDownColumnGroups,
            resources.numeric_shared_down);
    }
    if (validation_result == 0 && run_shared_fixture &&
        !numeric_task_bodies_identical(
            resources.numeric_shared_up,
            resources.numeric_shared_down)) {
        validation_result = kExitProducerPartition;
    }
    if (validation_result == 0 && run_shared_fixture) {
        validation_result = validate_producer_output(
            numeric_shared, kBn64UpColumnGroups,
            resources.numeric_shared_bn64_up);
    }
    if (validation_result == 0 && run_shared_fixture) {
        validation_result = validate_producer_output(
            numeric_shared, kBn64DownColumnGroups,
            resources.numeric_shared_bn64_down);
    }
    if (validation_result == 0 && run_shared_fixture &&
        (!numeric_task_bodies_identical(
             resources.numeric_shared_up,
             resources.numeric_shared_bn64_up) ||
         !numeric_task_bodies_identical(
             resources.numeric_shared_down,
             resources.numeric_shared_bn64_down))) {
        validation_result = kExitProducerPartition;
    }
    if (validation_result != 0) {
        return validation_result;
    }
    if (mode == "--benchmark" || mode == "--benchmark-r2" ||
        mode == "--benchmark-steel-down" ||
        mode == "--benchmark-steel" ||
        mode == "--benchmark-steel-bm32" ||
        mode == "--benchmark-steel-bk64" ||
        mode == "--benchmark-steel-bn64") {
        const BenchmarkArm treatment =
            mode == "--benchmark"
                ? BenchmarkArm::R1
                : mode == "--benchmark-r2"
                      ? BenchmarkArm::R2
                      : mode == "--benchmark-steel-down"
                            ? BenchmarkArm::SteelDown
                      : mode == "--benchmark-steel"
                            ? BenchmarkArm::Steel
                      : mode == "--benchmark-steel-bm32"
                            ? BenchmarkArm::SteelBm32
                      : mode == "--benchmark-steel-bk64"
                            ? BenchmarkArm::SteelBk64
                            : BenchmarkArm::SteelBn64;
        const int benchmark_result = run_benchmark(
            *device.device, *queue.command_queue, *library.library,
            pipelines, resources.high,
            treatment == BenchmarkArm::SteelBm32
                ? resources.high_bm32
            : treatment == BenchmarkArm::SteelBn64
                ? resources.high_bn64
                : resources.high,
            treatment);
        if (benchmark_result != 0) {
            return benchmark_result;
        }
        return 0;
    }
    capture_producer(resources.high);
    capture_producer(resources.high_bn64);
    capture_producer(resources.high_bm32);
    capture_producer(resources.numeric_up);
    capture_producer(resources.numeric_down);
    capture_producer(resources.numeric_bn64_up);
    capture_producer(resources.numeric_bn64_down);
    capture_producer(resources.numeric_bm32_up);
    capture_producer(resources.numeric_bm32_down);
    if (run_shared_fixture) {
        capture_producer(resources.numeric_shared_up);
        capture_producer(resources.numeric_shared_down);
        capture_producer(resources.numeric_shared_bn64_up);
        capture_producer(resources.numeric_shared_bn64_down);
        capture_producer(resources.numeric_shared_bm32_up);
        capture_producer(resources.numeric_shared_bm32_down);
    }
    const bool run_bn64_shared =
        mode == "--gpu-steel-bn64-shared";
    if (mode == "--gpu-steel-bn64" || run_bn64_shared) {
        const RouteFixture& routes =
            run_bn64_shared ? numeric_shared : numeric;
        const ProducerBuffers& control_up =
            run_bn64_shared
                ? resources.numeric_shared_up
                : resources.numeric_up;
        const ProducerBuffers& control_down =
            run_bn64_shared
                ? resources.numeric_shared_down
                : resources.numeric_down;
        const ProducerBuffers& treatment_up =
            run_bn64_shared
                ? resources.numeric_shared_bn64_up
                : resources.numeric_bn64_up;
        const ProducerBuffers& treatment_down =
            run_bn64_shared
                ? resources.numeric_shared_bn64_down
                : resources.numeric_bn64_down;
        int compute_result = run_staged_compute(
            *queue.command_queue, pipelines, routes,
            control_up, control_down, resources.compute,
            resources.compute.staged_hidden,
            resources.compute.staged_partials,
            command_buffers,
            /*steel_upgate=*/true,
            /*steel_down=*/true);
        if (compute_result != 0) {
            return compute_result;
        }
        compute_result = run_staged_compute(
            *queue.command_queue, pipelines, routes,
            treatment_up, treatment_down, resources.compute,
            resources.compute.bn64_hidden,
            resources.compute.bn64_partials,
            command_buffers,
            /*steel_upgate=*/false,
            /*steel_down=*/false,
            /*steel_bm32=*/false,
            /*steel_bk64=*/false,
            /*steel_bn64=*/true);
        if (compute_result != 0) {
            return compute_result;
        }
        const int output_result =
            validate_bn64_compute_output(
                resources, run_bn64_shared);
        if (output_result != 0) {
            return output_result;
        }
        const std::size_t task_count =
            run_bn64_shared
                ? kNumericSharedExpectedTasks
                : kNumericExpectedTasks;
        if (command_buffers != 3U ||
            direct_dispatches !=
                (run_bn64_shared ? 5U : 3U)) {
            return kExitSubmissionAccounting;
        }
        std::cout
            << "native routed QGEMM fixture: "
            << (run_bn64_shared
                    ? "PASS_GPU_STEEL_BN64_SHARED"
                    : "PASS_GPU_STEEL_BN64")
            << '\n'
            << "  device: " << device.device->name() << '\n'
            << "  command buffers submitted: "
            << command_buffers << '\n'
            << "  direct producer dispatches: "
            << direct_dispatches << '\n'
            << "  indirect compute dispatches: 4\n"
            << "  compute threadgroups: "
            << task_count *
                   (kUpColumnGroups + kDownColumnGroups +
                    kBn64UpColumnGroups +
                    kBn64DownColumnGroups)
            << '\n'
            << "  BN32/BN64 byte identity: yes\n"
            << "  warmups: 0\n"
            << "  timing samples: 0\n";
        return 0;
    }
    const bool run_bk64_shared =
        mode == "--gpu-steel-bk64-shared";
    if (mode == "--gpu-steel-bk64" || run_bk64_shared) {
        const RouteFixture& routes =
            run_bk64_shared ? numeric_shared : numeric;
        const ProducerBuffers& up =
            run_bk64_shared
                ? resources.numeric_shared_up
                : resources.numeric_up;
        const ProducerBuffers& down =
            run_bk64_shared
                ? resources.numeric_shared_down
                : resources.numeric_down;
        int compute_result = run_staged_compute(
            *queue.command_queue, pipelines, routes, up, down,
            resources.compute,
            resources.compute.staged_hidden,
            resources.compute.staged_partials,
            command_buffers,
            /*steel_upgate=*/true,
            /*steel_down=*/true);
        if (compute_result != 0) {
            return compute_result;
        }
        compute_result = run_staged_compute(
            *queue.command_queue, pipelines, routes, up, down,
            resources.compute,
            resources.compute.bk64_hidden,
            resources.compute.bk64_partials,
            command_buffers,
            /*steel_upgate=*/false,
            /*steel_down=*/false,
            /*steel_bm32=*/false,
            /*steel_bk64=*/true);
        if (compute_result != 0) {
            return compute_result;
        }
        const int output_result =
            validate_bk64_compute_output(
                resources, run_bk64_shared);
        if (output_result != 0) {
            return output_result;
        }
        const std::size_t task_count =
            run_bk64_shared
                ? kNumericSharedExpectedTasks
                : kNumericExpectedTasks;
        if (command_buffers != 3U ||
            direct_dispatches !=
                (run_bk64_shared ? 5U : 3U)) {
            return kExitSubmissionAccounting;
        }
        std::cout
            << "native routed QGEMM fixture: "
            << (run_bk64_shared
                    ? "PASS_GPU_STEEL_BK64_SHARED"
                    : "PASS_GPU_STEEL_BK64")
            << '\n'
            << "  device: " << device.device->name() << '\n'
            << "  command buffers submitted: "
            << command_buffers << '\n'
            << "  direct producer dispatches: "
            << direct_dispatches << '\n'
            << "  indirect compute dispatches: 4\n"
            << "  compute threadgroups: "
            << 2U * task_count *
                   (kUpColumnGroups + kDownColumnGroups)
            << '\n'
            << "  BK32/BK64 byte identity: yes\n"
            << "  warmups: 0\n"
            << "  timing samples: 0\n";
        return 0;
    }
    const bool run_bm32_shared =
        mode == "--gpu-steel-bm32-shared";
    if (mode == "--gpu-steel-bm32" || run_bm32_shared) {
        const RouteFixture& routes =
            run_bm32_shared ? numeric_shared_bm32
                            : numeric_bm32;
        const ProducerBuffers& up =
            run_bm32_shared
                ? resources.numeric_shared_bm32_up
                : resources.numeric_bm32_up;
        const ProducerBuffers& down =
            run_bm32_shared
                ? resources.numeric_shared_bm32_down
                : resources.numeric_bm32_down;
        const int compute_result = run_staged_compute(
            *queue.command_queue, pipelines, routes, up, down,
            resources.compute, resources.compute.bm32_hidden,
            resources.compute.bm32_partials, command_buffers,
            /*steel_upgate=*/false,
            /*steel_down=*/false,
            /*steel_bm32=*/true);
        if (compute_result != 0) {
            return compute_result;
        }
        const int output_result =
            validate_bm32_compute_output(
                numeric_fixture, resources, run_bm32_shared);
        if (output_result != 0) {
            return output_result;
        }
        const std::size_t task_count =
            run_bm32_shared
                ? kNumericSharedBm32ExpectedTasks
                : kNumericBm32ExpectedTasks;
        if (command_buffers != 2U ||
            direct_dispatches !=
                (run_bm32_shared ? 5U : 3U)) {
            return kExitSubmissionAccounting;
        }
        std::cout
            << "native routed QGEMM fixture: "
            << (run_bm32_shared
                    ? "PASS_GPU_STEEL_BM32_SHARED"
                    : "PASS_GPU_STEEL_BM32")
            << '\n'
            << "  device: " << device.device->name() << '\n'
            << "  command buffers submitted: "
            << command_buffers << '\n'
            << "  direct producer dispatches: "
            << direct_dispatches << '\n'
            << "  indirect compute dispatches: 2\n"
            << "  compute threadgroups: "
            << task_count *
                   (kUpColumnGroups + kDownColumnGroups)
            << '\n'
            << "  BM32 tail rows: "
            << (run_bm32_shared ? "17 shared" : "17 routed")
            << '\n'
            << "  warmups: 0\n"
            << "  timing samples: 0\n";
        return 0;
    }
    if (run_shared_fixture) {
        const int compute_result = run_staged_compute(
            *queue.command_queue, pipelines, numeric_shared,
            resources.numeric_shared_up,
            resources.numeric_shared_down, resources.compute,
            resources.compute.assimilated_hidden,
            resources.compute.assimilated_partials,
            command_buffers);
        if (compute_result != 0) {
            return compute_result;
        }
        const int output_result =
            validate_assimilated_compute_output(
                numeric_fixture, resources);
        if (output_result != 0) {
            return output_result;
        }
        if (command_buffers != 2U || direct_dispatches != 5U) {
            return kExitSubmissionAccounting;
        }
        std::cout
            << "native routed QGEMM fixture: PASS_GPU_R2_SHARED\n"
            << "  device: " << device.device->name() << '\n'
            << "  command buffers submitted: "
            << command_buffers << '\n'
            << "  direct producer dispatches: "
            << direct_dispatches << '\n'
            << "  indirect compute dispatches: 2\n"
            << "  compute threadgroups: "
            << kNumericSharedExpectedTasks *
                   (kUpColumnGroups + kDownColumnGroups)
            << '\n'
            << "  shared tail rows: 16,1\n"
            << "  warmups: 0\n"
            << "  timing samples: 0\n";
        return 0;
    }

    if (mode == "--gpu-r2" || mode == "--gpu-steel-down" ||
        mode == "--gpu-steel") {
        const bool steel_upgate = mode == "--gpu-steel";
        const bool steel_down =
            mode == "--gpu-steel-down" ||
            mode == "--gpu-steel";
        const int compute_result = run_staged_compute(
            *queue.command_queue, pipelines, numeric,
            resources.numeric_up, resources.numeric_down,
            resources.compute, resources.compute.staged_hidden,
            resources.compute.staged_partials,
            command_buffers, steel_upgate, steel_down);
        if (compute_result != 0) {
            return compute_result;
        }
        const int output_result =
            validate_staged_compute_output(
                numeric_fixture, resources);
        if (output_result != 0) {
            return output_result;
        }
        if (command_buffers != 2U || direct_dispatches != 3U) {
            return kExitSubmissionAccounting;
        }
        std::cout
            << "native routed QGEMM fixture: "
            << (steel_upgate ? "PASS_GPU_STEEL"
                             : steel_down
                                   ? "PASS_GPU_STEEL_DOWN"
                                   : "PASS_GPU_R2")
            << '\n'
            << "  device: " << device.device->name() << '\n'
            << "  command buffers submitted: "
            << command_buffers << '\n'
            << "  direct producer dispatches: "
            << direct_dispatches << '\n'
            << "  indirect compute dispatches: 2\n"
            << "  compute threadgroups: "
            << kNumericExpectedTasks *
                   (kUpColumnGroups + kDownColumnGroups)
            << '\n'
            << "  warmups: 0\n"
            << "  timing samples: 0\n";
        return 0;
    }

    const int compute_result = run_compute(
        *queue.command_queue, pipelines, numeric, resources,
        command_buffers, indirect_dispatches, barriers,
        indirect_threadgroups);
    if (compute_result != 0) {
        return compute_result;
    }
    const int output_result =
        validate_compute_output(numeric_fixture, resources);
    if (output_result != 0) {
        return output_result;
    }
    if (command_buffers != 2U || direct_dispatches != 3U ||
        indirect_dispatches != 4U || barriers != 2U ||
        indirect_threadgroups != kExpectedComputeThreadgroups) {
        return kExitSubmissionAccounting;
    }

    std::cout
        << "native routed QGEMM fixture: PASS_GPU\n"
        << "  device: " << device.device->name() << '\n'
        << "  command buffers submitted: " << command_buffers << '\n'
        << "  direct producer dispatches: " << direct_dispatches << '\n'
        << "  indirect compute dispatches: " << indirect_dispatches
        << '\n'
        << "  compute threadgroups: " << indirect_threadgroups << '\n'
        << "  warmups: 0\n"
        << "  timing samples: 0\n";
    return 0;
}
