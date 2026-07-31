#include "tatara/backend/metal/commands.h"
#include "tatara/backend/metal/pipeline.h"
#include "tatara/backend/metal/resources.h"
#include "tatara/generated/kernel_library.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace {

using namespace tatara::backend::metal;

constexpr std::uint32_t kQ4ValuesPerWord = 8;
constexpr std::uint32_t kGroupSize = 64;
constexpr std::uint32_t kThreads = 128;
constexpr std::uint32_t kTileRows = 32;
constexpr std::uint32_t kTileColumns = 32;
constexpr std::uint32_t kSteelThreads = 64;
constexpr std::uint32_t kSteelTileRows = 16;
constexpr std::size_t kGuardElements = 16;
constexpr std::uint16_t kOutputCanary = 0x5A5A;
constexpr std::uint32_t kPackedCanary = 0xD3C5A79BU;
constexpr std::uint16_t kInputCanary = 0x4B4B;
constexpr std::uint32_t kMaximumBfloatUlp = 2;
constexpr float kMaximumNormalizedError = 0.02F;
constexpr float kMaximumRelativeL2Error = 0.02F;
constexpr std::string_view kKernelName = "native_dense_qgemm_q4_bf16_n1";

constexpr int kExitUsage = 40;
constexpr int kExitCpuContract = 41;
constexpr int kExitDevice = 42;
constexpr int kExitLibrary = 43;
constexpr int kExitFunction = 44;
constexpr int kExitPipeline = 45;
constexpr int kExitBuffer = 46;
constexpr int kExitCommandBuffer = 47;
constexpr int kExitComputePass = 48;
constexpr int kExitEncode = 49;
constexpr int kExitCommit = 50;
constexpr int kExitExecution = 51;
constexpr int kExitNumerics = 52;
constexpr int kExitCanary = 53;

struct FixtureSpec {
    std::string_view name;
    std::uint32_t rows;
    std::uint32_t columns;
    std::uint32_t reduction;
    std::uint64_t activation_stride;
    std::uint64_t packed_stride_words;
    std::uint64_t parameter_stride;
    std::uint64_t output_stride;
};

constexpr std::array<FixtureSpec, 4> kFixtureSpecs{{
    {
        .name = "minimal",
        .rows = 1,
        .columns = 1,
        .reduction = 64,
        .activation_stride = 67,
        .packed_stride_words = 11,
        .parameter_stride = 3,
        .output_stride = 5,
    },
    {
        .name = "mn-tail",
        .rows = 33,
        .columns = 35,
        .reduction = 128,
        .activation_stride = 131,
        .packed_stride_words = 19,
        .parameter_stride = 5,
        .output_stride = 41,
    },
    {
        .name = "long-reduction",
        .rows = 32,
        .columns = 32,
        .reduction = 2048,
        .activation_stride = 2051,
        .packed_stride_words = 259,
        .parameter_stride = 35,
        .output_stride = 37,
    },
    {
        .name = "steel-aligned",
        .rows = 16,
        .columns = 32,
        .reduction = 2048,
        .activation_stride = 2048,
        .packed_stride_words = 256,
        .parameter_stride = 32,
        .output_stride = 32,
    },
}};

static_assert(kFixtureSpecs.size() == 4);
static_assert(kThreads == generated::kKernelLibraryNativeDenseQgemmN1Threads);
static_assert(kSteelThreads == 64);
static_assert(kTileRows == generated::kKernelLibraryNativeDenseQgemmN1TileRows);
static_assert(kTileColumns == generated::kKernelLibraryNativeDenseQgemmN1TileColumns);
static_assert(generated::kKernelLibraryGroupSize == kGroupSize);

std::uint16_t bfloat16(float value) noexcept {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t exponent = bits & 0x7F800000U;
    if (exponent != 0x7F800000U) {
        bits += 0x7FFFU + ((bits >> 16U) & 1U);
    }
    return static_cast<std::uint16_t>(bits >> 16U);
}

float from_bfloat16(std::uint16_t value) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

std::uint32_t ordered_bfloat(std::uint16_t value) noexcept {
    if ((value & 0x8000U) != 0U) {
        return 0x8000U - static_cast<std::uint32_t>(value & 0x7FFFU);
    }
    return 0x8000U + static_cast<std::uint32_t>(value);
}

std::uint32_t bfloat_ulp_distance(std::uint16_t left, std::uint16_t right) noexcept {
    const std::uint32_t ordered_left = ordered_bfloat(left);
    const std::uint32_t ordered_right = ordered_bfloat(right);
    return ordered_left > ordered_right ? ordered_left - ordered_right
                                        : ordered_right - ordered_left;
}

bool checked_multiply(std::size_t left, std::size_t right, std::size_t& result) noexcept {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

struct Fixture {
    FixtureSpec spec;
    std::vector<std::uint16_t> activations;
    std::vector<std::uint32_t> packed;
    std::vector<std::uint16_t> scales;
    std::vector<std::uint16_t> biases;
    std::vector<std::uint16_t> expected;
    std::vector<std::uint16_t> direct;
    std::vector<std::uint16_t> output_storage;
};

std::uint32_t quantized_value(std::uint32_t row, std::uint32_t reduction) noexcept {
    return (row * 11U + reduction * 5U + 3U) & 15U;
}

std::vector<std::uint16_t> reference(const Fixture& fixture, bool stage_weights) {
    const FixtureSpec& spec = fixture.spec;
    std::vector<std::uint16_t> result(
        static_cast<std::size_t>(spec.rows) * spec.columns);
    for (std::uint32_t row = 0; row < spec.rows; ++row) {
        for (std::uint32_t column = 0; column < spec.columns; ++column) {
            float accumulator = 0.0F;
            for (std::uint32_t reduction = 0; reduction < spec.reduction; ++reduction) {
                const std::size_t word_index =
                    static_cast<std::size_t>(column) * spec.packed_stride_words +
                    reduction / kQ4ValuesPerWord;
                const std::uint32_t word = fixture.packed[word_index];
                const std::uint32_t quantized =
                    (word >> (4U * (reduction % kQ4ValuesPerWord))) & 15U;
                const std::size_t parameter_index =
                    static_cast<std::size_t>(column) * spec.parameter_stride +
                    reduction / kGroupSize;
                const float scale = from_bfloat16(fixture.scales[parameter_index]);
                const float bias = from_bfloat16(fixture.biases[parameter_index]);
                const float dequantized =
                    std::fma(static_cast<float>(quantized), scale, bias);
                const float weight =
                    stage_weights ? from_bfloat16(bfloat16(dequantized)) : dequantized;
                const float activation = from_bfloat16(
                    fixture.activations[
                        static_cast<std::size_t>(row) * spec.activation_stride +
                        reduction]);
                accumulator = std::fma(activation, weight, accumulator);
            }
            result[static_cast<std::size_t>(row) * spec.columns + column] =
                bfloat16(accumulator);
        }
    }
    return result;
}

bool make_fixture(const FixtureSpec& spec, Fixture& fixture) {
    if (spec.rows == 0 || spec.columns == 0 || spec.reduction == 0 ||
        spec.reduction % kGroupSize != 0 ||
        spec.activation_stride < spec.reduction ||
        spec.packed_stride_words <
            spec.reduction / kQ4ValuesPerWord ||
        spec.parameter_stride < spec.reduction / kGroupSize ||
        spec.output_stride < spec.columns) {
        return false;
    }
    std::size_t activation_count = 0;
    std::size_t packed_count = 0;
    std::size_t parameter_count = 0;
    std::size_t output_body_count = 0;
    if (!checked_multiply(spec.rows, spec.activation_stride, activation_count) ||
        !checked_multiply(spec.columns, spec.packed_stride_words, packed_count) ||
        !checked_multiply(spec.columns, spec.parameter_stride, parameter_count) ||
        !checked_multiply(spec.rows, spec.output_stride, output_body_count) ||
        output_body_count >
            std::numeric_limits<std::size_t>::max() - 2U * kGuardElements) {
        return false;
    }

    fixture.spec = spec;
    fixture.activations.assign(activation_count, kInputCanary);
    fixture.packed.assign(packed_count, kPackedCanary);
    fixture.scales.assign(parameter_count, kInputCanary);
    fixture.biases.assign(parameter_count, kInputCanary);
    fixture.output_storage.assign(
        2U * kGuardElements + output_body_count, kOutputCanary);

    for (std::uint32_t row = 0; row < spec.rows; ++row) {
        for (std::uint32_t reduction = 0; reduction < spec.reduction; ++reduction) {
            const std::int32_t band =
                static_cast<std::int32_t>((row * 19U + reduction * 7U) % 37U) - 18;
            const float value =
                static_cast<float>(band) / 19.0F + 0.03125F;
            fixture.activations[
                static_cast<std::size_t>(row) * spec.activation_stride +
                reduction] = bfloat16(value);
        }
    }
    const std::uint32_t words_per_row =
        spec.reduction / kQ4ValuesPerWord;
    const std::uint32_t groups_per_row = spec.reduction / kGroupSize;
    for (std::uint32_t column = 0; column < spec.columns; ++column) {
        for (std::uint32_t word = 0; word < words_per_row; ++word) {
            fixture.packed[
                static_cast<std::size_t>(column) * spec.packed_stride_words +
                word] = 0U;
        }
        for (std::uint32_t reduction = 0; reduction < spec.reduction; ++reduction) {
            const std::size_t word_index =
                static_cast<std::size_t>(column) * spec.packed_stride_words +
                reduction / kQ4ValuesPerWord;
            fixture.packed[word_index] |=
                quantized_value(column, reduction)
                << (4U * (reduction % kQ4ValuesPerWord));
        }
        for (std::uint32_t group = 0; group < groups_per_row; ++group) {
            const std::size_t index =
                static_cast<std::size_t>(column) * spec.parameter_stride + group;
            fixture.scales[index] = bfloat16(
                0.071F + 0.013F * static_cast<float>(column % 7U) +
                0.019F * static_cast<float>(group % 5U));
            fixture.biases[index] = bfloat16(
                -0.137F + 0.017F * static_cast<float>(column % 5U) -
                0.011F * static_cast<float>(group % 3U));
        }
    }
    fixture.expected = reference(fixture, true);
    fixture.direct = reference(fixture, false);
    return true;
}

bool cpu_contract(std::span<const Fixture> fixtures) {
    std::size_t discriminating_outputs = 0;
    for (const Fixture& fixture : fixtures) {
        if (fixture.expected.size() != fixture.direct.size()) {
            return false;
        }
        for (std::size_t index = 0; index < fixture.expected.size(); ++index) {
            discriminating_outputs +=
                static_cast<std::size_t>(
                    fixture.expected[index] != fixture.direct[index]);
        }
    }
    return discriminating_outputs != 0;
}

template <typename Value>
bool upload(const MetalBuffer& buffer, std::span<const Value> values) {
    std::size_t bytes = 0;
    if (!checked_multiply(values.size(), sizeof(Value), bytes) ||
        bytes > buffer.size_bytes()) {
        return false;
    }
    std::memcpy(buffer.contents(), values.data(), bytes);
    return true;
}

struct NumericReport {
    float maximum_absolute = 0.0F;
    float maximum_normalized = 0.0F;
    float relative_l2 = 0.0F;
    std::uint32_t maximum_bfloat_ulp = 0;
    bool finite = true;
};

bool canaries_intact(const Fixture& fixture, std::span<const std::uint16_t> actual) {
    const FixtureSpec& spec = fixture.spec;
    for (std::size_t index = 0; index < kGuardElements; ++index) {
        if (actual[index] != kOutputCanary ||
            actual[actual.size() - kGuardElements + index] != kOutputCanary) {
            return false;
        }
    }
    const std::size_t body_begin = kGuardElements;
    for (std::uint32_t row = 0; row < spec.rows; ++row) {
        const std::size_t row_begin =
            body_begin + static_cast<std::size_t>(row) * spec.output_stride;
        for (std::uint64_t column = spec.columns;
             column < spec.output_stride; ++column) {
            if (actual[row_begin + column] != kOutputCanary) {
                return false;
            }
        }
    }
    return true;
}

NumericReport compare_output(
    const Fixture& fixture, std::span<const std::uint16_t> actual) {
    NumericReport report;
    double squared_error = 0.0;
    double squared_reference = 0.0;
    const FixtureSpec& spec = fixture.spec;
    const std::size_t body_begin = kGuardElements;
    for (std::uint32_t row = 0; row < spec.rows; ++row) {
        for (std::uint32_t column = 0; column < spec.columns; ++column) {
            const std::uint16_t actual_bits =
                actual[
                    body_begin +
                    static_cast<std::size_t>(row) * spec.output_stride +
                    column];
            const std::uint16_t expected_bits =
                fixture.expected[
                    static_cast<std::size_t>(row) * spec.columns + column];
            const float actual_value = from_bfloat16(actual_bits);
            const float expected_value = from_bfloat16(expected_bits);
            report.finite =
                report.finite && std::isfinite(actual_value) &&
                std::isfinite(expected_value);
            const float absolute = std::fabs(actual_value - expected_value);
            squared_error += static_cast<double>(absolute) *
                             static_cast<double>(absolute);
            squared_reference += static_cast<double>(expected_value) *
                                 static_cast<double>(expected_value);
            const float normalized =
                absolute / std::fmax(1.0F, std::fabs(expected_value));
            report.maximum_absolute =
                std::fmax(report.maximum_absolute, absolute);
            report.maximum_normalized =
                std::fmax(report.maximum_normalized, normalized);
            const std::uint32_t ulp =
                bfloat_ulp_distance(actual_bits, expected_bits);
            if (ulp > report.maximum_bfloat_ulp) {
                report.maximum_bfloat_ulp = ulp;
            }
        }
    }
    report.relative_l2 = static_cast<float>(
        std::sqrt(squared_error / std::fmax(1.0, squared_reference)));
    return report;
}

int run_gpu_case(
    const MetalDevice& device, const MetalCommandQueue& queue,
    const MetalComputePipeline& pipeline, Fixture& fixture,
    std::uint32_t& submissions) {
    const auto activation =
        create_shared_buffer(device, fixture.activations.size() * sizeof(std::uint16_t));
    const auto packed =
        create_shared_buffer(device, fixture.packed.size() * sizeof(std::uint32_t));
    const auto scales =
        create_shared_buffer(device, fixture.scales.size() * sizeof(std::uint16_t));
    const auto biases =
        create_shared_buffer(device, fixture.biases.size() * sizeof(std::uint16_t));
    const auto output =
        create_shared_buffer(device, fixture.output_storage.size() * sizeof(std::uint16_t));
    if (!activation || !packed || !scales || !biases || !output) {
        return kExitBuffer;
    }
    if (!upload(*activation.buffer, std::span<const std::uint16_t>(fixture.activations)) ||
        !upload(*packed.buffer, std::span<const std::uint32_t>(fixture.packed)) ||
        !upload(*scales.buffer, std::span<const std::uint16_t>(fixture.scales)) ||
        !upload(*biases.buffer, std::span<const std::uint16_t>(fixture.biases)) ||
        !upload(*output.buffer, std::span<const std::uint16_t>(fixture.output_storage))) {
        return kExitBuffer;
    }

    auto command = create_command_buffer(queue);
    if (!command) {
        return kExitCommandBuffer;
    }
    auto pass = begin_compute_pass(std::move(*command.command_buffer));
    if (!pass) {
        return kExitComputePass;
    }
    auto encode = [&](MetalCommandError error) {
        return error == MetalCommandError::None;
    };
    const FixtureSpec& spec = fixture.spec;
    if (!encode(set_compute_pipeline(*pass.compute_pass, pipeline)) ||
        !encode(set_buffer(*pass.compute_pass, *activation.buffer, 0, 0)) ||
        !encode(set_buffer(*pass.compute_pass, *packed.buffer, 0, 1)) ||
        !encode(set_buffer(*pass.compute_pass, *scales.buffer, 0, 2)) ||
        !encode(set_buffer(*pass.compute_pass, *biases.buffer, 0, 3)) ||
        !encode(set_buffer(
            *pass.compute_pass, *output.buffer,
            kGuardElements * sizeof(std::uint16_t), 4)) ||
        !encode(set_bytes(
            *pass.compute_pass, &spec.rows, sizeof(spec.rows), 5)) ||
        !encode(set_bytes(
            *pass.compute_pass, &spec.columns, sizeof(spec.columns), 6)) ||
        !encode(set_bytes(
            *pass.compute_pass, &spec.reduction, sizeof(spec.reduction), 7)) ||
        !encode(set_bytes(
            *pass.compute_pass, &spec.activation_stride,
            sizeof(spec.activation_stride), 8)) ||
        !encode(set_bytes(
            *pass.compute_pass, &spec.packed_stride_words,
            sizeof(spec.packed_stride_words), 9)) ||
        !encode(set_bytes(
            *pass.compute_pass, &spec.parameter_stride,
            sizeof(spec.parameter_stride), 10)) ||
        !encode(set_bytes(
            *pass.compute_pass, &spec.output_stride,
            sizeof(spec.output_stride), 11)) ||
        !encode(dispatch_threadgroups(
            *pass.compute_pass,
            {
                .width = (spec.columns + kTileColumns - 1U) / kTileColumns,
                .height = (spec.rows + kTileRows - 1U) / kTileRows,
                .depth = 1,
            },
            {.width = kThreads, .height = 1, .depth = 1}))) {
        return kExitEncode;
    }
    auto ended = end_compute_pass(std::move(*pass.compute_pass));
    if (!ended) {
        return kExitComputePass;
    }
    auto pending = commit(std::move(*ended.command_buffer));
    if (!pending) {
        return kExitCommit;
    }
    ++submissions;
    auto completed =
        wait_until_completed(std::move(*pending.pending_execution));
    if (!completed) {
        std::cerr << "N1 execution failed: "
                  << completed.failure_description.view() << '\n';
        return kExitExecution;
    }

    std::span<const std::uint16_t> actual(
        static_cast<const std::uint16_t*>(output.buffer->contents()),
        fixture.output_storage.size());
    if (!canaries_intact(fixture, actual)) {
        return kExitCanary;
    }
    const NumericReport report = compare_output(fixture, actual);
    std::cout << "tatara_native_dense_qgemm_case"
              << " name=" << spec.name
              << " rows=" << spec.rows
              << " columns=" << spec.columns
              << " reduction=" << spec.reduction
              << " max_abs=" << report.maximum_absolute
              << " max_norm=" << report.maximum_normalized
              << " max_bfloat_ulp=" << report.maximum_bfloat_ulp
              << '\n';
    if (!report.finite ||
        report.maximum_normalized > kMaximumNormalizedError ||
        report.maximum_bfloat_ulp > kMaximumBfloatUlp) {
        return kExitNumerics;
    }
    return 0;
}

template <typename Value>
bool buffer_equals(const MetalBuffer& buffer,
                   std::span<const Value> expected) noexcept {
    const std::size_t bytes = expected.size() * sizeof(Value);
    return bytes <= buffer.size_bytes() &&
           std::memcmp(buffer.contents(), expected.data(), bytes) == 0;
}

int run_gpu_steel_case(
    const MetalDevice& device, const MetalCommandQueue& queue,
    const MetalComputePipeline& pipeline, Fixture& fixture,
    bool dense, std::uint32_t& submissions) {
    const FixtureSpec& spec = fixture.spec;
    if (spec.rows != kSteelTileRows ||
        spec.columns != kTileColumns ||
        spec.activation_stride != spec.reduction ||
        spec.packed_stride_words != spec.reduction / kQ4ValuesPerWord ||
        spec.parameter_stride != spec.reduction / kGroupSize ||
        spec.output_stride != spec.columns) {
        return kExitCpuContract;
    }
    std::vector<std::uint32_t> indices(spec.rows, 0U);
    const auto activation = create_shared_buffer(
        device, fixture.activations.size() * sizeof(std::uint16_t));
    const auto packed = create_shared_buffer(
        device, fixture.packed.size() * sizeof(std::uint32_t));
    const auto scales = create_shared_buffer(
        device, fixture.scales.size() * sizeof(std::uint16_t));
    const auto biases = create_shared_buffer(
        device, fixture.biases.size() * sizeof(std::uint16_t));
    const auto index_buffer = create_shared_buffer(
        device, indices.size() * sizeof(std::uint32_t));
    const auto output = create_shared_buffer(
        device, fixture.output_storage.size() * sizeof(std::uint16_t));
    if (!activation || !packed || !scales || !biases ||
        !index_buffer || !output) {
        return kExitBuffer;
    }
    if (!upload(
            *activation.buffer,
            std::span<const std::uint16_t>(fixture.activations)) ||
        !upload(
            *packed.buffer,
            std::span<const std::uint32_t>(fixture.packed)) ||
        !upload(
            *scales.buffer,
            std::span<const std::uint16_t>(fixture.scales)) ||
        !upload(
            *biases.buffer,
            std::span<const std::uint16_t>(fixture.biases)) ||
        !upload(
            *index_buffer.buffer,
            std::span<const std::uint32_t>(indices)) ||
        !upload(
            *output.buffer,
            std::span<const std::uint16_t>(fixture.output_storage))) {
        return kExitBuffer;
    }

    auto command = create_command_buffer(queue);
    if (!command) {
        return kExitCommandBuffer;
    }
    auto pass = begin_compute_pass(std::move(*command.command_buffer));
    if (!pass) {
        return kExitComputePass;
    }
    const auto encode = [](MetalCommandError error) {
        return error == MetalCommandError::None;
    };
    const std::int32_t rows = static_cast<std::int32_t>(spec.rows);
    const std::int32_t columns = static_cast<std::int32_t>(spec.columns);
    const std::int32_t reduction =
        static_cast<std::int32_t>(spec.reduction);
    const std::int32_t output_stride =
        static_cast<std::int32_t>(spec.output_stride);
    if (!encode(set_compute_pipeline(*pass.compute_pass, pipeline))) {
        return kExitEncode;
    }
    const bool arguments_encoded =
        dense
        ? encode(set_buffer(
              *pass.compute_pass, *packed.buffer, 0, 0)) &&
          encode(set_buffer(
              *pass.compute_pass, *scales.buffer, 0, 1)) &&
          encode(set_buffer(
              *pass.compute_pass, *biases.buffer, 0, 2)) &&
          encode(set_buffer(
              *pass.compute_pass, *activation.buffer, 0, 3)) &&
          encode(set_buffer(
              *pass.compute_pass, *output.buffer,
              kGuardElements * sizeof(std::uint16_t), 4)) &&
          encode(set_bytes(
              *pass.compute_pass, &reduction,
              sizeof(reduction), 5)) &&
          encode(set_bytes(
              *pass.compute_pass, &columns,
              sizeof(columns), 6)) &&
          encode(set_bytes(
              *pass.compute_pass, &rows, sizeof(rows), 7)) &&
          encode(set_bytes(
              *pass.compute_pass, &output_stride,
              sizeof(output_stride), 8)) &&
          encode(dispatch_threadgroups(
              *pass.compute_pass,
              {.width = 1, .height = 1, .depth = 1},
              {.width = kThreads / 4U, .height = 2, .depth = 2}))
        : encode(set_buffer(
            *pass.compute_pass, *activation.buffer, 0, 0)) &&
          encode(set_buffer(
            *pass.compute_pass, *packed.buffer, 0, 1)) &&
          encode(set_buffer(
            *pass.compute_pass, *scales.buffer, 0, 2)) &&
          encode(set_buffer(
            *pass.compute_pass, *biases.buffer, 0, 3)) &&
          encode(set_buffer(
            *pass.compute_pass, *index_buffer.buffer, 0, 4)) &&
          encode(set_buffer(
            *pass.compute_pass, *output.buffer,
            kGuardElements * sizeof(std::uint16_t), 5)) &&
          encode(set_bytes(
            *pass.compute_pass, &rows, sizeof(rows), 6)) &&
          encode(set_bytes(
            *pass.compute_pass, &columns, sizeof(columns), 7)) &&
          encode(set_bytes(
            *pass.compute_pass, &reduction, sizeof(reduction), 8)) &&
          encode(dispatch_threadgroups(
            *pass.compute_pass,
            {.width = 1, .height = 1, .depth = 1},
            {.width = kSteelThreads / 2U,
             .height = 2,
             .depth = 1}));
    if (!arguments_encoded) {
        return kExitEncode;
    }
    auto ended = end_compute_pass(std::move(*pass.compute_pass));
    if (!ended) {
        return kExitComputePass;
    }
    auto pending = commit(std::move(*ended.command_buffer));
    if (!pending) {
        return kExitCommit;
    }
    ++submissions;
    auto completed =
        wait_until_completed(std::move(*pending.pending_execution));
    if (!completed) {
        std::cerr << "Steel execution failed: "
                  << completed.failure_description.view() << '\n';
        return kExitExecution;
    }

    if (!buffer_equals(
            *activation.buffer,
            std::span<const std::uint16_t>(fixture.activations)) ||
        !buffer_equals(
            *packed.buffer,
            std::span<const std::uint32_t>(fixture.packed)) ||
        !buffer_equals(
            *scales.buffer,
            std::span<const std::uint16_t>(fixture.scales)) ||
        !buffer_equals(
            *biases.buffer,
            std::span<const std::uint16_t>(fixture.biases)) ||
        !buffer_equals(
            *index_buffer.buffer,
            std::span<const std::uint32_t>(indices))) {
        return kExitCanary;
    }
    const std::span<const std::uint16_t> actual(
        static_cast<const std::uint16_t*>(output.buffer->contents()),
        fixture.output_storage.size());
    if (!canaries_intact(fixture, actual)) {
        return kExitCanary;
    }
    const NumericReport report = compare_output(fixture, actual);
    std::cout << "tatara_mlx_steel_qgemm_case"
              << " name=" << spec.name
              << " rows=" << spec.rows
              << " columns=" << spec.columns
              << " reduction=" << spec.reduction
              << " max_abs=" << report.maximum_absolute
              << " max_norm=" << report.maximum_normalized
              << " rel_l2=" << report.relative_l2
              << " max_bfloat_ulp=" << report.maximum_bfloat_ulp
              << '\n';
    if (!report.finite ||
        report.maximum_normalized > kMaximumNormalizedError ||
        report.relative_l2 > kMaximumRelativeL2Error) {
        return kExitNumerics;
    }
    return 0;
}

} // namespace

int main(int argument_count, char** arguments) {
    if (argument_count != 2) {
        std::cerr
            << "usage: tatara_native_dense_qgemm_probe "
               "--cpu-only|--gpu|--gpu-steel|--gpu-steel-dense\n";
        return kExitUsage;
    }
    const std::string_view mode(arguments[1]);
    if (mode != "--cpu-only" && mode != "--gpu" &&
        mode != "--gpu-steel" && mode != "--gpu-steel-dense") {
        std::cerr
            << "usage: tatara_native_dense_qgemm_probe "
               "--cpu-only|--gpu|--gpu-steel|--gpu-steel-dense\n";
        return kExitUsage;
    }

    std::array<Fixture, kFixtureSpecs.size()> fixtures;
    for (std::size_t index = 0; index < kFixtureSpecs.size(); ++index) {
        if (!make_fixture(kFixtureSpecs[index], fixtures[index])) {
            return kExitCpuContract;
        }
    }
    if (!cpu_contract(fixtures)) {
        return kExitCpuContract;
    }
    if (mode == "--cpu-only") {
        std::cout
            << "native dense QGEMM fixture: PASS_CPU\n"
            << "  cases: " << fixtures.size() << '\n'
            << "  max normalized error: " << kMaximumNormalizedError << '\n'
            << "  max bfloat ULP: " << kMaximumBfloatUlp << '\n'
            << "  command buffers submitted: 0\n";
        return 0;
    }

    auto device = create_system_device();
    if (!device) {
        return kExitDevice;
    }
    auto queue = create_command_queue(*device.device);
    if (!queue) {
        return kExitDevice;
    }
    auto library = create_library_with_source(
        *device.device, generated::kernel_library_source());
    if (!library) {
        std::cerr << library.failure_description << '\n';
        return kExitLibrary;
    }
    const bool steel =
        mode == "--gpu-steel" || mode == "--gpu-steel-dense";
    const bool steel_dense = mode == "--gpu-steel-dense";
    if (steel &&
        !generated::kKernelLibraryMlxSteelEnabled) {
        std::cerr << "MLX Steel probe closure is not enabled\n";
        return kExitFunction;
    }
    const std::string_view kernel_name =
        steel_dense
            ? std::string_view{
                  generated::kKernelLibraryMlxSteelDenseKernelName}
            : (steel
                   ? std::string_view{
                         generated::kKernelLibraryMlxSteelKernelName}
                   : kKernelName);
    auto function = create_function(*library.library, kernel_name);
    if (!function) {
        return kExitFunction;
    }
    auto pipeline =
        create_compute_pipeline(*device.device, *function.function);
    if (!pipeline) {
        std::cerr << pipeline.failure_description << '\n';
        return kExitPipeline;
    }

    std::uint32_t submissions = 0;
    if (steel) {
        const int result = run_gpu_steel_case(
            *device.device, *queue.command_queue, *pipeline.pipeline,
            fixtures.back(), steel_dense, submissions);
        if (result != 0) {
            return result;
        }
        if (submissions != 1U) {
            return kExitCpuContract;
        }
        std::cout << "native MLX Steel "
                  << (steel_dense ? "dense" : "gathered")
                  << " QGEMM fixture: PASS_GPU\n"
                  << "  device: " << device.device->name() << '\n'
                  << "  cases: 1\n"
                  << "  command buffers submitted: 1\n";
        return 0;
    }
    for (Fixture& fixture : fixtures) {
        const int result = run_gpu_case(
            *device.device, *queue.command_queue, *pipeline.pipeline,
            fixture, submissions);
        if (result != 0) {
            return result;
        }
    }
    if (submissions != fixtures.size()) {
        return kExitCpuContract;
    }
    std::cout << "native dense QGEMM fixture: PASS_GPU\n"
              << "  device: " << device.device->name() << '\n'
              << "  cases: " << fixtures.size() << '\n'
              << "  command buffers submitted: " << submissions << '\n';
    return 0;
}
