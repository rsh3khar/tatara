#include "tatara_benchmark.h"

#include "decode_harness.h"

#include "tatara/backend/metal/commands.h"
#include "tatara/generated/model_plan.h"
#include "tatara/runtime/decode_step.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace tatara::tools {
namespace {

using namespace tatara::backend::metal;
using namespace tatara::runtime;

// Warmups are discarded because the first token pays pipeline creation and
// residency, which is a load cost and not a decode cost. Reporting it inside
// the distribution would understate steady-state speed and overstate variance.
constexpr std::uint32_t kWarmupSteps = 4;
constexpr std::uint32_t kMeasuredSteps = 32;

constexpr int kExitOk = 0;
constexpr int kExitUsage = 2;
constexpr int kExitRuntime = 7;

double percentile(const std::vector<double>& sorted, double fraction) {
    if (sorted.empty()) {
        return 0.0;
    }
    const auto index =
        static_cast<std::size_t>(fraction * static_cast<double>(sorted.size() - 1) + 0.5);
    return sorted[std::min(index, sorted.size() - 1)];
}

std::uint32_t state_record_positions(std::span<const std::byte> record) {
    if (record.size() < 12) {
        return 0;
    }
    std::uint32_t count = 0;
    std::memcpy(&count, record.data() + 8, sizeof(count));
    return count;
}

} // namespace

int run_benchmark(std::vector<std::string_view> arguments) {
    bool as_json = false;
    std::vector<std::string_view> positional;
    for (const auto& argument : arguments) {
        if (argument == "--json") {
            as_json = true;
        } else {
            positional.push_back(argument);
        }
    }
    if (positional.size() != 3) {
        std::fprintf(stderr, "usage: tatara benchmark <record.tatara> <artifact-root> <states.bin> "
                             "[--json]\n");
        return kExitUsage;
    }

    const std::string record{positional[0]};
    const std::string artifact_root{positional[1]};
    const std::string states{positional[2]};
    const auto state_bytes = read_file(states.c_str());
    if (state_bytes.empty()) {
        std::fprintf(stderr, "benchmark: state record is empty or unreadable\n");
        return kExitRuntime;
    }

    auto harness = boot_decode(record.c_str(), artifact_root.c_str());
    if (!harness) {
        std::fprintf(stderr, "benchmark: model boot failed (%d)\n", harness.exit_code);
        return kExitRuntime;
    }
    DecodeStep& step = *harness.step;

    // Decode cost is dominated by attention over the live context, so measuring
    // from an empty cache would report a number no served request ever sees.
    const auto& layers = tatara::model::qwen36::generated::kModelPlan.layers;
    const std::span<const tatara::model::qwen36::LayerKind> schedule(layers.data(), layers.size());
    if (!load_state_record(state_bytes, step, schedule, harness.capacity)) {
        std::fprintf(stderr, "benchmark: state record could not be installed\n");
        return kExitRuntime;
    }
    const std::uint32_t prefilled = state_record_positions(state_bytes);
    if (prefilled == 0) {
        std::fprintf(stderr, "benchmark: state record declares no positions\n");
        return kExitRuntime;
    }

    std::vector<double> gpu_milliseconds;
    gpu_milliseconds.reserve(kMeasuredSteps);
    const auto& plan = tatara::model::qwen36::generated::kModelPlan;
    std::uint32_t token = 0;

    for (std::uint32_t index = 0; index < kWarmupSteps + kMeasuredSteps; ++index) {
        const auto started = std::chrono::steady_clock::now();
        auto command_buffer = create_command_buffer(*harness.queue);
        if (!command_buffer) {
            return kExitRuntime;
        }
        auto pass = begin_compute_pass(std::move(*command_buffer.command_buffer));
        if (!pass) {
            return kExitRuntime;
        }
        const std::uint32_t context = prefilled + index;
        if (encode_token(step, *pass.compute_pass, context) != MetalCommandError::None) {
            return kExitRuntime;
        }
        auto ended = end_compute_pass(std::move(*pass.compute_pass));
        if (!ended) {
            return kExitRuntime;
        }
        auto pending = commit(std::move(*ended.command_buffer));
        if (!pending) {
            return kExitRuntime;
        }
        if (auto execution = wait_until_completed(std::move(*pending.pending_execution));
            !execution) {
            std::fprintf(stderr, "benchmark: step %u failed: %s\n", index,
                         execution.failure_description.c_str());
            return kExitRuntime;
        }
        const double elapsed =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                .count();
        token = *static_cast<std::uint32_t*>(step.token_id.contents());
        if (token >= plan.dimensions.vocabulary) {
            std::fprintf(stderr, "benchmark: out-of-range token at step %u\n", index);
            return kExitRuntime;
        }
        if (index >= kWarmupSteps) {
            gpu_milliseconds.push_back(elapsed);
        }
        advance_decode_state(step);
    }

    std::sort(gpu_milliseconds.begin(), gpu_milliseconds.end());
    const double p10 = percentile(gpu_milliseconds, 0.10);
    const double p50 = percentile(gpu_milliseconds, 0.50);
    const double p90 = percentile(gpu_milliseconds, 0.90);
    const double p99 = percentile(gpu_milliseconds, 0.99);
    const double tokens_per_second = p50 > 0.0 ? 1000.0 / p50 : 0.0;

    if (as_json) {
        std::printf("{\n  \"schema_version\": 1,\n  \"measurement\": \"single-stream decode\",\n");
        std::printf("  \"warmup_steps\": %u,\n  \"measured_steps\": %u,\n", kWarmupSteps,
                    kMeasuredSteps);
        std::printf("  \"load_seconds\": %.3f,\n  \"context_tokens\": %u,\n", harness.load_seconds,
                    prefilled);
        std::printf("  \"milliseconds\": {\"p10\": %.3f, \"p50\": %.3f, \"p90\": %.3f, "
                    "\"p99\": %.3f, \"min\": %.3f, \"max\": %.3f},\n",
                    p10, p50, p90, p99, gpu_milliseconds.front(), gpu_milliseconds.back());
        std::printf("  \"tokens_per_second_p50\": %.2f\n}\n", tokens_per_second);
    } else {
        std::printf("model load: %.2f s\n", harness.load_seconds);
        std::printf("decode: %u measured steps after %u warmups, at context %u\n", kMeasuredSteps,
                    kWarmupSteps, prefilled);
        std::printf("  p50 %.3f ms (%.1f tok/s)\n", p50, tokens_per_second);
        std::printf("  p10 %.3f  p90 %.3f  p99 %.3f  range %.3f..%.3f ms\n", p10, p90, p99,
                    gpu_milliseconds.front(), gpu_milliseconds.back());
        // A median without its spread is not a performance claim; the
        // hyperoptimization doctrine requires the tails alongside it.
        std::printf("tails are reported because a faster median that spikes is not a win\n");
    }
    return kExitOk;
}

} // namespace tatara::tools
