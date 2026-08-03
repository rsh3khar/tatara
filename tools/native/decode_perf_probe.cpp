#include "decode_harness.h"

#include "tatara/backend/metal/commands.h"
#include "tatara/generated/model_plan.h"
#include "tatara/runtime/decode_step.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace {

using namespace tatara::backend::metal;
using namespace tatara::runtime;
using tatara::model::qwen36::LayerKind;

// The admitted decode fixture (reference.toml [token]): a 3,926-token prompt
// whose final token the block prefill probe's decode arm absorbs as the
// prompt handoff BEFORE dumping the state record. The record therefore
// declares all 3,926 positions, and the first id fed here is the handoff's
// own prediction -- the first sealed continuation token -- at context equal
// to the record's declared position count, exactly as the admitted
// continuation loop (context = positions + index - 1, feeding
// continuation[index - 1]) reads that record. Decoding continues greedily
// from there.
constexpr std::uint32_t kSealedPromptTokens = 3926;
constexpr std::uint32_t kHandoffPrediction = 4072;

// Timing-only probes still owe a correctness canary, because this timing is
// expert-routing dependent: mixture-of-experts decode streams the eight
// selected experts of 256 out of the model image, so a silently divergent
// trajectory -- particularly a degenerate one repeating a token and therefore
// re-selecting the same experts -- would be unrealistically cache-friendly and
// report an OPTIMISTIC median with nothing in the output to reveal it. The
// sealed continuation (reference.toml [token] expected_ids) begins with the
// handoff prediction, which the admitted arm produces BEFORE dumping the
// record -- so that id is FED, not expected, and the produced tokens checked
// here are the sealed entries after it. A mismatch is fatal: a number from a
// divergent trajectory is worse than no number. Tokens beyond the sealed
// record carry no expectation. An earlier revision instead re-fed the
// pre-handoff pending prompt token at context 3,925; against the post-handoff
// record that absorbs the handoff token twice into every gated-delta state
// (the attention K/V rewrite at 3,925 is byte-identical, masking it), and the
// trajectory held for two tokens before diverging at token 2 -- the frozen
// family misalignment this configuration reconciles.
constexpr std::uint32_t kExpectedProducedIds[15] = {279, 271, 248068, 198, 8160,
                                                    579, 264, 7047,   1817, 25,
                                                    271, 16,  13,     220,  2972};

// The reference timing protocol, replicated so the two engines' numbers are
// read the same way: 21 executions with the first four discarded, then the
// seventeen survivors sorted, median at index 8, p10 and p90 at 1 and 15.
constexpr std::size_t kWarmupTokens = 4;
constexpr std::size_t kMeasuredTokens = 17;
constexpr std::size_t kTotalTokens = kWarmupTokens + kMeasuredTokens;
constexpr std::size_t kMedianIndex = kMeasuredTokens / 2;
constexpr std::size_t kLowIndex = 1;
constexpr std::size_t kHighIndex = kMeasuredTokens - 2;

// Q36BRS01 header: the magic, then the u32 position count the admitted arm
// wrote at the prompt handoff.
constexpr char kRecordMagic[8] = {'Q', '3', '6', 'B', 'R', 'S', '0', '1'};

// One executed token, split so CPU-side cost separates from GPU execution.
// wall is what the reference wall_median measures and gpu is what its
// gpu_median measures; schedule is the driver's own window around the
// submission, which is where resource residency validation lands.
struct TokenTiming {
    double encode_milliseconds;
    double submit_milliseconds;
    double wall_milliseconds;
    double gpu_milliseconds;
    double schedule_milliseconds;
    bool gpu_timestamps_present;
};

struct Distribution {
    double median;
    double low;
    double high;
    double minimum;
    double maximum;
};

double milliseconds_between(std::chrono::steady_clock::time_point start,
                            std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

Distribution summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return {
        .median = samples[kMedianIndex],
        .low = samples[kLowIndex],
        .high = samples[kHighIndex],
        .minimum = samples.front(),
        .maximum = samples.back(),
    };
}

void report(const char* label, const Distribution& distribution, bool with_rate) {
    std::cout << label << " median=" << distribution.median << " ms";
    if (with_rate) {
        std::cout << " (" << std::setprecision(1) << 1000.0 / distribution.median << " tok/s)"
                  << std::setprecision(3);
    }
    std::cout << " p10=" << distribution.low << " p90=" << distribution.high
              << " range=" << distribution.minimum << ".." << distribution.maximum << " ms ("
              << kMeasuredTokens << " measured, " << kWarmupTokens << " warmup)\n";
}

bool parse_u32(std::string_view text, std::uint32_t& value) {
    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const auto parsed = std::from_chars(begin, end, value);
    return parsed.ec == std::errc{} && parsed.ptr == end;
}

bool parse_score_kernel(
    std::string_view text,
    tatara::tools::DecodeAttentionScoreKernel& score_kernel) {
    if (text == "score-adaptive") {
        score_kernel =
            tatara::tools::DecodeAttentionScoreKernel::Adaptive;
        return true;
    }
    if (text == "score-adaptive-fused") {
        score_kernel =
            tatara::tools::DecodeAttentionScoreKernel::AdaptiveA23;
        return true;
    }
    if (text == "score-gqa4") {
        score_kernel = tatara::tools::DecodeAttentionScoreKernel::Gqa4;
        return true;
    }
    if (text == "score-gqa4-simdreduce") {
        score_kernel =
            tatara::tools::DecodeAttentionScoreKernel::Gqa4SimdReduce;
        return true;
    }
    if (text == "score-gqa8") {
        score_kernel = tatara::tools::DecodeAttentionScoreKernel::Gqa8;
        return true;
    }
    if (text == "score-value-gqa8-fused") {
        score_kernel =
            tatara::tools::DecodeAttentionScoreKernel::Gqa8FusedPart;
        return true;
    }
    if (text == "score-vector-2pass") {
        score_kernel =
            tatara::tools::DecodeAttentionScoreKernel::
                IndependentHeadVector2Pass;
        return true;
    }
    return false;
}

bool parse_value_kernel(
    std::string_view text,
    tatara::tools::DecodeAttentionValueKernel& value_kernel) {
    if (text == "value-t1024") {
        value_kernel = tatara::tools::DecodeAttentionValueKernel::Gqa8T1024;
        return true;
    }
    if (text == "value-t512") {
        value_kernel = tatara::tools::DecodeAttentionValueKernel::Gqa8T512;
        return true;
    }
    return false;
}

const char* score_kernel_name(
    tatara::tools::DecodeAttentionScoreKernel score_kernel) {
    switch (score_kernel) {
    case tatara::tools::DecodeAttentionScoreKernel::Adaptive:
        return "score-adaptive";
    case tatara::tools::DecodeAttentionScoreKernel::AdaptiveA23:
        return "score-adaptive-fused";
    case tatara::tools::DecodeAttentionScoreKernel::Gqa4:
        return "score-gqa4";
    case tatara::tools::DecodeAttentionScoreKernel::Gqa4SimdReduce:
        return "score-gqa4-simdreduce";
    case tatara::tools::DecodeAttentionScoreKernel::Gqa8:
        return "score-gqa8";
    case tatara::tools::DecodeAttentionScoreKernel::Gqa8FusedPart:
        return "score-value-gqa8-fused";
    case tatara::tools::DecodeAttentionScoreKernel::
        IndependentHeadVector2Pass:
        return "score-vector-2pass";
    }
    return "score-invalid";
}

const char* value_kernel_name(
    tatara::tools::DecodeAttentionValueKernel value_kernel) {
    return value_kernel == tatara::tools::DecodeAttentionValueKernel::Gqa8T512
               ? "value-t512"
               : "value-t1024";
}

} // namespace

int main(int argument_count, char** arguments) {
    if (argument_count < 4 || argument_count > 7) {
        std::cerr << "usage: tatara_decode_perf_probe RECORD ARTIFACT_ROOT STATES"
                     " [FED_TOKEN] [score-adaptive|score-adaptive-fused|score-gqa4|"
                     "score-gqa4-simdreduce|score-gqa8|"
                     "score-value-gqa8-fused|score-vector-2pass]"
                     " [value-t1024|value-t512]\n";
        return 2;
    }
    // The state record is validated before the engine boots: a wrong-fixture
    // record must refuse fail-closed without ever creating a Metal device.
    const auto states_bytes = tatara::tools::read_file(arguments[3]);
    if (states_bytes.empty()) {
        std::cerr << "state record is empty or unreadable\n";
        return 8;
    }
    if (states_bytes.size() < 12 || std::memcmp(states_bytes.data(), kRecordMagic, 8) != 0) {
        std::cerr << "state record magic mismatch\n";
        return 19;
    }
    std::uint32_t declared_positions = 0;
    std::memcpy(&declared_positions, states_bytes.data() + 8, 4);
    const bool position_matches_sealed =
        declared_positions == kSealedPromptTokens;
    std::uint32_t initial_fed_token = kHandoffPrediction;
    bool explicit_fed_token = false;
    tatara::tools::DecodeAttentionScoreKernel score_kernel =
        tatara::tools::DecodeAttentionScoreKernel::Gqa4;
    tatara::tools::DecodeAttentionValueKernel value_kernel =
        tatara::tools::DecodeAttentionValueKernel::Gqa8T1024;
    int next_argument = 4;
    if (next_argument < argument_count) {
        std::uint32_t parsed_token = 0;
        if (parse_u32(arguments[next_argument], parsed_token)) {
            initial_fed_token = parsed_token;
            explicit_fed_token = true;
            ++next_argument;
        } else if (!position_matches_sealed) {
            std::cerr << "state record declares " << declared_positions
                      << " positions; a non-sealed state requires an explicit"
                         " FED_TOKEN before kernel selection\n";
            return 19;
        }
    }
    if (!position_matches_sealed && !explicit_fed_token) {
        std::cerr << "state record declares " << declared_positions
                  << " positions; a non-sealed state requires an explicit"
                     " FED_TOKEN\n";
        return 19;
    }
    // Position count alone is not fixture identity. An explicit fed token
    // opts into A/B trajectory comparison mode even at 3,926 positions; only
    // the default token path claims and checks the sealed continuation.
    const bool sealed_fixture =
        position_matches_sealed && !explicit_fed_token;
    bool score_selected = false;
    bool value_selected = false;
    for (; next_argument < argument_count; ++next_argument) {
        if (!score_selected &&
            parse_score_kernel(arguments[next_argument], score_kernel)) {
            score_selected = true;
        } else if (
            !value_selected &&
            parse_value_kernel(arguments[next_argument], value_kernel)) {
            value_selected = true;
        } else {
            std::cerr << "unknown or duplicate kernel selector: "
                      << arguments[next_argument] << '\n';
            return 2;
        }
    }
    const auto& plan = ::tatara::model::qwen36::generated::kModelPlan;
    const std::uint64_t required_capacity =
        std::uint64_t{declared_positions} + kTotalTokens;
    if (declared_positions == 0 ||
        required_capacity > plan.tokenizer.maximum_context ||
        required_capacity > std::numeric_limits<std::uint32_t>::max()) {
        std::cerr << "state record plus " << kTotalTokens
                  << " decode tokens exceeds the generated model bound "
                  << plan.tokenizer.maximum_context << '\n';
        return 19;
    }

    // Boot through the shared harness so kernel selection, geometry, bindings
    // and step construction are the admitted family's by construction. An
    // earlier revision cloned this ~140-line sequence, which is exactly the
    // drift channel the harness was extracted to close.
    auto harness = tatara::tools::boot_decode(
        arguments[1], arguments[2],
        static_cast<std::uint32_t>(required_capacity), score_kernel,
        value_kernel);
    if (!harness) {
        return harness.exit_code;
    }
    DecodeStep& step = *harness.step;
    const std::span<const LayerKind> schedule(plan.layers.data(), plan.layers.size());
    if (!tatara::tools::load_state_record(states_bytes, step, schedule, harness.capacity)) {
        return 18;
    }

    // Fixed token count, greedy feedback, no data-dependent exit: the loop
    // runs exactly kTotalTokens times whatever the model emits, and every
    // context stays far below the serving capacity. The per-submission policy
    // is the admitted arm's (block prefill probe, submit_decode_token plus its
    // continuation loop): write the fed token to the on-device id before every
    // submission, advance the state planes after the wait, and only then read
    // the produced token and feed it forward.
    std::vector<TokenTiming> timings;
    timings.reserve(kTotalTokens);
    // The gate policy requires the exact submission count; this probe commits
    // in two loops, the measured tokens and the empty sync-floor arm.
    std::uint32_t submissions = 0;
    std::uint32_t fed_token = initial_fed_token;
    for (std::size_t index = 0; index < kTotalTokens; ++index) {
        const std::uint32_t context =
            declared_positions + static_cast<std::uint32_t>(index);
        std::memcpy(step.token_id.contents(), &fed_token, sizeof(fed_token));
        const auto encode_start = std::chrono::steady_clock::now();
        auto command_buffer = create_command_buffer(*harness.queue);
        if (!command_buffer) {
            return 20;
        }
        auto pass = begin_compute_pass(std::move(*command_buffer.command_buffer));
        if (!pass) {
            return 21;
        }
        if (encode_token(step, *pass.compute_pass, context) != MetalCommandError::None) {
            std::cerr << "encode_token failed\n";
            return 22;
        }
        auto ended = end_compute_pass(std::move(*pass.compute_pass));
        if (!ended) {
            return 23;
        }
        const auto submit_start = std::chrono::steady_clock::now();
        auto pending = commit(std::move(*ended.command_buffer));
        if (!pending) {
            return 24;
        }
        ++submissions;
        auto execution = wait_until_completed_timed(std::move(*pending.pending_execution));
        const auto submit_end = std::chrono::steady_clock::now();
        if (!execution) {
            std::cerr << "token " << index
                      << " execution failed: " << execution.failure_description.view() << '\n';
            return 25;
        }
        const MetalExecutionTiming& stamps = execution.timing;
        const bool gpu_present = stamps.gpu_end_seconds > stamps.gpu_start_seconds;
        const bool schedule_present = stamps.schedule_end_seconds > stamps.schedule_start_seconds;
        const double encode_milliseconds = milliseconds_between(encode_start, submit_start);
        const double submit_milliseconds = milliseconds_between(submit_start, submit_end);
        timings.push_back({
            .encode_milliseconds = encode_milliseconds,
            .submit_milliseconds = submit_milliseconds,
            .wall_milliseconds = encode_milliseconds + submit_milliseconds,
            .gpu_milliseconds =
                gpu_present ? (stamps.gpu_end_seconds - stamps.gpu_start_seconds) * 1e3 : 0.0,
            .schedule_milliseconds =
                schedule_present
                    ? (stamps.schedule_end_seconds - stamps.schedule_start_seconds) * 1e3
                    : 0.0,
            .gpu_timestamps_present = gpu_present,
        });

        advance_decode_state(step);
        const std::uint32_t produced =
            *static_cast<const std::uint32_t*>(step.token_id.contents());
        if (produced >= plan.dimensions.vocabulary) {
            std::cerr << "token " << index << ": out of range id " << produced << '\n';
            return 26;
        }
        if (sealed_fixture && index < std::size(kExpectedProducedIds) &&
            produced != kExpectedProducedIds[index]) {
            std::cerr << "token " << index << ": trajectory diverged, produced " << produced
                      << " expected " << kExpectedProducedIds[index]
                      << " -- timing from a divergent trajectory is not comparable\n";
            return 28;
        }

        // One flushed line per token. Without it a mid-run SIGKILL leaves an
        // empty log, and the guard's CPU-stall arm can fire on a merely-slow
        // run because this loop blocks in the wait with near-zero CPU duty --
        // leaving no way to tell a wedge from slow tokens, which is precisely
        // the question this probe exists to answer.
        std::cout << (index < kWarmupTokens ? "warmup " : "measured ") << index << ": context "
                  << context << ", wall " << timings.back().wall_milliseconds << " ms, gpu "
                  << timings.back().gpu_milliseconds << " ms, token " << produced << '\n'
                  << std::flush;

        fed_token = produced;
    }

    // Round-trip floor for one empty command buffer on this queue: the part of
    // wall time that per-token submission and completion cost even with no
    // work encoded. It bounds the synchronization hypothesis; it does not
    // include the residency cost of the decode step's bound resources.
    std::vector<double> sync_floor;
    sync_floor.reserve(kMeasuredTokens);
    for (std::size_t index = 0; index < kTotalTokens; ++index) {
        const auto start = std::chrono::steady_clock::now();
        auto command_buffer = create_command_buffer(*harness.queue);
        if (!command_buffer) {
            return 40;
        }
        auto pass = begin_compute_pass(std::move(*command_buffer.command_buffer));
        if (!pass) {
            return 41;
        }
        auto ended = end_compute_pass(std::move(*pass.compute_pass));
        if (!ended) {
            return 43;
        }
        auto pending = commit(std::move(*ended.command_buffer));
        if (!pending) {
            return 44;
        }
        ++submissions;
        if (auto execution = wait_until_completed(std::move(*pending.pending_execution));
            !execution) {
            std::cerr << "empty command buffer " << index
                      << " execution failed: " << execution.failure_description.view() << '\n';
            return 45;
        }
        if (index >= kWarmupTokens) {
            sync_floor.push_back(milliseconds_between(start, std::chrono::steady_clock::now()));
        }
    }

    std::vector<double> wall;
    std::vector<double> gpu;
    std::vector<double> encode;
    std::vector<double> submit;
    std::vector<double> schedule_window;
    std::size_t gpu_samples = 0;
    for (std::size_t index = kWarmupTokens; index < kTotalTokens; ++index) {
        const TokenTiming& timing = timings[index];
        wall.push_back(timing.wall_milliseconds);
        encode.push_back(timing.encode_milliseconds);
        submit.push_back(timing.submit_milliseconds);
        schedule_window.push_back(timing.schedule_milliseconds);
        gpu.push_back(timing.gpu_milliseconds);
        gpu_samples += timing.gpu_timestamps_present ? 1u : 0u;
    }
    // Partial coverage is a typed failure, not a footnote. Substituting wall
    // time for a missing sample and still calling the result gpu_median
    // produces a blended number wearing a GPU label, which is then compared
    // against the reference genuine gpu_median -- a silent fallback that
    // rescues a measurement rather than reporting it failed. Anything short of
    // full coverage exits nonzero.
    if (gpu_samples != kMeasuredTokens) {
        std::cerr << "GPU timestamps present on only " << gpu_samples << " of " << kMeasuredTokens
                  << " measured tokens; a partial distribution cannot be reported as gpu_median\n";
        return 27;
    }

    const Distribution wall_distribution = summarize(wall);
    const Distribution gpu_distribution = summarize(gpu);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "decode perf: PASS\n"
              << "  device: " << harness.device->name() << '\n'
              << "  image load: " << harness.load_seconds
              << " s, capacity: " << harness.capacity << ", context: "
              << declared_positions << ".."
              << declared_positions + kTotalTokens - 1 << '\n'
              << "  score kernel: " << score_kernel_name(score_kernel) << '\n'
              << "  value kernel: " << value_kernel_name(value_kernel) << '\n'
              << "  gpu timestamp samples: " << gpu_samples << "/" << kMeasuredTokens << '\n'
              << "  command buffers submitted: " << submissions << '\n';
    report("  decode gpu:      ", gpu_distribution, true);
    report("  decode wall:     ", wall_distribution, true);
    report("  phase encode:    ", summarize(encode), false);
    report("  phase submit:    ", summarize(submit), false);
    report("  phase schedule:  ", summarize(schedule_window), false);
    report("  empty round trip:", summarize(sync_floor), false);
    std::cout << "  attribution: wall_median=" << wall_distribution.median
              << " ms gpu_median=" << gpu_distribution.median
              << " ms outside_gpu=" << wall_distribution.median - gpu_distribution.median << " ms ("
              << std::setprecision(1)
              << 100.0 * (wall_distribution.median - gpu_distribution.median) /
                     wall_distribution.median
              << "%)\n"
              << std::setprecision(3);

    std::cout << "  warmup (excluded from every distribution above):\n";
    for (std::size_t index = 0; index < kWarmupTokens; ++index) {
        const TokenTiming& timing = timings[index];
        std::cout << "    token " << index << ": wall=" << timing.wall_milliseconds
                  << " ms gpu=" << timing.gpu_milliseconds
                  << " ms encode=" << timing.encode_milliseconds
                  << " ms submit=" << timing.submit_milliseconds
                  << " ms schedule=" << timing.schedule_milliseconds << " ms\n";
    }
    return 0;
}
