#include "tatara/backend/metal/commands.h"
#include "tatara/backend/metal/pipeline.h"
#include "tatara/backend/metal/resources.h"
#include "tatara/generated/kernel_library.h"
#include "tatara/generated/model_plan.h"
#include "tatara/model/image_population.h"
#include "tatara/model/prepared_checkpoint.h"
#include "tatara/model/source_shards.h"
#include "tatara/runtime/decode_bindings.h"
#include "tatara/runtime/decode_geometry.h"
#include "tatara/runtime/decode_step.h"

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <span>
#include <vector>

namespace {

using namespace tatara::backend::metal;
using namespace tatara::runtime;
using tatara::model::qwen36::LayerKind;

// The sealed reference exam (reference.toml [token]): after the 3,926-token
// prompt — 3,925 tokens prefilled into the sealed state record plus the
// pending prompt token 321 — the reference engine continues with exactly these ids.
constexpr std::uint32_t kPendingToken = 321;
constexpr std::uint32_t kPrefilledTokens = 3925;
constexpr std::uint32_t kExpectedIds[16] = {4072, 279,  271, 248068, 198, 8160, 579, 264,
                                            7047, 1817, 25,  271,    16,  13,   220, 2972};

constexpr char kStateMagic[8] = {'Q', '3', '6', 'B', 'R', 'S', '0', '1'};

std::vector<std::byte> read_file(const char* path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        return {};
    }
    const std::streamsize size = stream.tellg();
    stream.seekg(0);
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    stream.read(reinterpret_cast<char*>(bytes.data()), size);
    return stream ? bytes : std::vector<std::byte>{};
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// Loads the sealed Q36BRS01 state record into the step's layer states:
// gated-delta conv (step-major, byte-identical layout) and float32
// recurrent states copy directly; attention K/V arrive contiguous per head
// at 3,925 positions and expand into the capacity-strided caches.
bool load_sealed_states(std::span<const std::byte> record, DecodeStep& step,
                        std::span<const LayerKind> schedule, std::uint32_t capacity) {
    if (record.size() < 12 || std::memcmp(record.data(), kStateMagic, 8) != 0) {
        std::cerr << "state record magic mismatch\n";
        return false;
    }
    std::uint32_t count = 0;
    std::memcpy(&count, record.data() + 8, 4);
    if (count != kPrefilledTokens) {
        std::cerr << "state record token count " << count << " != " << kPrefilledTokens << '\n';
        return false;
    }
    std::size_t cursor = 12;
    const auto take = [&](std::size_t bytes) -> const std::byte* {
        if (cursor + bytes > record.size()) {
            return nullptr;
        }
        const std::byte* view = record.data() + cursor;
        cursor += bytes;
        return view;
    };
    const std::uint64_t head_bytes = std::uint64_t{kPrefilledTokens} * 256 * 2;
    const std::uint64_t cache_head_stride = std::uint64_t{capacity} * 256 * 2;
    for (std::size_t layer = 0; layer < schedule.size(); ++layer) {
        DecodeLayerState& state = step.state.layers[layer];
        if (schedule[layer] == LayerKind::GatedDelta) {
            const std::byte* conv = take(step.geometry.gdn_conv_state_bytes);
            const std::byte* recurrent = take(step.geometry.gdn_recurrent_state_bytes);
            if (conv == nullptr || recurrent == nullptr) {
                std::cerr << "state record truncated at layer " << layer << '\n';
                return false;
            }
            std::memcpy(state.first.contents(), conv, step.geometry.gdn_conv_state_bytes);
            std::memcpy(state.second.contents(), recurrent,
                        step.geometry.gdn_recurrent_state_bytes);
        } else {
            const std::byte* keys = take(2 * head_bytes);
            const std::byte* values = take(2 * head_bytes);
            if (keys == nullptr || values == nullptr) {
                std::cerr << "state record truncated at layer " << layer << '\n';
                return false;
            }
            for (std::uint32_t head = 0; head < 2; ++head) {
                std::memcpy(static_cast<std::byte*>(state.first.contents()) +
                                head * cache_head_stride,
                            keys + head * head_bytes, head_bytes);
                std::memcpy(static_cast<std::byte*>(state.second.contents()) +
                                head * cache_head_stride,
                            values + head * head_bytes, head_bytes);
            }
        }
    }
    if (cursor != record.size()) {
        std::cerr << "state record has " << record.size() - cursor << " unconsumed bytes\n";
        return false;
    }
    return true;
}

// Writes the post-decode layer states in the same Q36BRS01 layout the sealed
// record uses, so the two are directly byte-comparable off-device. This is the
// exact inverse of load_sealed_states: gated-delta conv and recurrent planes
// copy straight out, attention K/V contract from the capacity-strided caches
// back to contiguous per-head positions, and K for both heads precedes V.
//
// After an even number of decoded tokens the ping-pong has returned the live
// state to first/second (advance_decode_state toggles once per token), and
// attention layers write in place, so first/second are correct for both kinds.
bool write_post_decode_states(const char* path, const DecodeStep& step,
                              std::span<const LayerKind> schedule, std::uint32_t capacity,
                              std::uint32_t positions) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        std::cerr << "post-decode dump: cannot open " << path << '\n';
        return false;
    }
    stream.write(kStateMagic, 8);
    stream.write(reinterpret_cast<const char*>(&positions), 4);

    const std::uint64_t head_bytes = std::uint64_t{positions} * 256 * 2;
    const std::uint64_t cache_head_stride = std::uint64_t{capacity} * 256 * 2;
    for (std::size_t layer = 0; layer < schedule.size(); ++layer) {
        const DecodeLayerState& state = step.state.layers[layer];
        if (schedule[layer] == LayerKind::GatedDelta) {
            stream.write(static_cast<const char*>(state.first.contents()),
                         static_cast<std::streamsize>(step.geometry.gdn_conv_state_bytes));
            stream.write(static_cast<const char*>(state.second.contents()),
                         static_cast<std::streamsize>(step.geometry.gdn_recurrent_state_bytes));
        } else {
            for (const MetalBuffer* plane : {&state.first, &state.second}) {
                for (std::uint32_t head = 0; head < 2; ++head) {
                    stream.write(static_cast<const char*>(plane->contents()) +
                                     head * cache_head_stride,
                                 static_cast<std::streamsize>(head_bytes));
                }
            }
        }
    }
    stream.flush();
    if (!stream) {
        std::cerr << "post-decode dump: write failed\n";
        return false;
    }
    return true;
}

} // namespace

int main(int argument_count, char** arguments) {
    if (argument_count != 4 && argument_count != 5) {
        std::cerr << "usage: tatara_token_exam_probe RECORD ARTIFACT_ROOT STATES [DUMP_OUT]\n";
        return 2;
    }
    const char* const dump_path = argument_count == 5 ? arguments[4] : nullptr;
    const auto record_bytes = read_file(arguments[1]);
    if (record_bytes.empty()) {
        std::cerr << "prepared checkpoint is empty or unreadable\n";
        return 3;
    }
    const auto parsed = tatara::model::parse_prepared_checkpoint(record_bytes);
    if (!parsed || !parsed.checkpoint) {
        std::cerr << "prepared checkpoint parse failed\n";
        return 4;
    }
    const tatara::model::PreparedCheckpointExpectation expectation = {
        .package_id = tatara::model::generated::kModelPackageId,
        .package_sha256 = tatara::model::generated::kModelPackageSha256,
        .artifact = tatara::model::generated::kArtifactIdentity,
    };
    if (tatara::model::validate_prepared_checkpoint_identity(*parsed.checkpoint, expectation) !=
        tatara::model::PreparedCheckpointIdentityError::None) {
        std::cerr << "prepared checkpoint identity failed\n";
        return 5;
    }
    const auto layout = tatara::model::plan_image_layout(parsed.checkpoint->tensors(),
                                                         tatara::model::kTensorAlignmentBytes);
    if (!layout || !layout.layout) {
        std::cerr << "image layout planning failed\n";
        return 6;
    }
    auto shards = tatara::model::open_source_shards(arguments[2], parsed.checkpoint->shards());
    if (!shards || !shards.shard_set) {
        std::cerr << "source shard open failed: path=" << shards.path << '\n';
        return 7;
    }
    const auto states_bytes = read_file(arguments[3]);
    if (states_bytes.empty()) {
        std::cerr << "state record is empty or unreadable\n";
        return 8;
    }

    auto device = create_system_device();
    if (!device) {
        std::cerr << "system Metal device creation failed\n";
        return 9;
    }
    auto queue = create_command_queue(*device.device);
    if (!queue) {
        std::cerr << "command queue creation failed\n";
        return 10;
    }
    auto library = create_library_with_source(*device.device, generated::kernel_library_source());
    if (!library) {
        std::cerr << "kernel library compilation failed:\n" << library.failure_description << '\n';
        return 11;
    }
    const char* names[] = {
        "embed_row_q4",
        "rms_only",
        "gdn_project",
        "gdn_prepare",
        "gdn_recurrence",
        "gdn_gate_norm",
        "gdn_outproj",
        "attn_project",
        "attn_qk_rope",
        "attention_decode",
        "attention_decode_scores_gqa4",
        "attention_decode_values_gqa8",
        "attention_decode_combine",
        "residual_rms",
        "router_q8",
        "router_select",
        "grouped_upgate",
        "grouped_down_res",
        "lmhead_q4",
        "logits_argmax_stage1",
        "logits_argmax_stage2",
    };
    DecodePipelines pipelines;
    MetalComputePipeline* slots[] = {
        &pipelines.embed,
        &pipelines.rms,
        &pipelines.gdn_project,
        &pipelines.gdn_prepare,
        &pipelines.gdn_recurrence,
        &pipelines.gdn_gate_norm,
        &pipelines.out_projection,
        &pipelines.attn_project,
        &pipelines.attn_qk_rope,
        &pipelines.attention_decode,
        &pipelines.attention_scores,
        &pipelines.attention_values,
        &pipelines.attention_combine,
        &pipelines.residual_rms,
        &pipelines.router,
        &pipelines.router_select,
        &pipelines.grouped_upgate,
        &pipelines.grouped_down_res,
        &pipelines.lmhead,
        &pipelines.argmax_stage1,
        &pipelines.argmax_stage2,
    };
    for (std::size_t i = 0; i < 21; ++i) {
        auto function = create_function(*library.library, names[i]);
        if (!function) {
            std::cerr << "function lookup failed: " << names[i] << '\n';
            return 12;
        }
        auto pipeline = create_compute_pipeline(*device.device, *function.function);
        if (!pipeline) {
            std::cerr << "pipeline creation failed: " << names[i] << '\n';
            return 13;
        }
        *slots[i] = std::move(*pipeline.pipeline);
    }

    const auto allocation_start = std::chrono::steady_clock::now();
    auto image = create_shared_buffer(*device.device, layout.layout->total_bytes);
    if (!image) {
        std::cerr << "model image allocation failed\n";
        return 14;
    }
    const std::span<std::byte> destination(static_cast<std::byte*>(image.buffer->contents()),
                                           static_cast<std::size_t>(layout.layout->total_bytes));
    const auto populated = tatara::model::populate_model_image(
        *parsed.checkpoint, *layout.layout, shards.shard_set->shards(), destination);
    if (!populated) {
        std::cerr << "image population failed: shard=" << populated.shard_index << '\n';
        return 15;
    }
    const double load_seconds = seconds_since(allocation_start);

    const auto& plan = ::tatara::model::qwen36::generated::kModelPlan;
    const std::span<const LayerKind> schedule(plan.layers.data(), plan.layers.size());
    auto bindings = resolve_decode_bindings(schedule, parsed.checkpoint->tensors());
    if (!bindings) {
        std::cerr << "binding resolution failed: " << bindings.missing_name << '\n';
        return 16;
    }
    const std::uint32_t capacity = plan.initial_serving_capacity;
    const auto geometry = make_decode_geometry(plan, capacity);
    auto step_result = create_decode_step(*device.device, geometry, capacity, schedule,
                                          std::move(*bindings.bindings), *image.buffer,
                                          layout.layout->tensor_offsets, std::move(pipelines));
    if (!step_result) {
        std::cerr << "decode step construction failed\n";
        return 17;
    }
    DecodeStep& step = *step_result.step;
    if (!load_sealed_states(states_bytes, step, schedule, capacity)) {
        return 18;
    }
    std::memcpy(step.token_id.contents(), &kPendingToken, 4);

    const auto decode_start = std::chrono::steady_clock::now();
    std::uint32_t produced[16] = {};
    std::uint32_t matched = 0;
    // The gate policy requires every probe to print its exact submission
    // count; this one and the perf probe are the two that touch real weights.
    std::uint32_t submissions = 0;
    for (std::uint32_t index = 0; index < 16; ++index) {
        const std::uint32_t context = kPrefilledTokens + index;
        auto command_buffer = create_command_buffer(*queue.command_queue);
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
        auto pending = commit(std::move(*ended.command_buffer));
        if (!pending) {
            return 24;
        }
        ++submissions;
        if (auto execution = wait_until_completed(std::move(*pending.pending_execution));
            !execution) {
            std::cerr << "token " << index
                      << " execution failed: " << execution.failure_description.view() << '\n';
            return 25;
        }
        produced[index] = *static_cast<std::uint32_t*>(step.token_id.contents());
        if (produced[index] >= plan.dimensions.vocabulary) {
            std::cout << "token " << index << ": OUT OF RANGE id " << produced[index] << '\n';
            return 26;
        }
        matched += produced[index] == kExpectedIds[index] ? 1 : 0;
        std::cout << "token " << index << ": produced " << produced[index] << ", expected "
                  << kExpectedIds[index]
                  << (produced[index] == kExpectedIds[index] ? "" : "  MISMATCH") << '\n';
        advance_decode_state(step);
    }
    const double decode_seconds = seconds_since(decode_start);

    std::cout << (matched == 16 ? "token exam: PASS\n" : "token exam: FAIL\n")
              << "  device: " << device.device->name() << '\n'
              << "  matched: " << matched << "/16\n"
              << "  image load: " << load_seconds << " s, 16-token decode: " << decode_seconds
              << " s\n"
              << "  command buffers submitted: " << submissions << '\n';

    // The dump runs even on a failed exam: a divergent state is diagnostic, and
    // this execution is a spent window either way.
    if (dump_path != nullptr) {
        const std::uint32_t positions = kPrefilledTokens + 16;
        if (!write_post_decode_states(dump_path, step, schedule, capacity, positions)) {
            return 31;
        }
        std::cout << "  post-decode states: written to " << dump_path << " at " << positions
                  << " positions\n";
    }
    return matched == 16 ? 0 : 30;
}
