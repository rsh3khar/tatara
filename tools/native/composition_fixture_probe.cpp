#include "fixture_batteries.h"
#include "tatara/backend/metal/commands.h"
#include "tatara/backend/metal/pipeline.h"
#include "tatara/backend/metal/resources.h"
#include "tatara/generated/kernel_library.h"
#include "tatara/generated/model_plan.h"
#include "tatara/model/image_population.h"
#include "tatara/runtime/decode_bindings.h"
#include "tatara/runtime/decode_geometry.h"
#include "tatara/runtime/decode_step.h"

#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using namespace tatara::backend::metal;
using namespace tatara::runtime;
using tatara::model::ImageLayout;
using tatara::model::plan_image_layout;
using tatara::model::TensorDataType;
using tatara::model::TensorRecord;
using tatara::model::qwen36::LayerKind;

constexpr std::uint32_t kHidden = generated::kKernelLibraryHidden;
constexpr std::uint32_t kVocabulary = generated::kKernelLibraryVocabulary;
constexpr std::uint32_t kCapacity = 512;
constexpr std::uint32_t kFirstToken = 7;
// Contexts 0-2 chain the short attention path; 256 and 257 run the
// partitioned GQA path (partitions = 2) in both executions.
constexpr std::uint32_t kContexts[] = {0, 1, 2, 256, 257};
constexpr std::uint32_t kTokens = 5;

int submissions = 0;

// Deterministic fill stream (xorshift64*), independent of kernel_reference
// so the probe stays self-contained.
struct Fill {
    std::uint64_t state;
    std::uint64_t next() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 0x2545F4914F6CDD1Dull;
    }
    std::uint16_t banded_bf16(std::uint16_t exponent_base) {
        const std::uint64_t draw = next();
        const auto sign = static_cast<std::uint16_t>((draw >> 15) & 0x8000u);
        const auto exponent = static_cast<std::uint16_t>((exponent_base + draw % 0x8u) << 7);
        const auto mantissa = static_cast<std::uint16_t>((draw >> 32) & 0x7Fu);
        return static_cast<std::uint16_t>(sign | exponent | mantissa);
    }
};

// Standard band spans about 2^-15..2^-8 magnitudes for stream-sized values;
// the low band (about 2^-31..2^-24) keeps every quantized product sub-unit
// so 40-layer residual chains stay finite across five tokens — the window-1
// finding made overflow-to-NaN a recorded fixture hazard.
constexpr std::uint16_t kStandardExponent = 0x70;
constexpr std::uint16_t kLowExponent = 0x60;

enum class FillKind : std::uint8_t { QuantWords, Bf16, LowBf16 };

struct SyntheticTensor {
    std::string name;
    FillKind kind;
    std::uint64_t count;
    // Tensors sharing an alias key share image bytes; big tables alias their
    // first-layer counterparts so the image stays near one gigabyte.
    // Per-layer norms, conv, and decay tensors stay unique so a cross-layer
    // binding mix-up still diverges.
    std::string alias;
};

std::vector<SyntheticTensor> synthetic_tensors(std::span<const LayerKind> schedule) {
    std::vector<SyntheticTensor> tensors;
    const auto quantized = [&](const std::string& stem, std::uint64_t rows,
                               std::uint64_t words_per_row, std::uint64_t groups_per_row,
                               const std::string& alias_stem) {
        tensors.push_back(
            {stem + ".weight", FillKind::QuantWords, rows * words_per_row, alias_stem + ".weight"});
        tensors.push_back(
            {stem + ".scales", FillKind::LowBf16, rows * groups_per_row, alias_stem + ".scales"});
        tensors.push_back(
            {stem + ".biases", FillKind::LowBf16, rows * groups_per_row, alias_stem + ".biases"});
    };
    const auto plain = [&](const std::string& name, std::uint64_t count) {
        tensors.push_back({name, FillKind::Bf16, count, name});
    };
    quantized("language_model.model.embed_tokens", kVocabulary, 256, 32,
              "language_model.model.embed_tokens");
    std::string first_gated;
    std::string first_attention;
    std::string first_mlp;
    for (std::size_t layer = 0; layer < schedule.size(); ++layer) {
        const std::string prefix = "language_model.model.layers." + std::to_string(layer) + ".";
        plain(prefix + "input_layernorm.weight", kHidden);
        plain(prefix + "post_attention_layernorm.weight", kHidden);
        if (schedule[layer] == LayerKind::GatedDelta) {
            const std::string gated = prefix + "linear_attn.";
            if (first_gated.empty()) {
                first_gated = gated;
            }
            quantized(gated + "in_proj_qkv", 8192, 256, 32, first_gated + "in_proj_qkv");
            quantized(gated + "in_proj_z", 4096, 256, 32, first_gated + "in_proj_z");
            quantized(gated + "in_proj_b", 32, 256, 32, first_gated + "in_proj_b");
            quantized(gated + "in_proj_a", 32, 256, 32, first_gated + "in_proj_a");
            quantized(gated + "out_proj", kHidden, 512, 64, first_gated + "out_proj");
            plain(gated + "conv1d.weight", 4 * 8192);
            plain(gated + "A_log", 32);
            plain(gated + "dt_bias", 32);
            plain(gated + "norm.weight", 128);
        } else {
            const std::string attention = prefix + "self_attn.";
            if (first_attention.empty()) {
                first_attention = attention;
            }
            quantized(attention + "q_proj", 8192, 256, 32, first_attention + "q_proj");
            quantized(attention + "k_proj", 512, 256, 32, first_attention + "k_proj");
            quantized(attention + "v_proj", 512, 256, 32, first_attention + "v_proj");
            quantized(attention + "o_proj", kHidden, 512, 64, first_attention + "o_proj");
            plain(attention + "q_norm.weight", 256);
            plain(attention + "k_norm.weight", 256);
        }
        const std::string mlp = prefix + "mlp.";
        if (first_mlp.empty()) {
            first_mlp = mlp;
        }
        quantized(mlp + "gate", 256, 512, 32, first_mlp + "gate");
        quantized(mlp + "shared_expert_gate", 1, 512, 32, first_mlp + "shared_expert_gate");
        quantized(mlp + "switch_mlp.gate_proj", 256 * 512, 256, 32,
                  first_mlp + "switch_mlp.gate_proj");
        quantized(mlp + "switch_mlp.up_proj", 256 * 512, 256, 32, first_mlp + "switch_mlp.up_proj");
        quantized(mlp + "switch_mlp.down_proj", 256 * 2048, 64, 8,
                  first_mlp + "switch_mlp.down_proj");
        quantized(mlp + "shared_expert.gate_proj", 512, 256, 32,
                  first_mlp + "shared_expert.gate_proj");
        quantized(mlp + "shared_expert.up_proj", 512, 256, 32, first_mlp + "shared_expert.up_proj");
        quantized(mlp + "shared_expert.down_proj", 2048, 64, 8,
                  first_mlp + "shared_expert.down_proj");
    }
    plain("language_model.model.norm.weight", kHidden);
    quantized("language_model.lm_head", kVocabulary, 256, 32, "language_model.lm_head");
    return tensors;
}

struct Pass {
    const MetalComputePipeline* pipeline;
    std::vector<const MetalBuffer*> buffers;
    std::vector<std::uint64_t> offsets;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> constants;
    MetalSize threadgroups;
    MetalSize threads;
    // Argument indices for buffers; empty means sequential from zero (used
    // where constants interleave into the argument table).
    std::vector<std::uint32_t> indices = {};
};

int run_batch(const MetalCommandQueue& queue, std::span<const Pass> passes, const char* label) {
    auto command_buffer = create_command_buffer(queue);
    if (!command_buffer) {
        return 1;
    }
    MetalCommandBuffer chain = std::move(*command_buffer.command_buffer);
    for (const Pass& pass : passes) {
        auto begun = begin_compute_pass(std::move(chain));
        if (!begun) {
            return 2;
        }
        if (set_compute_pipeline(*begun.compute_pass, *pass.pipeline) != MetalCommandError::None) {
            return 3;
        }
        for (std::uint32_t position = 0; position < pass.buffers.size(); ++position) {
            const std::uint32_t index = pass.indices.empty() ? position : pass.indices[position];
            if (set_buffer(*begun.compute_pass, *pass.buffers[position], pass.offsets[position],
                           index) != MetalCommandError::None) {
                return 4;
            }
        }
        for (const auto& [value, index] : pass.constants) {
            if (set_bytes(*begun.compute_pass, &value, 4, index) != MetalCommandError::None) {
                return 5;
            }
        }
        if (dispatch_threadgroups(*begun.compute_pass, pass.threadgroups, pass.threads) !=
            MetalCommandError::None) {
            return 6;
        }
        auto ended = end_compute_pass(std::move(*begun.compute_pass));
        if (!ended) {
            return 7;
        }
        chain = std::move(*ended.command_buffer);
    }
    auto pending = commit(std::move(chain));
    if (!pending) {
        return 8;
    }
    ++submissions;
    auto execution = wait_until_completed(std::move(*pending.pending_execution));
    if (!execution) {
        std::cerr << label << " execution failed: " << execution.failure_description.view() << '\n';
        return 9;
    }
    return 0;
}

std::uint32_t compare_bytes(const MetalBuffer& expected, const MetalBuffer& actual,
                            std::uint64_t bytes, const char* label) {
    if (std::memcmp(expected.contents(), actual.contents(), bytes) == 0) {
        return 0;
    }
    const auto* left = static_cast<const std::uint8_t*>(expected.contents());
    const auto* right = static_cast<const std::uint8_t*>(actual.contents());
    std::uint64_t first = 0;
    while (first < bytes && left[first] == right[first]) {
        ++first;
    }
    std::cout << "  " << label << " differs at byte " << first << '\n';
    return 1;
}

} // namespace

int run_composition_battery() {
    auto device = create_system_device();
    if (!device) {
        std::cerr << "system Metal device creation failed\n";
        return 1;
    }
    auto queue = create_command_queue(*device.device);
    if (!queue) {
        std::cerr << "command queue creation failed\n";
        return 2;
    }
    auto library = create_library_with_source(*device.device, generated::kernel_library_source());
    if (!library) {
        std::cerr << "kernel library compilation failed:\n" << library.failure_description << '\n';
        return 3;
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
    MetalComputePipeline pipelines[21];
    for (std::size_t i = 0; i < 21; ++i) {
        auto function = create_function(*library.library, names[i]);
        if (!function) {
            std::cerr << "function lookup failed: " << names[i] << '\n';
            return 4;
        }
        auto pipeline = create_compute_pipeline(*device.device, *function.function);
        if (!pipeline) {
            std::cerr << "pipeline creation failed: " << names[i] << '\n';
            return 5;
        }
        pipelines[i] = std::move(*pipeline.pipeline);
    }

    // Synthetic image: full 40-layer record set; unique tensors laid out
    // 256-aligned, aliased big tables mapped onto their first-layer
    // counterparts, deterministic fill in unique-tensor order.
    const auto& plan = ::tatara::model::qwen36::generated::kModelPlan;
    const std::span<const LayerKind> schedule(plan.layers.data(), plan.layers.size());
    const std::vector<SyntheticTensor> synthetic = synthetic_tensors(schedule);
    std::vector<TensorRecord> records;
    records.reserve(synthetic.size());
    std::vector<TensorRecord> unique_records;
    std::vector<const SyntheticTensor*> unique_tensors;
    std::unordered_map<std::string, std::size_t> unique_index;
    for (const SyntheticTensor& tensor : synthetic) {
        const std::uint64_t element_bytes = tensor.kind == FillKind::QuantWords ? 4 : 2;
        TensorRecord record{
            .name = tensor.name,
            .data_type =
                tensor.kind == FillKind::QuantWords ? TensorDataType::U32 : TensorDataType::BF16,
            .shape = {tensor.count},
            .shard = 0,
            .shard_offset_bytes = 0,
            .size_bytes = tensor.count * element_bytes,
        };
        if (unique_index.emplace(tensor.alias, unique_records.size()).second) {
            unique_records.push_back(record);
            unique_tensors.push_back(&tensor);
        }
        records.push_back(std::move(record));
    }
    const auto layout_result =
        plan_image_layout(unique_records, tatara::model::kTensorAlignmentBytes);
    if (!layout_result) {
        std::cerr << "image layout planning failed\n";
        return 6;
    }
    const ImageLayout& layout = *layout_result.layout;
    std::vector<std::uint64_t> tensor_offsets;
    tensor_offsets.reserve(synthetic.size());
    for (const SyntheticTensor& tensor : synthetic) {
        tensor_offsets.push_back(layout.tensor_offsets[unique_index.at(tensor.alias)]);
    }
    auto image = create_shared_buffer(*device.device, layout.total_bytes);
    if (!image) {
        std::cerr << "image allocation failed\n";
        return 7;
    }
    {
        Fill fill{.state = 0xC0117051710Eull};
        auto* base = static_cast<std::uint8_t*>(image.buffer->contents());
        for (std::size_t i = 0; i < unique_tensors.size(); ++i) {
            std::uint8_t* destination = base + layout.tensor_offsets[i];
            const SyntheticTensor& tensor = *unique_tensors[i];
            if (tensor.kind == FillKind::QuantWords) {
                auto* words = reinterpret_cast<std::uint32_t*>(destination);
                for (std::uint64_t w = 0; w < tensor.count; ++w) {
                    words[w] = static_cast<std::uint32_t>(fill.next());
                }
            } else {
                const std::uint16_t exponent =
                    tensor.kind == FillKind::LowBf16 ? kLowExponent : kStandardExponent;
                auto* values = reinterpret_cast<std::uint16_t*>(destination);
                for (std::uint64_t v = 0; v < tensor.count; ++v) {
                    values[v] = fill.banded_bf16(exponent);
                }
            }
        }
    }

    // Execution A: the DecodeStep executor over the full plan schedule.
    auto bindings_result = resolve_decode_bindings(schedule, records);
    if (!bindings_result) {
        std::cerr << "binding resolution failed: " << bindings_result.missing_name << '\n';
        return 8;
    }
    const auto geometry = make_decode_geometry(plan, kCapacity);
    DecodePipelines step_pipelines;
    step_pipelines.embed = std::move(pipelines[0]);
    step_pipelines.rms = std::move(pipelines[1]);
    step_pipelines.gdn_project = std::move(pipelines[2]);
    step_pipelines.gdn_prepare = std::move(pipelines[3]);
    step_pipelines.gdn_recurrence = std::move(pipelines[4]);
    step_pipelines.gdn_gate_norm = std::move(pipelines[5]);
    step_pipelines.out_projection = std::move(pipelines[6]);
    step_pipelines.attn_project = std::move(pipelines[7]);
    step_pipelines.attn_qk_rope = std::move(pipelines[8]);
    step_pipelines.attention_decode = std::move(pipelines[9]);
    step_pipelines.attention_scores = std::move(pipelines[10]);
    step_pipelines.attention_values = std::move(pipelines[11]);
    step_pipelines.attention_combine = std::move(pipelines[12]);
    step_pipelines.residual_rms = std::move(pipelines[13]);
    step_pipelines.router = std::move(pipelines[14]);
    step_pipelines.router_select = std::move(pipelines[15]);
    step_pipelines.grouped_upgate = std::move(pipelines[16]);
    step_pipelines.grouped_down_res = std::move(pipelines[17]);
    step_pipelines.lmhead = std::move(pipelines[18]);
    step_pipelines.argmax_stage1 = std::move(pipelines[19]);
    step_pipelines.argmax_stage2 = std::move(pipelines[20]);
    auto step_result = create_decode_step(*device.device, geometry, kCapacity, schedule,
                                          std::move(*bindings_result.bindings), *image.buffer,
                                          tensor_offsets, std::move(step_pipelines));
    if (!step_result) {
        std::cerr << "decode step construction failed\n";
        return 9;
    }
    DecodeStep& step = *step_result.step;
    std::memcpy(step.token_id.contents(), &kFirstToken, 4);

    // Execution B: independent per-kernel reference, one encoder per
    // dispatch, loop-built from the sealed encoder order with explicit
    // per-layer state arrays.
    const DecodePipelines& p = step.pipelines;
    const auto allocate = [&](std::uint64_t bytes, MetalBuffer& buffer) {
        auto result = create_shared_buffer(*device.device, bytes);
        if (!result) {
            return false;
        }
        buffer = std::move(*result.buffer);
        std::memset(buffer.contents(), 0, bytes);
        return true;
    };
    MetalBuffer r_input, r_normed, r_branch, r_residual, r_moe, r_layer, r_gdn_proj, r_gdn_qk,
        r_gdn_v, r_gdn_z, r_gdn_y, r_gdn_gated, r_attn_proj, r_attn_q, r_attn_gate, r_attended,
        r_partials, r_weights, r_logits_router, r_ids, r_coefs, r_shared, r_hidden, r_final,
        r_logits, r_vals, r_idx, r_token;
    bool allocated =
        allocate(geometry.hidden_bytes, r_input) && allocate(geometry.hidden_bytes, r_normed) &&
        allocate(geometry.layer_stream_bytes, r_branch) &&
        allocate(geometry.layer_stream_bytes, r_residual) &&
        allocate(geometry.layer_stream_bytes, r_moe) &&
        allocate(geometry.layer_stream_bytes, r_layer) &&
        allocate(geometry.gdn_projection_bytes, r_gdn_proj) &&
        allocate(geometry.gdn_qk_bytes, r_gdn_qk) && allocate(geometry.gdn_value_bytes, r_gdn_v) &&
        allocate(geometry.gdn_gate_bytes, r_gdn_z) && allocate(geometry.gdn_value_bytes, r_gdn_y) &&
        allocate(geometry.gdn_gate_bytes, r_gdn_gated) &&
        allocate(geometry.attn_projection_bytes, r_attn_proj) &&
        allocate(geometry.attn_query_bytes, r_attn_q) &&
        allocate(geometry.attn_query_bytes, r_attn_gate) &&
        allocate(geometry.attn_query_bytes, r_attended) &&
        allocate(geometry.attn_record_scratch_bytes, r_partials) &&
        allocate(geometry.attn_record_scratch_bytes, r_weights) &&
        allocate(geometry.router_logits_bytes, r_logits_router) &&
        allocate(geometry.expert_id_bytes, r_ids) &&
        allocate(geometry.expert_coefficient_bytes, r_coefs) && allocate(4, r_shared) &&
        allocate(geometry.expert_hidden_bytes, r_hidden) &&
        allocate(geometry.hidden_bytes, r_final) && allocate(geometry.logits_bytes, r_logits) &&
        allocate(geometry.argmax_value_bytes, r_vals) &&
        allocate(geometry.argmax_index_bytes, r_idx) && allocate(4, r_token);
    struct ReferenceState {
        MetalBuffer conv[2];
        MetalBuffer recurrent[2];
        MetalBuffer keys;
        MetalBuffer values;
    };
    std::vector<ReferenceState> reference_states(schedule.size());
    for (std::size_t layer = 0; layer < schedule.size() && allocated; ++layer) {
        if (schedule[layer] == LayerKind::GatedDelta) {
            allocated =
                allocated &&
                allocate(geometry.gdn_conv_state_bytes, reference_states[layer].conv[0]) &&
                allocate(geometry.gdn_conv_state_bytes, reference_states[layer].conv[1]) &&
                allocate(geometry.gdn_recurrent_state_bytes,
                         reference_states[layer].recurrent[0]) &&
                allocate(geometry.gdn_recurrent_state_bytes, reference_states[layer].recurrent[1]);
        } else {
            allocated = allocated &&
                        allocate(geometry.attn_cache_bytes, reference_states[layer].keys) &&
                        allocate(geometry.attn_cache_bytes, reference_states[layer].values);
        }
    }
    if (!allocated) {
        std::cerr << "reference allocation failed\n";
        return 10;
    }
    std::memcpy(r_token.contents(), &kFirstToken, 4);

    const MetalBuffer& img = *image.buffer;
    const DecodeBindings& bound = step.bindings;
    const auto weight = [&](std::uint32_t tensor) { return tensor_offsets[tensor]; };

    std::uint32_t total_mismatches = 0;
    for (std::uint32_t token = 0; token < kTokens; ++token) {
        const std::uint32_t context = kContexts[token];
        // Execution A: one pass, the executor.
        {
            auto command_buffer = create_command_buffer(*queue.command_queue);
            if (!command_buffer) {
                return 11;
            }
            auto pass = begin_compute_pass(std::move(*command_buffer.command_buffer));
            if (!pass) {
                return 12;
            }
            if (encode_token(step, *pass.compute_pass, context) != MetalCommandError::None) {
                std::cerr << "encode_token failed\n";
                return 13;
            }
            auto ended = end_compute_pass(std::move(*pass.compute_pass));
            if (!ended) {
                return 14;
            }
            auto pending = commit(std::move(*ended.command_buffer));
            if (!pending) {
                return 15;
            }
            ++submissions;
            if (auto execution = wait_until_completed(std::move(*pending.pending_execution));
                !execution) {
                std::cerr << "executor token failed: " << execution.failure_description.view()
                          << '\n';
                return 16;
            }
        }
        // Execution B: loop-built reference in the sealed order.
        const std::uint32_t parity = token & 1u;
        const std::uint32_t partitions = context + 1 <= 256 ? 1 : (context + 1 + 255) / 256;
        std::vector<Pass> passes;
        passes.push_back({&p.embed,
                          {&img, &img, &img, &r_token, &r_input},
                          {weight(bound.embedding.weight), weight(bound.embedding.scales),
                           weight(bound.embedding.biases), 0, 0},
                          {},
                          {8, 1, 1},
                          {256, 1, 1}});
        for (std::size_t layer = 0; layer < schedule.size(); ++layer) {
            const LayerBindings& lb = bound.layers[layer];
            ReferenceState& state = reference_states[layer];
            const MetalBuffer& layer_input = layer == 0 ? r_input : r_layer;
            const std::uint64_t input_offset = layer == 0 ? 0 : (layer - 1) * geometry.hidden_bytes;
            const std::uint64_t layer_offset = layer * geometry.hidden_bytes;
            passes.push_back({&p.rms,
                              {&layer_input, &img, &r_normed},
                              {input_offset, weight(lb.input_norm), 0},
                              {},
                              {1, 1, 1},
                              {512, 1, 1}});
            if (lb.kind == LayerKind::GatedDelta) {
                passes.push_back(
                    {&p.gdn_project,
                     {&r_normed, &img, &img, &img, &img, &img, &img, &img, &img, &img, &img, &img,
                      &img, &r_gdn_proj},
                     {0, weight(lb.gated_delta.qkv.weight), weight(lb.gated_delta.qkv.scales),
                      weight(lb.gated_delta.qkv.biases), weight(lb.gated_delta.z.weight),
                      weight(lb.gated_delta.z.scales), weight(lb.gated_delta.z.biases),
                      weight(lb.gated_delta.b.weight), weight(lb.gated_delta.b.scales),
                      weight(lb.gated_delta.b.biases), weight(lb.gated_delta.a.weight),
                      weight(lb.gated_delta.a.scales), weight(lb.gated_delta.a.biases), 0},
                     {},
                     {3088, 1, 1},
                     {128, 1, 1}});
                passes.push_back({&p.gdn_prepare,
                                  {&r_gdn_proj, &state.conv[parity], &img, &r_gdn_qk, &r_gdn_v,
                                   &r_gdn_z, &state.conv[parity ^ 1u]},
                                  {0, 0, weight(lb.gated_delta.conv_weight), 0, 0, 0, 0},
                                  {},
                                  {96, 1, 1},
                                  {128, 1, 1}});
                passes.push_back(
                    {&p.gdn_recurrence,
                     {&r_gdn_qk, &r_gdn_v, &r_gdn_proj, &img, &img, &state.recurrent[parity],
                      &r_gdn_y, &state.recurrent[parity ^ 1u]},
                     {0, 0, 0, weight(lb.gated_delta.a_log), weight(lb.gated_delta.dt_bias), 0, 0,
                      0},
                     {},
                     {1, 32, 32},
                     {32, 4, 1}});
                passes.push_back({&p.gdn_gate_norm,
                                  {&r_gdn_y, &r_gdn_z, &img, &r_gdn_gated},
                                  {0, 0, weight(lb.gated_delta.norm_weight), 0},
                                  {},
                                  {32, 1, 1},
                                  {128, 1, 1}});
                passes.push_back(
                    {&p.out_projection,
                     {&r_gdn_gated, &img, &img, &img, &r_branch},
                     {0, weight(lb.gated_delta.out.weight), weight(lb.gated_delta.out.scales),
                      weight(lb.gated_delta.out.biases), layer_offset},
                     {},
                     {1024, 1, 1},
                     {64, 1, 1}});
            } else {
                passes.push_back(
                    {&p.attn_project,
                     {&r_normed, &img, &img, &img, &img, &img, &img, &img, &img, &img,
                      &r_attn_proj},
                     {0, weight(lb.attention.query.weight), weight(lb.attention.query.scales),
                      weight(lb.attention.query.biases), weight(lb.attention.key.weight),
                      weight(lb.attention.key.scales), weight(lb.attention.key.biases),
                      weight(lb.attention.value.weight), weight(lb.attention.value.scales),
                      weight(lb.attention.value.biases), 0},
                     {},
                     {2304, 1, 1},
                     {128, 1, 1}});
                passes.push_back({&p.attn_qk_rope,
                                  {&r_attn_proj, &img, &img, &r_attn_q, &r_attn_gate, &state.keys,
                                   &state.values},
                                  {0, weight(lb.attention.query_norm),
                                   weight(lb.attention.key_norm), 0, 0, 0, 0},
                                  {{context, 7}, {kCapacity, 8}},
                                  {18, 1, 1},
                                  {256, 1, 1}});
                if (partitions == 1) {
                    passes.push_back(
                        {&p.attention_decode,
                         {&r_attn_q, &r_attn_gate, &state.keys, &state.values, &r_attended},
                         {0, 0, 0, 0, 0},
                         {{context, 4}, {kCapacity, 6}},
                         {16, 1, 1},
                         {256, 1, 1},
                         {0, 1, 2, 3, 5}});
                } else {
                    passes.push_back({&p.attention_scores,
                                      {&r_attn_q, &state.keys, &r_weights},
                                      {0, 0, 0},
                                      {{context, 2}, {kCapacity, 3}, {256, 4}, {partitions, 6}},
                                      {4, partitions, 1},
                                      {256, 4, 1},
                                      {0, 1, 5}});
                    passes.push_back({&p.attention_values,
                                      {&r_weights, &state.values, &r_partials},
                                      {0, 0, 0},
                                      {{context, 2}, {kCapacity, 3}, {256, 4}, {partitions, 6}},
                                      {2, partitions, 1},
                                      {32, 4, 8},
                                      {0, 1, 5}});
                    passes.push_back({&p.attention_combine,
                                      {&r_partials, &r_attn_gate, &r_attended},
                                      {0, 0, 0},
                                      {{partitions, 2}},
                                      {16, 1, 1},
                                      {256, 1, 1},
                                      {0, 1, 3}});
                }
                passes.push_back(
                    {&p.out_projection,
                     {&r_attended, &img, &img, &img, &r_branch},
                     {0, weight(lb.attention.out.weight), weight(lb.attention.out.scales),
                      weight(lb.attention.out.biases), layer_offset},
                     {},
                     {1024, 1, 1},
                     {64, 1, 1}});
            }
            passes.push_back({&p.residual_rms,
                              {&layer_input, &r_branch, &r_residual, &img, &r_normed},
                              {input_offset, layer_offset, layer_offset, weight(lb.post_norm), 0},
                              {},
                              {1, 1, 1},
                              {512, 1, 1}});
            passes.push_back({&p.router,
                              {&r_normed, &img, &img, &img, &img, &img, &img, &r_logits_router},
                              {0, weight(lb.router.weight), weight(lb.router.scales),
                               weight(lb.router.biases), weight(lb.shared_router.weight),
                               weight(lb.shared_router.scales), weight(lb.shared_router.biases), 0},
                              {},
                              {257, 1, 1},
                              {32, 1, 1}});
            passes.push_back({&p.router_select,
                              {&r_logits_router, &r_ids, &r_coefs, &r_shared},
                              {0, 0, 0, 0},
                              {{8, 4}},
                              {1, 1, 1},
                              {256, 1, 1}});
            passes.push_back({&p.grouped_upgate,
                              {&r_normed, &r_ids, &img, &img, &img, &img, &img, &img, &img, &img,
                               &img, &img, &img, &img, &r_hidden},
                              {0, 0, weight(lb.expert_gate.weight), weight(lb.expert_gate.scales),
                               weight(lb.expert_gate.biases), weight(lb.expert_up.weight),
                               weight(lb.expert_up.scales), weight(lb.expert_up.biases),
                               weight(lb.shared_gate.weight), weight(lb.shared_gate.scales),
                               weight(lb.shared_gate.biases), weight(lb.shared_up.weight),
                               weight(lb.shared_up.scales), weight(lb.shared_up.biases), 0},
                              {},
                              {4608, 1, 1},
                              {32, 1, 1}});
            passes.push_back(
                {&p.grouped_down_res,
                 {&r_hidden, &r_ids, &r_coefs, &r_shared, &img, &img, &img, &img, &img, &img,
                  &r_moe, &r_residual, &r_layer},
                 {0, 0, 0, 0, weight(lb.expert_down.weight), weight(lb.expert_down.scales),
                  weight(lb.expert_down.biases), weight(lb.shared_down.weight),
                  weight(lb.shared_down.scales), weight(lb.shared_down.biases), layer_offset,
                  layer_offset, layer_offset},
                 {},
                 {2048, 1, 1},
                 {32, 1, 1}});
        }
        passes.push_back(
            {&p.rms,
             {&r_layer, &img, &r_final},
             {(schedule.size() - 1) * geometry.hidden_bytes, weight(bound.final_norm), 0},
             {},
             {1, 1, 1},
             {512, 1, 1}});
        passes.push_back({&p.lmhead,
                          {&r_final, &img, &img, &img, &r_logits},
                          {0, weight(bound.head.weight), weight(bound.head.scales),
                           weight(bound.head.biases), 0},
                          {},
                          {kVocabulary, 1, 1},
                          {32, 1, 1}});
        passes.push_back({&p.argmax_stage1,
                          {&r_logits, &r_vals, &r_idx},
                          {0, 0, 0},
                          {{kVocabulary, 1}},
                          {256, 1, 1},
                          {256, 1, 1},
                          {0, 2, 3}});
        passes.push_back(
            {&p.argmax_stage2, {&r_vals, &r_idx, &r_token}, {0, 0, 0}, {}, {1, 1, 1}, {1, 1, 1}});
        if (const int rc = run_batch(*queue.command_queue, passes, "reference token"); rc != 0) {
            return 20 + rc;
        }

        // Byte equality at every boundary, then the finiteness gate: a
        // sentinel or out-of-range token means the chain degenerated and the
        // fixture is invalid regardless of equality.
        std::uint32_t mismatches = 0;
        mismatches +=
            compare_bytes(r_branch, step.branch_stream, geometry.layer_stream_bytes, "branch");
        mismatches += compare_bytes(r_residual, step.residual_stream, geometry.layer_stream_bytes,
                                    "residual");
        mismatches += compare_bytes(r_moe, step.moe_stream, geometry.layer_stream_bytes, "moe");
        mismatches +=
            compare_bytes(r_layer, step.layer_stream, geometry.layer_stream_bytes, "layer");
        for (std::size_t layer = 0; layer < schedule.size(); ++layer) {
            const DecodeLayerState& executor_state = step.state.layers[layer];
            if (schedule[layer] == LayerKind::GatedDelta) {
                const MetalBuffer& conv_out =
                    executor_state.swapped ? executor_state.first : executor_state.first_out;
                const MetalBuffer& recurrent_out =
                    executor_state.swapped ? executor_state.second : executor_state.second_out;
                mismatches += compare_bytes(reference_states[layer].conv[parity ^ 1u], conv_out,
                                            geometry.gdn_conv_state_bytes, "conv-state");
                mismatches +=
                    compare_bytes(reference_states[layer].recurrent[parity ^ 1u], recurrent_out,
                                  geometry.gdn_recurrent_state_bytes, "recurrent-state");
            } else {
                mismatches += compare_bytes(reference_states[layer].keys, executor_state.first,
                                            geometry.attn_cache_bytes, "keys");
                mismatches += compare_bytes(reference_states[layer].values, executor_state.second,
                                            geometry.attn_cache_bytes, "values");
            }
        }
        mismatches += compare_bytes(r_final, step.final_hidden, geometry.hidden_bytes, "final");
        mismatches += compare_bytes(r_logits, step.logits, geometry.logits_bytes, "logits");
        mismatches +=
            compare_bytes(r_vals, step.argmax_values, geometry.argmax_value_bytes, "argmax-values");
        mismatches += compare_bytes(r_idx, step.argmax_indices, geometry.argmax_index_bytes,
                                    "argmax-indices");
        mismatches += compare_bytes(r_token, step.token_id, 4, "token");
        const std::uint32_t executor_token = *static_cast<std::uint32_t*>(step.token_id.contents());
        std::cout << "token " << token << " (context " << context << ", partitions " << partitions
                  << "): mismatched boundaries " << mismatches << ", next token id "
                  << executor_token << '\n';
        total_mismatches += mismatches;
        if (executor_token >= kVocabulary) {
            std::cout << "composition fixtures: DEGENERATE (token out of range)\n";
            return 31;
        }
        advance_decode_state(step);
    }

    if (total_mismatches != 0) {
        std::cout << "composition fixtures: FAIL\n";
        return 30;
    }
    std::cout << "composition fixtures: PASS\n"
              << "  device: " << device.device->name() << '\n'
              << "  layers: " << schedule.size() << ", tokens chained: " << kTokens << '\n'
              << "  command buffers submitted: " << submissions << '\n';
    return 0;
}
