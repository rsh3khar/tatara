#include "tatara/runtime/decode_bindings.h"

#include "tatara/generated/model_plan.h"

#include <cstddef>
#include <initializer_list>
#include <string>
#include <string_view>
#include <unordered_map>

namespace tatara::runtime {

namespace {

using TensorIndex = std::unordered_map<std::string_view, std::uint32_t>;
using model::qwen36::TensorNaming;

std::string join(std::initializer_list<std::string_view> parts) {
    std::size_t size = 0;
    for (const std::string_view part : parts) {
        size += part.size();
    }
    std::string joined;
    joined.reserve(size);
    for (const std::string_view part : parts) {
        joined.append(part);
    }
    return joined;
}

// Latches the first missing name in resolution order, so a partial
// checkpoint reports the earliest tensor the schedule could not bind.
struct Resolver {
    const TensorIndex& index;
    std::span<const model::TensorRecord> tensors;
    const TensorNaming& naming;
    DecodeBindingError error = DecodeBindingError::None;
    std::string missing_name;

    std::uint32_t plain(const std::string& name) {
        if (error != DecodeBindingError::None) {
            return 0;
        }
        const auto found = index.find(name);
        if (found == index.end()) {
            error = DecodeBindingError::MissingTensor;
            missing_name = name;
            return 0;
        }
        return found->second;
    }

    std::uint64_t size_bytes(std::uint32_t tensor) const {
        return error == DecodeBindingError::None && tensor < tensors.size()
                   ? tensors[tensor].size_bytes
                   : 0;
    }

    QuantizedBinding quantized(std::string_view stem) {
        const std::uint32_t weight =
            plain(join({stem, naming.quantized_weight}));
        const std::uint32_t scales =
            plain(join({stem, naming.quantized_scales}));
        const std::uint32_t biases =
            plain(join({stem, naming.quantized_biases}));
        return QuantizedBinding{
            .weight = weight,
            .scales = scales,
            .biases = biases,
            .weight_size_bytes = size_bytes(weight),
            .scale_size_bytes = size_bytes(scales),
            .bias_size_bytes = size_bytes(biases),
        };
    }
};

} // namespace

DecodeBindingsResult resolve_decode_bindings(std::span<const model::qwen36::LayerKind> schedule,
                                             std::span<const model::TensorRecord> tensors) {
    if (tensors.empty()) {
        return {
            .error = DecodeBindingError::NoTensors, .missing_name = {}, .bindings = std::nullopt};
    }
    TensorIndex index;
    index.reserve(tensors.size());
    for (std::uint32_t i = 0; i < tensors.size(); ++i) {
        index.emplace(tensors[i].name, i);
    }
    const TensorNaming& naming = model::qwen36::generated::kTensorNaming;
    Resolver resolve{.index = index, .tensors = tensors, .naming = naming};

    DecodeBindings bindings;
    bindings.embedding = resolve.quantized(join({naming.model_prefix, naming.embedding_stem}));
    bindings.layers.reserve(schedule.size());
    // The package table cannot express two structural facts, so they live here:
    // the layer index renders as an unpadded base-ten integer, and every layer
    // carries a mixture-of-experts block.
    for (std::size_t layer = 0; layer < schedule.size(); ++layer) {
        const std::string prefix = join({naming.model_prefix, naming.layer_prefix,
                                         std::to_string(layer), naming.layer_index_suffix});
        LayerBindings bound{};
        bound.kind = schedule[layer];
        bound.input_norm = resolve.plain(join({prefix, naming.input_norm}));
        bound.post_norm = resolve.plain(join({prefix, naming.post_norm}));
        if (schedule[layer] == model::qwen36::LayerKind::GatedDelta) {
            const std::string gated = join({prefix, naming.gated_delta_prefix});
            bound.gated_delta = GatedDeltaBindings{
                .qkv = resolve.quantized(join({gated, naming.gated_delta_qkv_stem})),
                .z = resolve.quantized(join({gated, naming.gated_delta_z_stem})),
                .b = resolve.quantized(join({gated, naming.gated_delta_b_stem})),
                .a = resolve.quantized(join({gated, naming.gated_delta_a_stem})),
                .out = resolve.quantized(join({gated, naming.gated_delta_out_stem})),
                .conv_weight = resolve.plain(join({gated, naming.gated_delta_conv})),
                .a_log = resolve.plain(join({gated, naming.gated_delta_a_log})),
                .dt_bias = resolve.plain(join({gated, naming.gated_delta_dt_bias})),
                .norm_weight = resolve.plain(join({gated, naming.gated_delta_norm})),
            };
        } else {
            const std::string attention = join({prefix, naming.attention_prefix});
            bound.attention = AttentionBindings{
                .query = resolve.quantized(join({attention, naming.attention_query_stem})),
                .key = resolve.quantized(join({attention, naming.attention_key_stem})),
                .value = resolve.quantized(join({attention, naming.attention_value_stem})),
                .out = resolve.quantized(join({attention, naming.attention_out_stem})),
                .query_norm = resolve.plain(join({attention, naming.attention_query_norm})),
                .key_norm = resolve.plain(join({attention, naming.attention_key_norm})),
            };
        }
        const std::string mixture = join({prefix, naming.mixture_prefix});
        bound.router = resolve.quantized(join({mixture, naming.router_stem}));
        bound.shared_router = resolve.quantized(join({mixture, naming.shared_router_stem}));
        bound.expert_gate = resolve.quantized(join({mixture, naming.expert_gate_stem}));
        bound.expert_up = resolve.quantized(join({mixture, naming.expert_up_stem}));
        bound.expert_down = resolve.quantized(join({mixture, naming.expert_down_stem}));
        bound.shared_gate = resolve.quantized(join({mixture, naming.shared_gate_stem}));
        bound.shared_up = resolve.quantized(join({mixture, naming.shared_up_stem}));
        bound.shared_down = resolve.quantized(join({mixture, naming.shared_down_stem}));
        bindings.layers.push_back(bound);
    }
    bindings.final_norm = resolve.plain(join({naming.model_prefix, naming.final_norm}));
    bindings.head = resolve.quantized(naming.head_stem);

    if (resolve.error != DecodeBindingError::None) {
        return {.error = resolve.error,
                .missing_name = std::move(resolve.missing_name),
                .bindings = std::nullopt};
    }
    return {.error = DecodeBindingError::None, .missing_name = {}, .bindings = std::move(bindings)};
}

} // namespace tatara::runtime
