#pragma once

#include "tatara/model/checkpoint_layout.h"
#include "tatara/model/qwen36_plan.h"

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace tatara::runtime {

// Tensor indices into the prepared checkpoint's record span, resolved once
// at build time so the per-token walk performs no name lookups. A bound
// dispatch argument is the owned model image plus the indexed tensor's
// image offset.
struct QuantizedBinding {
    std::uint32_t weight;
    std::uint32_t scales;
    std::uint32_t biases;
    std::uint64_t weight_size_bytes;
    std::uint64_t scale_size_bytes;
    std::uint64_t bias_size_bytes;
};

struct GatedDeltaBindings {
    QuantizedBinding qkv;
    QuantizedBinding z;
    QuantizedBinding b;
    QuantizedBinding a;
    QuantizedBinding out;
    std::uint32_t conv_weight;
    std::uint32_t a_log;
    std::uint32_t dt_bias;
    std::uint32_t norm_weight;
};

struct AttentionBindings {
    QuantizedBinding query;
    QuantizedBinding key;
    QuantizedBinding value;
    QuantizedBinding out;
    std::uint32_t query_norm;
    std::uint32_t key_norm;
};

// One layer's complete binding set. Exactly one family member is
// meaningful, selected by kind; both are stored plainly because the walk
// branches on kind and nothing else.
struct LayerBindings {
    model::qwen36::LayerKind kind;
    std::uint32_t input_norm;
    std::uint32_t post_norm;
    GatedDeltaBindings gated_delta;
    AttentionBindings attention;
    QuantizedBinding router;
    QuantizedBinding shared_router;
    QuantizedBinding expert_gate;
    QuantizedBinding expert_up;
    QuantizedBinding expert_down;
    QuantizedBinding shared_gate;
    QuantizedBinding shared_up;
    QuantizedBinding shared_down;
};

struct DecodeBindings {
    QuantizedBinding embedding;
    std::vector<LayerBindings> layers;
    std::uint32_t final_norm;
    QuantizedBinding head;
};

enum class DecodeBindingError : std::uint8_t {
    None,
    NoTensors,
    MissingTensor,
};

struct DecodeBindingsResult {
    DecodeBindingError error;
    std::string missing_name;
    std::optional<DecodeBindings> bindings;

    explicit operator bool() const noexcept {
        return error == DecodeBindingError::None && bindings.has_value();
    }
};

// Resolves every tensor name the sealed decode schedule dispatches, in the
// sealed naming scheme, against the prepared checkpoint records. Any
// missing name is a typed error carrying that exact name.
DecodeBindingsResult resolve_decode_bindings(std::span<const model::qwen36::LayerKind> schedule,
                                             std::span<const model::TensorRecord> tensors);

} // namespace tatara::runtime
