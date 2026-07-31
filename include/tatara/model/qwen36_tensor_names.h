#pragma once

#include <string_view>

namespace tatara::model::qwen36 {

// The checkpoint tensor naming scheme a package ships with, so the runtime
// composes names instead of spelling them. Values live in the package record's
// `[tensor_names]` table and reach the runtime as
// `qwen36::generated::kTensorNaming`; this header owns only the shape.
//
// Names are flat dotted paths. Per-layer names read
// `<model_prefix><layer_prefix><decimal index><layer_index_suffix><block
// prefix><leaf>`. Quantized entries name a stem that the three affine-Q4
// component suffixes complete; plain entries name the leaf exactly, because the
// checkpoint spells some of them without a `.weight` tail.
//
// Two structural assumptions are not expressible in this table and stay in the
// composer in src/runtime/decode_bindings.cpp: the layer index is rendered as
// an unpadded base-ten integer, and every layer carries a mixture-of-experts
// block. A package that pads its layer index or interleaves dense layers needs
// the composer changed, not a new field here.
//
// Member order is the generator's field order, not the package record's key
// order -- TOML key order is free and the generator emits in its own order.
// C++20 designated initializers make a drift between this struct and the
// generator a compile error rather than a silent rename.
struct TensorNaming {
    std::string_view model_prefix;
    std::string_view layer_prefix;
    std::string_view layer_index_suffix;

    std::string_view embedding_stem;
    std::string_view final_norm;
    // The head sits outside the model prefix, so it carries a full stem.
    std::string_view head_stem;

    std::string_view input_norm;
    std::string_view post_norm;

    std::string_view gated_delta_prefix;
    std::string_view gated_delta_qkv_stem;
    std::string_view gated_delta_z_stem;
    std::string_view gated_delta_b_stem;
    std::string_view gated_delta_a_stem;
    std::string_view gated_delta_out_stem;
    std::string_view gated_delta_conv;
    std::string_view gated_delta_a_log;
    std::string_view gated_delta_dt_bias;
    std::string_view gated_delta_norm;

    std::string_view attention_prefix;
    std::string_view attention_query_stem;
    std::string_view attention_key_stem;
    std::string_view attention_value_stem;
    std::string_view attention_out_stem;
    std::string_view attention_query_norm;
    std::string_view attention_key_norm;

    std::string_view mixture_prefix;
    std::string_view router_stem;
    std::string_view shared_router_stem;
    std::string_view expert_gate_stem;
    std::string_view expert_up_stem;
    std::string_view expert_down_stem;
    std::string_view shared_gate_stem;
    std::string_view shared_up_stem;
    std::string_view shared_down_stem;

    std::string_view quantized_weight;
    std::string_view quantized_scales;
    std::string_view quantized_biases;
};

} // namespace tatara::model::qwen36
