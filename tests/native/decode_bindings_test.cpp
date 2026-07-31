#include "tatara/runtime/decode_bindings.h"

#include <string>
#include <vector>

namespace {

using namespace tatara::runtime;
using tatara::model::TensorRecord;
using tatara::model::qwen36::LayerKind;

TensorRecord record(std::string name) {
    return TensorRecord{
        .name = std::move(name),
        .data_type = tatara::model::TensorDataType::BF16,
        .shape = {1},
        .shard = 0,
        .shard_offset_bytes = 0,
        .size_bytes = 2,
    };
}

std::vector<TensorRecord> sealed_records() {
    std::vector<TensorRecord> records;
    const auto quantized = [&](const std::string& stem) {
        records.push_back(record(stem + ".weight"));
        records.push_back(record(stem + ".scales"));
        records.push_back(record(stem + ".biases"));
    };
    quantized("language_model.model.embed_tokens");
    const std::string gated = "language_model.model.layers.0.linear_attn.";
    records.push_back(record("language_model.model.layers.0.input_layernorm.weight"));
    records.push_back(record("language_model.model.layers.0.post_attention_layernorm.weight"));
    for (const char* stem : {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"}) {
        quantized(gated + stem);
    }
    records.push_back(record(gated + "conv1d.weight"));
    records.push_back(record(gated + "A_log"));
    records.push_back(record(gated + "dt_bias"));
    records.push_back(record(gated + "norm.weight"));
    const std::string attention = "language_model.model.layers.1.self_attn.";
    records.push_back(record("language_model.model.layers.1.input_layernorm.weight"));
    records.push_back(record("language_model.model.layers.1.post_attention_layernorm.weight"));
    for (const char* stem : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
        quantized(attention + stem);
    }
    records.push_back(record(attention + "q_norm.weight"));
    records.push_back(record(attention + "k_norm.weight"));
    for (const std::string layer : {"0", "1"}) {
        const std::string mlp = "language_model.model.layers." + layer + ".mlp.";
        quantized(mlp + "gate");
        quantized(mlp + "shared_expert_gate");
        quantized(mlp + "switch_mlp.gate_proj");
        quantized(mlp + "switch_mlp.up_proj");
        quantized(mlp + "switch_mlp.down_proj");
        quantized(mlp + "shared_expert.gate_proj");
        quantized(mlp + "shared_expert.up_proj");
        quantized(mlp + "shared_expert.down_proj");
    }
    records.push_back(record("language_model.model.norm.weight"));
    quantized("language_model.lm_head");
    return records;
}

} // namespace

int main() {
    const std::vector<TensorRecord> records = sealed_records();
    const LayerKind schedule[] = {LayerKind::GatedDelta, LayerKind::FullAttention};

    const DecodeBindingsResult resolved = resolve_decode_bindings(schedule, records);
    if (!resolved) {
        return 1;
    }
    const DecodeBindings& bindings = *resolved.bindings;
    if (bindings.layers.size() != 2) {
        return 2;
    }
    // Every resolved index must name exactly the tensor the schedule asked
    // for; spot-check across families and the shared tail.
    if (records[bindings.embedding.scales].name != "language_model.model.embed_tokens.scales") {
        return 3;
    }
    if (records[bindings.layers[0].gated_delta.conv_weight].name !=
        "language_model.model.layers.0.linear_attn.conv1d.weight") {
        return 4;
    }
    if (records[bindings.layers[1].attention.query.weight].name !=
        "language_model.model.layers.1.self_attn.q_proj.weight") {
        return 5;
    }
    if (records[bindings.layers[1].shared_down.biases].name !=
        "language_model.model.layers.1.mlp.shared_expert.down_proj.biases") {
        return 6;
    }
    if (records[bindings.final_norm].name != "language_model.model.norm.weight" ||
        records[bindings.head.weight].name != "language_model.lm_head.weight") {
        return 7;
    }
    if (bindings.layers[0].kind != LayerKind::GatedDelta ||
        bindings.layers[1].kind != LayerKind::FullAttention) {
        return 8;
    }
    if (bindings.layers[1].attention.query.weight_size_bytes != 2 ||
        bindings.layers[1].attention.query.scale_size_bytes != 2 ||
        bindings.layers[1].attention.query.bias_size_bytes != 2) {
        return 11;
    }

    // A missing tensor is a typed error carrying the exact name.
    std::vector<TensorRecord> incomplete = sealed_records();
    incomplete.erase(incomplete.begin() + 40);
    const std::string dropped = records[40].name;
    const DecodeBindingsResult missing = resolve_decode_bindings(schedule, incomplete);
    if (missing || missing.error != DecodeBindingError::MissingTensor ||
        missing.missing_name != dropped) {
        return 9;
    }

    const DecodeBindingsResult empty = resolve_decode_bindings(schedule, {});
    if (empty || empty.error != DecodeBindingError::NoTensors) {
        return 10;
    }
    return 0;
}
