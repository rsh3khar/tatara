#pragma once

#include "tatara/model/artifact_identity.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace tatara::model::qwen36 {

enum class LayerKind : std::uint8_t {
    GatedDelta,
    FullAttention,
};

enum class WeightFormat : std::uint8_t {
    AffineQ4,
};

enum class TokenizerKind : std::uint8_t {
    ByteLevelBpe,
};

enum class TokenizerNormalization : std::uint8_t {
    Nfc,
};

enum class TokenizerPretokenizer : std::uint8_t {
    QwenRegexByteLevelV1,
};

enum class TokenizerDecoder : std::uint8_t {
    ByteLevel,
};

enum class ChatTemplateKind : std::uint8_t {
    Qwen36TextV1,
};

inline constexpr std::size_t kMaximumStopTokens = 8;

struct ModelDimensions {
    std::uint32_t hidden;
    std::uint32_t vocabulary;
};

struct AttentionSpec {
    std::uint32_t query_heads;
    std::uint32_t key_value_heads;
    std::uint32_t head_dimension;
};

struct GatedDeltaSpec {
    std::uint32_t recurrent_heads;
    std::uint32_t state_dimension;
};

struct MixtureOfExpertsSpec {
    std::uint32_t experts;
    std::uint32_t active_experts;
    // Intermediate width of one expert's gate/up projection.
    std::uint32_t expert_dimension;
};

struct WeightSpec {
    WeightFormat format;
    std::uint32_t group_size;
};

struct TokenizerSpec {
    TokenizerKind kind;
    TokenizerNormalization normalization;
    TokenizerPretokenizer pretokenizer;
    TokenizerDecoder decoder;
    ChatTemplateKind template_kind;
    std::string_view data_path;
    std::string_view data_sha256;
    std::uint64_t data_size_bytes;
    std::string_view config_path;
    std::string_view config_sha256;
    std::uint64_t config_size_bytes;
    std::string_view template_path;
    std::string_view template_sha256;
    std::uint64_t template_size_bytes;
    std::string_view split_pattern;
    std::uint32_t vocabulary;
    std::uint32_t populated_vocabulary;
    std::uint32_t maximum_context;
    std::uint32_t end_of_text_id;
    std::uint32_t message_start_id;
    std::uint32_t message_end_id;
    std::uint32_t thinking_start_id;
    std::uint32_t thinking_end_id;
    std::uint32_t padding_id;
    std::array<std::uint32_t, kMaximumStopTokens> stop_token_ids;
    std::uint32_t stop_token_count;
    bool default_thinking;
};

template <std::size_t LayerCount, std::size_t Period, std::size_t First>
consteval std::array<LayerKind, LayerCount> make_hybrid_layer_schedule() {
    static_assert(LayerCount > 0);
    static_assert(Period > 0);
    static_assert(First < LayerCount);

    std::array<LayerKind, LayerCount> layers{};
    for (std::size_t index = 0; index < LayerCount; ++index) {
        layers[index] = index >= First && (index - First) % Period == 0 ? LayerKind::FullAttention
                                                                        : LayerKind::GatedDelta;
    }
    return layers;
}

template <std::size_t LayerCount> struct StaticModelPlan {
    std::string_view id;
    std::string_view family;
    std::string_view package_sha256;
    ArtifactIdentity artifact;
    ModelDimensions dimensions;
    AttentionSpec attention;
    GatedDeltaSpec gated_delta;
    MixtureOfExpertsSpec mixture_of_experts;
    WeightSpec weights;
    TokenizerSpec tokenizer;
    std::uint32_t initial_serving_capacity;
    std::array<LayerKind, LayerCount> layers;
};

template <std::size_t LayerCount>
consteval bool valid_model_plan(const StaticModelPlan<LayerCount>& plan) {
    if (LayerCount == 0 || plan.id.empty() || plan.family.empty() ||
        plan.package_sha256.size() != 64) {
        return false;
    }
    if (plan.artifact.id.empty() || plan.artifact.model_type.empty() ||
        plan.artifact.manifest_sha256.size() != 64 || plan.artifact.tensor_count == 0 ||
        plan.artifact.tensor_bytes == 0 || plan.artifact.file_count == 0 ||
        plan.artifact.weight_file_count == 0) {
        return false;
    }
    if (plan.dimensions.hidden == 0 || plan.dimensions.vocabulary == 0 ||
        plan.attention.query_heads == 0 || plan.attention.key_value_heads == 0 ||
        plan.attention.head_dimension == 0 ||
        plan.attention.query_heads % plan.attention.key_value_heads != 0) {
        return false;
    }
    if (plan.gated_delta.recurrent_heads == 0 || plan.gated_delta.state_dimension == 0 ||
        plan.mixture_of_experts.experts == 0 || plan.mixture_of_experts.active_experts == 0 ||
        plan.mixture_of_experts.active_experts > plan.mixture_of_experts.experts ||
        plan.mixture_of_experts.expert_dimension == 0 || plan.weights.group_size == 0 ||
        plan.initial_serving_capacity == 0) {
        return false;
    }
    const TokenizerSpec& tokenizer = plan.tokenizer;
    if (tokenizer.data_path.empty() || tokenizer.data_sha256.size() != 64 ||
        tokenizer.data_size_bytes == 0 || tokenizer.data_size_bytes > 64ULL * 1024ULL * 1024ULL ||
        tokenizer.config_path.empty() || tokenizer.config_sha256.size() != 64 ||
        tokenizer.config_size_bytes == 0 || tokenizer.template_path.empty() ||
        tokenizer.template_sha256.size() != 64 || tokenizer.template_size_bytes == 0 ||
        tokenizer.split_pattern.empty() || tokenizer.vocabulary != plan.dimensions.vocabulary ||
        tokenizer.populated_vocabulary == 0 ||
        tokenizer.populated_vocabulary > tokenizer.vocabulary ||
        tokenizer.maximum_context < plan.initial_serving_capacity ||
        tokenizer.stop_token_count == 0 ||
        tokenizer.stop_token_count > tokenizer.stop_token_ids.size() ||
        tokenizer.end_of_text_id >= tokenizer.populated_vocabulary ||
        tokenizer.message_start_id >= tokenizer.populated_vocabulary ||
        tokenizer.message_end_id >= tokenizer.populated_vocabulary ||
        tokenizer.thinking_start_id >= tokenizer.populated_vocabulary ||
        tokenizer.thinking_end_id >= tokenizer.populated_vocabulary ||
        tokenizer.padding_id >= tokenizer.populated_vocabulary) {
        return false;
    }
    for (std::size_t index = 0; index < tokenizer.stop_token_count; ++index) {
        if (tokenizer.stop_token_ids[index] >= tokenizer.populated_vocabulary) {
            return false;
        }
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (tokenizer.stop_token_ids[index] == tokenizer.stop_token_ids[previous]) {
                return false;
            }
        }
    }

    std::size_t full_attention_layers = 0;
    for (const LayerKind kind : plan.layers) {
        full_attention_layers += kind == LayerKind::FullAttention ? 1 : 0;
    }
    return full_attention_layers > 0 && full_attention_layers < LayerCount;
}

} // namespace tatara::model::qwen36
