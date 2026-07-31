#include "tatara/generated/model_plan.h"

#include <algorithm>
#include <cstddef>

namespace {

using tatara::model::qwen36::LayerKind;
using tatara::model::qwen36::generated::kModelPlan;

static_assert(kModelPlan.layers.size() == 40);
static_assert(kModelPlan.layers[0] == LayerKind::GatedDelta);
static_assert(kModelPlan.layers[3] == LayerKind::FullAttention);
static_assert(kModelPlan.layers[39] == LayerKind::FullAttention);
static_assert(kModelPlan.artifact.tensor_bytes == 20'401'929'952ULL);
static_assert(kModelPlan.package_sha256.size() == 64);
static_assert(kModelPlan.artifact.manifest_sha256.size() == 64);
static_assert(kModelPlan.tokenizer.vocabulary == 248'320);
static_assert(kModelPlan.tokenizer.populated_vocabulary == 248'077);
static_assert(kModelPlan.tokenizer.maximum_context == 262'144);
static_assert(kModelPlan.tokenizer.data_size_bytes == 19'989'343);
static_assert(kModelPlan.tokenizer.stop_token_count == 2);
static_assert(kModelPlan.tokenizer.stop_token_ids[0] == 248'046);
static_assert(kModelPlan.tokenizer.stop_token_ids[1] == 248'044);

} // namespace

int main() {
    const auto full_attention_layers =
        std::count(kModelPlan.layers.begin(), kModelPlan.layers.end(), LayerKind::FullAttention);
    const auto gated_delta_layers =
        std::count(kModelPlan.layers.begin(), kModelPlan.layers.end(), LayerKind::GatedDelta);

    if (full_attention_layers != 10 || gated_delta_layers != 30) {
        return 1;
    }
    if (kModelPlan.artifact.weight_file_count != 4 ||
        kModelPlan.initial_serving_capacity != 16'384) {
        return 2;
    }
    if (kModelPlan.tokenizer.data_path != "tokenizer.json" ||
        kModelPlan.tokenizer.template_path != "chat_template.jinja" ||
        !kModelPlan.tokenizer.default_thinking) {
        return 3;
    }
    return 0;
}
