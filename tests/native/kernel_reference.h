#pragma once

#include <cstdint>
#include <span>

namespace tatara::testing {

std::uint16_t bf16_from_f32(float value) noexcept;
float f32_from_bf16(std::uint16_t bits) noexcept;

// The device flushes bfloat subnormals to signed zero on arithmetic input
// and output; the oracle models that flush explicitly.
std::uint16_t flush_subnormal_bf16(std::uint16_t bits) noexcept;
float flush_subnormal_f32(float value) noexcept;

// Deterministic fixture source; never seeded from time or entropy.
class Xorshift64Star {
  public:
    explicit Xorshift64Star(std::uint64_t seed) noexcept;

    std::uint64_t next() noexcept;
    std::uint16_t next_finite_bf16() noexcept;
    // Exponent-banded values (about 2^-15 to 2^8) so summations stay
    // same-scale; full-range draws let one term absorb every other.
    std::uint16_t next_banded_bf16() noexcept;

  private:
    std::uint64_t state_;
};

enum class MultiplyAddOrder : std::uint8_t { Fused, Separate };
enum class SimdTreeShape : std::uint8_t {
    Halving,
    AdjacentPairs,
    Linear,
    QuadLinearHalving,
    QuadLinearAdjacent,
    QuadLinearLinear,
};
inline constexpr SimdTreeShape kAllTreeShapes[] = {
    SimdTreeShape::Halving,
    SimdTreeShape::AdjacentPairs,
    SimdTreeShape::Linear,
    SimdTreeShape::QuadLinearHalving,
    SimdTreeShape::QuadLinearAdjacent,
    SimdTreeShape::QuadLinearLinear,
};
enum class RsqrtOrder : std::uint8_t { ReciprocalOfSqrt, CorrectlyRounded };

float multiply_add_candidate(float quant, float scale, float bias, MultiplyAddOrder order) noexcept;
float rsqrt_candidate(float value, RsqrtOrder order) noexcept;
std::uint16_t bfloat_multiply_candidate(std::uint16_t left, std::uint16_t right) noexcept;

void embed_row_q4_reference(std::span<const std::uint32_t> quant_words,
                            std::span<const std::uint16_t> scales,
                            std::span<const std::uint16_t> biases, std::uint32_t token,
                            std::uint32_t hidden, std::uint32_t group_size, MultiplyAddOrder order,
                            std::span<std::uint16_t> row);

float simd_tree_sum(std::span<const float> lanes, SimdTreeShape shape);
float rms_total_of_squares(std::span<const std::uint16_t> input, std::uint32_t hidden,
                           SimdTreeShape shape);
float rms_mean_of_squares(std::span<const std::uint16_t> input, std::uint32_t hidden,
                          SimdTreeShape shape);

// Applies a caller-supplied inverse norm; the gated window queries the device
// for the exact precise::rsqrt bits because no CPU candidate reproduces them.
void rms_only_reference_with_inverse(std::span<const std::uint16_t> input,
                                     std::span<const std::uint16_t> weight, std::uint32_t hidden,
                                     float inverse, std::span<std::uint16_t> output);

void rms_only_reference(std::span<const std::uint16_t> input, std::span<const std::uint16_t> weight,
                        std::uint32_t hidden, float epsilon, SimdTreeShape shape,
                        RsqrtOrder rsqrt_order, std::span<std::uint16_t> output);

struct MultiplyAddDiscriminator {
    bool found;
    float quant;
    std::uint16_t scale;
    std::uint16_t bias;
};

// Inputs on which rival candidates disagree, proving a sweep can discriminate.
MultiplyAddDiscriminator find_multiply_add_discriminator(std::uint64_t seed, std::uint32_t trials);
bool find_rsqrt_discriminator(std::uint64_t seed, std::uint32_t trials, float& value);

bool find_rms_sum_discriminator(std::uint64_t seed, std::uint32_t trials, std::uint32_t hidden,
                                float epsilon, std::span<std::uint16_t> input,
                                std::span<std::uint16_t> weight);

// Positive banded 32-lane vector on which the two shapes' totals differ.
bool find_tree_shape_discriminator(std::uint64_t seed, std::uint32_t trials, SimdTreeShape first,
                                   SimdTreeShape second, std::span<float> lanes);

// GDN-boundary references. Chain candidates model multiply-add contraction
// uniformly per kernel; transcendental candidates model libm precision; a
// sweep pins each, falling back to device point queries.
enum class ChainOrder : std::uint8_t { Separate, Fused };
enum class TranscendentalModel : std::uint8_t { FloatLibm, DoubleLibm };

std::uint16_t bfloat_add_candidate(std::uint16_t left, std::uint16_t right) noexcept;
std::uint16_t bf16_sigmoid_candidate(std::uint16_t input, TranscendentalModel model) noexcept;
float f32_sigmoid_candidate(float input, TranscendentalModel model) noexcept;
float gdn_decay_candidate(std::uint16_t shifted, std::uint16_t a_log,
                          TranscendentalModel model) noexcept;
// The convolution chain has no contraction axis: bfloat16 weights times
// bfloat16 taps are exact in float, so fused and separate orders coincide.
float conv4_reference(std::span<const std::uint16_t> weights, std::span<const float> taps) noexcept;
float q4_dot_reference(std::span<const std::uint16_t> x, std::span<const std::uint32_t> words,
                       std::span<const std::uint16_t> scales, std::span<const std::uint16_t> biases,
                       std::uint32_t k_size, SimdTreeShape tree, ChainOrder order);

struct GdnGeometry {
    std::uint32_t key_heads;
    std::uint32_t value_heads;
    std::uint32_t head_dimension;
};

void gdn_conv_forward(std::span<const std::uint16_t> projection,
                      std::span<const std::uint16_t> conv_state,
                      std::span<const std::uint16_t> conv_weights, std::uint32_t channels,
                      std::span<std::uint16_t> convolved, std::span<std::uint16_t> new_conv_state);

float gdn_head_rsqrt_argument(std::span<const std::uint16_t> activated, std::uint32_t head,
                              std::uint32_t head_dimension, float epsilon, SimdTreeShape tree);

void gdn_qk_normalize(std::span<const std::uint16_t> activated,
                      std::span<const float> head_inverses, const GdnGeometry& geometry,
                      std::uint16_t query_scale, std::uint16_t key_scale,
                      std::span<std::uint16_t> qk);

void gdn_recurrence_reference(std::span<const std::uint16_t> qk,
                              std::span<const std::uint16_t> value, std::span<const float> decay,
                              std::span<const float> beta, std::span<const float> state_in,
                              const GdnGeometry& geometry, SimdTreeShape tree, ChainOrder order,
                              std::span<std::uint16_t> output, std::span<float> state_out);

void gdn_gate_norm_reference(std::span<const std::uint16_t> recurrence_out,
                             std::span<const std::uint16_t> gate,
                             std::span<const std::uint16_t> weight,
                             std::span<const float> head_inverses,
                             std::span<const float> gate_sigmoid, std::uint32_t heads,
                             std::uint32_t head_dimension, std::span<std::uint16_t> output);

struct AttnGeometry {
    std::uint32_t query_heads;
    std::uint32_t kv_heads;
    std::uint32_t head_dimension;
};

// Weighted 256-value RMS + RoPE per head. Trigonometry and rsqrt values are
// device-queried; the rotation two-term combine carries the chain axis.
void attn_qk_rope_reference(std::span<const std::uint16_t> projection,
                            std::span<const std::uint16_t> q_weight,
                            std::span<const std::uint16_t> k_weight, std::uint32_t position,
                            std::uint32_t capacity, const AttnGeometry& geometry,
                            std::span<const float> head_inverses, std::span<const float> cosines,
                            std::span<const float> sines, ChainOrder order,
                            std::span<std::uint16_t> q_out, std::span<std::uint16_t> gate_out,
                            std::span<std::uint16_t> keys, std::span<std::uint16_t> values);

float attn_rms_argument(std::span<const std::uint16_t> projection, std::uint32_t base,
                        std::uint32_t head_dimension, float epsilon);

// One decode step for one head over positions [0, context]: computes every
// exp argument for the probe to query, then consumes queried exp values.
void attention_decode_arguments(std::span<const std::uint16_t> q,
                                std::span<const std::uint16_t> keys, std::uint32_t head,
                                std::uint32_t context, std::uint32_t capacity,
                                const AttnGeometry& geometry, std::vector<float>& exp_arguments);

void attention_decode_reference(std::span<const std::uint16_t> q,
                                std::span<const std::uint16_t> gate,
                                std::span<const std::uint16_t> keys,
                                std::span<const std::uint16_t> values, std::uint32_t head,
                                std::uint32_t context, std::uint32_t capacity,
                                const AttnGeometry& geometry, std::span<const float> exp_values,
                                ChainOrder order, std::span<std::uint16_t> out);

void attention_scores_gqa4_arguments(std::span<const std::uint16_t> q,
                                     std::span<const std::uint16_t> keys, std::uint32_t head,
                                     std::uint32_t context, std::uint32_t capacity,
                                     std::uint32_t part, std::uint32_t partition,
                                     const AttnGeometry& geometry,
                                     std::vector<float>& exp_arguments);

void attention_scores_gqa4_reference(std::span<const std::uint16_t> q,
                                     std::span<const std::uint16_t> keys, std::uint32_t head,
                                     std::uint32_t context, std::uint32_t capacity,
                                     std::uint32_t part, std::uint32_t partition,
                                     const AttnGeometry& geometry,
                                     std::span<const float> exp_values, std::span<float> record);

void attention_values_gqa8_reference(std::span<const float> weight_record,
                                     std::span<const std::uint16_t> values, std::uint32_t kv_head,
                                     std::uint32_t context, std::uint32_t capacity,
                                     std::uint32_t part, std::uint32_t partition,
                                     const AttnGeometry& geometry, ChainOrder order,
                                     std::span<float> partial_record);

void attention_combine_arguments(std::span<const float> partial_records, std::uint32_t nparts,
                                 std::vector<float>& exp_arguments);

void attention_combine_reference(std::span<const float> partial_records,
                                 std::span<const std::uint16_t> gate, std::uint32_t head,
                                 std::uint32_t nparts, const AttnGeometry& geometry,
                                 std::span<const float> exp_values,
                                 std::span<const float> gate_exp_values, ChainOrder order,
                                 std::span<std::uint16_t> out);

// MoE-boundary references. The packed-Q4 and Q8 dots are pinned
// device-defined via same-source observation; these CPU candidates stand as
// evidence, mirroring the plain Q4 treatment.
float q4_dot_packed_reference(std::span<const std::uint16_t> x,
                              std::span<const std::uint32_t> words,
                              std::span<const std::uint16_t> scales,
                              std::span<const std::uint16_t> biases, std::uint32_t k_size,
                              SimdTreeShape tree, ChainOrder order);
float q8_dot_reference(std::span<const std::uint16_t> x, std::span<const std::uint8_t> bytes,
                       std::span<const std::uint16_t> scales, std::span<const std::uint16_t> biases,
                       std::uint32_t k_size, SimdTreeShape tree, ChainOrder order);

struct MoeGeometry {
    std::uint32_t experts;
    std::uint32_t active_experts;
    std::uint32_t expert_dimension;
};

inline constexpr std::uint32_t kMoeSentinelId = 0xffffffffu;

// f32 division on the device is a reciprocal-refinement class operation
// (labeled decision ): the oracle consumes point-queried
// quotients everywhere, and these candidates stand as evidence.
enum class DivisionModel : std::uint8_t { IeeeDivision, ReciprocalMultiply };

float f32_divide_candidate(float numerator, float denominator, DivisionModel model) noexcept;
bool find_divide_discriminator(std::uint64_t seed, std::uint32_t trials, float& numerator,
                               float& denominator);

// Softmax arguments for the router selection: `experts` entries of
// logit minus the tree maximum, then one negated shared logit for the
// sigmoid; the probe queries the device at every argument.
void router_select_arguments(std::span<const float> logits, std::uint32_t experts,
                             std::vector<float>& exp_arguments);

// Models router_select from source in query-plumbed phases: normalize
// division pairs (each exponential against the sealed halving-tree sum);
// the strict-greater argmax passes over the queried quotients, whose equal
// merges keep the lower slot (ties resolve by tree position — lowest
// bit-reversed index — not lowest index); renormalization and shared
// sigmoid division pairs; then the finalize consuming queried quotients
// with sentinel fill.
void router_select_normalize_arguments(std::span<const float> exp_values, std::uint32_t experts,
                                       std::vector<float>& numerators,
                                       std::vector<float>& denominators);
void router_select_pick(const MoeGeometry& geometry, std::uint32_t top_n,
                        std::span<const float> normalized, std::span<std::uint32_t> ids,
                        std::span<float> raw_coefficients);
// Pairs: top_n renormalizations, then the shared sigmoid pair last.
void router_select_renormalize_arguments(std::span<const float> raw_coefficients,
                                         std::uint32_t top_n, float shared_exp_value,
                                         std::vector<float>& numerators,
                                         std::vector<float>& denominators);
void router_select_finalize(const MoeGeometry& geometry, std::uint32_t top_n,
                            std::span<const float> quotients, std::span<std::uint32_t> ids,
                            std::span<float> coefficients, float& shared_coefficient);

// silu(gate) * up with one bfloat16 rounding; the silu quotient
// gate / (1 + exp(-gate)) arrives as a device-queried value.
std::uint16_t grouped_upgate_epilogue(float silu_quotient, float up) noexcept;

// Ascending slot-order accumulation over device-observed parts; sentinel
// slots skip. The multiply-add chain order is the adjudicated axis. parts
// holds one entry per slot with the shared expert last.
float grouped_down_total(std::span<const std::uint32_t> ids, std::span<const float> coefficients,
                         float shared_coefficient, std::span<const float> parts,
                         ChainOrder order) noexcept;
void grouped_down_res_epilogue(float total, std::uint16_t residual, std::uint16_t& moe_out,
                               std::uint16_t& layer_out) noexcept;

// The fused residual+RMS entry: residual sums round to bfloat16 exactly as
// the standalone add would, then the pinned RMS machinery consumes the
// rounded values; the rsqrt argument is device-queried.
void residual_rms_rounded_sums(std::span<const std::uint16_t> a, std::span<const std::uint16_t> b,
                               std::span<std::uint16_t> rounded);
float residual_rms_argument(std::span<const std::uint16_t> rounded, std::uint32_t hidden,
                            float epsilon, SimdTreeShape tree);

// Coefficient/part slot vectors on which the two chain orders' raw totals
// differ at float resolution.
bool find_down_chain_discriminator(std::uint64_t seed, std::uint32_t trials,
                                   std::span<float> coefficients, std::span<float> parts);

// Head-family references. The two-stage argmax is arithmetic-free beyond
// the bfloat16-to-float read — comparisons and shuffles round nothing — so
// the model is exact with no device queries. Every merge carries the
// explicit "greater, or equal with lower index" rule, and the stages
// compose to the global lowest-index argmax.
void logits_argmax_stage1_reference(std::span<const std::uint16_t> logits, std::uint32_t count,
                                    std::uint32_t threadgroup_id, float& group_value,
                                    std::uint32_t& group_index);
std::uint32_t logits_argmax_stage2_reference(std::span<const float> group_values,
                                             std::span<const std::uint32_t> group_indices);

} // namespace tatara::testing
