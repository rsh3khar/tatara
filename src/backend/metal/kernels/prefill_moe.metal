kernel void router_q8_block(
    device const bfloat* input [[buffer(0)]],
    device const uint* routed_words [[buffer(1)]],
    device const bfloat* routed_scales [[buffer(2)]],
    device const bfloat* routed_biases [[buffer(3)]],
    device const uint* shared_words [[buffer(4)]],
    device const bfloat* shared_scales [[buffer(5)]],
    device const bfloat* shared_biases [[buffer(6)]],
    device float* logits [[buffer(7)]],
    uint2 thread_position [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = thread_position.x / kSimdgroupWidth;
    if (row >= kMoeRouterRows) {
        return;
    }
    const uint position = thread_position.y;
    const bool shared = row == kMoeExperts;
    const ulong word_row =
        shared ? 0ul : ulong(row) * ulong(kHiddenDimension / 4u);
    const ulong group_row =
        shared ? 0ul : ulong(row) * ulong(kGroupsPerRow);
    const float result = q8_dot(
        input + position * kHiddenDimension,
        (shared ? shared_words : routed_words) + word_row,
        (shared ? shared_scales : routed_scales) + group_row,
        (shared ? shared_biases : routed_biases) + group_row,
        kHiddenDimension, lane);
    if (lane == 0u) {
        logits[position * kMoeRouterRows + row] =
            float(static_cast<bfloat>(result));
    }
}

// Deterministic oracle. One lane preserves the admitted halving reductions
// and strict-greater selection tree while every other lane remains idle.
// Equal candidates therefore resolve by tree position (lowest bit-reversed
// expert index), exactly like router_select, not by ascending expert index.
kernel void router_select_block(
    device const float* all_logits [[buffer(0)]],
    device uint* all_ids [[buffer(1)]],
    device float* all_coefficients [[buffer(2)]],
    device float* all_shared_coefficients [[buffer(3)]],
    constant uint& top_n [[buffer(4)]],
    uint position [[threadgroup_position_in_grid]],
    uint expert [[thread_position_in_threadgroup]]) {
    device const float* logits =
        all_logits + position * kMoeRouterRows;
    device uint* ids = all_ids + position * kMoeActiveExperts;
    device float* coefficients =
        all_coefficients + position * kMoeActiveExperts;
    threadgroup float values[kMoeExperts];
    threadgroup float reduction[kMoeExperts];
    threadgroup uint indices[kMoeExperts];
    if (expert == 0u) {
        for (uint index = 0u; index < kMoeExperts; ++index) {
            reduction[index] = logits[index];
        }
        for (uint offset = kMoeExperts / 2u; offset; offset >>= 1u) {
            for (uint index = 0u; index < offset; ++index) {
                reduction[index] =
                    max(reduction[index], reduction[index + offset]);
            }
        }
        const float maximum = reduction[0];
        for (uint index = 0u; index < kMoeExperts; ++index) {
            values[index] = exp(logits[index] - maximum);
            reduction[index] = values[index];
        }
        for (uint offset = kMoeExperts / 2u; offset; offset >>= 1u) {
            for (uint index = 0u; index < offset; ++index) {
                reduction[index] += reduction[index + offset];
            }
        }
        const float denominator = reduction[0];
        for (uint index = 0u; index < kMoeExperts; ++index) {
            values[index] /= denominator;
        }
        for (uint selected = 0u; selected < top_n; ++selected) {
            for (uint index = 0u; index < kMoeExperts; ++index) {
                reduction[index] = values[index];
                indices[index] = index;
            }
            for (uint offset = kMoeExperts / 2u; offset;
                 offset >>= 1u) {
                for (uint index = 0u; index < offset; ++index) {
                    if (reduction[index + offset] > reduction[index]) {
                        reduction[index] = reduction[index + offset];
                        indices[index] = indices[index + offset];
                    }
                }
            }
            const float best = reduction[0];
            const uint best_index = indices[0];
            ids[selected] = best_index;
            coefficients[selected] = best;
            values[best_index] = -1.0f;
        }
        float selected_sum = 0.0f;
        for (uint selected = 0u; selected < top_n; ++selected) {
            selected_sum += coefficients[selected];
        }
        for (uint selected = 0u; selected < top_n; ++selected) {
            coefficients[selected] /= selected_sum;
        }
        for (uint selected = top_n; selected < kMoeActiveExperts;
             ++selected) {
            ids[selected] = 0xffffffffu;
            coefficients[selected] = 0.0f;
        }
        all_shared_coefficients[position] =
            1.0f / (1.0f + exp(-logits[kMoeExperts]));
    }
}

// Parallel selector with byte-identical reduction and strict-greater
// selection trees matching the oracle and admitted decode path above.
kernel void router_select_block_parallel(
    device const float* all_logits [[buffer(0)]],
    device uint* all_ids [[buffer(1)]],
    device float* all_coefficients [[buffer(2)]],
    device float* all_shared_coefficients [[buffer(3)]],
    constant uint& top_n [[buffer(4)]],
    uint position [[threadgroup_position_in_grid]],
    uint expert [[thread_position_in_threadgroup]]) {
    static_assert((kMoeExperts & (kMoeExperts - 1u)) == 0u,
                  "parallel prefill router requires a power-of-two expert count");
    device const float* logits =
        all_logits + position * kMoeRouterRows;
    device uint* ids = all_ids + position * kMoeActiveExperts;
    device float* coefficients =
        all_coefficients + position * kMoeActiveExperts;
    threadgroup float values[kMoeExperts];
    threadgroup float reduction[kMoeExperts];
    threadgroup uint indices[kMoeExperts];
    reduction[expert] = logits[expert];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = kMoeExperts / 2u; offset; offset >>= 1u) {
        if (expert < offset) {
            reduction[expert] =
                max(reduction[expert], reduction[expert + offset]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float maximum = reduction[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    values[expert] = exp(logits[expert] - maximum);
    reduction[expert] = values[expert];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = kMoeExperts / 2u; offset; offset >>= 1u) {
        if (expert < offset) {
            reduction[expert] += reduction[expert + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    values[expert] /= reduction[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint selected = 0u; selected < top_n; ++selected) {
        reduction[expert] = values[expert];
        indices[expert] = expert;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = kMoeExperts / 2u; offset; offset >>= 1u) {
            if (expert < offset &&
                reduction[expert + offset] > reduction[expert]) {
                reduction[expert] = reduction[expert + offset];
                indices[expert] = indices[expert + offset];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (expert == 0u) {
            ids[selected] = indices[0];
            coefficients[selected] = reduction[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (expert == indices[0]) {
            values[expert] = -1.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (expert == 0u) {
        float selected_sum = 0.0f;
        for (uint selected = 0u; selected < top_n; ++selected) {
            selected_sum += coefficients[selected];
        }
        for (uint selected = 0u; selected < top_n; ++selected) {
            coefficients[selected] /= selected_sum;
        }
        for (uint selected = top_n; selected < kMoeActiveExperts;
             ++selected) {
            ids[selected] = 0xffffffffu;
            coefficients[selected] = 0.0f;
        }
        all_shared_coefficients[position] =
            1.0f / (1.0f + exp(-logits[kMoeExperts]));
    }
}

// Builds stable per-expert position lists and a compact active-expert vector.
// The two three-uint records are directly consumable by Metal indirect
// dispatch; no CPU readback is required between routing and expert kernels.
kernel void expert_union(
    device const uint* ids [[buffer(0)]],
    constant uint& block [[buffer(1)]],
    device uint* counts [[buffer(2)]],
    device uint* lists [[buffer(3)]],
    device uint* active [[buffer(4)]],
    device uint* upgate_arguments [[buffer(5)]],
    device uint* down_arguments [[buffer(6)]],
    constant uint& list_capacity [[buffer(7)]],
    constant uint& exact_rows_per_threadgroup [[buffer(8)]],
    uint expert [[thread_position_in_threadgroup]]) {
    if (expert < kMoeExperts) {
        uint count = 0u;
        for (uint position = 0u; position < block; ++position) {
            for (uint slot = 0u; slot < kMoeActiveExperts; ++slot) {
                if (ids[position * kMoeActiveExperts + slot] == expert) {
                    lists[expert * list_capacity + count++] =
                        (position << kPrefillPackedSlotBits) | slot;
                }
            }
        }
        counts[expert] = count;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (expert == 0u) {
        counts[kMoeExperts] = block;
        for (uint position = 0u; position < block; ++position) {
            lists[kMoeExperts * list_capacity + position] =
                (position << kPrefillPackedSlotBits) |
                kMoeActiveExperts;
        }
        uint active_count = 0u;
        for (uint index = 0u; index < kMoeExperts; ++index) {
            if (counts[index] != 0u) {
                active[active_count++] = index;
            }
        }
        active[active_count++] = kMoeExperts;
        for (uint index = active_count; index < kMoeRouterRows; ++index) {
            active[index] = 0xffffffffu;
        }
        upgate_arguments[0] = active_count;
        upgate_arguments[1] =
            (kMoeExpertDimension + exact_rows_per_threadgroup - 1u) /
            exact_rows_per_threadgroup;
        upgate_arguments[2] = 1u;
        down_arguments[0] = active_count;
        down_arguments[1] =
            (kHiddenDimension + exact_rows_per_threadgroup - 1u) /
            exact_rows_per_threadgroup;
        down_arguments[2] = 1u;
    }
}

kernel void block_upgate(
    device const bfloat* input [[buffer(0)]],
    device const uint* counts [[buffer(1)]],
    device const uint* lists [[buffer(2)]],
    device const uint* gate_words [[buffer(3)]],
    device const bfloat* gate_scales [[buffer(4)]],
    device const bfloat* gate_biases [[buffer(5)]],
    device const uint* up_words [[buffer(6)]],
    device const bfloat* up_scales [[buffer(7)]],
    device const bfloat* up_biases [[buffer(8)]],
    device bfloat* hidden_out [[buffer(9)]],
    device const uint* active [[buffer(10)]],
    device const uint* shared_gate_words [[buffer(11)]],
    device const bfloat* shared_gate_scales [[buffer(12)]],
    device const bfloat* shared_gate_biases [[buffer(13)]],
    device const uint* shared_up_words [[buffer(14)]],
    device const bfloat* shared_up_scales [[buffer(15)]],
    device const bfloat* shared_up_biases [[buffer(16)]],
    constant uint& list_capacity [[buffer(17)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint2 threads_per_group [[threads_per_threadgroup]]) {
    const uint expert = active[group.x];
    if (expert > kMoeExperts) {
        return;
    }
    const bool shared = expert == kMoeExperts;
    const uint count = counts[expert];
    const uint rows_per_group =
        threads_per_group.x / kSimdgroupWidth;
    const uint row = group.y * rows_per_group + simdgroup;
    if (row >= kMoeExpertDimension) {
        return;
    }
    const ulong word_row =
        shared ? ulong(row) *
                     ulong(kHiddenDimension / kQ4ValuesPerWord)
               : (ulong(expert) * kMoeExpertDimension + row) *
                     ulong(kHiddenDimension / kQ4ValuesPerWord);
    const ulong group_row =
        shared ? ulong(row) * ulong(kHiddenDimension / kQ4GroupSize)
               : (ulong(expert) * kMoeExpertDimension + row) *
                     ulong(kHiddenDimension / kQ4GroupSize);
    device const uint* gate_row =
        (shared ? shared_gate_words : gate_words) + word_row;
    device const uint* up_row =
        (shared ? shared_up_words : up_words) + word_row;
    device const bfloat* gate_scale_row =
        (shared ? shared_gate_scales : gate_scales) + group_row;
    device const bfloat* gate_bias_row =
        (shared ? shared_gate_biases : gate_biases) + group_row;
    device const bfloat* up_scale_row =
        (shared ? shared_up_scales : up_scales) + group_row;
    device const bfloat* up_bias_row =
        (shared ? shared_up_biases : up_biases) + group_row;
    for (uint first = 0u; first < count;
         first += kPrefillPositionBatch) {
        const uint position_count =
            min(count - first, kPrefillPositionBatch);
        float gate_accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float up_accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint k = 0u; k < kHiddenDimension; k += kQ4DotChunk) {
            const uint word_base =
                k / kQ4ValuesPerWord +
                lane * kQ4PackedWordsPerLane;
            uint gate_values[kQ4PackedWordsPerLane];
            uint up_values[kQ4PackedWordsPerLane];
            for (uint i = 0u; i < kQ4PackedWordsPerLane; ++i) {
                gate_values[i] = gate_row[word_base + i];
                up_values[i] = up_row[word_base + i];
            }
            const uint quant_group =
                (k >> 6u) + (lane >> 2u);
            const float gate_scale =
                float(gate_scale_row[quant_group]);
            const float gate_bias =
                float(gate_bias_row[quant_group]);
            const float up_scale = float(up_scale_row[quant_group]);
            const float up_bias = float(up_bias_row[quant_group]);
            const uint input_base = k + lane * kQ4LaneValues;
            for (uint item = 0u; item < position_count; ++item) {
                const uint packed =
                    lists[expert * list_capacity + first + item];
                device const bfloat* row_input =
                    input +
                    (packed >> kPrefillPackedSlotBits) *
                        kHiddenDimension;
                float sum = 0.0f;
                float gate_quantized = 0.0f;
                float up_quantized = 0.0f;
                for (uint word_index = 0u;
                     word_index < kQ4PackedWordsPerLane;
                     ++word_index) {
                    const uint offset =
                        word_index * kQ4ValuesPerWord;
                    const float4 even =
                        float4(float(row_input[input_base + offset]),
                               float(row_input[input_base + offset + 2u]),
                               float(row_input[input_base + offset + 4u]),
                               float(row_input[input_base + offset + 6u]));
                    const float4 odd =
                        float4(float(row_input[input_base + offset + 1u]),
                               float(row_input[input_base + offset + 3u]),
                               float(row_input[input_base + offset + 5u]),
                               float(row_input[input_base + offset + 7u]));
                    const float4 gate_even =
                        unpack_unorm4x8_to_float(
                            gate_values[word_index] & 0x0f0f0f0fu) *
                        255.0f;
                    const float4 gate_odd =
                        unpack_unorm4x8_to_float(
                            (gate_values[word_index] >> 4u) &
                            0x0f0f0f0fu) *
                        255.0f;
                    const float4 up_even =
                        unpack_unorm4x8_to_float(
                            up_values[word_index] & 0x0f0f0f0fu) *
                        255.0f;
                    const float4 up_odd =
                        unpack_unorm4x8_to_float(
                            (up_values[word_index] >> 4u) &
                            0x0f0f0f0fu) *
                        255.0f;
                    sum += dot(even + odd, float4(1.0f));
                    gate_quantized +=
                        dot(even, gate_even) + dot(odd, gate_odd);
                    up_quantized +=
                        dot(even, up_even) + dot(odd, up_odd);
                }
                gate_accumulators[item] +=
                    gate_scale * gate_quantized + sum * gate_bias;
                up_accumulators[item] +=
                    up_scale * up_quantized + sum * up_bias;
            }
        }
        for (uint item = 0u; item < position_count; ++item) {
            const float gate_value =
                simd_sum(gate_accumulators[item]);
            const float up_value = simd_sum(up_accumulators[item]);
            if (lane == 0u) {
                const uint packed =
                    lists[expert * list_capacity + first + item];
                const uint position =
                    packed >> kPrefillPackedSlotBits;
                const uint slot =
                    packed & ((1u << kPrefillPackedSlotBits) - 1u);
                hidden_out[(position * kMoeSlotCount + slot) *
                               kMoeExpertDimension +
                           row] =
                    static_cast<bfloat>(
                        (gate_value / (1.0f + exp(-gate_value))) *
                        up_value);
            }
        }
    }
}

kernel void block_down_partial(
    device const bfloat* hidden [[buffer(0)]],
    device const uint* counts [[buffer(1)]],
    device const uint* lists [[buffer(2)]],
    device const uint* down_words [[buffer(3)]],
    device const bfloat* down_scales [[buffer(4)]],
    device const bfloat* down_biases [[buffer(5)]],
    device float* partials [[buffer(6)]],
    device const uint* active [[buffer(7)]],
    device const uint* shared_down_words [[buffer(8)]],
    device const bfloat* shared_down_scales [[buffer(9)]],
    device const bfloat* shared_down_biases [[buffer(10)]],
    constant uint& list_capacity [[buffer(11)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint2 threads_per_group [[threads_per_threadgroup]]) {
    const uint expert = active[group.x];
    if (expert > kMoeExperts) {
        return;
    }
    const bool shared = expert == kMoeExperts;
    const uint count = counts[expert];
    const uint rows_per_group =
        threads_per_group.x / kSimdgroupWidth;
    const uint row = group.y * rows_per_group + simdgroup;
    if (row >= kHiddenDimension) {
        return;
    }
    const ulong word_row =
        shared ? ulong(row) *
                     ulong(kMoeExpertDimension / kQ4ValuesPerWord)
               : (ulong(expert) * kHiddenDimension + row) *
                     ulong(kMoeExpertDimension / kQ4ValuesPerWord);
    const ulong group_row =
        shared ? ulong(row) *
                     ulong(kMoeExpertDimension / kQ4GroupSize)
               : (ulong(expert) * kHiddenDimension + row) *
                     ulong(kMoeExpertDimension / kQ4GroupSize);
    device const uint* word_row_pointer =
        (shared ? shared_down_words : down_words) + word_row;
    device const bfloat* scale_row =
        (shared ? shared_down_scales : down_scales) + group_row;
    device const bfloat* bias_row =
        (shared ? shared_down_biases : down_biases) + group_row;
    for (uint first = 0u; first < count;
         first += kPrefillPositionBatch) {
        const uint position_count =
            min(count - first, kPrefillPositionBatch);
        float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint k = 0u; k < kMoeExpertDimension;
             k += kQ4DotChunk) {
            const uint word_base =
                k / kQ4ValuesPerWord +
                lane * kQ4PackedWordsPerLane;
            uint word_values[kQ4PackedWordsPerLane];
            for (uint i = 0u; i < kQ4PackedWordsPerLane; ++i) {
                word_values[i] = word_row_pointer[word_base + i];
            }
            const uint quant_group =
                (k >> 6u) + (lane >> 2u);
            const float scale = float(scale_row[quant_group]);
            const float bias = float(bias_row[quant_group]);
            const uint hidden_base = k + lane * kQ4LaneValues;
            for (uint item = 0u; item < position_count; ++item) {
                const uint packed =
                    lists[expert * list_capacity + first + item];
                const uint position =
                    packed >> kPrefillPackedSlotBits;
                const uint slot =
                    packed & ((1u << kPrefillPackedSlotBits) - 1u);
                device const bfloat* hidden_row =
                    hidden +
                    (position * kMoeSlotCount + slot) *
                        kMoeExpertDimension;
                float sum = 0.0f;
                float quantized = 0.0f;
                for (uint word_index = 0u;
                     word_index < kQ4PackedWordsPerLane;
                     ++word_index) {
                    const uint offset =
                        word_index * kQ4ValuesPerWord;
                    const float4 even =
                        float4(float(hidden_row[hidden_base + offset]),
                               float(hidden_row[hidden_base + offset + 2u]),
                               float(hidden_row[hidden_base + offset + 4u]),
                               float(hidden_row[hidden_base + offset + 6u]));
                    const float4 odd =
                        float4(float(hidden_row[hidden_base + offset + 1u]),
                               float(hidden_row[hidden_base + offset + 3u]),
                               float(hidden_row[hidden_base + offset + 5u]),
                               float(hidden_row[hidden_base + offset + 7u]));
                    const float4 quantized_even =
                        unpack_unorm4x8_to_float(
                            word_values[word_index] & 0x0f0f0f0fu) *
                        255.0f;
                    const float4 quantized_odd =
                        unpack_unorm4x8_to_float(
                            (word_values[word_index] >> 4u) &
                            0x0f0f0f0fu) *
                        255.0f;
                    sum += dot(even + odd, float4(1.0f));
                    quantized +=
                        dot(even, quantized_even) +
                        dot(odd, quantized_odd);
                }
                accumulators[item] += scale * quantized + sum * bias;
            }
        }
        for (uint item = 0u; item < position_count; ++item) {
            const float value = simd_sum(accumulators[item]);
            if (lane == 0u) {
                const uint packed =
                    lists[expert * list_capacity + first + item];
                const uint position =
                    packed >> kPrefillPackedSlotBits;
                const uint slot =
                    packed & ((1u << kPrefillPackedSlotBits) - 1u);
                partials[(position * kMoeSlotCount + slot) *
                             kHiddenDimension +
                         row] = value;
            }
        }
    }
}

kernel void block_down_combine(
    device const float* partials [[buffer(0)]],
    device const float* coefficients [[buffer(1)]],
    device const float* shared_coefficients [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    uint2 element [[thread_position_in_grid]]) {
    const uint row = element.x;
    const uint position = element.y;
    float total = 0.0f;
    for (uint slot = 0u; slot < kMoeActiveExperts; ++slot) {
        total +=
            coefficients[position * kMoeActiveExperts + slot] *
            partials[(position * kMoeSlotCount + slot) *
                         kHiddenDimension +
                     row];
    }
    total +=
        shared_coefficients[position] *
        partials[(position * kMoeSlotCount + kMoeActiveExperts) *
                     kHiddenDimension +
                 row];
    output[position * kHiddenDimension + row] =
        static_cast<bfloat>(total);
}

#include <metal_simdgroup_matrix>

struct NativeRoutedQgemmR1Task {
    uint expert_index;
    uint absolute_route_list_begin;
    uint row_count;
    uint output_row_begin;
};

static_assert(
    kNativeRoutedQgemmR1TileRows == 16u,
    "R1 requires a 16-row routed tile");
static_assert(
    kNativeRoutedQgemmR1TileColumns == 32u,
    "R1 requires a 32-column output tile");
static_assert(
    kNativeRoutedQgemmR1ReductionColumns == 32u,
    "R1 requires a 32-column reduction tile");
static_assert(
    kNativeRoutedQgemmR1Simdgroups == 2u &&
        kNativeRoutedQgemmR1Threads ==
            kNativeRoutedQgemmR1Simdgroups * kSimdgroupWidth,
    "R1 requires two complete simdgroups");
static_assert(
    kNativeRoutedQgemmR1TaskBytes == 16u &&
        sizeof(NativeRoutedQgemmR1Task) ==
            kNativeRoutedQgemmR1TaskBytes,
    "R1 task ABI must remain four packed uint fields");
static_assert(
    kNativeRoutedQgemmR1TaskCapacity == 4096u,
    "R1 task storage has a frozen bounded capacity");
static_assert(
    kNativeRoutedQgemmR1FusedAccumulatorElements ==
        2u * kNativeRoutedQgemmR1TileRows *
            kNativeRoutedQgemmR1TileColumns,
    "R1 fused accumulator accounting must cover gate and up");
static_assert(
    kNativeRoutedQgemmR1SingleAccumulatorElements ==
        kNativeRoutedQgemmR1TileRows *
            kNativeRoutedQgemmR1TileColumns,
    "R1 single accumulator accounting must cover one projection");
static_assert(
    kNativeRoutedQgemmR1AccumulatorElements ==
        kNativeRoutedQgemmR1FusedAccumulatorElements,
    "R1 compatibility accounting must name the maximum treatment");
static_assert(
    kNativeRoutedQgemmR1ThreadgroupMemoryBytes == 0u,
    "R1 register-direct kernels may not stage group-local memory");
static_assert(
    kNativeRoutedQgemmR1TileRows % 8u == 0u &&
        kNativeRoutedQgemmR1TileColumns % 8u == 0u &&
        kNativeRoutedQgemmR1ReductionColumns % 8u == 0u,
    "R1 tiles must contain complete 8x8 matrix fragments");
static_assert(
    kQ4GroupSize % kNativeRoutedQgemmR1ReductionColumns == 0u,
    "an R1 reduction tile may not split an affine group");
static_assert(
    kHiddenDimension % kNativeRoutedQgemmR1ReductionColumns == 0u &&
        kMoeExpertDimension %
                kNativeRoutedQgemmR1ReductionColumns ==
            0u,
    "R1 model dimensions must contain complete reduction tiles");
static_assert(
    kMoeSlotCount == kMoeActiveExperts + 1u &&
        kMoeSlotCount <= (1u << kPrefillPackedSlotBits),
    "R1 padded slot rows must cover every packed route slot");

inline uint2 native_routed_qgemm_r1_fragment_coordinate(uint lane) {
    const uint quarter = lane >> 2u;
    const uint row =
        (quarter & 4u) + ((lane >> 1u) & 3u);
    const uint column =
        (quarter & 2u) * 2u + (lane & 1u) * 2u;
    return uint2(row, column);
}

inline bool native_routed_qgemm_r1_dispatch_valid(
    uint task_count,
    uint3 group,
    uint simdgroup,
    uint simdgroup_width,
    uint3 threadgroup_shape) {
    return task_count <= kNativeRoutedQgemmR1TaskCapacity &&
           group.x < task_count &&
           group.z == 0u &&
           simdgroup < kNativeRoutedQgemmR1Simdgroups &&
           simdgroup_width == kSimdgroupWidth &&
           threadgroup_shape.x == kNativeRoutedQgemmR1Threads &&
           threadgroup_shape.y == 1u &&
           threadgroup_shape.z == 1u;
}

inline bool native_routed_qgemm_r1_task_valid(
    NativeRoutedQgemmR1Task task,
    uint route_list_expert_stride,
    uint route_list_capacity_per_expert,
    ulong route_list_total_extent) {
    if (task.expert_index >= kMoeExperts ||
        task.row_count == 0u ||
        task.row_count > kNativeRoutedQgemmR1TileRows ||
        task.output_row_begin != 0u ||
        route_list_capacity_per_expert >
            route_list_expert_stride) {
        return false;
    }
    const ulong expert_segment_begin =
        ulong(task.expert_index) *
        ulong(route_list_expert_stride);
    const ulong expert_segment_end =
        expert_segment_begin +
        ulong(route_list_capacity_per_expert);
    const ulong task_begin =
        ulong(task.absolute_route_list_begin);
    const ulong task_rows = ulong(task.row_count);
    return expert_segment_begin <= route_list_total_extent &&
           expert_segment_end <= route_list_total_extent &&
           task_begin >= expert_segment_begin &&
           task_begin <= expert_segment_end &&
           task_rows <= expert_segment_end - task_begin &&
           task_begin <= route_list_total_extent &&
           task_rows <= route_list_total_extent - task_begin;
}

inline bool native_routed_qgemm_r1_route(
    device const uint* route_list,
    NativeRoutedQgemmR1Task task,
    uint local_row,
    uint input_rows,
    thread uint& position,
    thread uint& slot) {
    if (local_row >= task.row_count) {
        return false;
    }
    const uint packed =
        route_list[
            ulong(task.absolute_route_list_begin) +
            ulong(local_row)];
    position = packed >> kPrefillPackedSlotBits;
    slot =
        packed & ((1u << kPrefillPackedSlotBits) - 1u);
    return position < input_rows &&
           slot < kMoeActiveExperts;
}

inline void native_routed_qgemm_r1_zero(
    thread simdgroup_matrix<float, 8, 8>& matrix) {
    matrix.thread_elements()[0] = 0.0f;
    matrix.thread_elements()[1] = 0.0f;
}

inline void native_routed_qgemm_r1_load_weight(
    thread simdgroup_matrix<float, 8, 8>& matrix,
    device const uint* packed_weights,
    device const bfloat* scales,
    device const bfloat* biases,
    ulong expert_row_begin,
    uint output_columns,
    uint reduction_columns,
    uint output_column_begin,
    uint reduction_column_begin,
    uint lane) {
    const uint2 coordinate =
        native_routed_qgemm_r1_fragment_coordinate(lane);
    const uint output_column =
        output_column_begin + coordinate.x;
    const uint reduction_column =
        reduction_column_begin + coordinate.y;
    if (output_column >= output_columns ||
        reduction_column + 1u >= reduction_columns) {
        native_routed_qgemm_r1_zero(matrix);
        return;
    }
    const ulong matrix_row =
        expert_row_begin + ulong(output_column);
    const ulong packed_words_per_row =
        ulong(reduction_columns / kQ4ValuesPerWord);
    const ulong parameters_per_row =
        ulong(reduction_columns / kQ4GroupSize);
    const ulong packed_index =
        matrix_row * packed_words_per_row +
        ulong(reduction_column / kQ4ValuesPerWord);
    const ulong parameter_index =
        matrix_row * parameters_per_row +
        ulong(reduction_column / kQ4GroupSize);
    const uint word = packed_weights[packed_index];
    const float scale = float(scales[parameter_index]);
    const float bias = float(biases[parameter_index]);
    const uint nibble = reduction_column % kQ4ValuesPerWord;
    matrix.thread_elements()[0] =
        float((word >> (4u * nibble)) & 15u) * scale + bias;
    matrix.thread_elements()[1] =
        float((word >> (4u * (nibble + 1u))) & 15u) *
            scale +
        bias;
}

inline void native_routed_qgemm_r1_load_position_rows(
    thread simdgroup_matrix<float, 8, 8>& matrix,
    device const bfloat* input,
    device const uint* route_list,
    NativeRoutedQgemmR1Task task,
    uint input_rows,
    uint local_row_begin,
    uint reduction_column_begin,
    uint lane) {
    const uint2 coordinate =
        native_routed_qgemm_r1_fragment_coordinate(lane);
    const uint reduction_column =
        reduction_column_begin + coordinate.x;
    const uint local_row =
        local_row_begin + coordinate.y;
    for (uint element = 0u; element < 2u; ++element) {
        uint position = 0u;
        uint slot = 0u;
        const bool valid =
            reduction_column < kHiddenDimension &&
            native_routed_qgemm_r1_route(
                route_list,
                task,
                local_row + element,
                input_rows,
                position,
                slot);
        matrix.thread_elements()[element] =
            valid
                ? float(input[
                      ulong(position) * ulong(kHiddenDimension) +
                      ulong(reduction_column)])
                : 0.0f;
    }
}

inline void native_routed_qgemm_r1_load_padded_slot_rows(
    thread simdgroup_matrix<float, 8, 8>& matrix,
    device const bfloat* hidden,
    device const uint* route_list,
    NativeRoutedQgemmR1Task task,
    uint input_rows,
    uint local_row_begin,
    uint reduction_column_begin,
    uint lane) {
    const uint2 coordinate =
        native_routed_qgemm_r1_fragment_coordinate(lane);
    const uint reduction_column =
        reduction_column_begin + coordinate.x;
    const uint local_row =
        local_row_begin + coordinate.y;
    for (uint element = 0u; element < 2u; ++element) {
        uint position = 0u;
        uint slot = 0u;
        const bool valid =
            reduction_column < kMoeExpertDimension &&
            native_routed_qgemm_r1_route(
                route_list,
                task,
                local_row + element,
                input_rows,
                position,
                slot);
        matrix.thread_elements()[element] =
            valid
                ? float(hidden[
                      (ulong(position) * ulong(kMoeSlotCount) +
                       ulong(slot)) *
                              ulong(kMoeExpertDimension) +
                      ulong(reduction_column)])
                : 0.0f;
    }
}

inline float native_routed_qgemm_r1_value(
    thread simdgroup_matrix<float, 8, 8>* accumulators,
    uint output_fragment,
    uint route_fragment,
    uint element) {
    return accumulators[
        output_fragment * 2u + route_fragment]
        .thread_elements()[element];
}

inline ulong native_routed_qgemm_r1_hidden_index(
    uint position,
    uint slot,
    uint expert_column) {
    return
        (ulong(position) * ulong(kMoeSlotCount) + ulong(slot)) *
            ulong(kMoeExpertDimension) +
        ulong(expert_column);
}

inline ulong native_routed_qgemm_r1_partial_index(
    uint position,
    uint slot,
    uint hidden_column) {
    return
        (ulong(position) * ulong(kMoeSlotCount) + ulong(slot)) *
            ulong(kHiddenDimension) +
        ulong(hidden_column);
}

kernel void native_routed_qgemm_r1_fused_upgate_swiglu(
    device const bfloat* input [[buffer(0)]],
    device const uint* route_list [[buffer(1)]],
    device const NativeRoutedQgemmR1Task* tasks [[buffer(2)]],
    device const uint* gate_words [[buffer(3)]],
    device const bfloat* gate_scales [[buffer(4)]],
    device const bfloat* gate_biases [[buffer(5)]],
    device const uint* up_words [[buffer(6)]],
    device const bfloat* up_scales [[buffer(7)]],
    device const bfloat* up_biases [[buffer(8)]],
    device bfloat* hidden [[buffer(9)]],
    constant uint& task_count [[buffer(10)]],
    constant uint& route_list_expert_stride [[buffer(11)]],
    constant uint& route_list_capacity_per_expert [[buffer(12)]],
    constant ulong& route_list_total_extent [[buffer(13)]],
    constant uint& input_rows [[buffer(14)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroup_width [[threads_per_simdgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    if (!native_routed_qgemm_r1_dispatch_valid(
            task_count,
            group,
            simdgroup,
            simdgroup_width,
            threadgroup_shape)) {
        return;
    }
    const NativeRoutedQgemmR1Task task = tasks[group.x];
    if (!native_routed_qgemm_r1_task_valid(
            task,
            route_list_expert_stride,
            route_list_capacity_per_expert,
            route_list_total_extent)) {
        return;
    }
    if (group.y >=
        (kMoeExpertDimension +
         kNativeRoutedQgemmR1TileColumns - 1u) /
            kNativeRoutedQgemmR1TileColumns) {
        return;
    }
    const ulong expert_row_begin =
        ulong(task.expert_index) *
        ulong(kMoeExpertDimension);
    const uint output_column_begin =
        group.y * kNativeRoutedQgemmR1TileColumns +
        simdgroup *
            (kNativeRoutedQgemmR1TileColumns /
             kNativeRoutedQgemmR1Simdgroups);

    simdgroup_matrix<float, 8, 8> gate_accumulators[4];
    simdgroup_matrix<float, 8, 8> up_accumulators[4];
    for (uint index = 0u; index < 4u; ++index) {
        native_routed_qgemm_r1_zero(gate_accumulators[index]);
        native_routed_qgemm_r1_zero(up_accumulators[index]);
    }
    for (uint reduction_tile = 0u;
         reduction_tile < kHiddenDimension;
         reduction_tile +=
             kNativeRoutedQgemmR1ReductionColumns) {
        for (uint reduction_fragment = 0u;
             reduction_fragment <
             kNativeRoutedQgemmR1ReductionColumns;
             reduction_fragment += 8u) {
            simdgroup_matrix<float, 8, 8> gate_a0;
            simdgroup_matrix<float, 8, 8> gate_a1;
            simdgroup_matrix<float, 8, 8> up_a0;
            simdgroup_matrix<float, 8, 8> up_a1;
            simdgroup_matrix<float, 8, 8> input_b0;
            simdgroup_matrix<float, 8, 8> input_b1;
            native_routed_qgemm_r1_load_weight(
                gate_a0, gate_words,
                gate_scales, gate_biases,
                expert_row_begin, kMoeExpertDimension,
                kHiddenDimension, output_column_begin,
                reduction_tile + reduction_fragment, lane);
            native_routed_qgemm_r1_load_weight(
                gate_a1, gate_words,
                gate_scales, gate_biases,
                expert_row_begin, kMoeExpertDimension,
                kHiddenDimension, output_column_begin + 8u,
                reduction_tile + reduction_fragment, lane);
            native_routed_qgemm_r1_load_weight(
                up_a0, up_words,
                up_scales, up_biases,
                expert_row_begin, kMoeExpertDimension,
                kHiddenDimension, output_column_begin,
                reduction_tile + reduction_fragment, lane);
            native_routed_qgemm_r1_load_weight(
                up_a1, up_words,
                up_scales, up_biases,
                expert_row_begin, kMoeExpertDimension,
                kHiddenDimension, output_column_begin + 8u,
                reduction_tile + reduction_fragment, lane);
            native_routed_qgemm_r1_load_position_rows(
                input_b0, input, route_list, task, input_rows,
                0u, reduction_tile + reduction_fragment, lane);
            native_routed_qgemm_r1_load_position_rows(
                input_b1, input, route_list, task, input_rows,
                8u, reduction_tile + reduction_fragment, lane);
            simdgroup_multiply_accumulate(
                gate_accumulators[0], gate_a0, input_b0,
                gate_accumulators[0]);
            simdgroup_multiply_accumulate(
                gate_accumulators[1], gate_a0, input_b1,
                gate_accumulators[1]);
            simdgroup_multiply_accumulate(
                gate_accumulators[2], gate_a1, input_b0,
                gate_accumulators[2]);
            simdgroup_multiply_accumulate(
                gate_accumulators[3], gate_a1, input_b1,
                gate_accumulators[3]);
            simdgroup_multiply_accumulate(
                up_accumulators[0], up_a0, input_b0,
                up_accumulators[0]);
            simdgroup_multiply_accumulate(
                up_accumulators[1], up_a0, input_b1,
                up_accumulators[1]);
            simdgroup_multiply_accumulate(
                up_accumulators[2], up_a1, input_b0,
                up_accumulators[2]);
            simdgroup_multiply_accumulate(
                up_accumulators[3], up_a1, input_b1,
                up_accumulators[3]);
        }
    }

    const uint2 coordinate =
        native_routed_qgemm_r1_fragment_coordinate(lane);
    for (uint output_fragment = 0u;
         output_fragment < 2u;
         ++output_fragment) {
        const uint output_column =
            output_column_begin + output_fragment * 8u +
            coordinate.x;
        for (uint route_fragment = 0u;
             route_fragment < 2u;
             ++route_fragment) {
            const uint local_row =
                route_fragment * 8u + coordinate.y;
            for (uint element = 0u; element < 2u; ++element) {
                uint position = 0u;
                uint slot = 0u;
                if (output_column >= kMoeExpertDimension ||
                    !native_routed_qgemm_r1_route(
                        route_list, task, local_row + element,
                        input_rows, position, slot)) {
                    continue;
                }
                const float gate =
                    native_routed_qgemm_r1_value(
                        gate_accumulators, output_fragment,
                        route_fragment, element);
                const float up =
                    native_routed_qgemm_r1_value(
                        up_accumulators, output_fragment,
                        route_fragment, element);
                hidden[native_routed_qgemm_r1_hidden_index(
                    position, slot, output_column)] =
                    static_cast<bfloat>(
                        (gate / (1.0f + exp(-gate))) * up);
            }
        }
    }
}

kernel void native_routed_qgemm_r1_gate(
    device const bfloat* input [[buffer(0)]],
    device const uint* route_list [[buffer(1)]],
    device const NativeRoutedQgemmR1Task* tasks [[buffer(2)]],
    device const uint* gate_words [[buffer(3)]],
    device const bfloat* gate_scales [[buffer(4)]],
    device const bfloat* gate_biases [[buffer(5)]],
    device bfloat* hidden [[buffer(6)]],
    constant uint& task_count [[buffer(7)]],
    constant uint& route_list_expert_stride [[buffer(8)]],
    constant uint& route_list_capacity_per_expert [[buffer(9)]],
    constant ulong& route_list_total_extent [[buffer(10)]],
    constant uint& input_rows [[buffer(11)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroup_width [[threads_per_simdgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    if (!native_routed_qgemm_r1_dispatch_valid(
            task_count,
            group,
            simdgroup,
            simdgroup_width,
            threadgroup_shape)) {
        return;
    }
    const NativeRoutedQgemmR1Task task = tasks[group.x];
    if (!native_routed_qgemm_r1_task_valid(
            task,
            route_list_expert_stride,
            route_list_capacity_per_expert,
            route_list_total_extent)) {
        return;
    }
    if (group.y >=
        (kMoeExpertDimension +
         kNativeRoutedQgemmR1TileColumns - 1u) /
            kNativeRoutedQgemmR1TileColumns) {
        return;
    }
    const ulong expert_row_begin =
        ulong(task.expert_index) *
        ulong(kMoeExpertDimension);
    const uint output_column_begin =
        group.y * kNativeRoutedQgemmR1TileColumns +
        simdgroup *
            (kNativeRoutedQgemmR1TileColumns /
             kNativeRoutedQgemmR1Simdgroups);

    simdgroup_matrix<float, 8, 8> accumulators[4];
    for (uint index = 0u; index < 4u; ++index) {
        native_routed_qgemm_r1_zero(accumulators[index]);
    }
    for (uint reduction_tile = 0u;
         reduction_tile < kHiddenDimension;
         reduction_tile +=
             kNativeRoutedQgemmR1ReductionColumns) {
        for (uint reduction_fragment = 0u;
             reduction_fragment <
             kNativeRoutedQgemmR1ReductionColumns;
             reduction_fragment += 8u) {
            simdgroup_matrix<float, 8, 8> weight_a0;
            simdgroup_matrix<float, 8, 8> weight_a1;
            simdgroup_matrix<float, 8, 8> input_b0;
            simdgroup_matrix<float, 8, 8> input_b1;
            native_routed_qgemm_r1_load_weight(
                weight_a0, gate_words, gate_scales,
                gate_biases, expert_row_begin,
                kMoeExpertDimension, kHiddenDimension,
                output_column_begin,
                reduction_tile + reduction_fragment, lane);
            native_routed_qgemm_r1_load_weight(
                weight_a1, gate_words, gate_scales,
                gate_biases, expert_row_begin,
                kMoeExpertDimension, kHiddenDimension,
                output_column_begin + 8u,
                reduction_tile + reduction_fragment, lane);
            native_routed_qgemm_r1_load_position_rows(
                input_b0, input, route_list, task, input_rows,
                0u, reduction_tile + reduction_fragment, lane);
            native_routed_qgemm_r1_load_position_rows(
                input_b1, input, route_list, task, input_rows,
                8u, reduction_tile + reduction_fragment, lane);
            simdgroup_multiply_accumulate(
                accumulators[0], weight_a0, input_b0,
                accumulators[0]);
            simdgroup_multiply_accumulate(
                accumulators[1], weight_a0, input_b1,
                accumulators[1]);
            simdgroup_multiply_accumulate(
                accumulators[2], weight_a1, input_b0,
                accumulators[2]);
            simdgroup_multiply_accumulate(
                accumulators[3], weight_a1, input_b1,
                accumulators[3]);
        }
    }

    const uint2 coordinate =
        native_routed_qgemm_r1_fragment_coordinate(lane);
    for (uint output_fragment = 0u;
         output_fragment < 2u;
         ++output_fragment) {
        const uint output_column =
            output_column_begin + output_fragment * 8u +
            coordinate.x;
        for (uint route_fragment = 0u;
             route_fragment < 2u;
             ++route_fragment) {
            const uint local_row =
                route_fragment * 8u + coordinate.y;
            for (uint element = 0u; element < 2u; ++element) {
                uint position = 0u;
                uint slot = 0u;
                if (output_column >= kMoeExpertDimension ||
                    !native_routed_qgemm_r1_route(
                        route_list, task, local_row + element,
                        input_rows, position, slot)) {
                    continue;
                }
                hidden[native_routed_qgemm_r1_hidden_index(
                    position, slot, output_column)] =
                    static_cast<bfloat>(
                        native_routed_qgemm_r1_value(
                            accumulators, output_fragment,
                            route_fragment, element));
            }
        }
    }
}

kernel void native_routed_qgemm_r1_up_swiglu(
    device const bfloat* input [[buffer(0)]],
    device const uint* route_list [[buffer(1)]],
    device const NativeRoutedQgemmR1Task* tasks [[buffer(2)]],
    device const uint* up_words [[buffer(3)]],
    device const bfloat* up_scales [[buffer(4)]],
    device const bfloat* up_biases [[buffer(5)]],
    device bfloat* hidden [[buffer(6)]],
    constant uint& task_count [[buffer(7)]],
    constant uint& route_list_expert_stride [[buffer(8)]],
    constant uint& route_list_capacity_per_expert [[buffer(9)]],
    constant ulong& route_list_total_extent [[buffer(10)]],
    constant uint& input_rows [[buffer(11)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroup_width [[threads_per_simdgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    if (!native_routed_qgemm_r1_dispatch_valid(
            task_count,
            group,
            simdgroup,
            simdgroup_width,
            threadgroup_shape)) {
        return;
    }
    const NativeRoutedQgemmR1Task task = tasks[group.x];
    if (!native_routed_qgemm_r1_task_valid(
            task,
            route_list_expert_stride,
            route_list_capacity_per_expert,
            route_list_total_extent)) {
        return;
    }
    if (group.y >=
        (kMoeExpertDimension +
         kNativeRoutedQgemmR1TileColumns - 1u) /
            kNativeRoutedQgemmR1TileColumns) {
        return;
    }
    const ulong expert_row_begin =
        ulong(task.expert_index) *
        ulong(kMoeExpertDimension);
    const uint output_column_begin =
        group.y * kNativeRoutedQgemmR1TileColumns +
        simdgroup *
            (kNativeRoutedQgemmR1TileColumns /
             kNativeRoutedQgemmR1Simdgroups);

    simdgroup_matrix<float, 8, 8> accumulators[4];
    for (uint index = 0u; index < 4u; ++index) {
        native_routed_qgemm_r1_zero(accumulators[index]);
    }
    for (uint reduction_tile = 0u;
         reduction_tile < kHiddenDimension;
         reduction_tile +=
             kNativeRoutedQgemmR1ReductionColumns) {
        for (uint reduction_fragment = 0u;
             reduction_fragment <
             kNativeRoutedQgemmR1ReductionColumns;
             reduction_fragment += 8u) {
            simdgroup_matrix<float, 8, 8> weight_a0;
            simdgroup_matrix<float, 8, 8> weight_a1;
            simdgroup_matrix<float, 8, 8> input_b0;
            simdgroup_matrix<float, 8, 8> input_b1;
            native_routed_qgemm_r1_load_weight(
                weight_a0, up_words, up_scales,
                up_biases, expert_row_begin,
                kMoeExpertDimension, kHiddenDimension,
                output_column_begin,
                reduction_tile + reduction_fragment, lane);
            native_routed_qgemm_r1_load_weight(
                weight_a1, up_words, up_scales,
                up_biases, expert_row_begin,
                kMoeExpertDimension, kHiddenDimension,
                output_column_begin + 8u,
                reduction_tile + reduction_fragment, lane);
            native_routed_qgemm_r1_load_position_rows(
                input_b0, input, route_list, task, input_rows,
                0u, reduction_tile + reduction_fragment, lane);
            native_routed_qgemm_r1_load_position_rows(
                input_b1, input, route_list, task, input_rows,
                8u, reduction_tile + reduction_fragment, lane);
            simdgroup_multiply_accumulate(
                accumulators[0], weight_a0, input_b0,
                accumulators[0]);
            simdgroup_multiply_accumulate(
                accumulators[1], weight_a0, input_b1,
                accumulators[1]);
            simdgroup_multiply_accumulate(
                accumulators[2], weight_a1, input_b0,
                accumulators[2]);
            simdgroup_multiply_accumulate(
                accumulators[3], weight_a1, input_b1,
                accumulators[3]);
        }
    }

    const uint2 coordinate =
        native_routed_qgemm_r1_fragment_coordinate(lane);
    for (uint output_fragment = 0u;
         output_fragment < 2u;
         ++output_fragment) {
        const uint output_column =
            output_column_begin + output_fragment * 8u +
            coordinate.x;
        for (uint route_fragment = 0u;
             route_fragment < 2u;
             ++route_fragment) {
            const uint local_row =
                route_fragment * 8u + coordinate.y;
            for (uint element = 0u; element < 2u; ++element) {
                uint position = 0u;
                uint slot = 0u;
                if (output_column >= kMoeExpertDimension ||
                    !native_routed_qgemm_r1_route(
                        route_list, task, local_row + element,
                        input_rows, position, slot)) {
                    continue;
                }
                const ulong hidden_index =
                    native_routed_qgemm_r1_hidden_index(
                        position, slot, output_column);
                const float gate = float(hidden[hidden_index]);
                const float up =
                    native_routed_qgemm_r1_value(
                        accumulators, output_fragment,
                        route_fragment, element);
                hidden[hidden_index] =
                    static_cast<bfloat>(
                        (gate / (1.0f + exp(-gate))) * up);
            }
        }
    }
}

kernel void native_routed_qgemm_r1_down_partial(
    device const bfloat* hidden [[buffer(0)]],
    device const uint* route_list [[buffer(1)]],
    device const NativeRoutedQgemmR1Task* tasks [[buffer(2)]],
    device const uint* down_words [[buffer(3)]],
    device const bfloat* down_scales [[buffer(4)]],
    device const bfloat* down_biases [[buffer(5)]],
    device float* partials [[buffer(6)]],
    constant uint& task_count [[buffer(7)]],
    constant uint& route_list_expert_stride [[buffer(8)]],
    constant uint& route_list_capacity_per_expert [[buffer(9)]],
    constant ulong& route_list_total_extent [[buffer(10)]],
    constant uint& input_rows [[buffer(11)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroup_width [[threads_per_simdgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    if (!native_routed_qgemm_r1_dispatch_valid(
            task_count,
            group,
            simdgroup,
            simdgroup_width,
            threadgroup_shape)) {
        return;
    }
    const NativeRoutedQgemmR1Task task = tasks[group.x];
    if (!native_routed_qgemm_r1_task_valid(
            task,
            route_list_expert_stride,
            route_list_capacity_per_expert,
            route_list_total_extent)) {
        return;
    }
    if (group.y >=
        (kHiddenDimension +
         kNativeRoutedQgemmR1TileColumns - 1u) /
            kNativeRoutedQgemmR1TileColumns) {
        return;
    }
    const ulong expert_row_begin =
        ulong(task.expert_index) *
        ulong(kHiddenDimension);
    const uint output_column_begin =
        group.y * kNativeRoutedQgemmR1TileColumns +
        simdgroup *
            (kNativeRoutedQgemmR1TileColumns /
             kNativeRoutedQgemmR1Simdgroups);

    simdgroup_matrix<float, 8, 8> accumulators[4];
    for (uint index = 0u; index < 4u; ++index) {
        native_routed_qgemm_r1_zero(accumulators[index]);
    }
    for (uint reduction_tile = 0u;
         reduction_tile < kMoeExpertDimension;
         reduction_tile +=
             kNativeRoutedQgemmR1ReductionColumns) {
        for (uint reduction_fragment = 0u;
             reduction_fragment <
             kNativeRoutedQgemmR1ReductionColumns;
             reduction_fragment += 8u) {
            simdgroup_matrix<float, 8, 8> weight_a0;
            simdgroup_matrix<float, 8, 8> weight_a1;
            simdgroup_matrix<float, 8, 8> hidden_b0;
            simdgroup_matrix<float, 8, 8> hidden_b1;
            native_routed_qgemm_r1_load_weight(
                weight_a0, down_words, down_scales,
                down_biases, expert_row_begin,
                kHiddenDimension, kMoeExpertDimension,
                output_column_begin,
                reduction_tile + reduction_fragment, lane);
            native_routed_qgemm_r1_load_weight(
                weight_a1, down_words, down_scales,
                down_biases, expert_row_begin,
                kHiddenDimension, kMoeExpertDimension,
                output_column_begin + 8u,
                reduction_tile + reduction_fragment, lane);
            native_routed_qgemm_r1_load_padded_slot_rows(
                hidden_b0, hidden, route_list, task, input_rows,
                0u, reduction_tile + reduction_fragment, lane);
            native_routed_qgemm_r1_load_padded_slot_rows(
                hidden_b1, hidden, route_list, task, input_rows,
                8u, reduction_tile + reduction_fragment, lane);
            simdgroup_multiply_accumulate(
                accumulators[0], weight_a0, hidden_b0,
                accumulators[0]);
            simdgroup_multiply_accumulate(
                accumulators[1], weight_a0, hidden_b1,
                accumulators[1]);
            simdgroup_multiply_accumulate(
                accumulators[2], weight_a1, hidden_b0,
                accumulators[2]);
            simdgroup_multiply_accumulate(
                accumulators[3], weight_a1, hidden_b1,
                accumulators[3]);
        }
    }

    const uint2 coordinate =
        native_routed_qgemm_r1_fragment_coordinate(lane);
    for (uint output_fragment = 0u;
         output_fragment < 2u;
         ++output_fragment) {
        const uint output_column =
            output_column_begin + output_fragment * 8u +
            coordinate.x;
        for (uint route_fragment = 0u;
             route_fragment < 2u;
             ++route_fragment) {
            const uint local_row =
                route_fragment * 8u + coordinate.y;
            for (uint element = 0u; element < 2u; ++element) {
                uint position = 0u;
                uint slot = 0u;
                if (output_column >= kHiddenDimension ||
                    !native_routed_qgemm_r1_route(
                        route_list, task, local_row + element,
                        input_rows, position, slot)) {
                    continue;
                }
                partials[native_routed_qgemm_r1_partial_index(
                    position, slot, output_column)] =
                    native_routed_qgemm_r1_value(
                        accumulators, output_fragment,
                        route_fragment, element);
            }
        }
    }
}

constant uint kNativeRoutedQgemmR2StageStride =
    kNativeRoutedQgemmR1ReductionColumns +
    16u / sizeof(bfloat);
constant uint kNativeRoutedQgemmR2InvalidRoute = 0xffffffffu;

static_assert(
    kNativeRoutedQgemmR1Threads * 8u ==
            kNativeRoutedQgemmR1TileRows *
                kNativeRoutedQgemmR1ReductionColumns &&
        kNativeRoutedQgemmR2StageStride % 8u == 0u &&
        kHiddenDimension % 8u == 0u &&
        kMoeExpertDimension % 8u == 0u,
    "R2 activation staging requires one aligned bfloat8 load per thread");

inline bool native_routed_qgemm_r2_task_valid(
    NativeRoutedQgemmR1Task task,
    uint route_list_expert_stride,
    uint route_list_capacity_per_expert,
    ulong route_list_total_extent,
    uint include_shared_expert) {
    if (include_shared_expert > 1u) {
        return false;
    }
    if (task.expert_index < kMoeExperts) {
        return native_routed_qgemm_r1_task_valid(
            task,
            route_list_expert_stride,
            route_list_capacity_per_expert,
            route_list_total_extent);
    }
    if (include_shared_expert != 1u ||
        task.expert_index != kMoeExperts ||
        task.row_count == 0u ||
        task.row_count > kNativeRoutedQgemmR1TileRows ||
        task.output_row_begin != 0u ||
        route_list_capacity_per_expert >
            route_list_expert_stride) {
        return false;
    }
    const ulong expert_segment_begin =
        ulong(kMoeExperts) *
        ulong(route_list_expert_stride);
    const ulong expert_segment_end =
        expert_segment_begin +
        ulong(route_list_capacity_per_expert);
    const ulong task_begin =
        ulong(task.absolute_route_list_begin);
    const ulong task_rows = ulong(task.row_count);
    return expert_segment_begin <= route_list_total_extent &&
           expert_segment_end <= route_list_total_extent &&
           task_begin >= expert_segment_begin &&
           task_begin <= expert_segment_end &&
           task_rows <= expert_segment_end - task_begin &&
           task_begin <= route_list_total_extent &&
           task_rows <= route_list_total_extent - task_begin;
}

inline bool native_routed_qgemm_r2_route(
    device const uint* route_list,
    NativeRoutedQgemmR1Task task,
    uint local_row,
    uint input_rows,
    uint include_shared_expert,
    thread uint& position,
    thread uint& slot) {
    if (local_row >= task.row_count) {
        return false;
    }
    const uint packed =
        route_list[
            ulong(task.absolute_route_list_begin) +
            ulong(local_row)];
    position = packed >> kPrefillPackedSlotBits;
    slot =
        packed & ((1u << kPrefillPackedSlotBits) - 1u);
    const bool shared = task.expert_index == kMoeExperts;
    const bool valid_slot =
        shared ? include_shared_expert == 1u &&
                     slot == kMoeActiveExperts
               : slot < kMoeActiveExperts;
    return position < input_rows && valid_slot;
}

inline void native_routed_qgemm_r2_copy_bfloat8(
    threadgroup bfloat* destination,
    device const bfloat* source) {
    *((threadgroup uint4*)destination) =
        *((device const uint4*)source);
}

inline void native_routed_qgemm_r2_zero_bfloat8(
    threadgroup bfloat* destination) {
    *((threadgroup uint4*)destination) = uint4(0u);
}

inline void native_routed_qgemm_r2_stage_routes(
    device const uint* route_list,
    NativeRoutedQgemmR1Task task,
    uint input_rows,
    uint include_shared_expert,
    threadgroup uint* staged_routes,
    uint thread_index) {
    if (thread_index >= kNativeRoutedQgemmR1TileRows) {
        return;
    }
    uint position = 0u;
    uint slot = 0u;
    staged_routes[thread_index] =
        native_routed_qgemm_r2_route(
            route_list,
            task,
            thread_index,
            input_rows,
            include_shared_expert,
            position,
            slot)
            ? (position << kPrefillPackedSlotBits) | slot
            : kNativeRoutedQgemmR2InvalidRoute;
}

constant uint kNativeRoutedQgemmBm32TileRows = 32u;
constant uint kNativeRoutedQgemmBm32Simdgroups = 4u;
constant uint kNativeRoutedQgemmBm32Threads =
    kNativeRoutedQgemmBm32Simdgroups * kSimdgroupWidth;

static_assert(
    kNativeRoutedQgemmBm32Threads * 8u ==
        kNativeRoutedQgemmBm32TileRows *
            kNativeRoutedQgemmR1ReductionColumns,
    "BM32 routed activation staging requires one bfloat8 per thread");

inline bool native_routed_qgemm_bm32_dispatch_valid(
    uint task_count,
    uint3 group,
    uint simdgroup,
    uint simdgroup_width,
    uint3 threadgroup_shape) {
    return task_count <= kNativeRoutedQgemmR1TaskCapacity &&
           group.x < task_count &&
           group.z == 0u &&
           simdgroup < kNativeRoutedQgemmBm32Simdgroups &&
           simdgroup_width == kSimdgroupWidth &&
           threadgroup_shape.x == kNativeRoutedQgemmBm32Threads &&
           threadgroup_shape.y == 1u &&
           threadgroup_shape.z == 1u;
}

inline bool native_routed_qgemm_bm32_task_valid(
    NativeRoutedQgemmR1Task task,
    uint route_list_expert_stride,
    uint route_list_capacity_per_expert,
    ulong route_list_total_extent,
    uint include_shared_expert) {
    if (include_shared_expert > 1u ||
        task.expert_index > kMoeExperts ||
        (task.expert_index == kMoeExperts &&
         include_shared_expert != 1u) ||
        task.row_count == 0u ||
        task.row_count > kNativeRoutedQgemmBm32TileRows ||
        task.output_row_begin != 0u ||
        route_list_capacity_per_expert >
            route_list_expert_stride) {
        return false;
    }
    const ulong expert_segment_begin =
        ulong(task.expert_index) *
        ulong(route_list_expert_stride);
    const ulong expert_segment_end =
        expert_segment_begin +
        ulong(route_list_capacity_per_expert);
    const ulong task_begin =
        ulong(task.absolute_route_list_begin);
    const ulong task_rows = ulong(task.row_count);
    return expert_segment_begin <= route_list_total_extent &&
           expert_segment_end <= route_list_total_extent &&
           task_begin >= expert_segment_begin &&
           task_begin <= expert_segment_end &&
           task_rows <= expert_segment_end - task_begin &&
           task_begin <= route_list_total_extent &&
           task_rows <= route_list_total_extent - task_begin;
}

inline void native_routed_qgemm_bm32_stage_routes(
    device const uint* route_list,
    NativeRoutedQgemmR1Task task,
    uint input_rows,
    uint include_shared_expert,
    threadgroup uint* staged_routes,
    uint thread_index) {
    if (thread_index >= kNativeRoutedQgemmBm32TileRows) {
        return;
    }
    uint position = 0u;
    uint slot = 0u;
    staged_routes[thread_index] =
        native_routed_qgemm_r2_route(
            route_list,
            task,
            thread_index,
            input_rows,
            include_shared_expert,
            position,
            slot)
            ? (position << kPrefillPackedSlotBits) | slot
            : kNativeRoutedQgemmR2InvalidRoute;
}

// A41b: BM=8 routed helpers — the R1 shape checks with the tile-row bound
// halved (WM/WN and threads unchanged from the base template).
inline bool native_routed_qgemm_bm8_dispatch_valid(
    uint task_count,
    uint3 group,
    uint simdgroup,
    uint simdgroup_width,
    uint3 threadgroup_shape) {
    return task_count <= kNativeRoutedQgemmR1TaskCapacity &&
           group.x < task_count &&
           group.z == 0u &&
           simdgroup < kNativeRoutedQgemmBm32Simdgroups &&
           simdgroup_width == kSimdgroupWidth &&
           threadgroup_shape.x == kNativeRoutedQgemmBm32Threads &&
           threadgroup_shape.y == 1u &&
           threadgroup_shape.z == 1u;
}

inline bool native_routed_qgemm_bm8_task_valid(
    NativeRoutedQgemmR1Task task,
    uint route_list_expert_stride,
    uint route_list_capacity_per_expert,
    ulong route_list_total_extent,
    uint include_shared_expert) {
    if (include_shared_expert > 1u ||
        task.expert_index > kMoeExperts ||
        (task.expert_index == kMoeExperts &&
         include_shared_expert != 1u) ||
        task.row_count == 0u ||
        task.row_count > kNativeRoutedQgemmBm32TileRows ||
        task.output_row_begin != 0u ||
        route_list_capacity_per_expert >
            route_list_expert_stride) {
        return false;
    }
    const ulong expert_segment_begin =
        ulong(task.expert_index) *
        ulong(route_list_expert_stride);
    const ulong expert_segment_end =
        expert_segment_begin +
        ulong(route_list_capacity_per_expert);
    const ulong task_begin =
        ulong(task.absolute_route_list_begin);
    const ulong task_rows = ulong(task.row_count);
    return expert_segment_begin <= route_list_total_extent &&
           expert_segment_end <= route_list_total_extent &&
           task_begin >= expert_segment_begin &&
           task_begin <= expert_segment_end &&
           task_rows <= expert_segment_end - task_begin &&
           task_begin <= route_list_total_extent &&
           task_rows <= route_list_total_extent - task_begin;
}

inline void native_routed_qgemm_bm8_stage_routes(
    device const uint* route_list,
    NativeRoutedQgemmR1Task task,
    uint input_rows,
    uint include_shared_expert,
    threadgroup uint* staged_routes,
    uint thread_index) {
    if (thread_index >= kNativeRoutedQgemmBm32TileRows) {
        return;
    }
    uint position = 0u;
    uint slot = 0u;
    staged_routes[thread_index] =
        native_routed_qgemm_r2_route(
            route_list,
            task,
            thread_index,
            input_rows,
            include_shared_expert,
            position,
            slot)
            ? (position << kPrefillPackedSlotBits) | slot
            : kNativeRoutedQgemmR2InvalidRoute;
}


inline void native_routed_qgemm_r2_stage_position_rows(
    device const bfloat* input,
    threadgroup const uint* staged_routes,
    uint reduction_tile,
    threadgroup bfloat* staged_input,
    uint thread_index) {
    const uint row = thread_index / 4u;
    const uint reduction = (thread_index % 4u) * 8u;
    threadgroup bfloat* destination =
        staged_input +
        row * kNativeRoutedQgemmR2StageStride + reduction;
    const uint packed = staged_routes[row];
    if (packed == kNativeRoutedQgemmR2InvalidRoute) {
        native_routed_qgemm_r2_zero_bfloat8(destination);
    } else {
        native_routed_qgemm_r2_copy_bfloat8(
            destination,
            input +
                ulong(packed >> kPrefillPackedSlotBits) *
                    ulong(kHiddenDimension) +
                ulong(reduction_tile + reduction));
    }
}

inline void native_routed_qgemm_r2_stage_hidden_rows(
    device const bfloat* hidden,
    threadgroup const uint* staged_routes,
    uint reduction_tile,
    threadgroup bfloat* staged_input,
    uint thread_index) {
    const uint row = thread_index / 4u;
    const uint reduction = (thread_index % 4u) * 8u;
    threadgroup bfloat* destination =
        staged_input +
        row * kNativeRoutedQgemmR2StageStride + reduction;
    const uint packed = staged_routes[row];
    if (packed == kNativeRoutedQgemmR2InvalidRoute) {
        native_routed_qgemm_r2_zero_bfloat8(destination);
    } else {
        const uint position =
            packed >> kPrefillPackedSlotBits;
        const uint slot =
            packed &
            ((1u << kPrefillPackedSlotBits) - 1u);
        native_routed_qgemm_r2_copy_bfloat8(
            destination,
            hidden +
                (ulong(position) * ulong(kMoeSlotCount) +
                 ulong(slot)) *
                    ulong(kMoeExpertDimension) +
                ulong(reduction_tile + reduction));
    }
}

inline void native_routed_qgemm_r2_stage_weights(
    device const uint* packed_weights,
    device const bfloat* scales,
    device const bfloat* biases,
    ulong expert_row_begin,
    uint output_columns,
    uint reduction_columns,
    uint output_tile,
    uint reduction_tile,
    threadgroup float* staged_weights,
    uint thread_index) {
    const uint packed_columns =
        kNativeRoutedQgemmR1ReductionColumns /
        kQ4ValuesPerWord;
    const uint packed_elements =
        kNativeRoutedQgemmR1TileColumns * packed_columns;
    for (uint linear = thread_index; linear < packed_elements;
         linear += kNativeRoutedQgemmR1Threads) {
        const uint output =
            linear / packed_columns;
        const uint packed_column = linear % packed_columns;
        const uint global_output = output_tile + output;
        const uint global_reduction =
            reduction_tile +
            packed_column * kQ4ValuesPerWord;
        uint word = 0u;
        float scale = 0.0f;
        float bias = 0.0f;
        if (global_output < output_columns &&
            global_reduction < reduction_columns) {
            const ulong matrix_row =
                expert_row_begin + ulong(global_output);
            word = packed_weights[
                matrix_row *
                    ulong(
                        reduction_columns /
                        kQ4ValuesPerWord) +
                ulong(
                    global_reduction /
                    kQ4ValuesPerWord)];
            const ulong parameter_index =
                matrix_row *
                    ulong(reduction_columns / kQ4GroupSize) +
                ulong(global_reduction / kQ4GroupSize);
            scale = float(scales[parameter_index]);
            bias = float(biases[parameter_index]);
        }
        const uint stage_begin =
            output * kNativeRoutedQgemmR2StageStride +
            packed_column * kQ4ValuesPerWord;
        for (uint value = 0u; value < kQ4ValuesPerWord; ++value) {
            staged_weights[stage_begin + value] =
                float((word >> (4u * value)) & 15u) * scale +
                bias;
        }
    }
}

inline void native_routed_qgemm_r2_load_weight(
    thread simdgroup_matrix<float, 8, 8>& matrix,
    threadgroup const float* staged_weights,
    uint simdgroup,
    uint output_fragment,
    uint reduction_fragment,
    uint lane) {
    const uint2 coordinate =
        native_routed_qgemm_r1_fragment_coordinate(lane);
    const uint output =
        simdgroup *
            (kNativeRoutedQgemmR1TileColumns /
             kNativeRoutedQgemmR1Simdgroups) +
        output_fragment * 8u + coordinate.x;
    const uint reduction =
        reduction_fragment + coordinate.y;
    const uint index =
        output * kNativeRoutedQgemmR2StageStride + reduction;
    matrix.thread_elements()[0] = staged_weights[index];
    matrix.thread_elements()[1] = staged_weights[index + 1u];
}

inline void native_routed_qgemm_r2_load_input(
    thread simdgroup_matrix<float, 8, 8>& matrix,
    threadgroup const bfloat* staged_input,
    uint route_fragment,
    uint reduction_fragment,
    uint lane) {
    const uint2 coordinate =
        native_routed_qgemm_r1_fragment_coordinate(lane);
    const uint reduction =
        reduction_fragment + coordinate.x;
    const uint row =
        route_fragment * 8u + coordinate.y;
    const uint index =
        row * kNativeRoutedQgemmR2StageStride + reduction;
    matrix.thread_elements()[0] =
        float(staged_input[index]);
    matrix.thread_elements()[1] =
        float(
            staged_input[
                index +
                kNativeRoutedQgemmR2StageStride]);
}

kernel void native_routed_qgemm_r2_fused_upgate_swiglu(
    device const bfloat* input [[buffer(0)]],
    device const uint* route_list [[buffer(1)]],
    device const NativeRoutedQgemmR1Task* tasks [[buffer(2)]],
    device const uint* gate_words [[buffer(3)]],
    device const bfloat* gate_scales [[buffer(4)]],
    device const bfloat* gate_biases [[buffer(5)]],
    device const uint* up_words [[buffer(6)]],
    device const bfloat* up_scales [[buffer(7)]],
    device const bfloat* up_biases [[buffer(8)]],
    device bfloat* hidden [[buffer(9)]],
    constant uint& task_count [[buffer(10)]],
    constant uint& route_list_expert_stride [[buffer(11)]],
    constant uint& route_list_capacity_per_expert [[buffer(12)]],
    constant ulong& route_list_total_extent [[buffer(13)]],
    constant uint& input_rows [[buffer(14)]],
    device const uint* shared_gate_words [[buffer(15)]],
    device const bfloat* shared_gate_scales [[buffer(16)]],
    device const bfloat* shared_gate_biases [[buffer(17)]],
    device const uint* shared_up_words [[buffer(18)]],
    device const bfloat* shared_up_scales [[buffer(19)]],
    device const bfloat* shared_up_biases [[buffer(20)]],
    constant uint& include_shared_expert [[buffer(21)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroup_width [[threads_per_simdgroup]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    if (!native_routed_qgemm_r1_dispatch_valid(
            task_count,
            group,
            simdgroup,
            simdgroup_width,
            threadgroup_shape)) {
        return;
    }
    const NativeRoutedQgemmR1Task task = tasks[group.x];
    if (!native_routed_qgemm_r2_task_valid(
            task,
            route_list_expert_stride,
            route_list_capacity_per_expert,
            route_list_total_extent,
            include_shared_expert) ||
        group.y >=
            (kMoeExpertDimension +
             kNativeRoutedQgemmR1TileColumns - 1u) /
                kNativeRoutedQgemmR1TileColumns) {
        return;
    }

    threadgroup uint
        staged_routes[kNativeRoutedQgemmR1TileRows];
    threadgroup bfloat staged_input[
        kNativeRoutedQgemmR1TileRows *
        kNativeRoutedQgemmR2StageStride];
    threadgroup float staged_gate[
        kNativeRoutedQgemmR1TileColumns *
        kNativeRoutedQgemmR2StageStride];
    threadgroup float staged_up[
        kNativeRoutedQgemmR1TileColumns *
        kNativeRoutedQgemmR2StageStride];

    native_routed_qgemm_r2_stage_routes(
        route_list,
        task,
        input_rows,
        include_shared_expert,
        staged_routes,
        thread_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> gate_accumulators[4];
    simdgroup_matrix<float, 8, 8> up_accumulators[4];
    for (uint index = 0u; index < 4u; ++index) {
        native_routed_qgemm_r1_zero(gate_accumulators[index]);
        native_routed_qgemm_r1_zero(up_accumulators[index]);
    }

    const bool shared =
        task.expert_index == kMoeExperts;
    device const uint* selected_gate_words =
        shared ? shared_gate_words : gate_words;
    device const bfloat* selected_gate_scales =
        shared ? shared_gate_scales : gate_scales;
    device const bfloat* selected_gate_biases =
        shared ? shared_gate_biases : gate_biases;
    device const uint* selected_up_words =
        shared ? shared_up_words : up_words;
    device const bfloat* selected_up_scales =
        shared ? shared_up_scales : up_scales;
    device const bfloat* selected_up_biases =
        shared ? shared_up_biases : up_biases;
    const ulong expert_row_begin =
        shared ? 0ul
               : ulong(task.expert_index) *
                     ulong(kMoeExpertDimension);
    const uint output_tile =
        group.y * kNativeRoutedQgemmR1TileColumns;
    for (uint reduction_tile = 0u;
         reduction_tile < kHiddenDimension;
         reduction_tile +=
             kNativeRoutedQgemmR1ReductionColumns) {
        if (reduction_tile != 0u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        native_routed_qgemm_r2_stage_position_rows(
            input,
            staged_routes,
            reduction_tile,
            staged_input,
            thread_index);
        native_routed_qgemm_r2_stage_weights(
            selected_gate_words,
            selected_gate_scales,
            selected_gate_biases,
            expert_row_begin,
            kMoeExpertDimension,
            kHiddenDimension,
            output_tile,
            reduction_tile,
            staged_gate,
            thread_index);
        native_routed_qgemm_r2_stage_weights(
            selected_up_words,
            selected_up_scales,
            selected_up_biases,
            expert_row_begin,
            kMoeExpertDimension,
            kHiddenDimension,
            output_tile,
            reduction_tile,
            staged_up,
            thread_index);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint reduction_fragment = 0u;
             reduction_fragment <
             kNativeRoutedQgemmR1ReductionColumns;
             reduction_fragment += 8u) {
            simdgroup_matrix<float, 8, 8> gate_a0;
            simdgroup_matrix<float, 8, 8> gate_a1;
            simdgroup_matrix<float, 8, 8> up_a0;
            simdgroup_matrix<float, 8, 8> up_a1;
            simdgroup_matrix<float, 8, 8> input_b0;
            simdgroup_matrix<float, 8, 8> input_b1;
            native_routed_qgemm_r2_load_weight(
                gate_a0,
                staged_gate,
                simdgroup,
                0u,
                reduction_fragment,
                lane);
            native_routed_qgemm_r2_load_weight(
                gate_a1,
                staged_gate,
                simdgroup,
                1u,
                reduction_fragment,
                lane);
            native_routed_qgemm_r2_load_weight(
                up_a0,
                staged_up,
                simdgroup,
                0u,
                reduction_fragment,
                lane);
            native_routed_qgemm_r2_load_weight(
                up_a1,
                staged_up,
                simdgroup,
                1u,
                reduction_fragment,
                lane);
            native_routed_qgemm_r2_load_input(
                input_b0,
                staged_input,
                0u,
                reduction_fragment,
                lane);
            native_routed_qgemm_r2_load_input(
                input_b1,
                staged_input,
                1u,
                reduction_fragment,
                lane);
            simdgroup_multiply_accumulate(
                gate_accumulators[0],
                gate_a0,
                input_b0,
                gate_accumulators[0]);
            simdgroup_multiply_accumulate(
                gate_accumulators[1],
                gate_a0,
                input_b1,
                gate_accumulators[1]);
            simdgroup_multiply_accumulate(
                gate_accumulators[2],
                gate_a1,
                input_b0,
                gate_accumulators[2]);
            simdgroup_multiply_accumulate(
                gate_accumulators[3],
                gate_a1,
                input_b1,
                gate_accumulators[3]);
            simdgroup_multiply_accumulate(
                up_accumulators[0],
                up_a0,
                input_b0,
                up_accumulators[0]);
            simdgroup_multiply_accumulate(
                up_accumulators[1],
                up_a0,
                input_b1,
                up_accumulators[1]);
            simdgroup_multiply_accumulate(
                up_accumulators[2],
                up_a1,
                input_b0,
                up_accumulators[2]);
            simdgroup_multiply_accumulate(
                up_accumulators[3],
                up_a1,
                input_b1,
                up_accumulators[3]);
        }
    }

    const uint2 coordinate =
        native_routed_qgemm_r1_fragment_coordinate(lane);
    for (uint output_fragment = 0u;
         output_fragment < 2u;
         ++output_fragment) {
        const uint output_column =
            output_tile +
            simdgroup *
                (kNativeRoutedQgemmR1TileColumns /
                 kNativeRoutedQgemmR1Simdgroups) +
            output_fragment * 8u + coordinate.x;
        for (uint route_fragment = 0u;
             route_fragment < 2u;
             ++route_fragment) {
            const uint local_row =
                route_fragment * 8u + coordinate.y;
            for (uint element = 0u; element < 2u; ++element) {
                const uint packed =
                    staged_routes[local_row + element];
                if (packed ==
                        kNativeRoutedQgemmR2InvalidRoute ||
                    output_column >= kMoeExpertDimension) {
                    continue;
                }
                const uint position =
                    packed >> kPrefillPackedSlotBits;
                const uint slot =
                    packed &
                    ((1u << kPrefillPackedSlotBits) - 1u);
                const float gate =
                    native_routed_qgemm_r1_value(
                        gate_accumulators,
                        output_fragment,
                        route_fragment,
                        element);
                const float up =
                    native_routed_qgemm_r1_value(
                        up_accumulators,
                        output_fragment,
                        route_fragment,
                        element);
                hidden[native_routed_qgemm_r1_hidden_index(
                    position, slot, output_column)] =
                    static_cast<bfloat>(
                        (gate / (1.0f + exp(-gate))) * up);
            }
        }
    }
}

kernel void native_routed_qgemm_r2_down_partial(
    device const bfloat* hidden [[buffer(0)]],
    device const uint* route_list [[buffer(1)]],
    device const NativeRoutedQgemmR1Task* tasks [[buffer(2)]],
    device const uint* down_words [[buffer(3)]],
    device const bfloat* down_scales [[buffer(4)]],
    device const bfloat* down_biases [[buffer(5)]],
    device float* partials [[buffer(6)]],
    constant uint& task_count [[buffer(7)]],
    constant uint& route_list_expert_stride [[buffer(8)]],
    constant uint& route_list_capacity_per_expert [[buffer(9)]],
    constant ulong& route_list_total_extent [[buffer(10)]],
    constant uint& input_rows [[buffer(11)]],
    device const uint* shared_down_words [[buffer(12)]],
    device const bfloat* shared_down_scales [[buffer(13)]],
    device const bfloat* shared_down_biases [[buffer(14)]],
    constant uint& include_shared_expert [[buffer(15)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroup_width [[threads_per_simdgroup]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    if (!native_routed_qgemm_r1_dispatch_valid(
            task_count,
            group,
            simdgroup,
            simdgroup_width,
            threadgroup_shape)) {
        return;
    }
    const NativeRoutedQgemmR1Task task = tasks[group.x];
    if (!native_routed_qgemm_r2_task_valid(
            task,
            route_list_expert_stride,
            route_list_capacity_per_expert,
            route_list_total_extent,
            include_shared_expert) ||
        group.y >=
            (kHiddenDimension +
             kNativeRoutedQgemmR1TileColumns - 1u) /
                kNativeRoutedQgemmR1TileColumns) {
        return;
    }

    threadgroup uint
        staged_routes[kNativeRoutedQgemmR1TileRows];
    threadgroup bfloat staged_hidden[
        kNativeRoutedQgemmR1TileRows *
        kNativeRoutedQgemmR2StageStride];
    threadgroup float staged_weights[
        kNativeRoutedQgemmR1TileColumns *
        kNativeRoutedQgemmR2StageStride];

    native_routed_qgemm_r2_stage_routes(
        route_list,
        task,
        input_rows,
        include_shared_expert,
        staged_routes,
        thread_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> accumulators[4];
    for (uint index = 0u; index < 4u; ++index) {
        native_routed_qgemm_r1_zero(accumulators[index]);
    }

    const bool shared =
        task.expert_index == kMoeExperts;
    device const uint* selected_down_words =
        shared ? shared_down_words : down_words;
    device const bfloat* selected_down_scales =
        shared ? shared_down_scales : down_scales;
    device const bfloat* selected_down_biases =
        shared ? shared_down_biases : down_biases;
    const ulong expert_row_begin =
        shared ? 0ul
               : ulong(task.expert_index) *
                     ulong(kHiddenDimension);
    const uint output_tile =
        group.y * kNativeRoutedQgemmR1TileColumns;
    for (uint reduction_tile = 0u;
         reduction_tile < kMoeExpertDimension;
         reduction_tile +=
             kNativeRoutedQgemmR1ReductionColumns) {
        if (reduction_tile != 0u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        native_routed_qgemm_r2_stage_hidden_rows(
            hidden,
            staged_routes,
            reduction_tile,
            staged_hidden,
            thread_index);
        native_routed_qgemm_r2_stage_weights(
            selected_down_words,
            selected_down_scales,
            selected_down_biases,
            expert_row_begin,
            kHiddenDimension,
            kMoeExpertDimension,
            output_tile,
            reduction_tile,
            staged_weights,
            thread_index);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint reduction_fragment = 0u;
             reduction_fragment <
             kNativeRoutedQgemmR1ReductionColumns;
             reduction_fragment += 8u) {
            simdgroup_matrix<float, 8, 8> weight_a0;
            simdgroup_matrix<float, 8, 8> weight_a1;
            simdgroup_matrix<float, 8, 8> hidden_b0;
            simdgroup_matrix<float, 8, 8> hidden_b1;
            native_routed_qgemm_r2_load_weight(
                weight_a0,
                staged_weights,
                simdgroup,
                0u,
                reduction_fragment,
                lane);
            native_routed_qgemm_r2_load_weight(
                weight_a1,
                staged_weights,
                simdgroup,
                1u,
                reduction_fragment,
                lane);
            native_routed_qgemm_r2_load_input(
                hidden_b0,
                staged_hidden,
                0u,
                reduction_fragment,
                lane);
            native_routed_qgemm_r2_load_input(
                hidden_b1,
                staged_hidden,
                1u,
                reduction_fragment,
                lane);
            simdgroup_multiply_accumulate(
                accumulators[0],
                weight_a0,
                hidden_b0,
                accumulators[0]);
            simdgroup_multiply_accumulate(
                accumulators[1],
                weight_a0,
                hidden_b1,
                accumulators[1]);
            simdgroup_multiply_accumulate(
                accumulators[2],
                weight_a1,
                hidden_b0,
                accumulators[2]);
            simdgroup_multiply_accumulate(
                accumulators[3],
                weight_a1,
                hidden_b1,
                accumulators[3]);
        }
    }

    const uint2 coordinate =
        native_routed_qgemm_r1_fragment_coordinate(lane);
    for (uint output_fragment = 0u;
         output_fragment < 2u;
         ++output_fragment) {
        const uint output_column =
            output_tile +
            simdgroup *
                (kNativeRoutedQgemmR1TileColumns /
                 kNativeRoutedQgemmR1Simdgroups) +
            output_fragment * 8u + coordinate.x;
        for (uint route_fragment = 0u;
             route_fragment < 2u;
             ++route_fragment) {
            const uint local_row =
                route_fragment * 8u + coordinate.y;
            for (uint element = 0u; element < 2u; ++element) {
                const uint packed =
                    staged_routes[local_row + element];
                if (packed ==
                        kNativeRoutedQgemmR2InvalidRoute ||
                    output_column >= kHiddenDimension) {
                    continue;
                }
                const uint position =
                    packed >> kPrefillPackedSlotBits;
                const uint slot =
                    packed &
                    ((1u << kPrefillPackedSlotBits) - 1u);
                partials[
                    native_routed_qgemm_r1_partial_index(
                        position, slot, output_column)] =
                    native_routed_qgemm_r1_value(
                        accumulators,
                        output_fragment,
                        route_fragment,
                        element);
            }
        }
    }
}

kernel void expert_union_fused_tasks(
    device const uint* ids [[buffer(0)]],
    constant uint& block [[buffer(1)]],
    device uint* counts [[buffer(2)]],
    device uint* lists [[buffer(3)]],
    device NativeRoutedQgemmR1Task* tasks [[buffer(4)]],
    device uint* task_count [[buffer(5)]],
    device uint* status [[buffer(6)]],
    constant uint& list_capacity [[buffer(7)]],
    constant uint& include_shared_expert [[buffer(8)]],
    constant ulong& route_list_total_extent [[buffer(9)]],
    constant uint& planned_task_capacity [[buffer(10)]],
    constant uint& packed_slot_bits [[buffer(11)]],
    uint3 thread_position [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    const uint expert = thread_position.x;
    if (!all(group == uint3(0u)) ||
        threadgroup_shape.x != kMoeExperts ||
        threadgroup_shape.y != 1u ||
        threadgroup_shape.z != 1u) {
        return;
    }

    threadgroup uint errors[kMoeExperts];
    threadgroup uint final_status;
    if (expert == 0u) {
        status[0] =
            kNativeRoutedQgemmR1TaskStatusNotProduced;
        task_count[0] = 0u;
        counts[kMoeExperts] = 0u;
        final_status =
            kNativeRoutedQgemmR1TaskStatusNotProduced;
    }

    const uint maximum_position_count =
        1u << (32u - kPrefillPackedSlotBits);
    const ulong required_list_extent =
        ulong(kMoeExperts + 1u) * ulong(list_capacity);
    uint local_error =
        kNativeRoutedQgemmR1TaskStatusNotProduced;
    if (block == 0u || list_capacity == 0u ||
        block > list_capacity ||
        list_capacity > maximum_position_count ||
        include_shared_expert > 1u ||
        planned_task_capacity == 0u ||
        planned_task_capacity >
            kNativeRoutedQgemmR1TaskCapacity ||
        packed_slot_bits != kPrefillPackedSlotBits) {
        local_error =
            kNativeRoutedQgemmR1TaskStatusCountOutOfRange;
    } else if (
        route_list_total_extent != required_list_extent ||
        route_list_total_extent > 0x100000000ul) {
        local_error =
            kNativeRoutedQgemmR1TaskStatusPackedSlotOutOfRange;
    }
    errors[expert] = local_error;
    threadgroup_barrier(
        mem_flags::mem_device | mem_flags::mem_threadgroup);

    uint count = 0u;
    if (local_error ==
        kNativeRoutedQgemmR1TaskStatusNotProduced) {
        for (uint position = 0u; position < block; ++position) {
            bool matched = false;
            for (uint slot = 0u; slot < kMoeActiveExperts;
                 ++slot) {
                const uint selected =
                    ids[position * kMoeActiveExperts + slot];
                if (selected >= kMoeExperts) {
                    local_error =
                        kNativeRoutedQgemmR1TaskStatusPackedSlotOutOfRange;
                }
                if (selected == expert) {
                    if (matched || count >= list_capacity) {
                        local_error =
                            kNativeRoutedQgemmR1TaskStatusRouteConservationFailure;
                    } else {
                        lists[expert * list_capacity + count] =
                            (position << kPrefillPackedSlotBits) |
                            slot;
                        ++count;
                    }
                    matched = true;
                }
            }
        }
    }
    counts[expert] = count;
    errors[expert] = local_error;
    threadgroup_barrier(
        mem_flags::mem_device | mem_flags::mem_threadgroup);

    if (expert == 0u) {
        uint result =
            kNativeRoutedQgemmR1TaskStatusNotProduced;
        for (uint index = 0u; index < kMoeExperts; ++index) {
            if (errors[index] !=
                kNativeRoutedQgemmR1TaskStatusNotProduced) {
                result = errors[index];
                break;
            }
        }

        if (result ==
            kNativeRoutedQgemmR1TaskStatusNotProduced) {
            counts[kMoeExperts] = block;
            for (uint position = 0u; position < block;
                 ++position) {
                lists[kMoeExperts * list_capacity + position] =
                    (position << kPrefillPackedSlotBits) |
                    kMoeActiveExperts;
            }

            ulong routed_count = 0ul;
            ulong required_task_count = 0ul;
            for (uint index = 0u; index < kMoeExperts;
                 ++index) {
                const uint expert_rows = counts[index];
                routed_count += ulong(expert_rows);
                required_task_count +=
                    (ulong(expert_rows) +
                     ulong(kNativeRoutedQgemmR1TileRows) -
                     1ul) /
                    ulong(kNativeRoutedQgemmR1TileRows);
            }

            const ulong expected_routed_count =
                ulong(block) * ulong(kMoeActiveExperts);
            if (routed_count != expected_routed_count) {
                result =
                    kNativeRoutedQgemmR1TaskStatusRouteConservationFailure;
            }
            if (include_shared_expert != 0u) {
                required_task_count +=
                    (ulong(block) +
                     ulong(kNativeRoutedQgemmR1TileRows) -
                     1ul) /
                    ulong(kNativeRoutedQgemmR1TileRows);
            }
            if (result ==
                    kNativeRoutedQgemmR1TaskStatusNotProduced &&
                required_task_count >
                    ulong(planned_task_capacity)) {
                result =
                    kNativeRoutedQgemmR1TaskStatusTaskCapacityExceeded;
            }

            if (result ==
                kNativeRoutedQgemmR1TaskStatusNotProduced) {
                uint next_task = 0u;
                const uint task_expert_count =
                    kMoeExperts + include_shared_expert;
                for (uint task_expert = 0u;
                     task_expert < task_expert_count &&
                     result ==
                         kNativeRoutedQgemmR1TaskStatusNotProduced;
                     ++task_expert) {
                    const uint expert_rows =
                        counts[task_expert];
                    const ulong segment_begin =
                        ulong(task_expert) *
                        ulong(list_capacity);
                    for (uint local_begin = 0u;
                         local_begin < expert_rows &&
                         result ==
                             kNativeRoutedQgemmR1TaskStatusNotProduced;
                         local_begin +=
                             kNativeRoutedQgemmR1TileRows) {
                        if (next_task >=
                            planned_task_capacity) {
                            result =
                                kNativeRoutedQgemmR1TaskStatusTaskCapacityExceeded;
                            break;
                        }
                        const uint row_count =
                            min(
                                kNativeRoutedQgemmR1TileRows,
                                expert_rows - local_begin);
                        tasks[next_task++] = {
                            task_expert,
                            uint(segment_begin +
                                 ulong(local_begin)),
                            row_count,
                            0u,
                        };
                    }
                }
                if (result ==
                    kNativeRoutedQgemmR1TaskStatusNotProduced) {
                    task_count[0] = next_task;
                    result =
                        kNativeRoutedQgemmR1TaskStatusReady;
                }
            }
        }
        final_status = result;
    }
    threadgroup_barrier(
        mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (expert == 0u) {
        status[0] = final_status;
    }
}
