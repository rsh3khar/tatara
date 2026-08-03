kernel void
attn_project(device const bfloat* input [[buffer(0)]], device const uint* qg_words [[buffer(1)]],
             device const bfloat* qg_scales [[buffer(2)]],
             device const bfloat* qg_biases [[buffer(3)]], device const uint* k_words [[buffer(4)]],
             device const bfloat* k_scales [[buffer(5)]],
             device const bfloat* k_biases [[buffer(6)]], device const uint* v_words [[buffer(7)]],
             device const bfloat* v_scales [[buffer(8)]],
             device const bfloat* v_biases [[buffer(9)]], device bfloat* projection [[buffer(10)]],
             uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tid >> 5u;
    if (row >= kAttnProjectionRows) {
        return;
    }
    device const uint* words;
    device const bfloat* scales;
    device const bfloat* biases;
    uint local_row;
    if (row < kAttnQGateRows) {
        words = qg_words;
        scales = qg_scales;
        biases = qg_biases;
        local_row = row;
    } else if (row < kAttnVRowOffset) {
        words = k_words;
        scales = k_scales;
        biases = k_biases;
        local_row = row - kAttnQGateRows;
    } else {
        words = v_words;
        scales = v_scales;
        biases = v_biases;
        local_row = row - kAttnVRowOffset;
    }
    const ulong word_row = ulong(local_row) * ulong(kQuantWordsPerRow);
    const ulong group_row = ulong(local_row) * ulong(kGroupsPerRow);
    const float acc = q4_dot(input, words + word_row, scales + group_row, biases + group_row,
                             kHiddenDimension, lane);
    if (lane == 0u) {
        projection[row] = static_cast<bfloat>(acc);
    }
}

kernel void attn_qk_rope(device const bfloat* projection [[buffer(0)]],
                         device const bfloat* q_weight [[buffer(1)]],
                         device const bfloat* k_weight [[buffer(2)]],
                         device bfloat* q_out [[buffer(3)]], device bfloat* gate_out [[buffer(4)]],
                         device bfloat* keys [[buffer(5)]], device bfloat* values [[buffer(6)]],
                         constant uint& context [[buffer(7)]],
                         constant uint& capacity [[buffer(8)]],
                         uint head [[threadgroup_position_in_grid]],
                         uint d [[thread_position_in_threadgroup]]) {
    // Exact reference semantics for the 256-value weighted q/k RMS: the
    // two-simdgroup 64x4 reduction with a zeroed 32-slot stage and precise
    // rsqrt, then the double-rounded output w * bf16(x * inv). RoPE reads
    // the bfloat16-rounded norm as a separate eager op, rotates float32
    // half-pairs, and rounds once on store.
    threadgroup float values_shared[256];
    threadgroup float stage[32];
    const bool is_query = head < kAttnQueryHeads;
    const uint local_head = is_query ? head : head - kAttnQueryHeads;
    const uint base = is_query ? head * (2u * kAttnHeadDimension)
                               : kAttnQGateRows + local_head * kAttnHeadDimension;
    const float x = float(projection[base + d]);
    float acc = 0.0f;
    if (d < 64u) {
        const uint reduce_base = d * 4u;
        for (uint j = 0; j < 4u; ++j) {
            const float value = float(projection[base + reduce_base + j]);
            acc += value * value;
        }
    }
    acc = simd_sum(acc);
    if (d < 32u) {
        stage[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d < 64u && (d & 31u) == 0u) {
        stage[d >> 5u] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d < 32u) {
        const float total = simd_sum(stage[d]);
        if (d == 0u) {
            stage[0] = precise::rsqrt(total / float(kAttnHeadDimension) + kRmsNormEpsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inverse = stage[0];
    const bfloat normed_b =
        (is_query ? q_weight[d] : k_weight[d]) * static_cast<bfloat>(x * inverse);
    values_shared[d] = float(normed_b);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rotated = values_shared[d];
    if (d < 64u) {
        const uint pair = d & 31u;
        const float theta =
            float(context) / pow(kAttnRopeBase, float(pair) / float(kAttnRopePairs));
        const float cosine = cos(theta);
        const float sine = sin(theta);
        rotated = d < 32u ? values_shared[d] * cosine - values_shared[d + 32u] * sine
                          : values_shared[d] * cosine + values_shared[d - 32u] * sine;
    }
    if (is_query) {
        q_out[head * kAttnHeadDimension + d] = static_cast<bfloat>(rotated);
        gate_out[head * kAttnHeadDimension + d] =
            projection[head * (2u * kAttnHeadDimension) + kAttnHeadDimension + d];
    } else {
        keys[(local_head * capacity + context) * kAttnHeadDimension + d] =
            static_cast<bfloat>(rotated);
        values[(local_head * capacity + context) * kAttnHeadDimension + d] =
            projection[kAttnVRowOffset + local_head * kAttnHeadDimension + d];
    }
}

// Row-batched attn_qk_rope over pooled KV caches: grid y is the batch row,
// state_table carries per-row {kv element offset, context}. Keys and values
// pools share the stripe layout so one offset serves both. Each row's
// arithmetic is exactly attn_qk_rope's.
kernel void attn_qk_rope_ms(device const bfloat* projection [[buffer(0)]],
                            device const bfloat* q_weight [[buffer(1)]],
                            device const bfloat* k_weight [[buffer(2)]],
                            device bfloat* q_out [[buffer(3)]],
                            device bfloat* gate_out [[buffer(4)]],
                            device bfloat* keys_pool [[buffer(5)]],
                            device bfloat* values_pool [[buffer(6)]],
                            constant uint& capacity [[buffer(7)]],
                            device const uint* state_table [[buffer(8)]],
                            constant uint& projection_stride [[buffer(9)]],
                            constant uint& query_stride [[buffer(10)]],
                            uint2 head2 [[threadgroup_position_in_grid]],
                            uint2 d2 [[thread_position_in_threadgroup]]) {
    const uint head = head2.x;
    const uint d = d2.x;
    const uint row = head2.y;
    const uint kv_offset = state_table[row * 4u];
    const uint context = state_table[row * 4u + 1u];
    device const bfloat* proj_row = projection + ulong(row) * ulong(projection_stride);
    device bfloat* q_row = q_out + ulong(row) * ulong(query_stride);
    device bfloat* gate_row = gate_out + ulong(row) * ulong(query_stride);
    device bfloat* keys = keys_pool + kv_offset;
    device bfloat* values = values_pool + kv_offset;
    threadgroup float values_shared[256];
    threadgroup float stage[32];
    const bool is_query = head < kAttnQueryHeads;
    const uint local_head = is_query ? head : head - kAttnQueryHeads;
    const uint base = is_query ? head * (2u * kAttnHeadDimension)
                               : kAttnQGateRows + local_head * kAttnHeadDimension;
    const float x = float(proj_row[base + d]);
    float acc = 0.0f;
    if (d < 64u) {
        const uint reduce_base = d * 4u;
        for (uint j = 0; j < 4u; ++j) {
            const float value = float(proj_row[base + reduce_base + j]);
            acc += value * value;
        }
    }
    acc = simd_sum(acc);
    if (d < 32u) {
        stage[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d < 64u && (d & 31u) == 0u) {
        stage[d >> 5u] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d < 32u) {
        const float total = simd_sum(stage[d]);
        if (d == 0u) {
            stage[0] = precise::rsqrt(total / float(kAttnHeadDimension) + kRmsNormEpsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inverse = stage[0];
    const bfloat normed_b =
        (is_query ? q_weight[d] : k_weight[d]) * static_cast<bfloat>(x * inverse);
    values_shared[d] = float(normed_b);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rotated = values_shared[d];
    if (d < 64u) {
        const uint pair = d & 31u;
        const float theta =
            float(context) / pow(kAttnRopeBase, float(pair) / float(kAttnRopePairs));
        const float cosine = cos(theta);
        const float sine = sin(theta);
        rotated = d < 32u ? values_shared[d] * cosine - values_shared[d + 32u] * sine
                          : values_shared[d] * cosine + values_shared[d - 32u] * sine;
    }
    if (is_query) {
        q_row[head * kAttnHeadDimension + d] = static_cast<bfloat>(rotated);
        gate_row[head * kAttnHeadDimension + d] =
            proj_row[head * (2u * kAttnHeadDimension) + kAttnHeadDimension + d];
    } else {
        keys[(local_head * capacity + context) * kAttnHeadDimension + d] =
            static_cast<bfloat>(rotated);
        values[(local_head * capacity + context) * kAttnHeadDimension + d] =
            proj_row[kAttnVRowOffset + local_head * kAttnHeadDimension + d];
    }
}

kernel void
attention_decode(device const bfloat* q [[buffer(0)]], device const bfloat* gate [[buffer(1)]],
                 device const bfloat* keys [[buffer(2)]], device const bfloat* values [[buffer(3)]],
                 constant uint& context [[buffer(4)]], device bfloat* out [[buffer(5)]],
                 constant uint& capacity [[buffer(6)]], uint head [[threadgroup_position_in_grid]],
                 uint tid [[thread_position_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]]) {
    // Tiled online softmax over arbitrary context. The extra barrier after
    // consuming the tile max is load-bearing: red[] is reused across
    // simdgroups and the overwrite would otherwise race a slower
    // simdgroup's read.
    threadgroup float scores[256];
    threadgroup float red[256];
    threadgroup float carry[2];
    const uint sg = tid >> 5u;
    const uint n = context + 1u;
    const uint kv = head >> 3u;
    if (tid == 0u) {
        carry[0] = -INFINITY;
        carry[1] = 0.0f;
    }
    float acc = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint tile = 0; tile < n; tile += 256u) {
        const uint count = min(n - tile, 256u);
        for (uint t = sg; t < count; t += 8u) {
            float dot = 0.0f;
            for (uint d = lane; d < kAttnHeadDimension; d += 32u) {
                dot += float(q[head * kAttnHeadDimension + d]) *
                       float(keys[(kv * capacity + tile + t) * kAttnHeadDimension + d]);
            }
            dot = simd_sum(dot);
            if (lane == 0u) {
                scores[t] = dot * kAttnScale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[tid] = tid < count ? scores[tid] : -INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 128u; off; off >>= 1u) {
            if (tid < off) {
                red[tid] = max(red[tid], red[tid + off]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float previous_max = carry[0];
        const float new_max = max(previous_max, red[0]);
        const float rescale = previous_max == -INFINITY ? 0.0f : exp(previous_max - new_max);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[tid] = tid < count ? exp(scores[tid] - new_max) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < count) {
            scores[tid] = red[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 128u; off; off >>= 1u) {
            if (tid < off) {
                red[tid] += red[tid + off];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float tile_sum = red[0];
        acc *= rescale;
        for (uint t = 0; t < count; ++t) {
            acc += scores[t] * float(values[(kv * capacity + tile + t) * kAttnHeadDimension + tid]);
        }
        if (tid == 0u) {
            carry[1] = carry[1] * rescale + tile_sum;
            carry[0] = new_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float value = acc / carry[1];
    const float g = float(gate[head * kAttnHeadDimension + tid]);
    value *= 1.0f / (1.0f + exp(-g));
    out[head * kAttnHeadDimension + tid] = static_cast<bfloat>(value);
}

// Bucket-batched attention_decode over pooled KV: grid y walks a host-built
// row list of the streams whose solo path is the single-kernel family, so
// no row ever changes kernel family versus its solo run. Per-row context
// and KV offsets come from the layer's state table. Each row's arithmetic
// is exactly attention_decode's.
kernel void
attention_decode_ms(device const bfloat* q [[buffer(0)]],
                    device const bfloat* gate [[buffer(1)]],
                    device const bfloat* keys_pool [[buffer(2)]],
                    device const bfloat* values_pool [[buffer(3)]],
                    device const uint* state_table [[buffer(4)]],
                    device bfloat* out [[buffer(5)]],
                    constant uint& capacity [[buffer(6)]],
                    device const uint* bucket_rows [[buffer(7)]],
                    constant uint& query_stride [[buffer(8)]],
                    uint2 head2 [[threadgroup_position_in_grid]],
                    uint2 tid2 [[thread_position_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    const uint head = head2.x;
    const uint tid = tid2.x;
    const uint row = bucket_rows[head2.y];
    const uint kv_offset = state_table[row * 4u];
    const uint context = state_table[row * 4u + 1u];
    device const bfloat* q_row = q + ulong(row) * ulong(query_stride);
    device const bfloat* gate_row = gate + ulong(row) * ulong(query_stride);
    device const bfloat* keys = keys_pool + kv_offset;
    device const bfloat* values = values_pool + kv_offset;
    device bfloat* out_row = out + ulong(row) * ulong(query_stride);
    threadgroup float scores[256];
    threadgroup float red[256];
    threadgroup float carry[2];
    const uint sg = tid >> 5u;
    const uint n = context + 1u;
    const uint kv = head >> 3u;
    if (tid == 0u) {
        carry[0] = -INFINITY;
        carry[1] = 0.0f;
    }
    float acc = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint tile = 0; tile < n; tile += 256u) {
        const uint count = min(n - tile, 256u);
        for (uint t = sg; t < count; t += 8u) {
            float dot = 0.0f;
            for (uint d = lane; d < kAttnHeadDimension; d += 32u) {
                dot += float(q_row[head * kAttnHeadDimension + d]) *
                       float(keys[(kv * capacity + tile + t) * kAttnHeadDimension + d]);
            }
            dot = simd_sum(dot);
            if (lane == 0u) {
                scores[t] = dot * kAttnScale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[tid] = tid < count ? scores[tid] : -INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 128u; off; off >>= 1u) {
            if (tid < off) {
                red[tid] = max(red[tid], red[tid + off]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float previous_max = carry[0];
        const float new_max = max(previous_max, red[0]);
        const float rescale = previous_max == -INFINITY ? 0.0f : exp(previous_max - new_max);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[tid] = tid < count ? exp(scores[tid] - new_max) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < count) {
            scores[tid] = red[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 128u; off; off >>= 1u) {
            if (tid < off) {
                red[tid] += red[tid + off];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float tile_sum = red[0];
        acc *= rescale;
        for (uint t = 0; t < count; ++t) {
            acc += scores[t] * float(values[(kv * capacity + tile + t) * kAttnHeadDimension + tid]);
        }
        if (tid == 0u) {
            carry[1] = carry[1] * rescale + tile_sum;
            carry[0] = new_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float value = acc / carry[1];
    const float g = float(gate_row[head * kAttnHeadDimension + tid]);
    value *= 1.0f / (1.0f + exp(-g));
    out_row[head * kAttnHeadDimension + tid] = static_cast<bfloat>(value);
}

kernel void attention_decode_scores_gqa4(
    device const bfloat* q [[buffer(0)]], device const bfloat* keys [[buffer(1)]],
    constant uint& context [[buffer(2)]], constant uint& capacity [[buffer(3)]],
    constant uint& part [[buffer(4)]], device float* weights [[buffer(5)]],
    constant uint& nparts [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
    uint3 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    // Four related query heads share every key load while keeping each
    // head's scalar dimension order, simd_sum tree, and the sealed
    // 256-thread softmax reductions.
    threadgroup float scores[4][256];
    threadgroup float red[4][256];
    const uint tid = tpos.x;
    const uint cohort = tpos.y;
    const uint sg = tid >> 5u;
    const uint kv = tg.x >> 1u;
    const uint head_base = kv * 8u + (tg.x & 1u) * 4u;
    const uint n = context + 1u;
    const uint start = tg.y * part;
    const uint count = start < n ? min(n - start, part) : 0u;

    for (uint t = cohort * 8u + sg; t < count; t += 32u) {
        float dot0 = 0.0f;
        float dot1 = 0.0f;
        float dot2 = 0.0f;
        float dot3 = 0.0f;
        for (uint d = lane; d < kAttnHeadDimension; d += 32u) {
            const float key_value =
                float(keys[(kv * capacity + start + t) * kAttnHeadDimension + d]);
            dot0 += float(q[(head_base + 0u) * kAttnHeadDimension + d]) * key_value;
            dot1 += float(q[(head_base + 1u) * kAttnHeadDimension + d]) * key_value;
            dot2 += float(q[(head_base + 2u) * kAttnHeadDimension + d]) * key_value;
            dot3 += float(q[(head_base + 3u) * kAttnHeadDimension + d]) * key_value;
        }
        dot0 = simd_sum(dot0);
        dot1 = simd_sum(dot1);
        dot2 = simd_sum(dot2);
        dot3 = simd_sum(dot3);
        if (lane == 0u) {
            scores[0][t] = dot0 * kAttnScale;
            scores[1][t] = dot1 * kAttnScale;
            scores[2][t] = dot2 * kAttnScale;
            scores[3][t] = dot3 * kAttnScale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint head = head_base + cohort;
    red[cohort][tid] = tid < count ? scores[cohort][tid] : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = 128u; off; off >>= 1u) {
        if (tid < off) {
            red[cohort][tid] = max(red[cohort][tid], red[cohort][tid + off]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float part_max = red[cohort][0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    red[cohort][tid] = tid < count ? exp(scores[cohort][tid] - part_max) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < count) {
        scores[cohort][tid] = red[cohort][tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = 128u; off; off >>= 1u) {
        if (tid < off) {
            red[cohort][tid] += red[cohort][tid + off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float* dst = weights + (head * nparts + tg.y) * 258u;
    dst[tid] = tid < count ? scores[cohort][tid] : 0.0f;
    if (tid == 0u) {
        dst[256] = part_max;
        dst[257] = red[cohort][0];
    }
}

inline float reduce_max_256_simd_tail_exact(
    threadgroup float* values, uint tid, uint lane) {
    for (uint off = 128u; off >= 32u; off >>= 1u) {
        if (tid < off) {
            values[tid] = max(values[tid], values[tid + off]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 32u) {
        float value = values[tid];
        for (uint off = 16u; off; off >>= 1u) {
            const float other = simd_shuffle_down(value, off);
            if (lane < off) {
                value = max(value, other);
            }
        }
        if (lane == 0u) {
            values[0] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return values[0];
}

inline float reduce_sum_256_simd_tail_exact(
    threadgroup float* values, uint tid, uint lane) {
    for (uint off = 128u; off >= 32u; off >>= 1u) {
        if (tid < off) {
            values[tid] += values[tid + off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 32u) {
        float value = values[tid];
        for (uint off = 16u; off; off >>= 1u) {
            const float other = simd_shuffle_down(value, off);
            if (lane < off) {
                value += other;
            }
        }
        if (lane == 0u) {
            values[0] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return values[0];
}

kernel void attention_decode_scores_gqa4_simdreduce(
    device const bfloat* q [[buffer(0)]], device const bfloat* keys [[buffer(1)]],
    constant uint& context [[buffer(2)]], constant uint& capacity [[buffer(3)]],
    constant uint& part [[buffer(4)]], device float* weights [[buffer(5)]],
    constant uint& nparts [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
    uint3 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[4][256];
    threadgroup float red[4][256];
    const uint tid = tpos.x;
    const uint cohort = tpos.y;
    const uint sg = tid >> 5u;
    const uint kv = tg.x >> 1u;
    const uint head_base = kv * 8u + (tg.x & 1u) * 4u;
    const uint n = context + 1u;
    const uint start = tg.y * part;
    const uint count = start < n ? min(n - start, part) : 0u;

    for (uint t = cohort * 8u + sg; t < count; t += 32u) {
        float dot0 = 0.0f;
        float dot1 = 0.0f;
        float dot2 = 0.0f;
        float dot3 = 0.0f;
        for (uint d = lane; d < kAttnHeadDimension; d += 32u) {
            const float key_value =
                float(keys[(kv * capacity + start + t) * kAttnHeadDimension + d]);
            dot0 += float(q[(head_base + 0u) * kAttnHeadDimension + d]) * key_value;
            dot1 += float(q[(head_base + 1u) * kAttnHeadDimension + d]) * key_value;
            dot2 += float(q[(head_base + 2u) * kAttnHeadDimension + d]) * key_value;
            dot3 += float(q[(head_base + 3u) * kAttnHeadDimension + d]) * key_value;
        }
        dot0 = simd_sum(dot0);
        dot1 = simd_sum(dot1);
        dot2 = simd_sum(dot2);
        dot3 = simd_sum(dot3);
        if (lane == 0u) {
            scores[0][t] = dot0 * kAttnScale;
            scores[1][t] = dot1 * kAttnScale;
            scores[2][t] = dot2 * kAttnScale;
            scores[3][t] = dot3 * kAttnScale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint head = head_base + cohort;
    red[cohort][tid] = tid < count ? scores[cohort][tid] : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float part_max =
        reduce_max_256_simd_tail_exact(red[cohort], tid, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    red[cohort][tid] =
        tid < count ? exp(scores[cohort][tid] - part_max) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < count) {
        scores[cohort][tid] = red[cohort][tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float part_sum =
        reduce_sum_256_simd_tail_exact(red[cohort], tid, lane);
    device float* dst = weights + (head * nparts + tg.y) * 258u;
    dst[tid] = tid < count ? scores[cohort][tid] : 0.0f;
    if (tid == 0u) {
        dst[256] = part_max;
        dst[257] = part_sum;
    }
}

kernel void attention_decode_scores_gqa8(
    device const bfloat* q [[buffer(0)]], device const bfloat* keys [[buffer(1)]],
    constant uint& context [[buffer(2)]], constant uint& capacity [[buffer(3)]],
    constant uint& part [[buffer(4)]], device float* weights [[buffer(5)]],
    constant uint& nparts [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
    uint3 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    // One KV-head group shares each key load across all eight related query
    // heads. The four 256-thread cohorts retain the GQA4 key-to-simdgroup
    // mapping and each cohort runs the unchanged 256-leaf reduction for two
    // disjoint heads in sequence.
    if (tg.x >= kAttnKvHeads) {
        return;
    }
    threadgroup float scores[8][256];
    threadgroup float red[8][256];
    const uint tid = tpos.x;
    const uint cohort = tpos.y;
    const uint sg = tid >> 5u;
    const uint kv = tg.x;
    const uint head_base = kv * 8u;
    const uint n = context + 1u;
    const uint start = tg.y * part;
    const uint count = start < n ? min(n - start, part) : 0u;

    for (uint t = cohort * 8u + sg; t < count; t += 32u) {
        float dot0 = 0.0f;
        float dot1 = 0.0f;
        float dot2 = 0.0f;
        float dot3 = 0.0f;
        float dot4 = 0.0f;
        float dot5 = 0.0f;
        float dot6 = 0.0f;
        float dot7 = 0.0f;
        for (uint d = lane; d < kAttnHeadDimension; d += 32u) {
            const float key_value =
                float(keys[(kv * capacity + start + t) * kAttnHeadDimension + d]);
            dot0 += float(q[(head_base + 0u) * kAttnHeadDimension + d]) * key_value;
            dot1 += float(q[(head_base + 1u) * kAttnHeadDimension + d]) * key_value;
            dot2 += float(q[(head_base + 2u) * kAttnHeadDimension + d]) * key_value;
            dot3 += float(q[(head_base + 3u) * kAttnHeadDimension + d]) * key_value;
            dot4 += float(q[(head_base + 4u) * kAttnHeadDimension + d]) * key_value;
            dot5 += float(q[(head_base + 5u) * kAttnHeadDimension + d]) * key_value;
            dot6 += float(q[(head_base + 6u) * kAttnHeadDimension + d]) * key_value;
            dot7 += float(q[(head_base + 7u) * kAttnHeadDimension + d]) * key_value;
        }
        dot0 = simd_sum(dot0);
        dot1 = simd_sum(dot1);
        dot2 = simd_sum(dot2);
        dot3 = simd_sum(dot3);
        dot4 = simd_sum(dot4);
        dot5 = simd_sum(dot5);
        dot6 = simd_sum(dot6);
        dot7 = simd_sum(dot7);
        if (lane == 0u) {
            scores[0][t] = dot0 * kAttnScale;
            scores[1][t] = dot1 * kAttnScale;
            scores[2][t] = dot2 * kAttnScale;
            scores[3][t] = dot3 * kAttnScale;
            scores[4][t] = dot4 * kAttnScale;
            scores[5][t] = dot5 * kAttnScale;
            scores[6][t] = dot6 * kAttnScale;
            scores[7][t] = dot7 * kAttnScale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint phase = 0u; phase < 2u; ++phase) {
        const uint local_head = cohort + phase * 4u;
        const uint head = head_base + local_head;
        red[local_head][tid] =
            tid < count ? scores[local_head][tid] : -INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 128u; off; off >>= 1u) {
            if (tid < off) {
                red[local_head][tid] =
                    max(red[local_head][tid],
                        red[local_head][tid + off]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float part_max = red[local_head][0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[local_head][tid] =
            tid < count
                ? exp(scores[local_head][tid] - part_max)
                : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < count) {
            scores[local_head][tid] = red[local_head][tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 128u; off; off >>= 1u) {
            if (tid < off) {
                red[local_head][tid] +=
                    red[local_head][tid + off];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        device float* dst =
            weights + (head * nparts + tg.y) * 258u;
        dst[tid] =
            tid < count ? scores[local_head][tid] : 0.0f;
        if (tid == 0u) {
            dst[256] = part_max;
            dst[257] = red[local_head][0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void attention_decode_scores_values_gqa8(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* keys [[buffer(1)]],
    device const bfloat* values [[buffer(2)]],
    constant uint& context [[buffer(3)]],
    constant uint& capacity [[buffer(4)]],
    constant uint& part [[buffer(5)]],
    device float* partials [[buffer(6)]],
    constant uint& nparts [[buffer(7)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tpos [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    // Exact fused long-context treatment. One group owns all eight query
    // heads related to one KV head. It retains the permanent score kernel's
    // 32 disjoint key simdgroups and exact 256-element softmax trees, then
    // consumes those normalized weights immediately while one staged V tile
    // serves all eight heads. No global probability record is materialized.
    if (tg.x >= kAttnKvHeads) {
        return;
    }
    threadgroup float scores[8][256];
    // Reduction and value staging are disjoint phases. One 16-KiB plane owns
    // both views explicitly, reducing total declared storage from 28 KiB to
    // 24 KiB without changing either phase's indexing or arithmetic.
    threadgroup uint shared_workspace[4096];
    threadgroup float* red =
        reinterpret_cast<threadgroup float*>(shared_workspace);
    threadgroup bfloat* value_tile =
        reinterpret_cast<threadgroup bfloat*>(shared_workspace);
    const uint tid = tpos.x;
    const uint cohort = tpos.y;
    const uint sg = tid >> 5u;
    const uint kv = tg.x;
    const uint head_base = kv * 8u;
    const uint n = context + 1u;
    const uint start = tg.y * part;
    const uint count = start < n ? min(n - start, part) : 0u;

    for (uint t = cohort * 8u + sg; t < count; t += 32u) {
        float dot0 = 0.0f;
        float dot1 = 0.0f;
        float dot2 = 0.0f;
        float dot3 = 0.0f;
        float dot4 = 0.0f;
        float dot5 = 0.0f;
        float dot6 = 0.0f;
        float dot7 = 0.0f;
        for (uint d = lane; d < kAttnHeadDimension; d += 32u) {
            const float key_value =
                float(keys[(kv * capacity + start + t) *
                               kAttnHeadDimension +
                           d]);
            dot0 +=
                float(q[(head_base + 0u) * kAttnHeadDimension + d]) *
                key_value;
            dot1 +=
                float(q[(head_base + 1u) * kAttnHeadDimension + d]) *
                key_value;
            dot2 +=
                float(q[(head_base + 2u) * kAttnHeadDimension + d]) *
                key_value;
            dot3 +=
                float(q[(head_base + 3u) * kAttnHeadDimension + d]) *
                key_value;
            dot4 +=
                float(q[(head_base + 4u) * kAttnHeadDimension + d]) *
                key_value;
            dot5 +=
                float(q[(head_base + 5u) * kAttnHeadDimension + d]) *
                key_value;
            dot6 +=
                float(q[(head_base + 6u) * kAttnHeadDimension + d]) *
                key_value;
            dot7 +=
                float(q[(head_base + 7u) * kAttnHeadDimension + d]) *
                key_value;
        }
        dot0 = simd_sum(dot0);
        dot1 = simd_sum(dot1);
        dot2 = simd_sum(dot2);
        dot3 = simd_sum(dot3);
        dot4 = simd_sum(dot4);
        dot5 = simd_sum(dot5);
        dot6 = simd_sum(dot6);
        dot7 = simd_sum(dot7);
        if (lane == 0u) {
            scores[0][t] = dot0 * kAttnScale;
            scores[1][t] = dot1 * kAttnScale;
            scores[2][t] = dot2 * kAttnScale;
            scores[3][t] = dot3 * kAttnScale;
            scores[4][t] = dot4 * kAttnScale;
            scores[5][t] = dot5 * kAttnScale;
            scores[6][t] = dot6 * kAttnScale;
            scores[7][t] = dot7 * kAttnScale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Four cohorts normalize all eight heads in two waves. Reusing the
    // four-head reduction plane keeps total threadgroup storage at 28 KiB.
    for (uint phase = 0u; phase < 2u; ++phase) {
        const uint local_head = cohort + phase * 4u;
        const uint head = head_base + local_head;
        red[cohort * 256u + tid] =
            tid < count ? scores[local_head][tid] : -INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 128u; off; off >>= 1u) {
            if (tid < off) {
                red[cohort * 256u + tid] =
                    max(red[cohort * 256u + tid],
                        red[cohort * 256u + tid + off]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float part_max = red[cohort * 256u];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[cohort * 256u + tid] =
            tid < count
                ? exp(scores[local_head][tid] - part_max)
                : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < count) {
            scores[local_head][tid] =
                red[cohort * 256u + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 128u; off; off >>= 1u) {
            if (tid < off) {
                red[cohort * 256u + tid] +=
                    red[cohort * 256u + tid + off];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) {
            device float* dst =
                partials + (head * nparts + tg.y) * 258u;
            dst[256] = part_max;
            dst[257] = red[cohort * 256u];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint flat = tid + 256u * cohort;
    const uint local_head = flat >> 7u;
    const uint output_dim = flat & 127u;
    const uint head = head_base + local_head;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint vb = 0u; vb < count; vb += 32u) {
        const uint vn = min(count - vb, 32u);
        const uint elems = vn * kAttnHeadDimension;
        device const bfloat* src =
            values +
            (kv * capacity + start + vb) * kAttnHeadDimension;
        for (uint i = flat; i < elems; i += 1024u) {
            value_tile[i] = src[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p = 0u; p < vn; ++p) {
            const float weight = scores[local_head][vb + p];
            acc0 +=
                weight *
                float(value_tile[p * kAttnHeadDimension + output_dim]);
            acc1 +=
                weight *
                float(value_tile[p * kAttnHeadDimension + output_dim +
                                 128u]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float* dst =
        partials + (head * nparts + tg.y) * 258u;
    dst[output_dim] = acc0;
    dst[output_dim + 128u] = acc1;
}

// Tatara adaptation of the MLX v0.32 q=1 vector mechanism (MIT): sequence
// parallelism is deliberately independent across the eight GQA query heads.
// Each simdgroup owns one head and carries an online softmax/value numerator;
// there is no cross-head communication and therefore no threadgroup barrier.
kernel void attention_decode_vector_2pass_part(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* keys [[buffer(1)]],
    device const bfloat* values [[buffer(2)]],
    device bfloat* partials [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    constant uint& context [[buffer(6)]],
    constant uint& capacity [[buffer(7)]],
    constant uint& blocks [[buffer(8)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tpos [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint kv = tg.x;
    const uint block = tg.y;
    const uint local_head = tpos.y;
    if (kv >= kAttnKvHeads || block >= blocks || local_head >= 8u) {
        return;
    }

    const uint head = kv * 8u + local_head;
    const uint dimension_base = lane * 8u;
    float query[8];
    float numerator[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f};
    for (uint i = 0u; i < 8u; ++i) {
        query[i] =
            kAttnScale *
            float(q[head * kAttnHeadDimension + dimension_base + i]);
    }

    float part_max = -INFINITY;
    float part_sum = 0.0f;
    const uint n = context + 1u;
    for (uint position = block; position < n; position += blocks) {
        const ulong kv_base =
            (ulong(kv) * ulong(capacity) + ulong(position)) *
            ulong(kAttnHeadDimension);
        float score = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            score += query[i] *
                     float(keys[kv_base + ulong(dimension_base + i)]);
        }
        score = simd_sum(score);

        const float new_max = max(part_max, score);
        const float old_factor = fast::exp(part_max - new_max);
        const float score_factor = fast::exp(score - new_max);
        part_max = new_max;
        part_sum = part_sum * old_factor + score_factor;
        for (uint i = 0u; i < 8u; ++i) {
            numerator[i] =
                numerator[i] * old_factor +
                score_factor *
                    float(values[kv_base + ulong(dimension_base + i)]);
        }
    }

    const ulong record =
        (ulong(head) * ulong(blocks) + ulong(block)) *
        ulong(kAttnHeadDimension);
    for (uint i = 0u; i < 8u; ++i) {
        partials[record + ulong(dimension_base + i)] =
            static_cast<bfloat>(numerator[i]);
    }
    if (lane == 0u) {
        const ulong scalar =
            ulong(head) * ulong(blocks) + ulong(block);
        sums[scalar] = part_sum;
        maxs[scalar] = part_max;
    }
}

kernel void attention_decode_vector_2pass_combine(
    device const bfloat* partials [[buffer(0)]],
    device const float* sums [[buffer(1)]],
    device const float* maxs [[buffer(2)]],
    device const bfloat* gate [[buffer(3)]],
    constant uint& blocks [[buffer(4)]],
    device bfloat* out [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    // One 32x32 transpose reduces 32 sequence-block residues while preserving
    // eight contiguous output dimensions per lane.
    threadgroup float transpose[32u * 32u];
    float numerator[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f};
    device const float* head_sums =
        sums + ulong(head) * ulong(blocks);
    device const float* head_maxs =
        maxs + ulong(head) * ulong(blocks);

    float global_max = -INFINITY;
    for (uint block_base = 0u; block_base < blocks; block_base += 32u) {
        global_max =
            max(global_max, head_maxs[block_base + lane]);
    }
    global_max = simd_max(global_max);

    float denominator = 0.0f;
    for (uint block_base = 0u; block_base < blocks; block_base += 32u) {
        const uint block = block_base + lane;
        denominator +=
            fast::exp(head_maxs[block] - global_max) *
            head_sums[block];
    }
    denominator = simd_sum(denominator);

    const uint dimension_base = lane * 8u;
    for (uint block = simdgroup; block < blocks; block += 32u) {
        const float factor =
            fast::exp(head_maxs[block] - global_max);
        const ulong record =
            (ulong(head) * ulong(blocks) + ulong(block)) *
                ulong(kAttnHeadDimension) +
            ulong(dimension_base);
        for (uint i = 0u; i < 8u; ++i) {
            numerator[i] +=
                factor * float(partials[record + ulong(i)]);
        }
    }

    for (uint i = 0u; i < 8u; ++i) {
        transpose[lane * 32u + simdgroup] = numerator[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float value =
            simd_sum(transpose[simdgroup * 32u + lane]) /
            denominator;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) {
            const uint dimension = simdgroup * 8u + i;
            const float g =
                float(gate[head * kAttnHeadDimension + dimension]);
            value *= 1.0f / (1.0f + exp(-g));
            out[head * kAttnHeadDimension + dimension] =
                static_cast<bfloat>(value);
        }
    }
}

kernel void attention_decode_values_gqa8(
    device const float* weights [[buffer(0)]], device const bfloat* values [[buffer(1)]],
    constant uint& context [[buffer(2)]], constant uint& capacity [[buffer(3)]],
    constant uint& part [[buffer(4)]], device float* partials [[buffer(5)]],
    constant uint& nparts [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
    uint3 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    // All eight query heads of one KV group reuse the same 32-key value
    // tile; four simdgroups per head, two output dimensions per thread,
    // ascending-key accumulation preserved exactly.
    threadgroup bfloat value_tile[32u * 256u];
    const uint tid = tpos.x + 32u * (tpos.y + 4u * tpos.z);
    const uint kv = tg.x;
    const uint head = kv * 8u + tpos.z;
    const uint output_dim = tpos.y * 32u + lane;
    const uint n = context + 1u;
    const uint start = tg.y * part;
    const uint count = start < n ? min(n - start, part) : 0u;
    device const float* w = weights + (head * nparts + tg.y) * 258u;
    float acc[2] = {0.0f, 0.0f};
    for (uint vb = 0u; vb < count; vb += 32u) {
        const uint vn = min(count - vb, 32u);
        const uint elems = vn * 256u;
        device const bfloat* src = values + (kv * capacity + start + vb) * kAttnHeadDimension;
        for (uint i = tid; i < elems; i += 1024u) {
            value_tile[i] = src[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p = 0u; p < vn; ++p) {
            const float wp = w[vb + p];
            acc[0] += wp * float(value_tile[p * 256u + output_dim]);
            acc[1] += wp * float(value_tile[p * 256u + output_dim + 128u]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float* dst = partials + (head * nparts + tg.y) * 258u;
    dst[output_dim] = acc[0];
    dst[output_dim + 128u] = acc[1];
    if (tpos.y == 0u && lane == 0u) {
        dst[256] = w[256];
        dst[257] = w[257];
    }
}

kernel void attention_decode_values_gqa8_t512(
    device const float* weights [[buffer(0)]], device const bfloat* values [[buffer(1)]],
    constant uint& context [[buffer(2)]], constant uint& capacity [[buffer(3)]],
    constant uint& part [[buffer(4)]], device float* partials [[buffer(5)]],
    constant uint& nparts [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
    uint3 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    // Same all-eight-head value sharing and ascending-key arithmetic as the
    // 1,024-thread control. Two output cohorts per head give 512 threads; each
    // thread owns four dimensions separated by 64.
    threadgroup bfloat value_tile[32u * 256u];
    const uint tid = tpos.x + 32u * (tpos.y + 2u * tpos.z);
    const uint kv = tg.x;
    const uint head = kv * 8u + tpos.z;
    const uint output_dim = tpos.y * 32u + lane;
    const uint n = context + 1u;
    const uint start = tg.y * part;
    const uint count = start < n ? min(n - start, part) : 0u;
    device const float* w = weights + (head * nparts + tg.y) * 258u;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint vb = 0u; vb < count; vb += 32u) {
        const uint vn = min(count - vb, 32u);
        const uint elems = vn * 256u;
        device const bfloat* src =
            values + (kv * capacity + start + vb) * kAttnHeadDimension;
        for (uint i = tid; i < elems; i += 512u) {
            value_tile[i] = src[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p = 0u; p < vn; ++p) {
            const float wp = w[vb + p];
            acc[0] += wp * float(value_tile[p * 256u + output_dim]);
            acc[1] += wp * float(value_tile[p * 256u + output_dim + 64u]);
            acc[2] += wp * float(value_tile[p * 256u + output_dim + 128u]);
            acc[3] += wp * float(value_tile[p * 256u + output_dim + 192u]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float* dst = partials + (head * nparts + tg.y) * 258u;
    dst[output_dim] = acc[0];
    dst[output_dim + 64u] = acc[1];
    dst[output_dim + 128u] = acc[2];
    dst[output_dim + 192u] = acc[3];
    if (tpos.y == 0u && lane == 0u) {
        dst[256] = w[256];
        dst[257] = w[257];
    }
}

kernel void attention_decode_combine(device const float* partials [[buffer(0)]],
                                     device const bfloat* gate [[buffer(1)]],
                                     constant uint& nparts [[buffer(2)]],
                                     device bfloat* out [[buffer(3)]],
                                     uint head [[threadgroup_position_in_grid]],
                                     uint tid [[thread_position_in_threadgroup]]) {
    device const float* base = partials + head * nparts * 258u;
    float global_max = -INFINITY;
    for (uint p = 0; p < nparts; ++p) {
        global_max = max(global_max, base[p * 258u + 256u]);
    }
    float acc = 0.0f;
    float denom = 0.0f;
    for (uint p = 0; p < nparts; ++p) {
        const float weight = exp(base[p * 258u + 256u] - global_max);
        acc += weight * base[p * 258u + tid];
        denom += weight * base[p * 258u + 257u];
    }
    float value = acc / denom;
    const float g = float(gate[head * kAttnHeadDimension + tid]);
    value *= 1.0f / (1.0f + exp(-g));
    out[head * kAttnHeadDimension + tid] = static_cast<bfloat>(value);
}

kernel void
attn_project_ms(device const bfloat* input [[buffer(0)]], device const uint* qg_words [[buffer(1)]],
                device const bfloat* qg_scales [[buffer(2)]],
                device const bfloat* qg_biases [[buffer(3)]],
                device const uint* k_words [[buffer(4)]],
                device const bfloat* k_scales [[buffer(5)]],
                device const bfloat* k_biases [[buffer(6)]],
                device const uint* v_words [[buffer(7)]],
                device const bfloat* v_scales [[buffer(8)]],
                device const bfloat* v_biases [[buffer(9)]],
                device bfloat* projection [[buffer(10)]],
                constant uint& row_count [[buffer(11)]],
                constant uint& row_tile [[buffer(12)]],
                uint2 tid2 [[thread_position_in_grid]],
                uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tid2.x >> 5u;
    const uint row_base = tid2.y * row_tile;
    if (row >= kAttnProjectionRows || row_base >= row_count) {
        return;
    }
    const uint tile_rows = min(row_tile, row_count - row_base);
    device const uint* words;
    device const bfloat* scales;
    device const bfloat* biases;
    uint local_row;
    if (row < kAttnQGateRows) {
        words = qg_words;
        scales = qg_scales;
        biases = qg_biases;
        local_row = row;
    } else if (row < kAttnVRowOffset) {
        words = k_words;
        scales = k_scales;
        biases = k_biases;
        local_row = row - kAttnQGateRows;
    } else {
        words = v_words;
        scales = v_scales;
        biases = v_biases;
        local_row = row - kAttnVRowOffset;
    }
    const ulong word_row = ulong(local_row) * ulong(kQuantWordsPerRow);
    const ulong group_row = ulong(local_row) * ulong(kGroupsPerRow);
    float batch[kQ4BatchRowsMax];
    q4_dot_ms(input + ulong(row_base) * ulong(kHiddenDimension), ulong(kHiddenDimension),
              tile_rows, words + word_row, scales + group_row, biases + group_row,
              kHiddenDimension, lane, batch);
    if (lane == 0u) {
        for (uint r = 0u; r < tile_rows; ++r) {
            projection[ulong(row_base + r) * ulong(kAttnProjectionRows) + row] =
                static_cast<bfloat>(batch[r]);
        }
    }
}


kernel void
attn_project_ms_v2(device const bfloat* input [[buffer(0)]], device const uint* qg_words [[buffer(1)]],
                device const bfloat* qg_scales [[buffer(2)]],
                device const bfloat* qg_biases [[buffer(3)]],
                device const uint* k_words [[buffer(4)]],
                device const bfloat* k_scales [[buffer(5)]],
                device const bfloat* k_biases [[buffer(6)]],
                device const uint* v_words [[buffer(7)]],
                device const bfloat* v_scales [[buffer(8)]],
                device const bfloat* v_biases [[buffer(9)]],
                device bfloat* projection [[buffer(10)]],
                constant uint& row_count [[buffer(11)]],
                constant uint& row_tile [[buffer(12)]],
                uint2 tid2 [[thread_position_in_grid]],
                uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tid2.x >> 5u;
    const uint row_base = tid2.y * row_tile;
    if (row >= kAttnProjectionRows || row_base >= row_count) {
        return;
    }
    const uint tile_rows = min(row_tile, row_count - row_base);
    device const uint* words;
    device const bfloat* scales;
    device const bfloat* biases;
    uint local_row;
    if (row < kAttnQGateRows) {
        words = qg_words;
        scales = qg_scales;
        biases = qg_biases;
        local_row = row;
    } else if (row < kAttnVRowOffset) {
        words = k_words;
        scales = k_scales;
        biases = k_biases;
        local_row = row - kAttnQGateRows;
    } else {
        words = v_words;
        scales = v_scales;
        biases = v_biases;
        local_row = row - kAttnVRowOffset;
    }
    const ulong word_row = ulong(local_row) * ulong(kQuantWordsPerRow);
    const ulong group_row = ulong(local_row) * ulong(kGroupsPerRow);
    float batch[kQ4BatchRowsMax];
    q4_dot_ms_v2(input + ulong(row_base) * ulong(kHiddenDimension), ulong(kHiddenDimension),
              tile_rows, words + word_row, scales + group_row, biases + group_row,
              kHiddenDimension, lane, batch);
    if (lane == 0u) {
        for (uint r = 0u; r < tile_rows; ++r) {
            projection[ulong(row_base + r) * ulong(kAttnProjectionRows) + row] =
                static_cast<bfloat>(batch[r]);
        }
    }
}
