kernel void attn_qk_rope_blk(
    device const bfloat* projection [[buffer(0)]],
    device const bfloat* query_weight [[buffer(1)]],
    device const bfloat* key_weight [[buffer(2)]],
    device bfloat* query_out [[buffer(3)]],
    device bfloat* gate_out [[buffer(4)]], device bfloat* keys [[buffer(5)]],
    device bfloat* values [[buffer(6)]],
    constant uint& context [[buffer(7)]],
    constant uint& capacity [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 thread_position [[thread_position_in_threadgroup]]) {
    threadgroup float normalized_values[kAttnHeadDimension];
    threadgroup float partials[kSimdgroupWidth];
    const uint head = group.x;
    const uint row = group.y;
    const uint dimension = thread_position.x;
    device const bfloat* projection_row =
        projection + row * kAttnProjectionRows;
    const uint sequence_position = context + row;
    const bool query = head < kAttnQueryHeads;
    const uint local_head = query ? head : head - kAttnQueryHeads;
    const uint base =
        query ? head * (2u * kAttnHeadDimension)
              : kAttnQGateRows + local_head * kAttnHeadDimension;
    const float input = float(projection_row[base + dimension]);
    float total = 0.0f;
    if (dimension < kAttnHeadDimension / 4u) {
        const uint reduce_base = dimension * 4u;
        for (uint i = 0u; i < 4u; ++i) {
            const float value =
                float(projection_row[base + reduce_base + i]);
            total += value * value;
        }
    }
    total = simd_sum(total);
    if (dimension < kSimdgroupWidth) {
        partials[dimension] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (dimension < kAttnHeadDimension / 4u &&
        dimension % kSimdgroupWidth == 0u) {
        partials[dimension / kSimdgroupWidth] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (dimension < kSimdgroupWidth) {
        const float reduced = simd_sum(partials[dimension]);
        if (dimension == 0u) {
            partials[0] = precise::rsqrt(
                reduced / float(kAttnHeadDimension) + kRmsNormEpsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bfloat normalized =
        (query ? query_weight[dimension] : key_weight[dimension]) *
        static_cast<bfloat>(input * partials[0]);
    normalized_values[dimension] = float(normalized);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rotated = normalized_values[dimension];
    if (dimension < 2u * kAttnRopePairs) {
        const uint pair = dimension % kAttnRopePairs;
        const float theta =
            float(sequence_position) /
            pow(kAttnRopeBase, float(pair) / float(kAttnRopePairs));
        const float cosine = cos(theta);
        const float sine = sin(theta);
        rotated =
            dimension < kAttnRopePairs
                ? normalized_values[dimension] * cosine -
                      normalized_values[dimension + kAttnRopePairs] * sine
                : normalized_values[dimension] * cosine +
                      normalized_values[dimension - kAttnRopePairs] * sine;
    }
    if (query) {
        query_out[(row * kAttnQueryHeads + head) *
                      kAttnHeadDimension +
                  dimension] = static_cast<bfloat>(rotated);
        gate_out[(row * kAttnQueryHeads + head) *
                     kAttnHeadDimension +
                 dimension] =
            projection_row[head * (2u * kAttnHeadDimension) +
                           kAttnHeadDimension + dimension];
    } else {
        keys[(local_head * capacity + sequence_position) *
                 kAttnHeadDimension +
             dimension] = static_cast<bfloat>(rotated);
        values[(local_head * capacity + sequence_position) *
                   kAttnHeadDimension +
               dimension] =
            projection_row[kAttnVRowOffset +
                           local_head * kAttnHeadDimension + dimension];
    }
}

// One query/head/partition record contains the unnormalized value numerator,
// local maximum, and local denominator. Query tiling is entirely a host
// schedule: row is local to the tile, while context carries the tile base.
kernel void attention_partial_blk(
    device const bfloat* query [[buffer(0)]],
    device const bfloat* keys [[buffer(1)]],
    device const bfloat* values [[buffer(2)]],
    constant uint& context [[buffer(3)]],
    constant uint& capacity [[buffer(4)]],
    constant uint& partition [[buffer(5)]],
    device float* partials [[buffer(6)]],
    constant uint& partition_count [[buffer(7)]],
    constant uint& block [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[kPrefillAttentionPartition];
    threadgroup float reduction[kPrefillAttentionPartition];
    threadgroup float carry[2];
    const uint tid = thread_position.x;
    const uint head = group.x;
    const uint row = group.z;
    const uint simdgroup = tid >> 5u;
    const uint visible = context + row + 1u;
    const uint start = group.y * partition;
    const uint end = min(visible, start + partition);
    const uint key_value_head = head / kAttnHeadsPerKv;
    device const bfloat* query_row =
        query + (row * kAttnQueryHeads + head) * kAttnHeadDimension;
    if (tid == 0u) {
        carry[0] = -INFINITY;
        carry[1] = 0.0f;
    }
    float accumulator = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint tile = start; tile < end;
         tile += kPrefillAttentionPartition) {
        const uint count =
            min(end - tile, kPrefillAttentionPartition);
        for (uint position = simdgroup; position < count;
             position += kPrefillAttentionPartition / kSimdgroupWidth) {
            float dot_product = 0.0f;
            for (uint dimension = lane; dimension < kAttnHeadDimension;
                 dimension += kSimdgroupWidth) {
                dot_product +=
                    float(query_row[dimension]) *
                    float(keys[(key_value_head * capacity + tile + position) *
                                   kAttnHeadDimension +
                               dimension]);
            }
            dot_product = simd_sum(dot_product);
            if (lane == 0u) {
                scores[position] = dot_product * kAttnScale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        reduction[tid] =
            tid < count ? scores[tid] : -INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = kPrefillAttentionPartition / 2u; offset;
             offset >>= 1u) {
            if (tid < offset) {
                reduction[tid] =
                    max(reduction[tid], reduction[tid + offset]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float previous_max = carry[0];
        const float next_max = max(previous_max, reduction[0]);
        const float rescale =
            previous_max == -INFINITY ? 0.0f
                                      : exp(previous_max - next_max);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        reduction[tid] =
            tid < count ? exp(scores[tid] - next_max) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < count) {
            scores[tid] = reduction[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = kPrefillAttentionPartition / 2u; offset;
             offset >>= 1u) {
            if (tid < offset) {
                reduction[tid] += reduction[tid + offset];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float tile_sum = reduction[0];
        accumulator *= rescale;
        for (uint position = 0u; position < count; ++position) {
            accumulator +=
                scores[position] *
                float(values[(key_value_head * capacity + tile + position) *
                                 kAttnHeadDimension +
                             tid]);
        }
        if (tid == 0u) {
            carry[1] = carry[1] * rescale + tile_sum;
            carry[0] = next_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float* destination =
        partials +
        ((head * block + row) * partition_count + group.y) *
            kPrefillAttentionRecordFloats;
    destination[tid] = accumulator;
    if (tid == 0u) {
        destination[kAttnHeadDimension] = carry[0];
        destination[kAttnHeadDimension + 1u] = carry[1];
    }
}

kernel void attention_combine_blk(
    device const float* partials [[buffer(0)]],
    device const bfloat* gate [[buffer(1)]],
    constant uint& partition_count [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& block [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 thread_position [[thread_position_in_threadgroup]]) {
    const uint head = group.x;
    const uint row = group.y;
    const uint dimension = thread_position.x;
    device const float* base =
        partials + (head * block + row) * partition_count *
                       kPrefillAttentionRecordFloats;
    float global_max = -INFINITY;
    for (uint partition = 0u; partition < partition_count; ++partition) {
        global_max =
            max(global_max,
                base[partition * kPrefillAttentionRecordFloats +
                     kAttnHeadDimension]);
    }
    float accumulator = 0.0f;
    float denominator = 0.0f;
    for (uint partition = 0u; partition < partition_count; ++partition) {
        const float local_max =
            base[partition * kPrefillAttentionRecordFloats +
                 kAttnHeadDimension];
        const float weight =
            local_max == -INFINITY ? 0.0f
                                   : exp(local_max - global_max);
        accumulator +=
            weight *
            base[partition * kPrefillAttentionRecordFloats + dimension];
        denominator +=
            weight *
            base[partition * kPrefillAttentionRecordFloats +
                 kAttnHeadDimension + 1u];
    }
    float value = accumulator / denominator;
    const float gate_value =
        float(gate[(row * kAttnQueryHeads + head) *
                       kAttnHeadDimension +
                   dimension]);
    value *= 1.0f / (1.0f + exp(-gate_value));
    output[row * kAttnQueryHeads * kAttnHeadDimension +
           head * kAttnHeadDimension + dimension] =
        static_cast<bfloat>(value);
}

static_assert(
    kPrefillStagedAttentionQueryTileRows == 16u &&
        kPrefillStagedAttentionKeyTileColumns == 32u &&
        kPrefillStagedAttentionOutputTileColumns == 32u,
    "staged attention requires 16x32 matrix tiles");
static_assert(
    kPrefillStagedAttentionSimdgroups == 2u &&
        kPrefillStagedAttentionThreads ==
            kPrefillStagedAttentionSimdgroups * kSimdgroupWidth,
    "staged attention requires two simdgroups");
static_assert(
    kPrefillStagedAttentionSoftmaxThreads == 256u &&
        kPrefillStagedAttentionSoftmaxThreads %
                kSimdgroupWidth ==
            0u,
    "staged attention softmax requires complete simdgroups");
static_assert(
    kAttnHeadDimension % 8u == 0u &&
        kAttnHeadDimension %
                kPrefillStagedAttentionOutputTileColumns ==
            0u,
    "staged attention requires complete matrix fragments");

inline uint2 prefill_staged_attention_fragment_coordinate(uint lane) {
    const uint quad = lane >> 2u;
    const uint row = (quad & 4u) + ((lane >> 1u) & 3u);
    const uint column =
        (quad & 2u) * 2u + (lane & 1u) * 2u;
    return uint2(row, column);
}

// S[m,h,n] = scale * Q[m,h,:] K[kv(h),n,:]^T. The full visible
// key extent is staged because the following in-place causal softmax zeros
// future positions before the value GEMM.
kernel void attention_staged_scores_blk(
    device const bfloat* query [[buffer(0)]],
    device const bfloat* keys [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& visible [[buffer(3)]],
    constant uint& capacity [[buffer(4)]],
    constant uint& block [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroup_width [[threads_per_simdgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    if (simdgroup_width != kSimdgroupWidth ||
        threadgroup_shape.x !=
            kPrefillStagedAttentionThreads ||
        threadgroup_shape.y != 1u ||
        threadgroup_shape.z != 1u ||
        visible == 0u || visible > capacity || block == 0u) {
        return;
    }
    const uint head = group.z % kAttnQueryHeads;
    const uint row_begin =
        (group.z / kAttnQueryHeads) *
        kPrefillStagedAttentionQueryTileRows;
    const uint key_begin =
        group.x * kPrefillStagedAttentionKeyTileColumns +
        simdgroup * 16u;
    if (row_begin >= block || key_begin >= visible) {
        return;
    }
    const uint key_value_head = head / kAttnHeadsPerKv;
    device const bfloat* key_base =
        keys + ulong(key_value_head) * capacity *
                   kAttnHeadDimension;
    const uint row_count =
        min(block - row_begin,
            kPrefillStagedAttentionQueryTileRows);
    const uint key_count = min(visible - key_begin, 16u);
    simdgroup_float8x8 c00 =
        make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 c01 =
        make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 c10 =
        make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 c11 =
        make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint2 coordinate =
        prefill_staged_attention_fragment_coordinate(lane);
    for (uint reduction = 0u;
         reduction < kAttnHeadDimension;
         reduction += 8u) {
        simdgroup_float8x8 key0;
        simdgroup_float8x8 key1;
        const uint key_row = key_begin + coordinate.x;
        key0.thread_elements()[0] =
            key_row < key_begin + key_count
                ? float(key_base[
                      ulong(key_row) * kAttnHeadDimension +
                      reduction + coordinate.y])
                : 0.0f;
        key0.thread_elements()[1] =
            key_row < key_begin + key_count
                ? float(key_base[
                      ulong(key_row) * kAttnHeadDimension +
                      reduction + coordinate.y + 1u])
                : 0.0f;
        key1.thread_elements()[0] =
            key_row + 8u < key_begin + key_count
                ? float(key_base[
                      ulong(key_row + 8u) *
                          kAttnHeadDimension +
                      reduction + coordinate.y])
                : 0.0f;
        key1.thread_elements()[1] =
            key_row + 8u < key_begin + key_count
                ? float(key_base[
                      ulong(key_row + 8u) *
                          kAttnHeadDimension +
                      reduction + coordinate.y + 1u])
                : 0.0f;

        const uint reduction_row = reduction + coordinate.x;
        simdgroup_float8x8 query0;
        simdgroup_float8x8 query1;
        query0.thread_elements()[0] =
            coordinate.y < row_count
                ? float(query[
                      (ulong(row_begin + coordinate.y) *
                           kAttnQueryHeads +
                       head) *
                              kAttnHeadDimension +
                          reduction_row])
                : 0.0f;
        query0.thread_elements()[1] =
            coordinate.y + 1u < row_count
                ? float(query[
                      (ulong(row_begin + coordinate.y + 1u) *
                           kAttnQueryHeads +
                       head) *
                              kAttnHeadDimension +
                          reduction_row])
                : 0.0f;
        query1.thread_elements()[0] =
            coordinate.y + 8u < row_count
                ? float(query[
                      (ulong(row_begin + coordinate.y + 8u) *
                           kAttnQueryHeads +
                       head) *
                              kAttnHeadDimension +
                          reduction_row])
                : 0.0f;
        query1.thread_elements()[1] =
            coordinate.y + 9u < row_count
                ? float(query[
                      (ulong(row_begin + coordinate.y + 9u) *
                           kAttnQueryHeads +
                       head) *
                              kAttnHeadDimension +
                          reduction_row])
                : 0.0f;
        simdgroup_multiply_accumulate(c00, key0, query0, c00);
        simdgroup_multiply_accumulate(c01, key0, query1, c01);
        simdgroup_multiply_accumulate(c10, key1, query0, c10);
        simdgroup_multiply_accumulate(c11, key1, query1, c11);
    }
    for (uint key_fragment = 0u;
         key_fragment < 2u; ++key_fragment) {
        const uint local_key =
            coordinate.x + key_fragment * 8u;
        if (key_begin + local_key >= visible) {
            continue;
        }
        for (uint row_fragment = 0u;
             row_fragment < 2u; ++row_fragment) {
            const uint local_row =
                coordinate.y + row_fragment * 8u;
            for (uint element = 0u; element < 2u; ++element) {
                if (local_row + element >= row_count) {
                    continue;
                }
                const float value =
                    key_fragment == 0u
                        ? (row_fragment == 0u
                               ? c00.thread_elements()[element]
                               : c01.thread_elements()[element])
                        : (row_fragment == 0u
                               ? c10.thread_elements()[element]
                               : c11.thread_elements()[element]);
                scores[
                    (ulong(row_begin + local_row + element) *
                         kAttnQueryHeads +
                     head) *
                            visible +
                        key_begin + local_key] =
                    value * kAttnScale;
            }
        }
    }
}

kernel void attention_staged_softmax_blk(
    device float* scores [[buffer(0)]],
    constant uint& visible [[buffer(1)]],
    constant uint& query_context [[buffer(2)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroup_width [[threads_per_simdgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    threadgroup float reduction[
        kPrefillStagedAttentionSoftmaxThreads /
        kSimdgroupWidth];
    if (simdgroup_width != kSimdgroupWidth ||
        threadgroup_shape.x !=
            kPrefillStagedAttentionSoftmaxThreads ||
        threadgroup_shape.y != 1u ||
        threadgroup_shape.z != 1u ||
        group.y >= kAttnQueryHeads) {
        return;
    }
    const uint row = group.x;
    const uint head = group.y;
    const uint limit = query_context + row + 1u;
    if (visible == 0u || limit > visible) {
        return;
    }
    device float* row_scores =
        scores +
        (ulong(row) * kAttnQueryHeads + head) * visible;
    float maximum = -INFINITY;
    for (uint key = thread_index; key < limit;
         key += kPrefillStagedAttentionSoftmaxThreads) {
        maximum = max(maximum, row_scores[key]);
    }
    maximum = simd_max(maximum);
    if (lane == 0u) {
        reduction[simdgroup] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        float value =
            lane <
                    kPrefillStagedAttentionSoftmaxThreads /
                        kSimdgroupWidth
                ? reduction[lane]
                : -INFINITY;
        value = simd_max(value);
        if (lane == 0u) {
            reduction[0] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum = reduction[0];
    float denominator = 0.0f;
    for (uint key = thread_index; key < limit;
         key += kPrefillStagedAttentionSoftmaxThreads) {
        const float value = exp(row_scores[key] - maximum);
        row_scores[key] = value;
        denominator += value;
    }
    denominator = simd_sum(denominator);
    // Every simdgroup must finish consuming reduction[0] as the row maximum
    // before simdgroup 0 reuses that slot for its denominator.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        reduction[simdgroup] = denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        float value =
            lane <
                    kPrefillStagedAttentionSoftmaxThreads /
                        kSimdgroupWidth
                ? reduction[lane]
                : 0.0f;
        value = simd_sum(value);
        if (lane == 0u) {
            reduction[0] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inverse = 1.0f / reduction[0];
    for (uint key = thread_index; key < limit;
         key += kPrefillStagedAttentionSoftmaxThreads) {
        row_scores[key] *= inverse;
    }
    for (uint key = limit + thread_index; key < visible;
         key += kPrefillStagedAttentionSoftmaxThreads) {
        row_scores[key] = 0.0f;
    }
}

kernel void attention_staged_softmax_bf16_blk(
    device bfloat* scores [[buffer(0)]],
    constant uint& visible [[buffer(1)]],
    constant uint& query_context [[buffer(2)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroup_width [[threads_per_simdgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    threadgroup float reduction[
        kPrefillStagedAttentionSoftmaxThreads /
        kSimdgroupWidth];
    if (simdgroup_width != kSimdgroupWidth ||
        threadgroup_shape.x !=
            kPrefillStagedAttentionSoftmaxThreads ||
        threadgroup_shape.y != 1u ||
        threadgroup_shape.z != 1u ||
        group.y >= kAttnQueryHeads) {
        return;
    }
    const uint row = group.x;
    const uint head = group.y;
    const uint limit = query_context + row + 1u;
    if (visible == 0u || limit > visible) {
        return;
    }
    device bfloat* row_scores =
        scores +
        (ulong(row) * kAttnQueryHeads + head) * visible;
    float maximum = -INFINITY;
    for (uint key = thread_index; key < limit;
         key += kPrefillStagedAttentionSoftmaxThreads) {
        maximum = max(maximum, float(row_scores[key]) * kAttnScale);
    }
    maximum = simd_max(maximum);
    if (lane == 0u) {
        reduction[simdgroup] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        float value =
            lane <
                    kPrefillStagedAttentionSoftmaxThreads /
                        kSimdgroupWidth
                ? reduction[lane]
                : -INFINITY;
        value = simd_max(value);
        if (lane == 0u) {
            reduction[0] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum = reduction[0];
    float denominator = 0.0f;
    for (uint key = thread_index; key < limit;
         key += kPrefillStagedAttentionSoftmaxThreads) {
        const float value =
            exp(float(row_scores[key]) * kAttnScale - maximum);
        row_scores[key] = static_cast<bfloat>(value);
        denominator += value;
    }
    denominator = simd_sum(denominator);
    // Every simdgroup must finish consuming reduction[0] as the row maximum
    // before simdgroup 0 reuses that slot for its denominator.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        reduction[simdgroup] = denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        float value =
            lane <
                    kPrefillStagedAttentionSoftmaxThreads /
                        kSimdgroupWidth
                ? reduction[lane]
                : 0.0f;
        value = simd_sum(value);
        if (lane == 0u) {
            reduction[0] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inverse = 1.0f / reduction[0];
    for (uint key = thread_index; key < limit;
         key += kPrefillStagedAttentionSoftmaxThreads) {
        row_scores[key] =
            static_cast<bfloat>(float(row_scores[key]) * inverse);
    }
    for (uint key = limit + thread_index; key < visible;
         key += kPrefillStagedAttentionSoftmaxThreads) {
        row_scores[key] = bfloat(0.0f);
    }
}

kernel void attention_staged_values_blk(
    device const float* probabilities [[buffer(0)]],
    device const bfloat* values [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant uint& visible [[buffer(3)]],
    constant uint& capacity [[buffer(4)]],
    constant uint& block [[buffer(5)]],
    device const bfloat* gate [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroup_width [[threads_per_simdgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    if (simdgroup_width != kSimdgroupWidth ||
        threadgroup_shape.x !=
            kPrefillStagedAttentionThreads ||
        threadgroup_shape.y != 1u ||
        threadgroup_shape.z != 1u ||
        visible == 0u || visible > capacity || block == 0u) {
        return;
    }
    const uint head = group.z % kAttnQueryHeads;
    const uint row_begin =
        (group.z / kAttnQueryHeads) *
        kPrefillStagedAttentionQueryTileRows;
    const uint dimension_begin =
        group.x *
            kPrefillStagedAttentionOutputTileColumns +
        simdgroup * 16u;
    if (row_begin >= block ||
        dimension_begin >= kAttnHeadDimension) {
        return;
    }
    const uint key_value_head = head / kAttnHeadsPerKv;
    device const bfloat* value_base =
        values + ulong(key_value_head) * capacity *
                     kAttnHeadDimension;
    const uint row_count =
        min(block - row_begin,
            kPrefillStagedAttentionQueryTileRows);
    simdgroup_float8x8 c00 =
        make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 c01 =
        make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 c10 =
        make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 c11 =
        make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint2 coordinate =
        prefill_staged_attention_fragment_coordinate(lane);
    const uint reduction_end = (visible + 7u) & ~7u;
    for (uint reduction = 0u;
         reduction < reduction_end; reduction += 8u) {
        simdgroup_float8x8 value0;
        simdgroup_float8x8 value1;
        const uint key = reduction + coordinate.y;
        value0.thread_elements()[0] =
            key < visible
                ? float(value_base[
                      ulong(key) * kAttnHeadDimension +
                      dimension_begin + coordinate.x])
                : 0.0f;
        value0.thread_elements()[1] =
            key + 1u < visible
                ? float(value_base[
                      ulong(key + 1u) *
                          kAttnHeadDimension +
                      dimension_begin + coordinate.x])
                : 0.0f;
        value1.thread_elements()[0] =
            key < visible
                ? float(value_base[
                      ulong(key) * kAttnHeadDimension +
                      dimension_begin + coordinate.x + 8u])
                : 0.0f;
        value1.thread_elements()[1] =
            key + 1u < visible
                ? float(value_base[
                      ulong(key + 1u) *
                          kAttnHeadDimension +
                      dimension_begin + coordinate.x + 8u])
                : 0.0f;

        const uint probability_key =
            reduction + coordinate.x;
        simdgroup_float8x8 probability0;
        simdgroup_float8x8 probability1;
        probability0.thread_elements()[0] =
            probability_key < visible &&
                    coordinate.y < row_count
                ? probabilities[
                      (ulong(row_begin + coordinate.y) *
                           kAttnQueryHeads +
                       head) *
                              visible +
                          probability_key]
                : 0.0f;
        probability0.thread_elements()[1] =
            probability_key < visible &&
                    coordinate.y + 1u < row_count
                ? probabilities[
                      (ulong(row_begin + coordinate.y + 1u) *
                           kAttnQueryHeads +
                       head) *
                              visible +
                          probability_key]
                : 0.0f;
        probability1.thread_elements()[0] =
            probability_key < visible &&
                    coordinate.y + 8u < row_count
                ? probabilities[
                      (ulong(row_begin + coordinate.y + 8u) *
                           kAttnQueryHeads +
                       head) *
                              visible +
                          probability_key]
                : 0.0f;
        probability1.thread_elements()[1] =
            probability_key < visible &&
                    coordinate.y + 9u < row_count
                ? probabilities[
                      (ulong(row_begin + coordinate.y + 9u) *
                           kAttnQueryHeads +
                       head) *
                              visible +
                          probability_key]
                : 0.0f;
        simdgroup_multiply_accumulate(
            c00, value0, probability0, c00);
        simdgroup_multiply_accumulate(
            c01, value0, probability1, c01);
        simdgroup_multiply_accumulate(
            c10, value1, probability0, c10);
        simdgroup_multiply_accumulate(
            c11, value1, probability1, c11);
    }
    for (uint dimension_fragment = 0u;
         dimension_fragment < 2u; ++dimension_fragment) {
        const uint local_dimension =
            coordinate.x + dimension_fragment * 8u;
        for (uint row_fragment = 0u;
             row_fragment < 2u; ++row_fragment) {
            const uint local_row =
                coordinate.y + row_fragment * 8u;
            for (uint element = 0u; element < 2u; ++element) {
                if (local_row + element >= row_count) {
                    continue;
                }
                const float value =
                    dimension_fragment == 0u
                        ? (row_fragment == 0u
                               ? c00.thread_elements()[element]
                               : c01.thread_elements()[element])
                        : (row_fragment == 0u
                               ? c10.thread_elements()[element]
                               : c11.thread_elements()[element]);
                const ulong output_index =
                    (ulong(row_begin + local_row + element) *
                         kAttnQueryHeads +
                     head) *
                            kAttnHeadDimension +
                        dimension_begin + local_dimension;
                const float gate_value = float(gate[output_index]);
                output[output_index] = static_cast<bfloat>(
                    value /
                    (1.0f + exp(-gate_value)));
            }
        }
    }
}

static_assert(
    kPrefillStreamingAttentionQueryTileRows == 16u &&
        kPrefillStreamingAttentionKeysPerSimdgroup == 16u &&
        kPrefillStreamingAttentionOutputColumnsPerSimdgroup == 32u,
    "streaming attention requires 16-query, 16-key, 32-output fragments");
static_assert(
    kPrefillStreamingAttentionKeyTileColumns ==
            kPrefillStreamingAttentionSimdgroups *
                kPrefillStreamingAttentionKeysPerSimdgroup &&
        kPrefillStreamingAttentionThreads ==
            kPrefillStreamingAttentionSimdgroups *
                kSimdgroupWidth,
    "streaming attention tile and thread geometry must agree");
static_assert(
    kAttnHeadDimension ==
        kPrefillStreamingAttentionSimdgroups *
            kPrefillStreamingAttentionOutputColumnsPerSimdgroup,
    "streaming attention simdgroups must cover the complete head");
static_assert(
    kPrefillStreamingAttentionThreadgroupMemoryBytes <=
        kMinimumThreadgroupMemoryBytes,
    "streaming attention exceeds the threadgroup-memory guarantee");

inline void prefill_streaming_attention_zero(
    thread simdgroup_matrix<float, 8, 8>& matrix) {
    matrix.thread_elements()[0] = 0.0f;
    matrix.thread_elements()[1] = 0.0f;
}

inline void prefill_streaming_attention_rescale(
    thread simdgroup_matrix<float, 8, 8>& matrix,
    threadgroup const float* row_rescale,
    uint query_fragment,
    uint row_count,
    uint lane) {
    const uint2 coordinate =
        prefill_staged_attention_fragment_coordinate(lane);
    const uint row =
        query_fragment * 8u + coordinate.y;
    matrix.thread_elements()[0] *=
        row < row_count ? row_rescale[row] : 0.0f;
    matrix.thread_elements()[1] *=
        row + 1u < row_count ? row_rescale[row + 1u] : 0.0f;
}

// Flash-style exact-model attention: one threadgroup owns 16 queries for one
// query head. It streams the complete visible key extent in bounded tiles,
// maintains online-softmax state, and keeps the 16xhead-dimension numerator in
// simdgroup accumulators. No score or probability matrix reaches device
// memory. The permanent partial/combine and staged-GEMM families remain the
// numerical and performance controls.
kernel void attention_streaming_blk(
    device const bfloat* query [[buffer(0)]],
    device const bfloat* keys [[buffer(1)]],
    device const bfloat* values [[buffer(2)]],
    device const bfloat* gate [[buffer(3)]],
    device bfloat* output [[buffer(4)]],
    constant uint& visible [[buffer(5)]],
    constant uint& capacity [[buffer(6)]],
    constant uint& block [[buffer(7)]],
    constant uint& query_context [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroup_width [[threads_per_simdgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    threadgroup float scores[
        kPrefillStreamingAttentionQueryTileRows *
        kPrefillStreamingAttentionKeyTileColumns];
    threadgroup float row_maximum[
        kPrefillStreamingAttentionQueryTileRows];
    threadgroup float row_denominator[
        kPrefillStreamingAttentionQueryTileRows];
    threadgroup float row_rescale[
        kPrefillStreamingAttentionQueryTileRows];

    if (simdgroup_width != kSimdgroupWidth ||
        threadgroup_shape.x !=
            kPrefillStreamingAttentionThreads ||
        threadgroup_shape.y != 1u ||
        threadgroup_shape.z != 1u ||
        group.x >= kAttnQueryHeads ||
        visible == 0u || visible > capacity ||
        block == 0u || query_context >= visible) {
        return;
    }
    const uint head = group.x;
    const uint row_begin =
        group.y * kPrefillStreamingAttentionQueryTileRows;
    if (row_begin >= block) {
        return;
    }
    const uint row_count =
        min(block - row_begin,
            kPrefillStreamingAttentionQueryTileRows);
    if (query_context + row_begin + row_count > visible) {
        return;
    }
    const uint key_value_head = head / kAttnHeadsPerKv;
    device const bfloat* key_base =
        keys + ulong(key_value_head) * capacity *
                   kAttnHeadDimension;
    device const bfloat* value_base =
        values + ulong(key_value_head) * capacity *
                     kAttnHeadDimension;
    const uint dimension_begin =
        simdgroup *
        kPrefillStreamingAttentionOutputColumnsPerSimdgroup;

    if (thread_index < kPrefillStreamingAttentionQueryTileRows) {
        row_maximum[thread_index] = -INFINITY;
        row_denominator[thread_index] = 0.0f;
        row_rescale[thread_index] = 0.0f;
    }
    simdgroup_matrix<float, 8, 8> accumulators[8];
    for (uint index = 0u; index < 8u; ++index) {
        prefill_streaming_attention_zero(accumulators[index]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint2 coordinate =
        prefill_staged_attention_fragment_coordinate(lane);
    for (uint key_tile_begin = 0u;
         key_tile_begin < visible;
         key_tile_begin +=
             kPrefillStreamingAttentionKeyTileColumns) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint key_count =
            min(visible - key_tile_begin,
                kPrefillStreamingAttentionKeyTileColumns);
        const uint simdgroup_key_begin =
            key_tile_begin +
            simdgroup *
                kPrefillStreamingAttentionKeysPerSimdgroup;

        simdgroup_matrix<float, 8, 8> score_accumulators[4];
        for (uint index = 0u; index < 4u; ++index) {
            prefill_streaming_attention_zero(
                score_accumulators[index]);
        }
        for (uint reduction = 0u;
             reduction < kAttnHeadDimension;
             reduction += 8u) {
            simdgroup_matrix<float, 8, 8> key0;
            simdgroup_matrix<float, 8, 8> key1;
            const uint key_row =
                simdgroup_key_begin + coordinate.x;
            for (uint element = 0u; element < 2u; ++element) {
                const uint reduction_column =
                    reduction + coordinate.y + element;
                key0.thread_elements()[element] =
                    key_row < visible
                        ? float(key_base[
                              ulong(key_row) *
                                      kAttnHeadDimension +
                                  reduction_column])
                        : 0.0f;
                key1.thread_elements()[element] =
                    key_row + 8u < visible
                        ? float(key_base[
                              ulong(key_row + 8u) *
                                      kAttnHeadDimension +
                                  reduction_column])
                        : 0.0f;
            }

            simdgroup_matrix<float, 8, 8> query0;
            simdgroup_matrix<float, 8, 8> query1;
            const uint reduction_row =
                reduction + coordinate.x;
            for (uint element = 0u; element < 2u; ++element) {
                const uint local_row =
                    coordinate.y + element;
                query0.thread_elements()[element] =
                    local_row < row_count
                        ? float(query[
                              (ulong(row_begin + local_row) *
                                       kAttnQueryHeads +
                                   head) *
                                      kAttnHeadDimension +
                                  reduction_row])
                        : 0.0f;
                query1.thread_elements()[element] =
                    local_row + 8u < row_count
                        ? float(query[
                              (ulong(
                                   row_begin + local_row + 8u) *
                                       kAttnQueryHeads +
                                   head) *
                                      kAttnHeadDimension +
                                  reduction_row])
                        : 0.0f;
            }
            simdgroup_multiply_accumulate(
                score_accumulators[0], key0, query0,
                score_accumulators[0]);
            simdgroup_multiply_accumulate(
                score_accumulators[1], key0, query1,
                score_accumulators[1]);
            simdgroup_multiply_accumulate(
                score_accumulators[2], key1, query0,
                score_accumulators[2]);
            simdgroup_multiply_accumulate(
                score_accumulators[3], key1, query1,
                score_accumulators[3]);
        }

        for (uint key_fragment = 0u;
             key_fragment < 2u; ++key_fragment) {
            const uint local_key =
                simdgroup *
                    kPrefillStreamingAttentionKeysPerSimdgroup +
                coordinate.x + key_fragment * 8u;
            if (local_key >= key_count) {
                continue;
            }
            for (uint query_fragment = 0u;
                 query_fragment < 2u; ++query_fragment) {
                const uint local_row =
                    coordinate.y + query_fragment * 8u;
                for (uint element = 0u;
                     element < 2u; ++element) {
                    const uint row = local_row + element;
                    if (row >= row_count) {
                        continue;
                    }
                    const float score =
                        score_accumulators[
                            key_fragment * 2u +
                            query_fragment]
                            .thread_elements()[element] *
                        kAttnScale;
                    const uint absolute_key =
                        key_tile_begin + local_key;
                    const uint causal_limit =
                        query_context + row_begin + row + 1u;
                    scores[
                        row *
                            kPrefillStreamingAttentionKeyTileColumns +
                        local_key] =
                        absolute_key < causal_limit
                            ? score
                            : -INFINITY;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint row = simdgroup; row < row_count;
             row += kPrefillStreamingAttentionSimdgroups) {
            threadgroup float* row_scores =
                scores +
                row *
                    kPrefillStreamingAttentionKeyTileColumns;
            float tile_maximum = -INFINITY;
            for (uint key = lane; key < key_count;
                 key += kSimdgroupWidth) {
                tile_maximum =
                    max(tile_maximum, row_scores[key]);
            }
            tile_maximum = simd_max(tile_maximum);
            const float previous_maximum = row_maximum[row];
            const float next_maximum =
                max(previous_maximum, tile_maximum);
            const float rescale =
                previous_maximum == -INFINITY
                    ? 0.0f
                    : exp(previous_maximum - next_maximum);
            float tile_denominator = 0.0f;
            for (uint key = lane; key < key_count;
                 key += kSimdgroupWidth) {
                const float weight =
                    row_scores[key] == -INFINITY
                        ? 0.0f
                        : exp(
                              row_scores[key] -
                              next_maximum);
                row_scores[key] = weight;
                tile_denominator += weight;
            }
            tile_denominator =
                simd_sum(tile_denominator);
            if (lane == 0u) {
                row_maximum[row] = next_maximum;
                row_denominator[row] =
                    row_denominator[row] * rescale +
                    tile_denominator;
                row_rescale[row] = rescale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint output_fragment = 0u;
             output_fragment < 4u; ++output_fragment) {
            for (uint query_fragment = 0u;
                 query_fragment < 2u; ++query_fragment) {
                prefill_streaming_attention_rescale(
                    accumulators[
                        output_fragment * 2u +
                        query_fragment],
                    row_rescale, query_fragment, row_count,
                    lane);
            }
        }

        const uint reduction_end =
            (key_count + 7u) & ~7u;
        for (uint reduction = 0u;
             reduction < reduction_end;
             reduction += 8u) {
            simdgroup_matrix<float, 8, 8>
                probability0;
            simdgroup_matrix<float, 8, 8>
                probability1;
            const uint probability_key =
                reduction + coordinate.x;
            for (uint element = 0u; element < 2u;
                 ++element) {
                const uint local_row =
                    coordinate.y + element;
                probability0.thread_elements()[element] =
                    probability_key < key_count &&
                            local_row < row_count
                        ? scores[
                              local_row *
                                      kPrefillStreamingAttentionKeyTileColumns +
                                  probability_key]
                        : 0.0f;
                probability1.thread_elements()[element] =
                    probability_key < key_count &&
                            local_row + 8u < row_count
                        ? scores[
                              (local_row + 8u) *
                                      kPrefillStreamingAttentionKeyTileColumns +
                                  probability_key]
                        : 0.0f;
            }
            for (uint output_fragment = 0u;
                 output_fragment < 4u;
                 ++output_fragment) {
                simdgroup_matrix<float, 8, 8>
                    value_matrix;
                const uint dimension =
                    dimension_begin +
                    output_fragment * 8u +
                    coordinate.x;
                const uint local_key =
                    reduction + coordinate.y;
                for (uint element = 0u;
                     element < 2u; ++element) {
                    const uint value_key =
                        local_key + element;
                    value_matrix.thread_elements()[element] =
                        value_key < key_count &&
                                dimension <
                                    kAttnHeadDimension
                            ? float(value_base[
                                  ulong(
                                      key_tile_begin +
                                      value_key) *
                                          kAttnHeadDimension +
                                      dimension])
                            : 0.0f;
                }
                simdgroup_multiply_accumulate(
                    accumulators[
                        output_fragment * 2u],
                    value_matrix, probability0,
                    accumulators[
                        output_fragment * 2u]);
                simdgroup_multiply_accumulate(
                    accumulators[
                        output_fragment * 2u + 1u],
                    value_matrix, probability1,
                    accumulators[
                        output_fragment * 2u + 1u]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint output_fragment = 0u;
         output_fragment < 4u; ++output_fragment) {
        const uint dimension =
            dimension_begin +
            output_fragment * 8u +
            coordinate.x;
        if (dimension >= kAttnHeadDimension) {
            continue;
        }
        for (uint query_fragment = 0u;
             query_fragment < 2u; ++query_fragment) {
            const uint local_row =
                query_fragment * 8u + coordinate.y;
            for (uint element = 0u; element < 2u;
                 ++element) {
                const uint row = local_row + element;
                if (row >= row_count) {
                    continue;
                }
                const ulong output_index =
                    (ulong(row_begin + row) *
                         kAttnQueryHeads +
                     head) *
                            kAttnHeadDimension +
                        dimension;
                const float numerator =
                    accumulators[
                        output_fragment * 2u +
                        query_fragment]
                        .thread_elements()[element];
                const float gate_value =
                    float(gate[output_index]);
                output[output_index] =
                    static_cast<bfloat>(
                        (numerator /
                         row_denominator[row]) /
                        (1.0f + exp(-gate_value)));
            }
        }
    }
}

kernel void attention_flash_v2_blk(
    device const bfloat* query [[buffer(0)]],
    device const bfloat* keys [[buffer(1)]],
    device const bfloat* values [[buffer(2)]],
    device const bfloat* gate [[buffer(3)]],
    device bfloat* output [[buffer(4)]],
    constant uint& visible [[buffer(5)]],
    constant uint& capacity [[buffer(6)]],
    constant uint& block [[buffer(7)]],
    constant uint& query_context [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroup_width [[threads_per_simdgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    threadgroup float scores[
        kPrefillStreamingAttentionQueryTileRows *
        kPrefillFlashV2KeyTileColumns];
    threadgroup bfloat query_tile[
        kPrefillStreamingAttentionQueryTileRows * kAttnHeadDimension];
    threadgroup bfloat key_staging[
        kPrefillFlashV2KeyTileColumns * kAttnHeadDimension];
    threadgroup float row_maximum[
        kPrefillStreamingAttentionQueryTileRows];
    threadgroup float row_denominator[
        kPrefillStreamingAttentionQueryTileRows];
    threadgroup float row_rescale[
        kPrefillStreamingAttentionQueryTileRows];

    if (simdgroup_width != kSimdgroupWidth ||
        threadgroup_shape.x !=
            kPrefillStreamingAttentionThreads ||
        threadgroup_shape.y != 1u ||
        threadgroup_shape.z != 1u ||
        group.x >= kAttnQueryHeads ||
        visible == 0u || visible > capacity ||
        block == 0u || query_context >= visible) {
        return;
    }
    const uint head = group.x;
    const uint row_begin =
        group.y * kPrefillStreamingAttentionQueryTileRows;
    if (row_begin >= block) {
        return;
    }
    const uint row_count =
        min(block - row_begin,
            kPrefillStreamingAttentionQueryTileRows);
    if (query_context + row_begin + row_count > visible) {
        return;
    }
    const uint key_value_head = head / kAttnHeadsPerKv;
    device const bfloat* key_base =
        keys + ulong(key_value_head) * capacity *
                   kAttnHeadDimension;
    device const bfloat* value_base =
        values + ulong(key_value_head) * capacity *
                     kAttnHeadDimension;
    const uint dimension_begin =
        simdgroup *
        kPrefillStreamingAttentionOutputColumnsPerSimdgroup;

    if (thread_index < kPrefillStreamingAttentionQueryTileRows) {
        row_maximum[thread_index] = -INFINITY;
        row_denominator[thread_index] = 0.0f;
        row_rescale[thread_index] = 0.0f;
    }
    // v2b delta (the ONLY change from attention_streaming_blk): the query
    // block is staged once, replacing per-key-tile scattered device reads
    // of the same 8 KiB. Rows beyond row_count are zero-filled so fragment
    // reads stay in-bounds without per-element ternaries.
    for (uint element = thread_index;
         element < kPrefillStreamingAttentionQueryTileRows *
                       kAttnHeadDimension;
         element += kPrefillStreamingAttentionThreads) {
        const uint row = element / kAttnHeadDimension;
        const uint dimension = element % kAttnHeadDimension;
        query_tile[element] =
            row < row_count
                ? query[(ulong(row_begin + row) * kAttnQueryHeads + head) *
                            kAttnHeadDimension +
                        dimension]
                : bfloat(0.0f);
    }
    simdgroup_matrix<float, 8, 8> accumulators[8];
    for (uint index = 0u; index < 8u; ++index) {
        prefill_streaming_attention_zero(accumulators[index]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint2 coordinate =
        prefill_staged_attention_fragment_coordinate(lane);
    for (uint key_tile_begin = 0u;
         key_tile_begin < visible;
         key_tile_begin += kPrefillFlashV2KeyTileColumns) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint key_count =
            min(visible - key_tile_begin, kPrefillFlashV2KeyTileColumns);
        // v2c delta: the key tile is staged once with coalesced cooperative
        // loads; the score phase reads it from threadgroup memory instead
        // of scattered 2-byte device fragments.
        for (uint element = thread_index;
             element < kPrefillFlashV2KeyTileColumns * kAttnHeadDimension;
             element += kPrefillStreamingAttentionThreads) {
            const uint key = element / kAttnHeadDimension;
            const uint dimension = element % kAttnHeadDimension;
            key_staging[element] =
                key < key_count
                    ? key_base[ulong(key_tile_begin + key) *
                                   kAttnHeadDimension +
                               dimension]
                    : bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            const uint row_fragment = simdgroup / 4u;
            const uint key_fragment = simdgroup % 4u;
            simdgroup_matrix<float, 8, 8> score_accumulator =
                make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint reduction = 0u; reduction < kAttnHeadDimension;
                 reduction += 8u) {
                simdgroup_matrix<float, 8, 8> key_matrix;
                simdgroup_matrix<float, 8, 8> query_matrix;
                for (uint element = 0u; element < 2u; ++element) {
                    const uint reduction_column =
                        reduction + coordinate.y + element;
                    key_matrix.thread_elements()[element] = float(
                        key_staging[(key_fragment * 8u + coordinate.x) *
                                        kAttnHeadDimension +
                                    reduction_column]);
                }
                const uint reduction_row = reduction + coordinate.x;
                for (uint element = 0u; element < 2u; ++element) {
                    const uint local_row =
                        row_fragment * 8u + coordinate.y + element;
                    query_matrix.thread_elements()[element] = float(
                        query_tile[local_row * kAttnHeadDimension +
                                   reduction_row]);
                }
                simdgroup_multiply_accumulate(
                    score_accumulator, key_matrix, query_matrix,
                    score_accumulator);
            }
            const uint local_key_base = key_fragment * 8u + coordinate.x;
            for (uint element = 0u; element < 2u; ++element) {
                const uint row =
                    row_fragment * 8u + coordinate.y + element;
                if (row >= row_count || local_key_base >= key_count) {
                    continue;
                }
                const float score =
                    score_accumulator.thread_elements()[element] *
                    kAttnScale;
                const uint absolute_key = key_tile_begin + local_key_base;
                const uint causal_limit =
                    query_context + row_begin + row + 1u;
                scores[row * kPrefillFlashV2KeyTileColumns +
                       local_key_base] =
                    absolute_key < causal_limit ? score : -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint row = simdgroup; row < row_count;
             row += kPrefillStreamingAttentionSimdgroups) {
            threadgroup float* row_scores =
                scores + row * kPrefillFlashV2KeyTileColumns;
            float tile_maximum = -INFINITY;
            for (uint key = lane; key < key_count;
                 key += kSimdgroupWidth) {
                tile_maximum =
                    max(tile_maximum, row_scores[key]);
            }
            tile_maximum = simd_max(tile_maximum);
            const float previous_maximum = row_maximum[row];
            const float next_maximum =
                max(previous_maximum, tile_maximum);
            const float rescale =
                previous_maximum == -INFINITY
                    ? 0.0f
                    : exp(previous_maximum - next_maximum);
            float tile_denominator = 0.0f;
            for (uint key = lane; key < key_count;
                 key += kSimdgroupWidth) {
                const float weight =
                    row_scores[key] == -INFINITY
                        ? 0.0f
                        : exp(
                              row_scores[key] -
                              next_maximum);
                row_scores[key] = weight;
                tile_denominator += weight;
            }
            tile_denominator =
                simd_sum(tile_denominator);
            if (lane == 0u) {
                row_maximum[row] = next_maximum;
                row_denominator[row] =
                    row_denominator[row] * rescale +
                    tile_denominator;
                row_rescale[row] = rescale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint output_fragment = 0u;
             output_fragment < 4u; ++output_fragment) {
            for (uint query_fragment = 0u;
                 query_fragment < 2u; ++query_fragment) {
                prefill_streaming_attention_rescale(
                    accumulators[
                        output_fragment * 2u +
                        query_fragment],
                    row_rescale, query_fragment, row_count,
                    lane);
            }
        }

        const uint reduction_end = (key_count + 7u) & ~7u;
        for (uint reduction = 0u;
             reduction < reduction_end;
             reduction += 8u) {
            simdgroup_matrix<float, 8, 8>
                probability0;
            simdgroup_matrix<float, 8, 8>
                probability1;
            const uint probability_key =
                reduction + coordinate.x;
            for (uint element = 0u; element < 2u;
                 ++element) {
                const uint local_row =
                    coordinate.y + element;
                probability0.thread_elements()[element] =
                    probability_key < key_count &&
                            local_row < row_count
                        ? scores[
                              local_row * kPrefillFlashV2KeyTileColumns +
                                  probability_key]
                        : 0.0f;
                probability1.thread_elements()[element] =
                    probability_key < key_count &&
                            local_row + 8u < row_count
                        ? scores[
                              (local_row + 8u) *
                                      kPrefillFlashV2KeyTileColumns +
                                  probability_key]
                        : 0.0f;
            }
            for (uint output_fragment = 0u;
                 output_fragment < 4u;
                 ++output_fragment) {
                simdgroup_matrix<float, 8, 8>
                    value_matrix;
                const uint dimension =
                    dimension_begin +
                    output_fragment * 8u +
                    coordinate.x;
                const uint local_key =
                    reduction + coordinate.y;
                for (uint element = 0u;
                     element < 2u; ++element) {
                    const uint value_key =
                        local_key + element;
                    value_matrix.thread_elements()[element] =
                        value_key < key_count &&
                                dimension <
                                    kAttnHeadDimension
                            ? float(value_base[
                                  ulong(
                                      key_tile_begin +
                                      value_key) *
                                          kAttnHeadDimension +
                                      dimension])
                            : 0.0f;
                }
                simdgroup_multiply_accumulate(
                    accumulators[
                        output_fragment * 2u],
                    value_matrix, probability0,
                    accumulators[
                        output_fragment * 2u]);
                simdgroup_multiply_accumulate(
                    accumulators[
                        output_fragment * 2u + 1u],
                    value_matrix, probability1,
                    accumulators[
                        output_fragment * 2u + 1u]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint output_fragment = 0u;
         output_fragment < 4u; ++output_fragment) {
        const uint dimension =
            dimension_begin +
            output_fragment * 8u +
            coordinate.x;
        if (dimension >= kAttnHeadDimension) {
            continue;
        }
        for (uint query_fragment = 0u;
             query_fragment < 2u; ++query_fragment) {
            const uint local_row =
                query_fragment * 8u + coordinate.y;
            for (uint element = 0u; element < 2u;
                 ++element) {
                const uint row = local_row + element;
                if (row >= row_count) {
                    continue;
                }
                const ulong output_index =
                    (ulong(row_begin + row) *
                         kAttnQueryHeads +
                     head) *
                            kAttnHeadDimension +
                        dimension;
                const float numerator =
                    accumulators[
                        output_fragment * 2u +
                        query_fragment]
                        .thread_elements()[element];
                const float gate_value =
                    float(gate[output_index]);
                output[output_index] =
                    static_cast<bfloat>(
                        (numerator /
                         row_denominator[row]) /
                        (1.0f + exp(-gate_value)));
            }
        }
    }
}

kernel void attention_gate_apply_blk(
    device bfloat* attended [[buffer(0)]],
    device const bfloat* gate [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint3 position [[thread_position_in_grid]]) {
    if (position.x >= element_count) {
        return;
    }
    const float value = float(attended[position.x]);
    const float gate_value = float(gate[position.x]);
    attended[position.x] =
        static_cast<bfloat>(value / (1.0f + exp(-gate_value)));
}

// A41a: conditioning-capture copy. After a captured layer's unit, copies
// the band rows' residual (hidden_slab) into the capture buffer at the
// layer's slot, [row][slot][hidden] layout matching the engine's
// cycle_features. Pure bf16 copy inside the command graph — replaces the
// nine-segment blit path that cost the verify band its efficiency.
kernel void capture_rows_blk(
    device const bfloat* hidden [[buffer(0)]],
    device bfloat* captures [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& slot [[buffer(3)]],
    constant uint& rows_capacity [[buffer(4)]],
    uint3 position [[thread_position_in_grid]]) {
    const uint row = position.y;
    const uint dimension = position.x;
    if (rows > rows_capacity || row >= rows || dimension >= 2048u ||
        slot >= 8u) {
        return;
    }
    captures[(ulong(row) * 8u + slot) * 2048u + dimension] =
        hidden[ulong(row) * 2048u + dimension];
}
