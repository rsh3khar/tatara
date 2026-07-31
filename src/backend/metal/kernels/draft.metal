// DFlash draft-model kernel family (TATARA-EXACT-SPEC-1, A36 U3).
//
// The draft is a 6-layer bf16 dense transformer conditioned on projected
// target hidden states. Draft numerics can never change committed output —
// acceptance is exact token-id equality against the target — so this family
// carries the numerical-family label `draft-v1`; its only product-facing
// contract is boundedness and the block-drafting dataflow.
//
// v1 shapes are chosen for boundedness and hot-path byte-optimality:
// weights stream exactly once per <= kDraftMaxBlockRows-row batch in the
// dense kernel, and attention stages key/value tiles through threadgroup
// memory so cached context is read once per key tile regardless of the
// number of query rows. All extents are dispatch-derived; every kernel
// guards its row/column bounds, so partial trailing threadgroups are safe.

// y[m, n] = sum_k x[m, k] * w[n, k]   (bf16 in, f32 accumulate, bf16 out)
// One output row per simdgroup, up to kDraftMaxBlockRows input rows per
// dispatch: the weight row is loaded once per k-chunk and reused across all
// m, so weight traffic is amortized by the row batch.
kernel void draft_dense_bf16(device const bfloat* activations [[buffer(0)]],
                             device const bfloat* weights [[buffer(1)]],
                             device bfloat* outputs [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& reduction [[buffer(4)]],
                             constant uint& columns [[buffer(5)]],
                             uint3 group [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simdgroup [[simdgroup_index_in_threadgroup]],
                             uint simdgroup_width [[threads_per_simdgroup]],
                             uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    if (simdgroup_width != 32u ||
        threadgroup_shape.x != kDraftDenseThreads || threadgroup_shape.y != 1u ||
        threadgroup_shape.z != 1u || rows == 0u ||
        rows > kDraftMaxBlockRows || reduction == 0u ||
        (reduction & 3u) != 0u || columns == 0u) {
        return;
    }
    const uint column = group.x * (kDraftDenseThreads / 32u) + simdgroup;
    if (column >= columns) {
        return;
    }
    float accumulators[kDraftMaxBlockRows];
    for (uint m = 0; m < kDraftMaxBlockRows; ++m) {
        accumulators[m] = 0.0f;
    }
    device const bfloat* weight_row = weights + ulong(column) * reduction;
    for (uint k = lane * 4u; k < reduction; k += 32u * 4u) {
        const float w0 = float(weight_row[k + 0u]);
        const float w1 = float(weight_row[k + 1u]);
        const float w2 = float(weight_row[k + 2u]);
        const float w3 = float(weight_row[k + 3u]);
        for (uint m = 0; m < rows; ++m) {
            device const bfloat* activation_row =
                activations + ulong(m) * reduction + k;
            accumulators[m] = fma(float(activation_row[0]), w0,
                              fma(float(activation_row[1]), w1,
                              fma(float(activation_row[2]), w2,
                              fma(float(activation_row[3]), w3,
                                  accumulators[m]))));
        }
    }
    for (uint m = 0; m < rows; ++m) {
        const float total = simd_sum(accumulators[m]);
        if (lane == 0u) {
            outputs[ulong(m) * columns + column] = static_cast<bfloat>(total);
        }
    }
}

// Fragment coordinate for one lane of an Apple 8x8 simdgroup matrix: the
// lane owns elements (row, column) and (row, column + 1). Identical bit
// mapping to native_dense_qgemm_n1_fragment_coordinate (the admitted N1
// MMA head) and to the vendored MLX Steel BaseMMAFrag::get_coord.
inline ushort2 draft_gemm16_fragment_coordinate(uint lane) {
    const ushort row =
        ushort(((lane >> 2u) & 4u) | ((lane >> 1u) & 3u));
    const ushort column =
        ushort(((lane >> 1u) & 4u) | ((lane & 1u) << 1u));
    return ushort2(column, row);
}

// y[m, n] = sum_k x[m, k] * w[n, k] for the exact 16-row draft block
// (bf16 in, f32 simdgroup-MMA accumulate, bf16 out). v2 of the dense
// family: A36 attribution measured the m-loop GEMV kernel at the per-lane
// issue wall (~28 GB/s effective on the 6-layer forward); 8x8 simdgroup
// MMA escapes it. One simdgroup owns kDraftGemmColumnsPerSimdgroup output
// columns for BOTH 8-row tiles, so each weight fragment is loaded exactly
// once per simdgroup; activations re-reads are 64 KiB and cache-served.
// Guards refuse every shape this kernel is not written for; the m-loop
// kernel remains the fallback for those dispatches.
kernel void draft_gemm16_bf16(device const bfloat* activations [[buffer(0)]],
                              device const bfloat* weights [[buffer(1)]],
                              device bfloat* outputs [[buffer(2)]],
                              constant uint& rows [[buffer(3)]],
                              constant uint& reduction [[buffer(4)]],
                              constant uint& columns [[buffer(5)]],
                              uint3 group [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simdgroup [[simdgroup_index_in_threadgroup]],
                              uint simdgroup_width [[threads_per_simdgroup]],
                              uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    if (simdgroup_width != 32u ||
        threadgroup_shape.x != kDraftGemmThreads ||
        threadgroup_shape.y != 1u || threadgroup_shape.z != 1u ||
        rows != kDraftMaxBlockRows || reduction == 0u ||
        (reduction & 7u) != 0u || columns == 0u ||
        (columns % kDraftGemmColumnsPerSimdgroup) != 0u) {
        return;
    }
    const uint column_base =
        (group.x * (kDraftGemmThreads / 32u) + simdgroup) *
        kDraftGemmColumnsPerSimdgroup;
    if (column_base >= columns) {
        return;
    }
    const ushort2 coordinate = draft_gemm16_fragment_coordinate(lane);
    const uint fragment_row = uint(coordinate.y);
    const uint fragment_column = uint(coordinate.x);

    simdgroup_matrix<float, 8, 8> accumulators[2][2];
    for (uint row_tile = 0; row_tile < 2u; ++row_tile) {
        for (uint column_tile = 0; column_tile < 2u; ++column_tile) {
            accumulators[row_tile][column_tile] =
                make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    device const bfloat* activation_low =
        activations + ulong(fragment_row) * reduction;
    device const bfloat* activation_high =
        activations + ulong(fragment_row + 8u) * reduction;
    // Transposed weight fragments: fragment element (i, j) must hold
    // w[column_base + tile * 8 + j][k + i].
    device const bfloat* weight_low =
        weights + ulong(column_base + fragment_column) * reduction;
    device const bfloat* weight_high =
        weights + ulong(column_base + 8u + fragment_column) * reduction;
    const ulong weight_next_column = ulong(reduction);
    for (uint k = 0; k < reduction; k += 8u) {
        simdgroup_matrix<float, 8, 8> block_rows[2];
        block_rows[0].thread_elements()[0] =
            float(activation_low[k + fragment_column]);
        block_rows[0].thread_elements()[1] =
            float(activation_low[k + fragment_column + 1u]);
        block_rows[1].thread_elements()[0] =
            float(activation_high[k + fragment_column]);
        block_rows[1].thread_elements()[1] =
            float(activation_high[k + fragment_column + 1u]);
        simdgroup_matrix<float, 8, 8> weight_tiles[2];
        weight_tiles[0].thread_elements()[0] =
            float(weight_low[k + fragment_row]);
        weight_tiles[0].thread_elements()[1] =
            float(weight_low[weight_next_column + k + fragment_row]);
        weight_tiles[1].thread_elements()[0] =
            float(weight_high[k + fragment_row]);
        weight_tiles[1].thread_elements()[1] =
            float(weight_high[weight_next_column + k + fragment_row]);
        for (uint row_tile = 0; row_tile < 2u; ++row_tile) {
            for (uint column_tile = 0; column_tile < 2u; ++column_tile) {
                simdgroup_multiply_accumulate(
                    accumulators[row_tile][column_tile],
                    block_rows[row_tile],
                    weight_tiles[column_tile],
                    accumulators[row_tile][column_tile]);
            }
        }
    }
    for (uint row_tile = 0; row_tile < 2u; ++row_tile) {
        for (uint column_tile = 0; column_tile < 2u; ++column_tile) {
            const uint out_row = row_tile * 8u + fragment_row;
            const uint out_column =
                column_base + column_tile * 8u + fragment_column;
            outputs[ulong(out_row) * columns + out_column] =
                static_cast<bfloat>(
                    accumulators[row_tile][column_tile].thread_elements()[0]);
            outputs[ulong(out_row) * columns + out_column + 1u] =
                static_cast<bfloat>(
                    accumulators[row_tile][column_tile].thread_elements()[1]);
        }
    }
}

// Row RMS norm with weight: dimension is dispatch-derived (2048 hidden rows
// or 128 per-head rows), threads = dimension / 4, one row per threadgroup.
kernel void draft_rms_rows(device const bfloat* inputs [[buffer(0)]],
                           device const bfloat* norm_weight [[buffer(1)]],
                           device bfloat* outputs [[buffer(2)]],
                           constant uint& rows [[buffer(3)]],
                           constant uint& dimension [[buffer(4)]],
                           uint3 group [[threadgroup_position_in_grid]],
                           uint thread_index [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint simdgroup [[simdgroup_index_in_threadgroup]],
                           uint simdgroup_width [[threads_per_simdgroup]],
                           uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    threadgroup float partials[kDraftRmsMaxSimdgroups];
    const uint threads = threadgroup_shape.x;
    if (simdgroup_width != 32u || threadgroup_shape.y != 1u ||
        threadgroup_shape.z != 1u || rows == 0u || dimension == 0u ||
        (dimension & 3u) != 0u || threads * 4u != dimension ||
        threads / 32u > kDraftRmsMaxSimdgroups || group.x >= rows) {
        return;
    }
    device const bfloat* row = inputs + ulong(group.x) * dimension;
    device bfloat* out_row = outputs + ulong(group.x) * dimension;
    const uint base = thread_index * 4u;
    float4 values;
    values.x = float(row[base + 0u]);
    values.y = float(row[base + 1u]);
    values.z = float(row[base + 2u]);
    values.w = float(row[base + 3u]);
    float sum = values.x * values.x + values.y * values.y +
                values.z * values.z + values.w * values.w;
    sum = simd_sum(sum);
    const uint simdgroups = threads / 32u;
    if (simdgroups > 1u) {
        if (lane == 0u) {
            partials[simdgroup] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = 0.0f;
        for (uint index = 0; index < simdgroups; ++index) {
            sum += partials[index];
        }
    }
    const float scale = rsqrt(sum / float(dimension) + kDraftRmsEpsilon);
    out_row[base + 0u] = static_cast<bfloat>(values.x * scale * float(norm_weight[base + 0u]));
    out_row[base + 1u] = static_cast<bfloat>(values.y * scale * float(norm_weight[base + 1u]));
    out_row[base + 2u] = static_cast<bfloat>(values.z * scale * float(norm_weight[base + 2u]));
    out_row[base + 3u] = static_cast<bfloat>(values.w * scale * float(norm_weight[base + 3u]));
}

// Full-head 128-dimension NeoX rotation in place: pair (i, i + 64), angle
// position * base^(-i/64). Positions are contiguous from first_position.
// Grid: (heads, rows); 64 threads, one pair per thread.
kernel void draft_rope128(device bfloat* vectors [[buffer(0)]],
                          constant uint& rows [[buffer(1)]],
                          constant uint& heads [[buffer(2)]],
                          constant uint& first_position [[buffer(3)]],
                          uint3 group [[threadgroup_position_in_grid]],
                          uint thread_index [[thread_index_in_threadgroup]],
                          uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    if (threadgroup_shape.x != kDraftHeadDim / 2u || threadgroup_shape.y != 1u ||
        threadgroup_shape.z != 1u || group.x >= heads || group.y >= rows ||
        heads == 0u || rows == 0u) {
        return;
    }
    const uint pair = thread_index;
    device bfloat* head_vector =
        vectors + (ulong(group.y) * heads + group.x) * kDraftHeadDim;
    const float position = float(first_position + group.y);
    const float exponent = -2.0f * float(pair) / float(kDraftHeadDim);
    const float theta = position * pow(kDraftRopeBase, exponent);
    const float cos_theta = cos(theta);
    const float sin_theta = sin(theta);
    const float first = float(head_vector[pair]);
    const float second = float(head_vector[pair + kDraftHeadDim / 2u]);
    head_vector[pair] = static_cast<bfloat>(first * cos_theta - second * sin_theta);
    head_vector[pair + kDraftHeadDim / 2u] =
        static_cast<bfloat>(second * cos_theta + first * sin_theta);
}

// Block attention over cached context plus in-flight noise keys. One
// threadgroup per KV head; key/value tiles are staged through threadgroup
// memory once and consumed by every (query row, query head) pair, so
// context bytes are read once per tile regardless of block width. Online
// softmax runs redundantly across the simdgroup's lanes (values identical
// by construction); each lane owns four output dimensions per pair.
//
// Masking: full-attention layers attend everything; sliding layers attend
// keys with q_pos >= k_pos && q_pos < k_pos + window. Query row m sits at
// absolute position query_offset + m; key positions arrive explicitly so
// sink eviction keeps absolute coordinates.
kernel void draft_block_attention(device const bfloat* queries [[buffer(0)]],
                                  device const bfloat* keys [[buffer(1)]],
                                  device const bfloat* values [[buffer(2)]],
                                  device const int* key_positions [[buffer(3)]],
                                  device bfloat* outputs [[buffer(4)]],
                                  constant uint& key_count [[buffer(5)]],
                                  constant uint& block_rows [[buffer(6)]],
                                  constant uint& query_offset [[buffer(7)]],
                                  constant uint& full_attention [[buffer(8)]],
                                  constant uint& key_capacity [[buffer(9)]],
                                  uint3 group [[threadgroup_position_in_grid]],
                                  uint thread_index [[thread_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint simdgroup [[simdgroup_index_in_threadgroup]],
                                  uint simdgroup_width [[threads_per_simdgroup]],
                                  uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    threadgroup bfloat key_tile[kDraftAttnKeyTile * kDraftHeadDim];
    threadgroup bfloat value_tile[kDraftAttnKeyTile * kDraftHeadDim];
    threadgroup int position_tile[kDraftAttnKeyTile];
    if (simdgroup_width != 32u ||
        threadgroup_shape.x != kDraftAttnThreads || threadgroup_shape.y != 1u ||
        threadgroup_shape.z != 1u || group.x >= kDraftKvHeads ||
        block_rows == 0u || block_rows > kDraftMaxBlockRows ||
        key_count == 0u || key_count > key_capacity) {
        return;
    }
    const uint kv_head = group.x;
    const uint group_width = kDraftQHeads / kDraftKvHeads;      // 4
    const uint pairs = block_rows * group_width;                // <= 64
    const uint simdgroups = kDraftAttnThreads / 32u;            // 4
    // Online state per simdgroup-local pair (pair = simdgroup + local *
    // simdgroups), lane-owned output dims: bounded register footprint.
    float row_max[kDraftAttnLocalPairs];
    float row_sum[kDraftAttnLocalPairs];
    float acc[kDraftAttnLocalPairs][4];
    for (uint local = 0; local < kDraftAttnLocalPairs; ++local) {
        row_max[local] = -INFINITY;
        row_sum[local] = 0.0f;
        acc[local][0] = 0.0f;
        acc[local][1] = 0.0f;
        acc[local][2] = 0.0f;
        acc[local][3] = 0.0f;
    }
    const float scale = rsqrt(float(kDraftHeadDim));
    for (uint tile_begin = 0; tile_begin < key_count;
         tile_begin += kDraftAttnKeyTile) {
        const uint tile_rows = min(uint(kDraftAttnKeyTile), key_count - tile_begin);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint element = thread_index;
             element < tile_rows * kDraftHeadDim;
             element += kDraftAttnThreads) {
            const uint tile_row = element / kDraftHeadDim;
            const uint dimension = element % kDraftHeadDim;
            // Token-major cache layout [position, kv_head, dim]: context
            // K/V projections land here directly with no transpose.
            const ulong source =
                (ulong(tile_begin + tile_row) * kDraftKvHeads + kv_head) *
                    kDraftHeadDim + dimension;
            key_tile[tile_row * kDraftHeadDim + dimension] = keys[source];
            value_tile[tile_row * kDraftHeadDim + dimension] = values[source];
        }
        for (uint tile_row = thread_index; tile_row < tile_rows;
             tile_row += kDraftAttnThreads) {
            position_tile[tile_row] = key_positions[tile_begin + tile_row];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint pair = simdgroup; pair < pairs; pair += simdgroups) {
            const uint local = (pair - simdgroup) / simdgroups;
            const uint row = pair / group_width;
            const uint query_head = kv_head * group_width + pair % group_width;
            device const bfloat* query_vector =
                queries + (ulong(row) * kDraftQHeads + query_head) * kDraftHeadDim;
            const float query_a = float(query_vector[lane * 4u + 0u]);
            const float query_b = float(query_vector[lane * 4u + 1u]);
            const float query_c = float(query_vector[lane * 4u + 2u]);
            const float query_d = float(query_vector[lane * 4u + 3u]);
            const int query_position = int(query_offset + row);
            for (uint tile_row = 0; tile_row < tile_rows; ++tile_row) {
                const int key_position = position_tile[tile_row];
                bool visible = true;
                if (full_attention == 0u) {
                    visible = query_position >= key_position &&
                              query_position <
                                  key_position + int(kDraftWindowPositions);
                }
                const threadgroup bfloat* key_vector =
                    key_tile + tile_row * kDraftHeadDim + lane * 4u;
                float dot = float(key_vector[0]) * query_a +
                            float(key_vector[1]) * query_b +
                            float(key_vector[2]) * query_c +
                            float(key_vector[3]) * query_d;
                dot = simd_sum(dot) * scale;
                if (!visible) {
                    continue;
                }
                const float previous_max = row_max[local];
                const float new_max = max(previous_max, dot);
                const float correction =
                    previous_max == -INFINITY ? 0.0f
                                              : exp(previous_max - new_max);
                const float weight = exp(dot - new_max);
                row_sum[local] = row_sum[local] * correction + weight;
                const threadgroup bfloat* value_vector =
                    value_tile + tile_row * kDraftHeadDim + lane * 4u;
                acc[local][0] = acc[local][0] * correction +
                               weight * float(value_vector[0]);
                acc[local][1] = acc[local][1] * correction +
                               weight * float(value_vector[1]);
                acc[local][2] = acc[local][2] * correction +
                               weight * float(value_vector[2]);
                acc[local][3] = acc[local][3] * correction +
                               weight * float(value_vector[3]);
                row_max[local] = new_max;
            }
        }
    }
    for (uint pair = simdgroup; pair < pairs; pair += simdgroups) {
        const uint local = (pair - simdgroup) / simdgroups;
        const uint row = pair / group_width;
        const uint query_head = kv_head * group_width + pair % group_width;
        const float denominator = row_sum[local];
        device bfloat* out_vector =
            outputs + (ulong(row) * kDraftQHeads + query_head) * kDraftHeadDim +
            lane * 4u;
        const float inverse = denominator == 0.0f ? 0.0f : 1.0f / denominator;
        out_vector[0] = static_cast<bfloat>(acc[local][0] * inverse);
        out_vector[1] = static_cast<bfloat>(acc[local][1] * inverse);
        out_vector[2] = static_cast<bfloat>(acc[local][2] * inverse);
        out_vector[3] = static_cast<bfloat>(acc[local][3] * inverse);
    }
}

// y = silu(gate) * up, elementwise over rows x dimension.
kernel void draft_swiglu(device const bfloat* gates [[buffer(0)]],
                         device const bfloat* ups [[buffer(1)]],
                         device bfloat* outputs [[buffer(2)]],
                         constant uint& rows [[buffer(3)]],
                         constant uint& dimension [[buffer(4)]],
                         uint3 position [[thread_position_in_grid]]) {
    if (position.y >= rows || position.x >= dimension) {
        return;
    }
    const ulong index = ulong(position.y) * dimension + position.x;
    const float gate = float(gates[index]);
    const float up = float(ups[index]);
    outputs[index] = static_cast<bfloat>((gate / (1.0f + exp(-gate))) * up);
}

// residual += branch, elementwise (bf16 storage, f32 add).
kernel void draft_residual_add(device bfloat* residuals [[buffer(0)]],
                               device const bfloat* branches [[buffer(1)]],
                               constant uint& rows [[buffer(2)]],
                               constant uint& dimension [[buffer(3)]],
                               uint3 position [[thread_position_in_grid]]) {
    if (position.y >= rows || position.x >= dimension) {
        return;
    }
    const ulong index = ulong(position.y) * dimension + position.x;
    residuals[index] = static_cast<bfloat>(float(residuals[index]) +
                                           float(branches[index]));
}
