kernel void residual_rms(device const bfloat* a [[buffer(0)]], device const bfloat* b [[buffer(1)]],
                         device bfloat* residual_out [[buffer(2)]],
                         device const bfloat* weight [[buffer(3)]],
                         device bfloat* normed_out [[buffer(4)]],
                         uint thread_in_group [[thread_position_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    // The residual sum rounds to bfloat16 and is written back exactly as the
    // standalone add would round it; the RMS reduction consumes those
    // rounded values with the admitted double-rounded output.
    threadgroup float inverse_shared[1];
    threadgroup float stage[32];

    const uint base = thread_in_group * kRmsValuesPerThread;
    bfloat rounded[4];
    float acc = 0.0f;
    for (uint i = 0; i < kRmsValuesPerThread; ++i) {
        const uint element = base + i;
        rounded[i] = static_cast<bfloat>(float(a[element]) + float(b[element]));
        residual_out[element] = rounded[i];
        const float value = float(rounded[i]);
        acc += value * value;
    }
    acc = simd_sum(acc);
    if (simdgroup == 0u) {
        stage[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        stage[simdgroup] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        acc = simd_sum(stage[lane]);
        if (lane == 0u) {
            inverse_shared[0] = precise::rsqrt(acc / float(kHiddenDimension) + kRmsNormEpsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inverse = inverse_shared[0];
    for (uint i = 0; i < kRmsValuesPerThread; ++i) {
        const uint element = base + i;
        normed_out[element] = weight[element] * static_cast<bfloat>(float(rounded[i]) * inverse);
    }
}

kernel void router_q8(device const bfloat* input [[buffer(0)]],
                      device const uint* routed_words [[buffer(1)]],
                      device const bfloat* routed_scales [[buffer(2)]],
                      device const bfloat* routed_biases [[buffer(3)]],
                      device const uint* shared_words [[buffer(4)]],
                      device const bfloat* shared_scales [[buffer(5)]],
                      device const bfloat* shared_biases [[buffer(6)]],
                      device float* logits [[buffer(7)]], uint tid [[thread_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tid >> 5u;
    if (row >= kMoeExperts + 1u) {
        return;
    }
    const bool shared = row == kMoeExperts;
    const ulong word_row = shared ? 0ul : ulong(row) * ulong(kQuantWordsPerRow * 2u);
    const ulong group_row = shared ? 0ul : ulong(row) * ulong(kGroupsPerRow);
    const float acc =
        q8_dot(input, (shared ? shared_words : routed_words) + word_row,
               (shared ? shared_scales : routed_scales) + group_row,
               (shared ? shared_biases : routed_biases) + group_row, kHiddenDimension, lane);
    if (lane == 0u) {
        logits[row] = float(static_cast<bfloat>(acc));
    }
}

kernel void router_select(device const float* logits [[buffer(0)]], device uint* ids [[buffer(1)]],
                          device float* coefficients [[buffer(2)]],
                          device float* shared_coefficient [[buffer(3)]],
                          constant uint& top_n [[buffer(4)]],
                          uint e [[thread_position_in_threadgroup]]) {
    // Sized from the router width itself, not a literal: the dispatch is
    // {groups = 1, threads = experts}, so a family with a different expert
    // count would leave the upper slots of a fixed-256 array never written
    // while the reduction still read them.
    //
    // The tree halving requires a power of two -- with, say, 6 experts the
    // off=3 pass merges slots 3..5 into 0..2 and the off=1 pass then never
    // merges slot 2. The generator rejects a non-power-of-two expert count
    // before emitting this source; this assert is the second line of defence
    // for anyone compiling the kernel by hand.
    static_assert((kMoeExperts & (kMoeExperts - 1u)) == 0u,
                  "router_select's tree reduction requires a power-of-two expert count");
    threadgroup float values[kMoeExperts];
    threadgroup float red[kMoeExperts];
    threadgroup uint indices[kMoeExperts];
    red[e] = logits[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = kMoeExperts / 2u; off; off >>= 1u) {
        if (e < off) {
            red[e] = max(red[e], red[e + off]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float maximum = red[0];
    // Every thread reads red[0] above, and thread 0 overwrites it below. Without
    // this barrier those two are a data race: thread 0 may store values[0] into
    // red[0] before another simdgroup has read the maximum, so that thread
    // exponentiates against a wrong maximum and the whole softmax -- and with it
    // the top-k selection -- changes. It is rare because the window is a few
    // instructions after a barrier, which is exactly why it presented as
    // occasional run-to-run nondeterminism rather than a reproducible wrong
    // answer. The sum reduction below already has the equivalent guard (line
    // 99), which is what made the omission visible.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    values[e] = exp(logits[e] - maximum);
    red[e] = values[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = kMoeExperts / 2u; off; off >>= 1u) {
        if (e < off) {
            red[e] += red[e + off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    values[e] /= red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Iterative argmax: on equal merges the strict-greater comparison keeps
    // the lower tree slot, so exact ties resolve by tree position (lowest
    // bit-reversed index), not by lowest expert index.
    for (uint k = 0; k < top_n; ++k) {
        red[e] = values[e];
        indices[e] = e;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = kMoeExperts / 2u; off; off >>= 1u) {
            if (e < off && red[e + off] > red[e]) {
                red[e] = red[e + off];
                indices[e] = indices[e + off];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (e == 0u) {
            ids[k] = indices[0];
            coefficients[k] = red[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (e == indices[0]) {
            values[e] = -1.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (e == 0u) {
        // norm_topk_prob: selected probabilities renormalize by their sum.
        float selected_sum = 0.0f;
        for (uint k = 0; k < top_n; ++k) {
            selected_sum += coefficients[k];
        }
        for (uint k = 0; k < top_n; ++k) {
            coefficients[k] /= selected_sum;
        }
        for (uint k = top_n; k < kMoeActiveExperts; ++k) {
            ids[k] = 0xffffffffu;
            coefficients[k] = 0.0f;
        }
        shared_coefficient[0] = 1.0f / (1.0f + exp(-logits[kMoeExperts]));
    }
}

// Row-batched router_select: grid y is the batch row; each row runs the
// full threadgroup selection independently against its own logits slice.
// Selection order, tie handling, and renormalization are exactly
// router_select's.
kernel void router_select_ms(device const float* logits [[buffer(0)]],
                             device uint* ids [[buffer(1)]],
                             device float* coefficients [[buffer(2)]],
                             device float* shared_coefficient [[buffer(3)]],
                             constant uint& top_n [[buffer(4)]],
                             constant uint& logits_stride [[buffer(5)]],
                             constant uint& ids_stride [[buffer(6)]],
                             constant uint& coefficient_stride [[buffer(7)]],
                             constant uint& shared_stride [[buffer(8)]],
                             uint2 tg2 [[threadgroup_position_in_grid]],
                             uint2 e2 [[thread_position_in_threadgroup]]) {
    static_assert((kMoeExperts & (kMoeExperts - 1u)) == 0u,
                  "router_select_ms's tree reduction requires a power-of-two expert count");
    const uint e = e2.x;
    const uint row = tg2.y;
    device const float* logits_row = logits + ulong(row) * ulong(logits_stride);
    device uint* ids_row = ids + ulong(row) * ulong(ids_stride);
    device float* coefficients_row = coefficients + ulong(row) * ulong(coefficient_stride);
    device float* shared_row = shared_coefficient + ulong(row) * ulong(shared_stride);
    threadgroup float values[kMoeExperts];
    threadgroup float red[kMoeExperts];
    threadgroup uint indices[kMoeExperts];
    red[e] = logits_row[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = kMoeExperts / 2u; off; off >>= 1u) {
        if (e < off) {
            red[e] = max(red[e], red[e + off]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float maximum = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    values[e] = exp(logits_row[e] - maximum);
    red[e] = values[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = kMoeExperts / 2u; off; off >>= 1u) {
        if (e < off) {
            red[e] += red[e + off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    values[e] /= red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < top_n; ++k) {
        red[e] = values[e];
        indices[e] = e;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = kMoeExperts / 2u; off; off >>= 1u) {
            if (e < off && red[e + off] > red[e]) {
                red[e] = red[e + off];
                indices[e] = indices[e + off];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (e == 0u) {
            ids_row[k] = indices[0];
            coefficients_row[k] = red[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (e == indices[0]) {
            values[e] = -1.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (e == 0u) {
        float selected_sum = 0.0f;
        for (uint k = 0; k < top_n; ++k) {
            selected_sum += coefficients_row[k];
        }
        for (uint k = 0; k < top_n; ++k) {
            coefficients_row[k] /= selected_sum;
        }
        for (uint k = top_n; k < kMoeActiveExperts; ++k) {
            ids_row[k] = 0xffffffffu;
            coefficients_row[k] = 0.0f;
        }
        shared_row[0] = 1.0f / (1.0f + exp(-logits_row[kMoeExperts]));
    }
}

kernel void grouped_upgate(
    device const bfloat* input [[buffer(0)]], device const uint* ids [[buffer(1)]],
    device const uint* gate_words [[buffer(2)]], device const bfloat* gate_scales [[buffer(3)]],
    device const bfloat* gate_biases [[buffer(4)]], device const uint* up_words [[buffer(5)]],
    device const bfloat* up_scales [[buffer(6)]], device const bfloat* up_biases [[buffer(7)]],
    device const uint* shared_gate_words [[buffer(8)]],
    device const bfloat* shared_gate_scales [[buffer(9)]],
    device const bfloat* shared_gate_biases [[buffer(10)]],
    device const uint* shared_up_words [[buffer(11)]],
    device const bfloat* shared_up_scales [[buffer(12)]],
    device const bfloat* shared_up_biases [[buffer(13)]], device bfloat* hidden_out [[buffer(14)]],
    uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint sg = tid >> 5u;
    if (sg >= (kMoeActiveExperts + 1u) * kMoeExpertDimension) {
        return;
    }
    const uint slot = sg / kMoeExpertDimension;
    const uint row = sg % kMoeExpertDimension;
    const bool shared = slot == kMoeActiveExperts;
    const uint expert = shared ? 0u : ids[slot];
    // Unused slots carry a sentinel id; their output zeroes so downstream
    // reads stay finite.
    if (!shared && expert == 0xffffffffu) {
        if (lane == 0u) {
            hidden_out[sg] = bfloat(0.0f);
        }
        return;
    }
    const ulong word_row = shared ? ulong(row) * ulong(kQuantWordsPerRow)
                                  : (ulong(expert) * ulong(kMoeExpertDimension) + ulong(row)) *
                                        ulong(kQuantWordsPerRow);
    const ulong group_row =
        shared ? ulong(row) * ulong(kGroupsPerRow)
               : (ulong(expert) * ulong(kMoeExpertDimension) + ulong(row)) * ulong(kGroupsPerRow);
    const float gate = q4_dot_packed(input, (shared ? shared_gate_words : gate_words) + word_row,
                                     (shared ? shared_gate_scales : gate_scales) + group_row,
                                     (shared ? shared_gate_biases : gate_biases) + group_row,
                                     kHiddenDimension, lane);
    const float up =
        q4_dot_packed(input, (shared ? shared_up_words : up_words) + word_row,
                      (shared ? shared_up_scales : up_scales) + group_row,
                      (shared ? shared_up_biases : up_biases) + group_row, kHiddenDimension, lane);
    if (lane == 0u) {
        hidden_out[sg] = static_cast<bfloat>((gate / (1.0f + exp(-gate))) * up);
    }
}

kernel void grouped_down_res(
    device const bfloat* hidden_in [[buffer(0)]], device const uint* ids [[buffer(1)]],
    device const float* coefficients [[buffer(2)]],
    device const float* shared_coefficient [[buffer(3)]],
    device const uint* down_words [[buffer(4)]], device const bfloat* down_scales [[buffer(5)]],
    device const bfloat* down_biases [[buffer(6)]],
    device const uint* shared_down_words [[buffer(7)]],
    device const bfloat* shared_down_scales [[buffer(8)]],
    device const bfloat* shared_down_biases [[buffer(9)]], device bfloat* moe_out [[buffer(10)]],
    device const bfloat* residual [[buffer(11)]], device bfloat* layer_out [[buffer(12)]],
    uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tid >> 5u;
    if (row >= kHiddenDimension) {
        return;
    }
    float total = 0.0f;
    for (uint slot = 0; slot < kMoeActiveExperts + 1u; ++slot) {
        const bool shared = slot == kMoeActiveExperts;
        const uint expert = shared ? 0u : ids[slot];
        if (!shared && expert == 0xffffffffu) {
            continue;
        }
        const ulong word_row = shared ? ulong(row) * ulong(kMoeExpertDimension / 8u)
                                      : (ulong(expert) * ulong(kHiddenDimension) + ulong(row)) *
                                            ulong(kMoeExpertDimension / 8u);
        const ulong group_row = shared ? ulong(row) * ulong(kMoeExpertDimension / kQ4GroupSize)
                                       : (ulong(expert) * ulong(kHiddenDimension) + ulong(row)) *
                                             ulong(kMoeExpertDimension / kQ4GroupSize);
        const float part = q4_dot_packed(hidden_in + slot * kMoeExpertDimension,
                                         (shared ? shared_down_words : down_words) + word_row,
                                         (shared ? shared_down_scales : down_scales) + group_row,
                                         (shared ? shared_down_biases : down_biases) + group_row,
                                         kMoeExpertDimension, lane);
        if (lane == 0u) {
            total += (shared ? shared_coefficient[0] : coefficients[slot]) * part;
        }
    }
    if (lane == 0u) {
        const bfloat moe = static_cast<bfloat>(total);
        moe_out[row] = moe;
        layer_out[row] = static_cast<bfloat>(float(residual[row]) + float(moe));
    }
}

// Batched MoE with expert-union weight sharing. Union table layout in
// u32 units: [0,256) row masks per expert; [256,768) slot_of bytes
// (expert*8 + row, 0xff = not routed); [768] routed-union count;
// [769,...) the compacted ascending work list.

kernel void moe_union_build(device const uint* ids [[buffer(0)]],
                            device uint* table [[buffer(1)]],
                            constant uint& row_count [[buffer(2)]],
                            constant uint& ids_stride_words [[buffer(3)]],
                            uint e [[thread_position_in_grid]]) {
    if (e >= kMoeExperts) {
        return;
    }
    device uchar* slots = reinterpret_cast<device uchar*>(table + kMoeExperts);
    uint mask = 0u;
    for (uint r = 0u; r < row_count; ++r) {
        uchar slot = 0xffu;
        for (uint k = 0u; k < kMoeActiveExperts; ++k) {
            if (ids[r * ids_stride_words + k] == e) {
                slot = uchar(k);
                mask |= (1u << r);
            }
        }
        slots[e * 8u + r] = slot;
    }
    table[e] = mask;
}

kernel void moe_union_compact(device uint* table [[buffer(0)]],
                              uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) {
        return;
    }
    uint count = 0u;
    for (uint e = 0u; e < kMoeExperts; ++e) {
        if (table[e] != 0u) {
            table[769u + count] = e;
            ++count;
        }
    }
    table[768u] = count;
}

kernel void grouped_upgate_union_ms(
    device const bfloat* input [[buffer(0)]], device const uint* table [[buffer(1)]],
    device const uint* gate_words [[buffer(2)]], device const bfloat* gate_scales [[buffer(3)]],
    device const bfloat* gate_biases [[buffer(4)]], device const uint* up_words [[buffer(5)]],
    device const bfloat* up_scales [[buffer(6)]], device const bfloat* up_biases [[buffer(7)]],
    device const uint* shared_gate_words [[buffer(8)]],
    device const bfloat* shared_gate_scales [[buffer(9)]],
    device const bfloat* shared_gate_biases [[buffer(10)]],
    device const uint* shared_up_words [[buffer(11)]],
    device const bfloat* shared_up_scales [[buffer(12)]],
    device const bfloat* shared_up_biases [[buffer(13)]],
    device bfloat* hidden_out [[buffer(14)]], constant uint& row_count [[buffer(15)]],
    constant uint& input_stride [[buffer(16)]], constant uint& hidden_stride [[buffer(17)]],
    uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint sg = tid >> 5u;
    const uint item = sg / kMoeExpertDimension;
    const uint row = sg % kMoeExpertDimension;
    const uint union_count = table[768u];
    if (item > union_count) {
        return;
    }
    const bool shared = item == union_count;
    const uint eidx = shared ? 0u : table[769u + item];
    device const uchar* slots = reinterpret_cast<device const uchar*>(table + kMoeExperts);
    uint offsets[kQ4BatchRowsMax];
    uint outs[kQ4BatchRowsMax];
    uint count = 0u;
    if (shared) {
        for (uint r = 0u; r < row_count; ++r) {
            offsets[count] = r * input_stride;
            outs[count] =
                r * hidden_stride + kMoeActiveExperts * kMoeExpertDimension + row;
            ++count;
        }
    } else {
        const uint mask = table[eidx];
        for (uint r = 0u; r < row_count; ++r) {
            if ((mask & (1u << r)) == 0u) {
                continue;
            }
            offsets[count] = r * input_stride;
            outs[count] = r * hidden_stride +
                          uint(slots[eidx * 8u + r]) * kMoeExpertDimension + row;
            ++count;
        }
    }
    const ulong word_row = shared ? ulong(row) * ulong(kQuantWordsPerRow)
                                  : (ulong(eidx) * ulong(kMoeExpertDimension) + ulong(row)) *
                                        ulong(kQuantWordsPerRow);
    const ulong group_row =
        shared ? ulong(row) * ulong(kGroupsPerRow)
               : (ulong(eidx) * ulong(kMoeExpertDimension) + ulong(row)) * ulong(kGroupsPerRow);
    float gate[kQ4BatchRowsMax];
    float up[kQ4BatchRowsMax];
    q4_dot_packed_ms(input, offsets, count,
                     (shared ? shared_gate_words : gate_words) + word_row,
                     (shared ? shared_gate_scales : gate_scales) + group_row,
                     (shared ? shared_gate_biases : gate_biases) + group_row,
                     kHiddenDimension, lane, gate);
    q4_dot_packed_ms(input, offsets, count,
                     (shared ? shared_up_words : up_words) + word_row,
                     (shared ? shared_up_scales : up_scales) + group_row,
                     (shared ? shared_up_biases : up_biases) + group_row,
                     kHiddenDimension, lane, up);
    if (lane == 0u) {
        for (uint i = 0u; i < count; ++i) {
            hidden_out[outs[i]] =
                static_cast<bfloat>((gate[i] / (1.0f + exp(-gate[i]))) * up[i]);
        }
    }
}

kernel void grouped_down_parts_union_ms(
    device const bfloat* hidden_in [[buffer(0)]], device const uint* table [[buffer(1)]],
    device const uint* down_words [[buffer(2)]], device const bfloat* down_scales [[buffer(3)]],
    device const bfloat* down_biases [[buffer(4)]],
    device const uint* shared_down_words [[buffer(5)]],
    device const bfloat* shared_down_scales [[buffer(6)]],
    device const bfloat* shared_down_biases [[buffer(7)]],
    device float* parts [[buffer(8)]], constant uint& row_count [[buffer(9)]],
    constant uint& hidden_stride [[buffer(10)]],
    uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint sg = tid >> 5u;
    const uint item = sg / kHiddenDimension;
    const uint row = sg % kHiddenDimension;
    const uint union_count = table[768u];
    if (item > union_count) {
        return;
    }
    const bool shared = item == union_count;
    const uint eidx = shared ? 0u : table[769u + item];
    device const uchar* slots = reinterpret_cast<device const uchar*>(table + kMoeExperts);
    uint offsets[kQ4BatchRowsMax];
    uint outs[kQ4BatchRowsMax];
    uint count = 0u;
    if (shared) {
        for (uint r = 0u; r < row_count; ++r) {
            offsets[count] = r * hidden_stride + kMoeActiveExperts * kMoeExpertDimension;
            outs[count] = (r * (kMoeActiveExperts + 1u) + kMoeActiveExperts) *
                              kHiddenDimension + row;
            ++count;
        }
    } else {
        const uint mask = table[eidx];
        for (uint r = 0u; r < row_count; ++r) {
            if ((mask & (1u << r)) == 0u) {
                continue;
            }
            const uint slot = uint(slots[eidx * 8u + r]);
            offsets[count] = r * hidden_stride + slot * kMoeExpertDimension;
            outs[count] = (r * (kMoeActiveExperts + 1u) + slot) * kHiddenDimension + row;
            ++count;
        }
    }
    const ulong word_row = shared ? ulong(row) * ulong(kMoeExpertDimension / 8u)
                                  : (ulong(eidx) * ulong(kHiddenDimension) + ulong(row)) *
                                        ulong(kMoeExpertDimension / 8u);
    const ulong group_row = shared ? ulong(row) * ulong(kMoeExpertDimension / kQ4GroupSize)
                                   : (ulong(eidx) * ulong(kHiddenDimension) + ulong(row)) *
                                         ulong(kMoeExpertDimension / kQ4GroupSize);
    float part[kQ4BatchRowsMax];
    q4_dot_packed_ms(hidden_in, offsets, count,
                     (shared ? shared_down_words : down_words) + word_row,
                     (shared ? shared_down_scales : down_scales) + group_row,
                     (shared ? shared_down_biases : down_biases) + group_row,
                     kMoeExpertDimension, lane, part);
    if (lane == 0u) {
        for (uint i = 0u; i < count; ++i) {
            parts[outs[i]] = part[i];
        }
    }
}

// Per-row combine in exact slot order: identical accumulation sequence to
// grouped_down_res, consuming the float parts verbatim.
kernel void grouped_down_combine_ms(
    device const float* parts [[buffer(0)]], device const uint* ids [[buffer(1)]],
    device const float* coefficients [[buffer(2)]],
    device const float* shared_coefficients [[buffer(3)]],
    device bfloat* moe_out [[buffer(4)]], device const bfloat* residual [[buffer(5)]],
    device bfloat* layer_out [[buffer(6)]], constant uint& row_count [[buffer(7)]],
    constant uint& ids_stride_words [[buffer(8)]],
    constant uint& coeff_stride_floats [[buffer(9)]],
    constant uint& stream_stride [[buffer(10)]],
    constant uint& layer_offset_elems [[buffer(11)]],
    uint tid [[thread_position_in_grid]]) {
    const uint r = tid / kHiddenDimension;
    const uint row = tid % kHiddenDimension;
    if (r >= row_count) {
        return;
    }
    float total = 0.0f;
    for (uint slot = 0u; slot < kMoeActiveExperts + 1u; ++slot) {
        const bool shared = slot == kMoeActiveExperts;
        if (!shared && ids[r * ids_stride_words + slot] == 0xffffffffu) {
            continue;
        }
        const float part =
            parts[(r * (kMoeActiveExperts + 1u) + slot) * kHiddenDimension + row];
        total += (shared ? shared_coefficients[r]
                         : coefficients[r * coeff_stride_floats + slot]) *
                 part;
    }
    const uint stream_at = r * stream_stride + layer_offset_elems + row;
    const bfloat moe = static_cast<bfloat>(total);
    moe_out[stream_at] = moe;
    layer_out[stream_at] =
        static_cast<bfloat>(float(residual[stream_at]) + float(moe));
}

kernel void residual_rms_ms(device const bfloat* a [[buffer(0)]],
                            device const bfloat* b [[buffer(1)]],
                            device bfloat* residual_out [[buffer(2)]],
                            device const bfloat* weight [[buffer(3)]],
                            device bfloat* normed_out [[buffer(4)]],
                            constant uint& a_stride [[buffer(5)]],
                            constant uint& lane_stride [[buffer(6)]],
                            constant uint& normed_stride [[buffer(7)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint2 thread_in_group2 [[thread_position_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint thread_in_group = thread_in_group2.x;
    device const bfloat* a_row = a + ulong(tg.y) * a_stride;
    device const bfloat* b_row = b + ulong(tg.y) * lane_stride;
    device bfloat* residual_row = residual_out + ulong(tg.y) * lane_stride;
    device bfloat* normed_row = normed_out + ulong(tg.y) * normed_stride;
    threadgroup float inverse_shared[1];
    threadgroup float stage[32];
    const uint base = thread_in_group * kRmsValuesPerThread;
    bfloat rounded[4];
    float acc = 0.0f;
    for (uint i = 0; i < kRmsValuesPerThread; ++i) {
        const uint element = base + i;
        rounded[i] = static_cast<bfloat>(float(a_row[element]) + float(b_row[element]));
        residual_row[element] = rounded[i];
        const float value = float(rounded[i]);
        acc += value * value;
    }
    acc = simd_sum(acc);
    if (simdgroup == 0u) {
        stage[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        stage[simdgroup] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        acc = simd_sum(stage[lane]);
        if (lane == 0u) {
            inverse_shared[0] = precise::rsqrt(acc / float(kHiddenDimension) + kRmsNormEpsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inverse = inverse_shared[0];
    for (uint i = 0; i < kRmsValuesPerThread; ++i) {
        const uint element = base + i;
        normed_row[element] =
            weight[element] * static_cast<bfloat>(float(rounded[i]) * inverse);
    }
}

kernel void router_q8_ms(device const bfloat* input [[buffer(0)]],
                         device const uint* routed_words [[buffer(1)]],
                         device const bfloat* routed_scales [[buffer(2)]],
                         device const bfloat* routed_biases [[buffer(3)]],
                         device const uint* shared_words [[buffer(4)]],
                         device const bfloat* shared_scales [[buffer(5)]],
                         device const bfloat* shared_biases [[buffer(6)]],
                         device float* logits [[buffer(7)]],
                         constant uint& row_count [[buffer(8)]],
                         constant uint& input_stride [[buffer(9)]],
                         constant uint& logits_stride [[buffer(10)]],
                         uint tid [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tid >> 5u;
    if (row >= kMoeExperts + 1u) {
        return;
    }
    const bool shared = row == kMoeExperts;
    const ulong word_row = shared ? 0ul : ulong(row) * ulong(kQuantWordsPerRow * 2u);
    const ulong group_row = shared ? 0ul : ulong(row) * ulong(kGroupsPerRow);
    float batch[kQ4BatchRowsMax];
    q8_dot_ms(input, ulong(input_stride), row_count,
              (shared ? shared_words : routed_words) + word_row,
              (shared ? shared_scales : routed_scales) + group_row,
              (shared ? shared_biases : routed_biases) + group_row,
              kHiddenDimension, lane, batch);
    if (lane == 0u) {
        for (uint r = 0u; r < row_count; ++r) {
            logits[r * logits_stride + row] = float(static_cast<bfloat>(batch[r]));
        }
    }
}

kernel void grouped_upgate_rows_ms(
    device const bfloat* input [[buffer(0)]], device const uint* ids [[buffer(1)]],
    device const uint* gate_words [[buffer(2)]], device const bfloat* gate_scales [[buffer(3)]],
    device const bfloat* gate_biases [[buffer(4)]], device const uint* up_words [[buffer(5)]],
    device const bfloat* up_scales [[buffer(6)]], device const bfloat* up_biases [[buffer(7)]],
    device const uint* shared_gate_words [[buffer(8)]],
    device const bfloat* shared_gate_scales [[buffer(9)]],
    device const bfloat* shared_gate_biases [[buffer(10)]],
    device const uint* shared_up_words [[buffer(11)]],
    device const bfloat* shared_up_scales [[buffer(12)]],
    device const bfloat* shared_up_biases [[buffer(13)]], device bfloat* hidden_out [[buffer(14)]],
    constant uint& input_stride [[buffer(15)]], constant uint& ids_stride [[buffer(16)]],
    constant uint& hidden_stride [[buffer(17)]],
    uint2 tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint sg = tid.x >> 5u;
    const uint batch_row = tid.y;
    if (sg >= (kMoeActiveExperts + 1u) * kMoeExpertDimension) {
        return;
    }
    device const bfloat* in_row = input + ulong(batch_row) * input_stride;
    device const uint* ids_row = ids + ulong(batch_row) * ids_stride;
    device bfloat* out_row = hidden_out + ulong(batch_row) * hidden_stride;
    const uint slot = sg / kMoeExpertDimension;
    const uint row = sg % kMoeExpertDimension;
    const bool shared = slot == kMoeActiveExperts;
    const uint expert = shared ? 0u : ids_row[slot];
    if (!shared && expert == 0xffffffffu) {
        if (lane == 0u) {
            out_row[sg] = bfloat(0.0f);
        }
        return;
    }
    const ulong word_row = shared ? ulong(row) * ulong(kQuantWordsPerRow)
                                  : (ulong(expert) * ulong(kMoeExpertDimension) + ulong(row)) *
                                        ulong(kQuantWordsPerRow);
    const ulong group_row =
        shared ? ulong(row) * ulong(kGroupsPerRow)
               : (ulong(expert) * ulong(kMoeExpertDimension) + ulong(row)) * ulong(kGroupsPerRow);
    const float gate = q4_dot_packed(in_row, (shared ? shared_gate_words : gate_words) + word_row,
                                     (shared ? shared_gate_scales : gate_scales) + group_row,
                                     (shared ? shared_gate_biases : gate_biases) + group_row,
                                     kHiddenDimension, lane);
    const float up =
        q4_dot_packed(in_row, (shared ? shared_up_words : up_words) + word_row,
                      (shared ? shared_up_scales : up_scales) + group_row,
                      (shared ? shared_up_biases : up_biases) + group_row, kHiddenDimension, lane);
    if (lane == 0u) {
        out_row[sg] = static_cast<bfloat>((gate / (1.0f + exp(-gate))) * up);
    }
}

kernel void grouped_down_res_rows_ms(
    device const bfloat* hidden_in [[buffer(0)]], device const uint* ids [[buffer(1)]],
    device const float* coefficients [[buffer(2)]],
    device const float* shared_coefficient [[buffer(3)]],
    device const uint* down_words [[buffer(4)]], device const bfloat* down_scales [[buffer(5)]],
    device const bfloat* down_biases [[buffer(6)]],
    device const uint* shared_down_words [[buffer(7)]],
    device const bfloat* shared_down_scales [[buffer(8)]],
    device const bfloat* shared_down_biases [[buffer(9)]], device bfloat* moe_out [[buffer(10)]],
    device const bfloat* residual [[buffer(11)]], device bfloat* layer_out [[buffer(12)]],
    constant uint& hidden_stride [[buffer(13)]], constant uint& ids_stride [[buffer(14)]],
    constant uint& coeff_stride [[buffer(15)]], constant uint& stream_stride [[buffer(16)]],
    uint2 tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tid.x >> 5u;
    const uint batch_row = tid.y;
    if (row >= kHiddenDimension) {
        return;
    }
    device const bfloat* hidden_row = hidden_in + ulong(batch_row) * hidden_stride;
    device const uint* ids_row = ids + ulong(batch_row) * ids_stride;
    device const float* coeff_row = coefficients + ulong(batch_row) * coeff_stride;
    const ulong stream_at = ulong(batch_row) * stream_stride + row;
    float total = 0.0f;
    for (uint slot = 0; slot < kMoeActiveExperts + 1u; ++slot) {
        const bool shared = slot == kMoeActiveExperts;
        const uint expert = shared ? 0u : ids_row[slot];
        if (!shared && expert == 0xffffffffu) {
            continue;
        }
        const ulong word_row = shared ? ulong(row) * ulong(kMoeExpertDimension / 8u)
                                      : (ulong(expert) * ulong(kHiddenDimension) + ulong(row)) *
                                            ulong(kMoeExpertDimension / 8u);
        const ulong group_row = shared ? ulong(row) * ulong(kMoeExpertDimension / kQ4GroupSize)
                                       : (ulong(expert) * ulong(kHiddenDimension) + ulong(row)) *
                                             ulong(kMoeExpertDimension / kQ4GroupSize);
        const float part = q4_dot_packed(hidden_row + slot * kMoeExpertDimension,
                                         (shared ? shared_down_words : down_words) + word_row,
                                         (shared ? shared_down_scales : down_scales) + group_row,
                                         (shared ? shared_down_biases : down_biases) + group_row,
                                         kMoeExpertDimension, lane);
        if (lane == 0u) {
            total += (shared ? shared_coefficient[batch_row] : coeff_row[slot]) * part;
        }
    }
    if (lane == 0u) {
        const bfloat moe = static_cast<bfloat>(total);
        moe_out[stream_at] = moe;
        layer_out[stream_at] = static_cast<bfloat>(float(residual[stream_at]) + float(moe));
    }
}
