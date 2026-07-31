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
